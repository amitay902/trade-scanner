#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel-Up Scanner — daily scan for US mid-cap+ stocks in an ascending channel,
currently sitting near the channel bottom (pullback within an uptrend).

Strategy (inferred from manual finviz workflow):
  - Uptrend: linear-regression channel over ~3-5 months, positive slope, good fit (R2)
  - Entry zone: price in the lower third of the channel
  - Pullback: price below SMA20 (and ideally SMA50), RSI(14) <= ~45-47
  - Trend intact: price above SMA200, channel floor not broken
  - TP: projected channel top (capped near 52w high) ; SL: ~1.2x daily ATR
    (wider, ~1.7x ATR, for high-volatility names where ATR% > 5)

Outputs data/picks.json (+ dated copy in data/history/) and sends a ntfy push.
Also reads data/positions.json (synced from the PWA) and emits TP/SL/earnings
proximity alerts for open positions.
"""

import json
import math
import os
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT
HIST = ROOT / "history"

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").strip()
APP_URL = os.environ.get("APP_URL", "").strip()

# ---- tunables -------------------------------------------------------------
MIN_MARKET_CAP = 2_000_000_000          # mid-cap and above
MIN_DOLLAR_VOL = 10_000_000             # avg daily $ volume (20d)
LOOKBACKS = [45, 60, 75, 90, 110]       # candidate channel lengths (trading days)
MIN_R2 = 0.55                           # channel fit quality
MIN_TREND_GAIN = 0.08                   # regression rise over the window
MAX_CHANNEL_POS = 0.35                  # lower third of the channel
MIN_CHANNEL_POS_CAND = -0.15            # slightly below the floor still shown as candidate
MIN_CHANNEL_POS_STRICT = -0.05          # strict picks must be essentially inside the channel
EARNINGS_MIN_DAYS = 6                   # strict picks must not have earnings within N days
MAX_RSI_CAND = 47.0
MAX_RSI_PICK = 45.0
MAX_PICKS = 6
ALERT_PROXIMITY = 0.015                 # within 1.5% of TP/SL
EARNINGS_ALERT_DAYS = 5
# ---------------------------------------------------------------------------


def log(*a):
    print(*a, flush=True)


# ---------------------------- universe -------------------------------------

def universe_from_nasdaq():
    """Full US-listed universe with market caps from the NASDAQ screener API."""
    import requests
    url = "https://api.nasdaq.com/api/screener/stocks"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    rows = []
    r = requests.get(url, params={"tableonly": "true", "limit": 25000, "download": "true"},
                     headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()["data"]["rows"]
    for row in data:
        try:
            cap = float(row.get("marketCap") or 0)
        except ValueError:
            continue
        country = (row.get("country") or "").strip()
        sym = (row.get("symbol") or "").strip()
        if not sym or "^" in sym or "/" in sym:
            continue
        if cap >= MIN_MARKET_CAP and country in ("United States", ""):
            rows.append({
                "ticker": sym.replace(".", "-"),
                "name": (row.get("name") or "").strip(),
                "sector": (row.get("sector") or "").strip() or "Unknown",
                "cap": cap,
            })
    if len(rows) < 300:
        raise RuntimeError(f"nasdaq universe too small: {len(rows)}")
    return rows


def universe_from_wikipedia():
    """Fallback: S&P 500 + S&P 400 constituents."""
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (channel-up-scanner)"}
    rows = []
    for url in (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    ):
        html = requests.get(url, headers=headers, timeout=60).text
        for table in pd.read_html(html):
            cols = [str(c).lower() for c in table.columns]
            if any("symbol" in c for c in cols):
                sym_col = table.columns[[i for i, c in enumerate(cols) if "symbol" in c][0]]
                name_col = None
                for cand in ("Security", "Company"):
                    if cand in table.columns:
                        name_col = cand
                sec_col = None
                for cand in ("GICS Sector", "GICS\xa0Sector"):
                    if cand in table.columns:
                        sec_col = cand
                for _, r in table.iterrows():
                    sym = str(r[sym_col]).strip().replace(".", "-")
                    if sym and sym != "nan":
                        rows.append({
                            "ticker": sym,
                            "name": str(r[name_col]) if name_col else sym,
                            "sector": str(r[sec_col]) if sec_col else "Unknown",
                            "cap": None,
                        })
                break
    if len(rows) < 300:
        raise RuntimeError("wikipedia universe too small")
    return rows


def load_universe():
    try:
        rows = universe_from_nasdaq()
        log(f"universe: nasdaq api -> {len(rows)} tickers")
    except Exception as e:
        log(f"nasdaq universe failed ({e}); falling back to wikipedia")
        try:
            rows = universe_from_wikipedia()
            log(f"universe: wikipedia -> {len(rows)} tickers")
        except Exception as e2:
            cache = DATA / "universe.json"
            if cache.exists():
                rows = json.loads(cache.read_text())["rows"]
                log(f"universe: cached -> {len(rows)} tickers")
            else:
                raise RuntimeError(f"no universe available: {e2}")
    # de-dup, cache
    seen, out = set(), []
    for r in rows:
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            out.append(r)
    (DATA / "universe.json").write_text(
        json.dumps({"updated": dt.date.today().isoformat(), "rows": out}))
    return out


# ---------------------------- indicators -----------------------------------

def rsi_wilder(close: pd.Series, n=14) -> float:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    r = 100 - 100 / (1 + rs)
    return float(r.iloc[-1])


def atr_wilder(high, low, close, n=14) -> float:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def best_channel(close: pd.Series):
    """Try several lookbacks; return the best-fit ascending regression channel."""
    best = None
    for lb in LOOKBACKS:
        if len(close) < lb + 5:
            continue
        y = close.iloc[-lb:].to_numpy(dtype=float)
        x = np.arange(lb, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * x + intercept
        ss_res = float(((y - fit) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        gain = (fit[-1] - fit[0]) / fit[0] if fit[0] > 0 else 0.0
        if slope <= 0 or gain < MIN_TREND_GAIN or r2 < MIN_R2:
            continue
        resid = y - fit
        upper_off = float(np.quantile(resid, 0.97))
        lower_off = float(np.quantile(resid, 0.03))
        width = upper_off - lower_off
        if width <= 0 or width / y[-1] < 0.02:
            continue
        pos = (resid[-1] - lower_off) / width
        wp = width / y[-1]
        # prefer tight fit; penalize very wide channels (not tradeable short-term)
        score = r2 + min(gain, 0.2) - max(0.0, wp - 0.15) * 1.5
        cand = {
            "lookback": lb, "slope": slope, "r2": round(r2, 3),
            "gain": round(gain, 4), "width_pct": round(width / y[-1], 4),
            "pos": round(float(pos), 3),
            "upper_now": float(fit[-1] + upper_off),
            "lower_now": float(fit[-1] + lower_off),
            "score": score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# ---------------------------- price data -----------------------------------

def download_prices(tickers, period="10mo"):
    import yfinance as yf
    frames = {}
    CH = 150
    for i in range(0, len(tickers), CH):
        chunk = tickers[i:i + CH]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d",
                                 group_by="ticker", auto_adjust=True,
                                 threads=True, progress=False)
                break
            except Exception as e:
                log(f"chunk {i} attempt {attempt} failed: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            continue
        if len(chunk) == 1:
            frames[chunk[0]] = df
        else:
            for t in chunk:
                try:
                    sub = df[t].dropna(how="all")
                    if len(sub) > 0:
                        frames[t] = sub
                except KeyError:
                    pass
        log(f"downloaded {min(i+CH, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    return frames


def enrich(ticker):
    """Earnings date + analyst target for a finalist (best effort)."""
    import yfinance as yf
    out = {"earnings_date": None, "target_mean": None, "name": None, "sector": None}
    try:
        tk = yf.Ticker(ticker)
        try:
            cal = tk.calendar
            edates = None
            if isinstance(cal, dict):
                edates = cal.get("Earnings Date")
            if edates:
                d0 = edates[0] if isinstance(edates, (list, tuple)) else edates
                if hasattr(d0, "isoformat"):
                    out["earnings_date"] = d0.isoformat()[:10]
        except Exception:
            pass
        try:
            info = tk.info or {}
            out["target_mean"] = info.get("targetMeanPrice")
            out["name"] = info.get("shortName") or info.get("longName")
            out["sector"] = info.get("sector")
        except Exception:
            pass
    except Exception:
        pass
    return out


# ---------------------------- analysis -------------------------------------

def analyze(ticker, df, meta):
    df = df.dropna(subset=["Close"])
    if len(df) < 120:
        return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(close.iloc[-1])
    if price < 3:
        return None
    dollar_vol = float((close * vol).tail(20).mean())
    if dollar_vol < MIN_DOLLAR_VOL:
        return None

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi = rsi_wilder(close)
    atr = atr_wilder(high, low, close)
    atr_pct = atr / price

    ch = best_channel(close)
    if ch is None:
        return None

    # gate: pullback inside an intact rising channel
    if ch["pos"] > MAX_CHANNEL_POS or ch["pos"] < MIN_CHANNEL_POS_CAND:
        return None
    if rsi > MAX_RSI_CAND:
        return None
    if price >= sma20:
        return None
    if sma200 is not None and price < sma200:
        return None

    tail = close.tail(252)
    hi52 = float(tail.max())
    hi52_age = len(tail) - 1 - int(np.argmax(tail.to_numpy()))  # trading days since the high

    # --- TP: today's channel top. Cap just under the 52w high only when that high
    # is OLD overhead resistance (set before this channel leg) — a recent high made
    # inside the channel is expected to be exceeded as the channel rises.
    tp_price = ch["upper_now"]
    if hi52 > price and hi52 < tp_price and hi52_age > ch["lookback"]:
        tp_price = hi52 * 0.998
    tp_pct = tp_price / price - 1
    if tp_pct < 0.04:
        return None
    # cap TP by volatility: short-term swing target, not the full 110-day channel width
    tp_cap = max(0.05, min(0.15, 3.5 * atr_pct))
    tp_pct = min(tp_pct, tp_cap)
    tp_price = price * (1 + tp_pct)

    # --- SL: volatility-scaled
    k = 1.7 if atr_pct > 0.05 else 1.2
    sl_pct = max(k * atr_pct, 0.02)
    sl_pct = min(sl_pct, 0.15)
    sl_price = price * (1 - sl_pct)
    # keep the stop below the channel floor when it's close
    if ch["lower_now"] < price and (price - ch["lower_now"]) / price < sl_pct:
        sl_price = min(sl_price, ch["lower_now"] * 0.995)
        sl_pct = 1 - sl_price / price

    rr = tp_pct / sl_pct if sl_pct > 0 else 0

    strict = (rsi <= MAX_RSI_PICK) and (price < sma50) and (ch["pos"] >= MIN_CHANNEL_POS_STRICT)
    pos_quality = 1.0 - min(1.0, abs(ch["pos"] - 0.12) / 0.5)   # best: just above the floor
    score = (
        (MAX_RSI_CAND - rsi) * 1.0          # oversold helps...
        + pos_quality * 12                   # ...but inside the channel matters more
        + ch["r2"] * 20                      # channel quality
        + min(rr, 4) * 6                     # risk/reward
        + (10 if strict else 0)
    )

    return {
        "ticker": ticker,
        "name": meta.get("name") or ticker,
        "sector": meta.get("sector") or "Unknown",
        "cap": meta.get("cap"),
        "price": round(price, 2),
        "rsi": round(rsi, 1),
        "atr_pct": round(atr_pct, 4),
        "sma20_dist": round(price / sma20 - 1, 4),
        "sma50_dist": round(price / sma50 - 1, 4),
        "sma200_dist": round(price / sma200 - 1, 4) if sma200 else None,
        "hi52_dist": round(price / hi52 - 1, 4),
        "dollar_vol": round(dollar_vol),
        "channel": {k2: ch[k2] for k2 in ("lookback", "r2", "gain", "width_pct", "pos")},
        "tp_price": round(tp_price, 2),
        "tp_pct": round(tp_pct, 4),
        "sl_price": round(sl_price, 2),
        "sl_pct": round(sl_pct, 4),
        "rr": round(rr, 2),
        "strict": strict,
        "score": round(score, 1),
        "spark": [round(float(v), 3) for v in close.tail(40).tolist()],
    }


# ---------------------------- positions & alerts ---------------------------

def load_positions():
    p = DATA / "positions.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("positions", [])
        except Exception:
            return []
    return []


def position_alerts(positions, quotes, earnings_map):
    alerts = []
    today = dt.date.today()
    for pos in positions:
        if pos.get("status") != "open":
            continue
        t = pos.get("ticker")
        q = quotes.get(t)
        if not q:
            continue
        price = q["price"]
        entry = float(pos.get("entry_price") or 0)
        tp = float(pos.get("tp_price") or 0)
        sl = float(pos.get("sl_price") or 0)
        chg = (price / entry - 1) if entry else 0
        if tp:
            if price >= tp:
                alerts.append({"ticker": t, "type": "tp_hit", "price": price,
                               "level": tp, "msg": f"{t} הגיעה ליעד הרווח ({price:.2f} ≥ {tp:.2f}) — שקול לממש"})
            elif (tp - price) / price <= ALERT_PROXIMITY:
                alerts.append({"ticker": t, "type": "tp_near", "price": price,
                               "level": tp, "msg": f"{t} מתקרבת ליעד הרווח: {price:.2f} מול {tp:.2f} ({chg:+.1%} מהכניסה)"})
        if sl:
            if price <= sl:
                alerts.append({"ticker": t, "type": "sl_hit", "price": price,
                               "level": sl, "msg": f"{t} פרצה את הסטופ ({price:.2f} ≤ {sl:.2f}) — בדוק את הפוזיציה"})
            elif (price - sl) / price <= ALERT_PROXIMITY:
                alerts.append({"ticker": t, "type": "sl_near", "price": price,
                               "level": sl, "msg": f"{t} מתקרבת לסטופ: {price:.2f} מול {sl:.2f} ({chg:+.1%} מהכניסה)"})
        ed = earnings_map.get(t) or pos.get("earnings_date")
        if ed:
            try:
                d = dt.date.fromisoformat(str(ed)[:10])
                days = (d - today).days
                if 0 <= days <= EARNINGS_ALERT_DAYS:
                    alerts.append({"ticker": t, "type": "earnings", "price": price,
                                   "days": days,
                                   "msg": f"{t}: דוחות בעוד {days} ימים ({d.isoformat()}) — החלט אם להישאר"})
            except ValueError:
                pass
    return alerts


# ---------------------------- notify ---------------------------------------

def notify(title, message, tags=None, priority=3):
    if not NTFY_TOPIC:
        log("no NTFY_TOPIC set; skipping notification")
        return
    import requests
    payload = {"topic": NTFY_TOPIC, "title": title, "message": message,
               "priority": priority, "tags": tags or ["chart_with_upwards_trend"]}
    if APP_URL:
        payload["click"] = APP_URL
    try:
        requests.post(NTFY_SERVER, json=payload, timeout=30).raise_for_status()
        log("ntfy notification sent")
    except Exception as e:
        log(f"ntfy failed: {e}")


# ---------------------------- main -----------------------------------------

def main():
    DATA.mkdir(exist_ok=True)
    HIST.mkdir(exist_ok=True)

    universe = load_universe()
    meta = {u["ticker"]: u for u in universe}
    tickers = [u["ticker"] for u in universe]

    positions = load_positions()
    pos_tickers = [p["ticker"] for p in positions if p.get("status") == "open"]
    for t in pos_tickers:
        if t not in meta:
            meta[t] = {"ticker": t, "name": t, "sector": "Unknown", "cap": None}
            tickers.append(t)

    log(f"downloading prices for {len(tickers)} tickers...")
    frames = download_prices(tickers)
    log(f"got data for {len(frames)} tickers")

    quotes = {}
    for t, df in frames.items():
        d = df.dropna(subset=["Close"])
        if len(d) >= 2:
            quotes[t] = {"price": round(float(d['Close'].iloc[-1]), 2),
                         "prev_close": round(float(d['Close'].iloc[-2]), 2)}

    candidates = []
    for t, df in frames.items():
        try:
            res = analyze(t, df, meta.get(t, {}))
            if res:
                candidates.append(res)
        except Exception as e:
            log(f"analyze {t} failed: {e}")
    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[:20]

    log(f"{len(candidates)} candidates; enriching finalists + positions...")
    earnings_map = {}
    for c in candidates:
        extra = enrich(c["ticker"])
        if extra["earnings_date"]:
            c["earnings_date"] = extra["earnings_date"]
            earnings_map[c["ticker"]] = extra["earnings_date"]
            try:
                c["days_to_earnings"] = (dt.date.fromisoformat(extra["earnings_date"]) - dt.date.today()).days
            except ValueError:
                pass
        if extra["target_mean"]:
            c["target_mean"] = extra["target_mean"]
        if (c["name"] == c["ticker"]) and extra["name"]:
            c["name"] = extra["name"]
        if c["sector"] == "Unknown" and extra["sector"]:
            c["sector"] = extra["sector"]
        time.sleep(0.4)
    for t in pos_tickers:
        if t not in earnings_map:
            extra = enrich(t)
            if extra["earnings_date"]:
                earnings_map[t] = extra["earnings_date"]
            time.sleep(0.4)

    # earnings too close -> not a channel trade, it's an earnings bet
    for c in candidates:
        d = c.get("days_to_earnings")
        if d is not None and 0 <= d < EARNINGS_MIN_DAYS:
            c["strict"] = False
            c["earnings_soon"] = True

    picks = [c for c in candidates if c["strict"]][:MAX_PICKS]
    if len(picks) < 3:
        for c in candidates:
            if c not in picks:
                picks.append(c)
            if len(picks) >= 3:
                break
    pick_set = {c["ticker"] for c in picks}
    for c in candidates:
        c["is_pick"] = c["ticker"] in pick_set

    alerts = position_alerts(positions, quotes, earnings_map)

    prev = {}
    prev_file = DATA / "picks.json"
    if prev_file.exists():
        try:
            prev = json.loads(prev_file.read_text())
        except Exception:
            pass
    prev_picks = set(prev.get("picks", []))
    new_picks = [t for t in pick_set if t not in prev_picks]

    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "scan_date": dt.date.today().isoformat(),
        "universe_size": len(universe),
        "analyzed": len(frames),
        "picks": sorted(pick_set),
        "new_picks": sorted(new_picks),
        "candidates": candidates,
        "quotes": {t: quotes[t] for t in
                   set(list(pick_set) + pos_tickers + [c["ticker"] for c in candidates])
                   if t in quotes},
        "earnings": earnings_map,
        "alerts": alerts,
    }
    prev_file.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (HIST / f"{out['scan_date']}.json").write_text(json.dumps(out, ensure_ascii=False))
    log(f"picks: {sorted(pick_set)} | new: {sorted(new_picks)} | alerts: {len(alerts)}")

    # --- notifications
    lines = []
    if new_picks:
        for c in picks:
            if c["ticker"] in new_picks:
                lines.append(f"🎯 {c['ticker']} — {c['price']:.2f}$ | TP {c['tp_pct']:+.1%} | SL -{c['sl_pct']:.1%} | RSI {c['rsi']:.0f}")
    for a in alerts:
        lines.append(("⚠️ " if a["type"] in ("sl_hit", "sl_near") else "🔔 ") + a["msg"])
    if lines:
        title = []
        if new_picks:
            title.append(f"{len(new_picks)} מניות חדשות")
        if alerts:
            title.append(f"{len(alerts)} התראות")
        notify(" + ".join(title), "\n".join(lines),
               priority=4 if any(a["type"] in ("sl_hit", "tp_hit") for a in alerts) else 3)
    else:
        log("nothing new today; no notification")


if __name__ == "__main__":
    main()
