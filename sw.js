/* Channel-Up Scanner — service worker */
const VERSION = 'cs-v1';
const SHELL = ['./', 'index.html', 'manifest.webmanifest',
               'icon-192.png', 'icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;

  // data: network-first (fresh scan), fall back to cache when offline
  if (url.pathname.endsWith('.json') && !url.pathname.endsWith('manifest.webmanifest')) {
    e.respondWith(
      fetch(e.request).then(r => {
        const copy = r.clone();
        caches.open(VERSION).then(c => c.put(e.request, copy));
        return r;
      }).catch(() => caches.match(e.request))
    );
    return;
  }
  // shell: cache-first
  e.respondWith(
    caches.match(e.request, {ignoreSearch: true}).then(hit =>
      hit || fetch(e.request).then(r => {
        const copy = r.clone();
        caches.open(VERSION).then(c => c.put(e.request, copy));
        return r;
      })
    ).catch(() => caches.match('index.html'))
  );
});
