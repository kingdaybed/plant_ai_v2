self.addEventListener('install', function (event) {
    console.log('PWA Installed');
});

self.addEventListener('fetch', function (event) {
    event.respondWith(fetch(event.request));
});