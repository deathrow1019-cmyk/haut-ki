// Offline-Speicher: App-Dateien beim Installieren, das Modell beim ersten Laden. Danach alles aus dem Cache.
const VERSION = "haut-ki-v8-2";
const SHELL = [
  "./", "index.html", "style.css", "app.js", "lernen.js", "verlauf.js", "manifest.webmanifest",
  "icons/icon-180.png", "icons/icon-192.png", "icons/icon-512.png",
  "ort/ort.wasm.min.mjs", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm",
  // Das Modell gehört zur Version: erst wenn alles da ist, löst die neue Version die alte ab (sonst nach einem
  // Update offline kein Modell, oder neue Klassenliste mit altem Modell).
  "modell/classes.json", "modell/bericht.json", "modell/warnung.json", "modell/haut-ki.onnx",
];

self.addEventListener("install", (e) => {
  e.waitUntil(// cache: "reload" umgeht den Browser-Zwischenspeicher: sonst kann eine veraltete Datei in die neue Version rutschen
    caches.open(VERSION).then((c) => c.addAll(SHELL.map((u) => new Request(u, { cache: "reload" })))).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  // Ausdrücklich frisch angefordert (App repariert ein unvollständiges Update): Netz statt Cache
  const frisch = e.request.cache === "reload";
  e.respondWith(
    (frisch ? Promise.resolve(null) : caches.match(e.request, { ignoreSearch: true })).then((hit) => hit || fetch(e.request).then((res) => {
      if (res.ok && res.status === 200) {
        const copy = res.clone();
        caches.open(VERSION).then((c) => c.put(e.request, copy));
      }
      return res;
    })),
  );
});
