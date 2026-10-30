// Haut-KI auf dem iPhone: Modell läuft lokal per onnxruntime-web (WebAssembly), kein Foto verlässt das Gerät.
import * as ort from "./ort/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
// Mehrere Threads bräuchten Cross-Origin-Isolation, die statische Hosts nicht setzen können.
ort.env.wasm.numThreads = 1;

const MODEL_URL = "modell/haut-ki.onnx";
const UNSICHER = 0.35;     // Spitzenreiter darunter: Ergebnis gilt als unsicher
const KREBS_SCHWELLE = 0.25; // Hautkrebs-Klassen zusammen ab hier: rote Ampel

const HINWEIS = {
  hoch: "🔴 <strong>Möglicher Hautkrebs.</strong> Bitte zeitnah einen Hautarzt aufsuchen und die Stelle untersuchen lassen.",
  mittel: "🟠 <strong>Sollte ärztlich abgeklärt werden</strong>, z. B. weil es eine Vorstufe, ansteckend oder behandlungsbedürftig sein kann.",
  niedrig: "🟢 <strong>Meist harmlos oder gut behandelbar.</strong> Bei Beschwerden, Veränderung oder Unsicherheit trotzdem zum Arzt.",
};
const QUELLE = { isic: "Dermatoskop-Aufnahmen", fitz: "Klinische Fotos", scin: "Handyfotos" };

const $ = (id) => document.getElementById(id);
let meta, session;

function zeige(id, an = true) { $(id).hidden = !an; }
function fehler(text) { $("fehler").textContent = text; zeige("fehler"); }
const prozent = (p) => (p < 0.005 ? "< 1 %" : `${Math.round(p * 100)} %`);

async function ladeMitFortschritt(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const teile = [];
  let geladen = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    teile.push(value);
    geladen += value.length;
    onProgress(Math.min(1, geladen / total));
  }
  const out = new Uint8Array(geladen);
  let pos = 0;
  for (const t of teile) { out.set(t, pos); pos += t.length; }
  return out;
}

async function start() {
  meta = await (await fetch("modell/classes.json")).json();
  zeigeQualitaet();
  const imCache = "caches" in window && (await caches.match(MODEL_URL));
  if (!imCache) zeige("laden");
  const bytes = await ladeMitFortschritt(MODEL_URL, (p) => {
    $("laden-balken").style.width = `${Math.round(p * 100)}%`;
    $("laden-text").textContent = `Modell wird geladen … ${Math.round(p * 100)} %`;
  });
  $("laden-text").textContent = "Rechenkern wird gestartet …";
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  zeige("laden", false);
  $("status").textContent = navigator.serviceWorker?.controller ? "✓ Offline bereit" : "";
  if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});
}

// Bild → Tensor wie beim Training: kürzere Seite auf 255 (Resize), Mitte 224×224 (CenterCrop), normalisiert.
async function dekodiere(file) {
  try {
    return await createImageBitmap(file);
  } catch {
    // z. B. HEIC: über <img> dekodieren lassen
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();
    return createImageBitmap(img);
  }
}

function verkleinere(quelle, w, h) {
  // In Halbierungsschritten verkleinern: glättet wie das Antialiasing beim Training
  let src = quelle, sw = quelle.width, sh = quelle.height;
  while (sw / 2 > w && sh / 2 > h) {
    const c = document.createElement("canvas");
    c.width = Math.floor(sw / 2); c.height = Math.floor(sh / 2);
    const g = c.getContext("2d");
    g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, c.width, c.height);
    src = c; sw = c.width; sh = c.height;
  }
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingQuality = "high";
  g.drawImage(src, 0, 0, w, h);
  return g;
}

function alsTensor(bmp) {
  const C = meta.img_size, S = Math.floor(C * 1.14);
  const kurz = Math.min(bmp.width, bmp.height);
  const w = bmp.width === kurz ? S : Math.floor((S * bmp.width) / kurz);
  const h = bmp.height === kurz ? S : Math.floor((S * bmp.height) / kurz);
  const g = verkleinere(bmp, w, h);
  const px = g.getImageData(Math.round((w - C) / 2), Math.round((h - C) / 2), C, C).data;
  const { mean, std } = meta;
  const n = C * C, data = new Float32Array(2 * 3 * n);
  // Batch aus Original und gespiegelter Version: das Ergebnis wird gemittelt, etwas stabiler
  for (let y = 0; y < C; y++) {
    for (let x = 0; x < C; x++) {
      const i = (y * C + x) * 4, o = y * C + x, f = y * C + (C - 1 - x);
      for (let k = 0; k < 3; k++) {
        const v = (px[i + k] / 255 - mean[k]) / std[k];
        data[k * n + o] = v;
        data[3 * n + k * n + f] = v;
      }
    }
  }
  return new ort.Tensor("float32", data, [2, 3, C, C]);
}

function softmax(row) {
  const m = Math.max(...row);
  const e = row.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

async function analysiere(file) {
  zeige("fehler", false);
  zeige("ergebnis", false);
  $("vorschau-bild").src = URL.createObjectURL(file);
  zeige("vorschau");
  $("pick").classList.add("busy");
  $("pick-titel").textContent = "Wird analysiert …";
  try {
    if (!session) await bereit;
    const bmp = await dekodiere(file);
    const out = await session.run({ bild: alsTensor(bmp) });
    bmp.close?.();
    const K = meta.classes.length, logits = Array.from(out.logits.data);
    const a = softmax(logits.slice(0, K)), b = softmax(logits.slice(K, 2 * K));
    zeigeErgebnis(a.map((v, i) => (v + b[i]) / 2));
  } catch (e) {
    console.error(e);
    fehler(`Das Bild konnte nicht ausgewertet werden (${e.message || e}). Bitte ein anderes Foto versuchen.`);
  } finally {
    $("pick").classList.remove("busy");
    $("pick-titel").textContent = "Neues Foto aufnehmen oder auswählen";
  }
}

function zeigeErgebnis(probs) {
  const ranked = meta.classes.map((c, i) => ({ ...c, p: probs[i] })).sort((x, y) => y.p - x.p);
  const top = ranked[0];
  const krebs = ranked.filter((c) => c.risiko === "hoch").reduce((s, c) => s + c.p, 0);
  const risiko = krebs >= KREBS_SCHWELLE ? "hoch" : top.risiko;

  $("ampel").className = `ampel ${risiko}`;
  $("top-name").textContent = top.name;
  $("top-prozent").textContent = `Am wahrscheinlichsten · ${prozent(top.p)}`;
  $("hinweis").innerHTML = HINWEIS[risiko] +
    (risiko === "hoch" && top.risiko !== "hoch" ? ` Die Hautkrebs-Arten zusammen kommen auf <strong>${prozent(krebs)}</strong>.` : "");
  zeige("unsicher", top.p < UNSICHER);

  const liste = $("liste");
  liste.replaceChildren(...ranked.slice(0, 5).map((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<div class="row"><span></span><b>${prozent(c.p)}</b></div><div class="mini"><div></div></div>`;
    li.querySelector("span").textContent = c.name;
    li.querySelector(".mini > div").style.width = `${Math.max(1, c.p * 100)}%`;
    return li;
  }));
  zeige("ergebnis");
  $("ergebnis").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function zeigeQualitaet() {
  const box = $("qualitaet");
  try {
    const r = await (await fetch("modell/bericht.json")).json();
    const t = r.test, name = Object.fromEntries(meta.classes.map((c) => [c.key, c.name]));
    const zeile = (a, b) => `<tr><td>${a}</td><td>${b}</td></tr>`;
    box.innerHTML =
      `<p>Getestet an Bildern, die das Modell beim Training nie gesehen hat:</p><table>` +
      zeile("Richtige Antwort an 1. Stelle", prozent(t.genauigkeit)) +
      zeile("Richtige Antwort unter den Top 3", prozent(t.top3)) +
      Object.entries(t.pro_quelle).map(([k, v]) => zeile(`… bei ${QUELLE[k] || k}`, prozent(v))).join("") +
      `</table><p class="hint">Bei eigenen Handyfotos ist die App deutlich unsicherer als bei Arzt-Aufnahmen.</p>` +
      `<h3>Pro Krankheit (wie oft richtig erkannt)</h3><table>` +
      Object.entries(t.pro_klasse).sort((a, b) => b[1] - a[1]).map(([k, v]) => zeile(name[k] || k, prozent(v))).join("") +
      `</table>`;
  } catch {
    box.innerHTML = "<p>Keine Testwerte vorhanden.</p>";
  }
}

const bereit = start().catch((e) => {
  console.error(e);
  zeige("laden", false);
  fehler(`Das Modell konnte nicht geladen werden (${e.message || e}). Beim ersten Start wird Internet gebraucht.`);
  throw e;
});

$("datei").addEventListener("change", (ev) => {
  const f = ev.target.files?.[0];
  ev.target.value = "";
  if (f) analysiere(f);
});

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service Worker:", e));
}
