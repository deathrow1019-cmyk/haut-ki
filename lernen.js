// Dazulernen auf dem Gerät: Arzt-Befunde werden mit ihrem Merkmalsvektor in IndexedDB gespeichert.
// Ähnliche neue Fotos (Kosinus-Ähnlichkeit ab SIM_MIN) bekommen den Befund beigemischt. Nichts verlässt das Gerät.

const DB_NAME = "haut-ki", STORE = "befunde";
// Gemessen: dieselbe Stelle anders fotografiert meist ≥ 0,79, verschiedene Stellen höchstens 0,67
export const SIM_MIN = 0.7;
const SIM_VOLL = 0.95;     // ab hier zählt ein Befund voll
const ALPHA_MAX = 0.6;     // eigene Befunde machen höchstens 60 % des Ergebnisses aus

function db() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(out?.result ?? out);
    t.onerror = t.onabort = () => reject(t.error || new Error("Speicher-Fehler"));
  });
}

// Bilder als rohe Bytes speichern: nicht jede WebKit-Version kann Blobs in IndexedDB ablegen
const zuSpeicher = async (b) => ({ ...b, bild: b.bild && { bytes: await b.bild.arrayBuffer(), typ: b.bild.type } });
const ausSpeicher = (b) => ({ ...b, bild: b.bild && new Blob([b.bild.bytes], { type: b.bild.typ }) });

export const alle = async () => (await tx("readonly", (s) => s.getAll())).map(ausSpeicher);
export const speichere = async (b) => { const roh = await zuSpeicher(b); return tx("readwrite", (s) => s.put(roh)); };
export const loesche = (id) => tx("readwrite", (s) => s.delete(id));
export const loescheAlle = () => tx("readwrite", (s) => s.clear());

export function normiere(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
}

function aehnlichkeit(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Mischt passende eigene Befunde in die Modell-Wahrscheinlichkeiten. Gibt null zurück, wenn keiner ähnlich genug ist. */
export function beimischen(probs, merkmale, befunde, klassen) {
  const idx = Object.fromEntries(klassen.map((c, i) => [c.key, i]));
  const dist = new Array(probs.length).fill(0);
  let summe = 0, anzahl = 0, maxSim = 0;
  for (const b of befunde) {
    if (!b.merkmale || !(b.label in idx)) continue;
    const s = aehnlichkeit(merkmale, b.merkmale);
    if (s < SIM_MIN) continue;
    const w = Math.min(1, (s - SIM_MIN) / (SIM_VOLL - SIM_MIN));
    dist[idx[b.label]] += w;
    summe += w; anzahl++; maxSim = Math.max(maxSim, s);
  }
  if (!summe) return null;
  const alpha = Math.min(ALPHA_MAX, summe / (summe + 0.5));
  return {
    probs: probs.map((p, i) => (1 - alpha) * p + (alpha * dist[i]) / summe),
    anzahl, alpha, maxSim,
  };
}

/** Kleines JPEG des Fotos (kürzere Seite 320 px): zum erneuten Durchrechnen nach Modell-Updates und für den Export. */
export async function kleinesBild(bmp, kurz = 320) {
  const k = Math.min(1, kurz / Math.min(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  const g = c.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(bmp, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.88));
}

const alsDataUrl = (blob) => new Promise((r) => { const f = new FileReader(); f.onload = () => r(f.result); f.readAsDataURL(blob); });

/** Exportdatei für das nächste große Training. Über das Teilen-Menü (iPhone: „In Dateien sichern“) oder als Download. */
export async function exportiere(befunde, modellVersion) {
  const daten = {
    app: "haut-ki", format: 1, exportiert: new Date().toISOString(), modell: modellVersion,
    befunde: await Promise.all(befunde.map(async (b) => ({
      id: b.id, datum: b.datum, label: b.label, notiz: b.notiz || "", bild: await alsDataUrl(b.bild),
    }))),
  };
  const name = `haut-ki-befunde-${new Date().toISOString().slice(0, 10)}.json`;
  const file = new File([JSON.stringify(daten)], name, { type: "application/json" });
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: "Haut-KI Befunde" }); return; } catch (e) { if (e.name === "AbortError") return; }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(file); a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}
