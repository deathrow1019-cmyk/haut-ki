// Verlauf: Fotos derselben Hautstelle (z. B. ein Muttermal) über Wochen und Monate sammeln und vergleichen.
// Eigene IndexedDB, getrennt von den Arzt-Befunden. Nichts verlässt das Gerät.

const DB_NAME = "haut-ki-verlauf", STORE = "eintraege";
const SIM_MIN = 0.7;          // wie beim Dazulernen: ab hier gilt ein Foto als „wahrscheinlich dieselbe Stelle“
export const ANSTIEG = 0.15;  // Hautkrebs-Anteil um so viel höher als beim niedrigsten früheren Foto -> Hinweis

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
const zuSpeicher = async (e) => ({ ...e, bild: e.bild && { bytes: await e.bild.arrayBuffer(), typ: e.bild.type } });
const ausSpeicher = (e) => ({ ...e, bild: e.bild && new Blob([e.bild.bytes], { type: e.bild.typ }) });

export const alle = async () => (await tx("readonly", (s) => s.getAll())).map(ausSpeicher);
export const speichere = async (e) => { const roh = await zuSpeicher(e); return tx("readwrite", (s) => s.put(roh)); };
export const loesche = (id) => tx("readwrite", (s) => s.delete(id));

/** Nach Stelle gruppiert: je Stelle die Fotos von alt nach neu; die zuletzt fotografierte Stelle steht oben. */
export function gruppiere(eintraege) {
  const g = new Map();
  for (const e of [...eintraege].sort((a, b) => a.datum.localeCompare(b.datum))) {
    if (!g.has(e.stelle)) g.set(e.stelle, []);
    g.get(e.stelle).push(e);
  }
  return [...g.entries()].map(([stelle, fotos]) => ({ stelle, fotos }))
    .sort((a, b) => b.fotos.at(-1).datum.localeCompare(a.fotos.at(-1).datum));
}

/** Zu welcher gespeicherten Stelle passt das neue Foto am besten? null, wenn keine ähnlich genug ist. */
export function passendeStelle(merkmale, eintraege, modell) {
  let beste = null, max = SIM_MIN;
  for (const e of eintraege) {
    if (!e.merkmale || e.modell !== modell || e.merkmale.length !== merkmale.length) continue;
    let s = 0;
    for (let i = 0; i < merkmale.length; i++) s += merkmale[i] * e.merkmale[i];
    if (s >= max) { max = s; beste = e.stelle; }
  }
  return beste;
}

/** Hat sich die Einschätzung einer Stelle verschlechtert? Vergleicht das neueste Foto mit den früheren.
 *  Gibt "rot" (jetzt rote Warnung, früher nicht), "gestiegen" oder null zurück. */
export function veraenderung(fotos, schwelleRot) {
  if (fotos.length < 2) return null;
  const neu = fotos.at(-1).krebs, frueher = fotos.slice(0, -1).map((f) => f.krebs);
  if (neu >= schwelleRot && frueher.every((k) => k < schwelleRot)) return "rot";
  if (neu - Math.min(...frueher) >= ANSTIEG) return "gestiegen";
  return null;
}

const alsDataUrl = (blob) => new Promise((r) => { const f = new FileReader(); f.onload = () => r(f.result); f.readAsDataURL(blob); });

function ausDataUrl(url) {
  const [kopf, daten] = url.split(",", 2);
  const roh = atob(daten), bytes = new Uint8Array(roh.length);
  for (let i = 0; i < roh.length; i++) bytes[i] = roh.charCodeAt(i);
  return new Blob([bytes], { type: /data:([^;]+)/.exec(kopf)?.[1] || "image/jpeg" });
}

/** Sicherungsdatei mit allen Verlaufs-Fotos. Über das Teilen-Menü (iPhone: „In Dateien sichern“) oder als Download. */
export async function exportiere(eintraege) {
  const daten = {
    app: "haut-ki", typ: "verlauf", format: 1, exportiert: new Date().toISOString(),
    eintraege: await Promise.all(eintraege.map(async (e) => ({
      id: e.id, datum: e.datum, stelle: e.stelle, top: e.top, p: e.p, krebs: e.krebs, bild: e.bild ? await alsDataUrl(e.bild) : "",
    }))),
  };
  const name = `haut-ki-verlauf-${new Date().toISOString().slice(0, 10)}.json`;
  const file = new File([JSON.stringify(daten)], name, { type: "application/json" });
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: "Haut-KI Verlauf" }); return; } catch (e) { if (e.name === "AbortError") return; }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(file); a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

/** Liest eine Sicherungsdatei ein. Vorhandene Fotos (gleiche Kennung) werden nicht doppelt angelegt. Gibt die Anzahl zurück.
 *  Die Einschätzung wird danach mit dem aktuellen Modell neu berechnet (modell bleibt leer). */
export async function importiere(file) {
  const daten = JSON.parse(await file.text());
  if (daten?.app !== "haut-ki" || daten.typ !== "verlauf" || !Array.isArray(daten.eintraege)) {
    throw new Error("das ist keine Verlaufs-Sicherung von Haut-KI");
  }
  let n = 0;
  for (const e of daten.eintraege) {
    if (!e?.id || !e.stelle || !e.datum || typeof e.bild !== "string" || !e.bild.startsWith("data:image/")) continue;
    await speichere({ id: String(e.id), datum: String(e.datum), stelle: String(e.stelle).slice(0, 40), bild: ausDataUrl(e.bild),
      top: String(e.top || ""), p: Number(e.p) || 0, krebs: Number(e.krebs) || 0, merkmale: null, modell: "" });
    n++;
  }
  return n;
}
