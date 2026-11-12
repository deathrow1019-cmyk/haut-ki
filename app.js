// Haut-KI auf dem iPhone: Modell läuft lokal per onnxruntime-web (WebAssembly), kein Foto verlässt das Gerät.
import * as ort from "./ort/ort.wasm.min.mjs";
import * as lernen from "./lernen.js";
import * as verlauf from "./verlauf.js";

ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
// Mehrere Threads bräuchten Cross-Origin-Isolation, die statische Hosts nicht setzen können.
ort.env.wasm.numThreads = 1;

const MODEL_URL = "modell/haut-ki.onnx";
const UNSICHER = 0.35;     // Spitzenreiter darunter: Ergebnis gilt als unsicher
const KLAR = 0.5;          // harmlose Krankheit darunter: „Kein klarer Befund“ statt Krankheitsname
const GESUND_MIT = 0.15;   // … oder wenn „gesunde Haut“ mindestens so wahrscheinlich ist
const UNKLAR_TEXT = "Das Modell erkennt hier nichts Eindeutiges. Sieht die Stelle für dich unauffällig aus, " +
  "ist das sehr wahrscheinlich gesunde Haut. Wenn sie juckt, schmerzt, sich verändert oder dir Sorgen macht, zeig sie einem Arzt.";
// Hautkrebs-Klassen zusammen: ab "rot" rote Ampel, ab "gelb" mindestens „abklären“. Die Werte kommen mit dem Modell
// (modell/warnung.json, gemessen an Testbildern); das hier sind nur die Ersatzwerte, falls die Datei fehlt.
let warnung = { schwelle_rot: 0.25, schwelle_gelb: 0.10 };

const HINWEIS = {
  hoch: "🔴 <strong>Möglicher Hautkrebs.</strong> Bitte zeitnah einen Hautarzt aufsuchen und die Stelle untersuchen lassen.",
  mittel: "🟠 <strong>Sollte ärztlich abgeklärt werden</strong>, z. B. weil es eine Vorstufe, ansteckend oder behandlungsbedürftig sein kann.",
  niedrig: "🟢 <strong>Meist harmlos oder gut behandelbar.</strong> Bei Beschwerden, Veränderung oder Unsicherheit trotzdem zum Arzt.",
};
// Ab Modell v8 kennt die App auch „kein Befund“: gesunde Haut und Fotos, die gar keine Haut zeigen
const GESUND_TEXT = "🟢 <strong>Keine auffällige Hautveränderung erkannt.</strong> Sehr kleine oder blasse Stellen kann die App " +
  "übersehen – bei Beschwerden oder wenn sich etwas verändert, trotzdem zum Arzt.";
const KEIN_BILD_TEXT = "📷 <strong>Das sieht nicht nach einer Hautstelle aus.</strong> Bitte die Stelle nah, scharf und gut " +
  "beleuchtet fotografieren, sodass sie in der Bildmitte liegt.";
const QUELLE = { isic: "Dermatoskop-Aufnahmen", fitz: "Klinische Fotos", scin: "Handyfotos (SCIN)", pad: "Handyfotos (PAD-UFES)", mednode: "Klinische Fotos (MED-NODE)",
  dermnet: "Klinische Fotos (DermNet)", milk: "Nahaufnahmen (MILK10k)", milkderm: "Dermatoskop-Aufnahmen (MILK10k)", eigene: "Eigene Befunde",
  gesund: "Fotos gesunder Haut", coco: "Alltagsfotos (keine Haut)" };

const $ = (id) => document.getElementById(id);
let meta, session;
let befunde = [];   // eigene Arzt-Befunde (IndexedDB)
let eintraege = []; // Verlauf: gespeicherte Fotos beobachteter Stellen (IndexedDB)
let vorschauUrl = null;
let letzte = null;  // letzte Analyse: Merkmale + kleines Bild, falls ein Befund dazu eingetragen wird
let serie = null;   // mehrere Fotos derselben Stelle: { summe: Wahrscheinlichkeiten aufsummiert, n: Anzahl }
let anhaengen = false; // nächstes Foto gehört zur selben Stelle (Knopf „Weiteres Foto“)

function zeige(id, an = true) { $(id).hidden = !an; }
function fehler(text) { $("fehler").textContent = text; zeige("fehler"); }
const prozent = (p) => (p < 0.005 ? "< 1 %" : `${Math.round(p * 100)} %`);

async function ladeMitFortschritt(url, onProgress, opt) {
  const res = await fetch(url, opt);
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

/** Kurzer Fingerabdruck der Modelldatei – steht auch in classes.json (model_version). */
async function fingerabdruck(bytes) {
  if (!crypto.subtle) return null;
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(h.slice(0, 6), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Lädt Klassenliste und Modell. Passen sie nicht zusammen (halb fertiges Update), wird einmal frisch aus dem Netz
 *  geladen – sonst würden Krankheitsnamen und Modell-Ausgaben vertauscht, ohne dass man es merkt. */
async function ladeModell() {
  for (const frisch of [false, true]) {
    const opt = frisch ? { cache: "reload" } : undefined;
    meta = await (await fetch("modell/classes.json", opt)).json();
    const bytes = await ladeMitFortschritt(MODEL_URL, (p) => {
      $("laden-balken").style.width = `${Math.round(p * 100)}%`;
      $("laden-text").textContent = `Modell wird geladen … ${Math.round(p * 100)} %`;
    }, opt);
    const ist = meta.model_version && (await fingerabdruck(bytes));
    if (!ist || ist === meta.model_version) {
      try {   // Schwellen und Messwerte der Hautkrebs-Warnung – nur wenn sie zu genau diesem Modell gehören
        const w = await (await fetch("modell/warnung.json", opt)).json();
        if (w.modell === meta.model_version && w.schwelle_rot > 0 && w.schwelle_gelb > 0) warnung = w;
      } catch { /* Ersatzwerte bleiben */ }
      return bytes;
    }
    console.warn(`Modell (${ist}) passt nicht zur Klassenliste (${meta.model_version})${frisch ? "" : " – lade neu"}`);
    zeige("laden");
  }
  throw new Error("Modell-Update unvollständig – bitte die App mit Internet erneut öffnen");
}

async function start() {
  const imCache = "caches" in window && (await caches.match(MODEL_URL));
  if (!imCache) zeige("laden");
  const bytes = await ladeModell();
  zeigeQualitaet();
  $("laden-text").textContent = "Rechenkern wird gestartet …";
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  zeige("laden", false);
  await ladeBefunde();
  await ladeVerlauf();
  $("status").textContent = navigator.serviceWorker?.controller ? "✓ Offline bereit" : "";
  if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});
}

// Bild → Tensor wie beim Training: kürzere Seite auf img_size × 1,14 (Resize), Mitte ausschneiden (CenterCrop), normalisiert.
async function dekodiere(file) {
  try {
    return await createImageBitmap(file);
  } catch {
    // z. B. HEIC: über <img> dekodieren lassen
    const img = new Image();
    const url = URL.createObjectURL(file);
    try {
      img.src = url;
      await img.decode();
      return await createImageBitmap(img);
    } finally {
      URL.revokeObjectURL(url);
    }
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

/** Grobe Prüfung der Fotoqualität (auf 256 px verkleinert): zu dunkel, überbelichtet, unscharf, zu klein.
 *  Schwellen an scharfen und künstlich verschlechterten Bildern eingestellt; bewusst vorsichtig (lieber kein Fehlalarm). */
function pruefeFoto(bmp) {
  const probleme = [];
  const kurz = Math.min(bmp.width, bmp.height);
  if (kurz < 200) probleme.push("sehr klein (wenig Details)");
  const f = 256 / kurz, w = Math.max(8, Math.round(bmp.width * f)), h = Math.max(8, Math.round(bmp.height * f));
  const px = verkleinere(bmp, w, h).getImageData(0, 0, w, h).data;
  const y = new Float32Array(w * h);
  let summe = 0, hell = 0;
  for (let i = 0; i < w * h; i++) {
    y[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
    summe += y[i];
    if (y[i] > 245) hell++;
  }
  const mittel = summe / (w * h);
  if (mittel < 55) probleme.push("zu dunkel");
  else if (mittel > 225 || hell / (w * h) > 0.4) probleme.push("überbelichtet");
  // Schärfe: Streuung des Laplace-Filters (Kanten). Scharfe Fotos liegen bei 15–50, deutlich unscharfe unter 2.
  let s1 = 0, s2 = 0, n = 0;
  for (let r = 1; r < h - 1; r++) {
    for (let c = 1; c < w - 1; c++) {
      const i = r * w + c, v = 4 * y[i] - y[i - 1] - y[i + 1] - y[i - w] - y[i + w];
      s1 += v; s2 += v * v; n++;
    }
  }
  const schaerfe = s2 / n - (s1 / n) ** 2;
  if (schaerfe < 3 && !probleme.length) probleme.push("unscharf");
  return { probleme, mittel, schaerfe };
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
  if (vorschauUrl) URL.revokeObjectURL(vorschauUrl);
  vorschauUrl = URL.createObjectURL(file);
  $("vorschau-bild").src = vorschauUrl;
  zeige("vorschau");
  $("pick").classList.add("busy");
  $("pick-titel").textContent = "Wird analysiert …";
  try {
    if (!session) await bereit;
    const bmp = await dekodiere(file);
    const foto = pruefeFoto(bmp);
    let { probs, merkmale } = await rechne(bmp);
    letzte = { merkmale, bild: await lernen.kleinesBild(bmp), gross: await lernen.kleinesBild(bmp, 640) };
    bmp.close?.();
    zeige("gesund-ok", false);
    // Mehrere Fotos derselben Stelle: Wahrscheinlichkeiten mitteln (ein einzelnes Foto kann täuschen)
    if (anhaengen && serie && serie.summe.length === probs.length) {
      serie = { summe: serie.summe.map((v, i) => v + probs[i]), n: serie.n + 1 };
    } else {
      serie = { summe: probs.slice(), n: 1 };
    }
    anhaengen = false;
    probs = serie.summe.map((v) => v / serie.n);
    letzte.modellProbs = probs;
    zeigeErgebnis(probs, merkmale);
    $("serie-hinweis").hidden = serie.n < 2;
    $("serie-hinweis").textContent = `📷 Mittel aus ${serie.n} Fotos derselben Stelle.`;
    $("foto-warnung").hidden = !foto.probleme.length;
    $("foto-warnung").innerHTML = `⚠️ <strong>Das Foto ist ${foto.probleme.join(" und ")}.</strong> Das Ergebnis ist dadurch ` +
      `unzuverlässiger – am besten bei Tageslicht, scharf gestellt und nah genug noch einmal aufnehmen.`;
  } catch (e) {
    anhaengen = false;
    console.error(e);
    fehler(`Das Bild konnte nicht ausgewertet werden (${e.message || e}). Bitte ein anderes Foto versuchen.`);
  } finally {
    $("pick").classList.remove("busy");
    $("pick-titel").textContent = "Neues Foto aufnehmen oder auswählen";
  }
}

/** Ein Bild durchs Modell: Wahrscheinlichkeiten und Merkmalsvektor (je Mittel aus Original und Spiegelbild). */
async function rechne(bmp) {
  const out = await session.run({ bild: alsTensor(bmp) });
  const K = meta.classes.length, logits = Array.from(out.logits.data);
  const a = softmax(logits.slice(0, K)), b = softmax(logits.slice(K, 2 * K));
  const f = out.merkmale.data, D = f.length / 2;
  return {
    probs: a.map((v, i) => (v + b[i]) / 2),
    merkmale: lernen.normiere(Array.from({ length: D }, (_, i) => (f[i] + f[D + i]) / 2)),
  };
}

const krebsAnteil = (probs) => meta.classes.reduce((s, c, i) => s + (c.risiko === "hoch" ? probs[i] : 0), 0);

function zeigeErgebnis(modellProbs, merkmale) {
  const eigen = lernen.beimischen(modellProbs, merkmale, befunde, meta.classes);
  const probs = eigen ? eigen.probs : modellProbs;
  const ranked = meta.classes.map((c, i) => ({ ...c, p: probs[i] })).sort((x, y) => y.p - x.p);
  const top = ranked[0];
  // Sicherheitsregel: eigene Befunde dürfen eine Hautkrebs-Warnung des Modells nie abschwächen
  const krebs = Math.max(krebsAnteil(modellProbs), krebsAnteil(probs));
  // Grauzone: Hautkrebs nicht wahrscheinlich, aber auch nicht ausgeschlossen -> nie „grün“
  const grauzone = krebs < warnung.schwelle_rot && krebs >= warnung.schwelle_gelb;
  const risiko = krebs >= warnung.schwelle_rot ? "hoch" : grauzone && top.risiko === "niedrig" ? "mittel" : top.risiko;
  $("eigen-hinweis").hidden = !eigen;
  if (eigen) {
    $("eigen-hinweis").textContent = `🧠 Berücksichtigt ${eigen.anzahl === 1 ? "einen eigenen Befund" : `${eigen.anzahl} eigene Befunde`}` +
      ` einer sehr ähnlichen Stelle (Anteil ${prozent(eigen.alpha)}).`;
  }
  befundFormular(false);
  verlaufFormular(false);
  if (letzte) letzte.ergebnis = { top: top.key, p: top.p, krebs };

  // „Kein Befund“ nur, wenn Hautkrebs auch nicht in der Grauzone liegt – die Warnung geht immer vor
  const keinBild = top.key === "kein_hautbild" && krebs < warnung.schwelle_gelb;
  const gesund = top.key === "gesunde_haut" && risiko === "niedrig";
  $("ampel").className = keinBild ? "ampel" : `ampel ${risiko}`;
  // Harmlose Krankheit nur schwach erkannt (oder „gesunde Haut“ spielt mit) -> keinen Krankheitsnamen als Ergebnis zeigen
  const pGesund = probs[meta.classes.findIndex((c) => c.key === "gesunde_haut")] || 0;
  const unklar = !keinBild && !gesund && risiko === "niedrig" && top.key !== "kein_hautbild" &&
    (top.p < KLAR || pGesund >= GESUND_MIT);
  $("top-name").textContent = unklar ? "Kein klarer Befund" : top.name;
  $("top-prozent").textContent = unklar ? `Am ehesten: ${top.name} · ${prozent(top.p)}` : `Am wahrscheinlichsten · ${prozent(top.p)}`;
  $("hinweis").innerHTML = (keinBild ? KEIN_BILD_TEXT : gesund ? GESUND_TEXT : unklar ? UNKLAR_TEXT : HINWEIS[risiko]) +
    (risiko === "hoch" && top.risiko !== "hoch" ? ` Die Hautkrebs-Arten zusammen kommen auf <strong>${prozent(krebs)}</strong>.` : "") +
    (grauzone ? ` Hautkrebs ist nicht sicher ausgeschlossen (zusammen <strong>${prozent(krebs)}</strong>) – bitte beobachten und im Zweifel zeigen.` : "");
  zeige("unsicher", top.p < UNSICHER);
  // „Das ist gesunde Haut“ nur anbieten, wenn das Modell Hautkrebs nicht einmal in der Grauzone sieht
  zeige("gesund-box", KENNT_GESUND() && !gesund && krebsAnteil(modellProbs) < warnung.schwelle_gelb);

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

// Wie verlässlich ist die Hautkrebs-Warnung? (gemessen an Testbildern, siehe modell/warnung.json)
function warnungText() {
  const t = warnung.test;
  if (!t) return "";
  const von100 = (p) => `${Math.round(p * 100)} von 100`;
  const zeile = (a, b) => `<tr><td>${a}</td><td>${b}</td></tr>`;
  return `<h3>Hautkrebs-Warnung</h3><p>Gemessen an ${t.n_krebs} Hautkrebs- und ${t.n_gutartig} gutartigen Testbildern:</p><table>` +
    zeile("Hautkrebs-Bilder mit roter Warnung", von100(t.rot.erkannt)) +
    zeile("… mit roter Warnung oder „nicht ausgeschlossen“", von100(t.gelb.erkannt)) +
    zeile("Gutartige Bilder mit roter Warnung (Fehlalarm)", von100(t.rot.fehlalarm)) +
    zeile("… mit roter Warnung oder „nicht ausgeschlossen“", von100(t.gelb.fehlalarm)) +
    (t.handy ? zeile("Nur Handyfotos: Hautkrebs mit roter Warnung", von100(t.handy.rot.erkannt)) +
      zeile("Nur Handyfotos: Fehlalarm (rot)", von100(t.handy.rot.fehlalarm)) : "") +
    `</table><p class="hint">Die App übersieht also einzelne Hautkrebs-Fälle und warnt oft auch bei harmlosen Stellen. ` +
    `Eine grüne Ampel ist keine Entwarnung.</p>`;
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
      warnungText() +
      `<h3>Pro Krankheit (wie oft richtig erkannt)</h3><table>` +
      Object.entries(t.pro_klasse).sort((a, b) => b[1] - a[1]).map(([k, v]) => zeile(name[k] || k, prozent(v))).join("") +
      `</table>`;
  } catch {
    box.innerHTML = "<p>Keine Testwerte vorhanden.</p>";
  }
}

// ---------- Dazulernen: Arzt-Befunde ----------

async function ladeBefunde() {
  try {
    befunde = await lernen.alle();
  } catch (e) {
    console.warn("Befunde nicht lesbar:", e);
    befunde = [];
  }
  // Neues Modell: gespeicherte Befunde mit dem neuen Modell neu durchrechnen, sonst passen die Merkmale nicht
  const alt = befunde.filter((b) => b.modell !== meta.model_version && b.bild);
  for (const b of alt) {
    try {
      const bmp = await createImageBitmap(b.bild);
      b.merkmale = (await rechne(bmp)).merkmale;
      bmp.close?.();
      b.modell = meta.model_version;
      await lernen.speichere(b);
    } catch (e) {
      console.warn("Befund nicht neu berechnet:", e);
    }
  }
  zeigeBefunde();
}

function befundFormular(offen) {
  zeige("befund-form", offen);
  zeige("befund-auf", !offen);
  if (!offen) zeige("befund-ok", false);
}

function fuelleAuswahl() {
  const sel = $("befund-label");
  const leer = new Option("Bitte wählen …", "");
  leer.disabled = true; leer.selected = true;
  const gruppen = {};
  for (const c of meta.classes) if (c.key !== "kein_hautbild") (gruppen[c.gruppe] ||= []).push(c);
  sel.replaceChildren(leer, ...Object.entries(gruppen).map(([g, cs]) => {
    const og = document.createElement("optgroup");
    og.label = g;
    og.append(...cs.map((c) => new Option(c.name, c.key)));
    return og;
  }), new Option("Etwas anderes (nicht in der Liste)", "andere"));
}

async function speichereBefund(ev) {
  ev.preventDefault();
  if (!letzte) return;
  const label = $("befund-label").value;
  if (!label || !$("befund-arzt").checked) return;
  const b = {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    datum: new Date().toISOString(),
    label,
    notiz: $("befund-notiz").value.trim(),
    merkmale: letzte.merkmale,
    bild: letzte.bild,
    modell: meta.model_version,
  };
  try {
    await lernen.speichere(b);
  } catch (e) {
    fehler(`Befund konnte nicht gespeichert werden (${e?.message || e || "unbekannter Fehler"}).`);
    return;
  }
  befunde.push(b);
  zeigeBefunde();
  zeige("befund-form", false);
  $("befund-ok").textContent = label === "andere"
    ? "✓ Gespeichert. Für den Export vorgemerkt (die App kennt diese Krankheit noch nicht)."
    : "✓ Gespeichert. Ähnliche Fotos berücksichtigen diesen Befund ab sofort.";
  zeige("befund-ok");
}

// ---------- Dazulernen: gesunde Haut (ein Tipp, ohne Arzt) ----------

const KENNT_GESUND = () => meta.classes.some((c) => c.key === "gesunde_haut");

function gesundBefund(merkmale, bild) {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    datum: new Date().toISOString(), label: "gesunde_haut", notiz: "selbst als gesund markiert",
    merkmale, bild, modell: meta.model_version,
  };
}

async function merkeGesund() {
  if (!letzte?.modellProbs) return;
  const b = gesundBefund(letzte.merkmale, letzte.bild);
  try {
    await lernen.speichere(b);
  } catch (e) {
    fehler(`Konnte nicht gespeichert werden (${e?.message || e || "unbekannter Fehler"}).`);
    return;
  }
  befunde.push(b);
  zeigeBefunde();
  zeigeErgebnis(letzte.modellProbs, letzte.merkmale);
  zeige("gesund-box", false);
  $("gesund-ok").textContent = "✓ Gemerkt. Ähnliche Fotos gelten ab sofort als gesunde Haut (unter „Dazugelernt“ wieder löschbar).";
  zeige("gesund-ok");
}

/** Mehrere Fotos unauffälliger Haut auf einmal einlernen. Fotos mit Hautkrebs-Verdacht werden nie als gesund gespeichert. */
async function lerneGesundeHaut(dateien) {
  const meldung = { set textContent(t) { $("einlernen-meldung").textContent = $("einrichten-meldung").textContent = t; } };
  zeige("einrichten-meldung");
  let gelernt = 0, verdacht = 0, keinBild = 0, kaputt = 0;
  zeige("fehler", false);
  zeige("einlernen-meldung");
  $("einlernen-knopf").disabled = true;
  try {
    if (!session) await bereit;
    const iKein = meta.classes.findIndex((c) => c.key === "kein_hautbild");
    for (const [n, f] of dateien.entries()) {
      meldung.textContent = `Foto ${n + 1} von ${dateien.length} wird eingelernt …`;
      try {
        const bmp = await dekodiere(f);
        const { probs, merkmale } = await rechne(bmp);
        const bild = await lernen.kleinesBild(bmp);
        bmp.close?.();
        if (krebsAnteil(probs) >= warnung.schwelle_gelb) { verdacht++; continue; }
        if (iKein >= 0 && probs[iKein] >= 0.5) { keinBild++; continue; }
        const b = gesundBefund(merkmale, bild);
        await lernen.speichere(b);
        befunde.push(b);
        gelernt++;
      } catch (e) {
        console.warn("Einlernen:", e);
        kaputt++;
      }
    }
  } finally {
    $("einlernen-knopf").disabled = false;
  }
  zeigeBefunde();
  meldung.textContent = `✓ ${gelernt} ${gelernt === 1 ? "Foto" : "Fotos"} als gesunde Haut eingelernt.` +
    (verdacht ? ` ${verdacht} nicht übernommen, weil das Modell dort Hautkrebs nicht sicher ausschließt – diese Stelle bitte einzeln prüfen und im Zweifel einem Arzt zeigen.` : "") +
    (keinBild ? ` ${keinBild} nicht übernommen (keine Haut erkannt).` : "") +
    (kaputt ? ` ${kaputt} konnten nicht gelesen werden.` : "");
}

function zeigeEinrichten() {
  let zu = false;
  try { zu = localStorage.getItem("einrichten-zu") === "1"; } catch { /* privater Modus */ }
  const n = befunde.filter((b) => b.label === "gesunde_haut").length;
  zeige("einrichten", KENNT_GESUND() && !zu && (n < 5 || !$("einrichten-meldung").hidden));
}

function zeigeBefunde() {
  zeigeEinrichten();
  const name = Object.fromEntries(meta.classes.map((c) => [c.key, c.name]));
  name.andere = "Etwas anderes";
  $("gelernt-zahl").textContent = befunde.length ? `(${befunde.length})` : "";
  zeige("gelernt-leer", !befunde.length);
  zeige("gelernt-aktionen", befunde.length > 0);
  const sortiert = [...befunde].sort((a, b) => b.datum.localeCompare(a.datum));
  $("gelernt-liste").replaceChildren(...sortiert.map((b) => {
    const li = document.createElement("li");
    li.className = "befund-eintrag";
    const img = document.createElement("img");
    img.alt = "";
    if (b.bild) {
      img.src = URL.createObjectURL(b.bild);
      img.onload = img.onerror = () => URL.revokeObjectURL(img.src);
    }
    const text = document.createElement("div");
    const titel = document.createElement("strong");
    titel.textContent = name[b.label] || b.label;
    const info = document.createElement("span");
    info.textContent = new Date(b.datum).toLocaleDateString("de-DE") + (b.notiz ? ` · ${b.notiz}` : "");
    text.append(titel, info);
    const weg = document.createElement("button");
    weg.className = "btn klein";
    weg.textContent = "Löschen";
    weg.addEventListener("click", async () => {
      try {
        await lernen.loesche(b.id);
      } catch (e) {
        fehler(`Befund konnte nicht gelöscht werden (${e?.message || e}).`);
        return;
      }
      befunde = befunde.filter((x) => x.id !== b.id);
      zeigeBefunde();
    });
    li.append(img, text, weg);
    return li;
  }));
}

// ---------- Verlauf: dieselbe Stelle über die Zeit beobachten ----------

async function ladeVerlauf() {
  try {
    eintraege = await verlauf.alle();
  } catch (err) {
    console.warn("Verlauf nicht lesbar:", err);
    eintraege = [];
  }
  // Neues Modell: alte Fotos neu einschätzen, sonst wären die Werte im Verlauf nicht vergleichbar
  for (const v of eintraege.filter((x) => x.modell !== meta.model_version && x.bild)) {
    try {
      const bmp = await createImageBitmap(v.bild);
      const { probs, merkmale } = await rechne(bmp);
      bmp.close?.();
      const i = probs.indexOf(Math.max(...probs));
      Object.assign(v, { merkmale, top: meta.classes[i].key, p: probs[i], krebs: krebsAnteil(probs), modell: meta.model_version });
      await verlauf.speichere(v);
    } catch (err) {
      console.warn("Verlaufs-Foto nicht neu berechnet:", err);
    }
  }
  zeigeVerlauf();
}

function verlaufFormular(offen) {
  zeige("verlauf-form", offen);
  zeige("verlauf-auf", !offen);
  if (!offen) zeige("verlauf-ok", false);
}

function oeffneVerlaufFormular() {
  if (!letzte) return;
  const sel = $("verlauf-stelle");
  const stellen = verlauf.gruppiere(eintraege);
  sel.replaceChildren(...stellen.map((s) => new Option(`${s.stelle} (${s.fotos.length} ${s.fotos.length === 1 ? "Foto" : "Fotos"})`, s.stelle)),
    new Option("➕ Neue Stelle", ""));
  // Sieht das Foto einer gespeicherten Stelle sehr ähnlich, wird sie vorgeschlagen
  sel.value = verlauf.passendeStelle(letzte.merkmale, eintraege, meta.model_version) ?? "";
  $("verlauf-name").value = "";
  zeige("verlauf-neu-feld", sel.value === "");
  verlaufFormular(true);
}

async function speichereVerlauf(ev) {
  ev.preventDefault();
  if (!letzte?.ergebnis) return;
  const stelle = $("verlauf-stelle").value || $("verlauf-name").value.trim();
  if (!stelle) { $("verlauf-name").focus(); return; }
  const v = {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    datum: new Date().toISOString(),
    stelle,
    bild: letzte.gross,
    merkmale: letzte.merkmale,
    modell: meta.model_version,
    ...letzte.ergebnis,
  };
  try {
    await verlauf.speichere(v);
  } catch (err) {
    fehler(`Foto konnte nicht im Verlauf gespeichert werden (${err?.message || err || "unbekannter Fehler"}).`);
    return;
  }
  eintraege.push(v);
  zeigeVerlauf();
  zeige("verlauf-form", false);
  const n = eintraege.filter((x) => x.stelle === stelle).length;
  $("verlauf-ok").textContent = `✓ Gespeichert unter „${stelle}“ (${n}. Foto). Zum Vergleichen unten „Verlauf“ öffnen.`;
  zeige("verlauf-ok");
}

function zeigeVerlauf() {
  const name = Object.fromEntries(meta.classes.map((c) => [c.key, c.name]));
  const stellen = verlauf.gruppiere(eintraege);
  $("verlauf-zahl").textContent = stellen.length ? `(${stellen.length})` : "";
  zeige("verlauf-leer", !stellen.length);
  zeige("verlauf-export", stellen.length > 0);
  $("verlauf-liste").replaceChildren(...stellen.map(({ stelle, fotos }) => {
    const box = document.createElement("div");
    box.className = "verlauf-stelle";
    const titel = document.createElement("h3");
    titel.textContent = `${stelle} · ${fotos.length} ${fotos.length === 1 ? "Foto" : "Fotos"}`;
    box.append(titel);
    const art = verlauf.veraenderung(fotos, warnung.schwelle_rot);
    if (art) {
      const p = document.createElement("p");
      p.className = "warn";
      p.innerHTML = art === "rot"
        ? "🔴 <strong>Beim neuesten Foto erscheint die Hautkrebs-Warnung, vorher nicht.</strong> Bitte die Stelle einem Hautarzt zeigen."
        : "⚠️ <strong>Die Einschätzung ist schlechter als bei einem früheren Foto.</strong> Das kann auch am Foto liegen (Licht, Abstand) – " +
          "noch einmal aufnehmen und im Zweifel einem Hautarzt zeigen.";
      box.append(p);
    }
    const reihe = document.createElement("div");
    reihe.className = "verlauf-reihe";
    reihe.append(...fotos.map((v) => {
      const karte = document.createElement("div");
      karte.className = "verlauf-foto";
      const img = document.createElement("img");
      img.alt = `${stelle}, ${new Date(v.datum).toLocaleDateString("de-DE")}`;
      if (v.bild) {
        img.src = URL.createObjectURL(v.bild);
        img.onload = img.onerror = () => URL.revokeObjectURL(img.src);
      }
      const datum = document.createElement("strong");
      datum.textContent = new Date(v.datum).toLocaleDateString("de-DE");
      const top = document.createElement("span");
      top.textContent = `${name[v.top] || v.top} · ${prozent(v.p)}`;
      const krebs = document.createElement("span");
      krebs.textContent = `Hautkrebs-Anteil ${prozent(v.krebs)}`;
      if (v.krebs >= warnung.schwelle_rot) krebs.className = "rot";
      const weg = document.createElement("button");
      weg.className = "btn klein";
      weg.textContent = "Löschen";
      weg.addEventListener("click", async () => {
        if (!confirm(`Foto vom ${datum.textContent} aus „${stelle}“ löschen?`)) return;
        try {
          await verlauf.loesche(v.id);
        } catch (err) {
          fehler(`Foto konnte nicht gelöscht werden (${err?.message || err}).`);
          return;
        }
        eintraege = eintraege.filter((x) => x.id !== v.id);
        zeigeVerlauf();
      });
      karte.append(img, datum, top, krebs, weg);
      return karte;
    }));
    box.append(reihe);
    requestAnimationFrame(() => { reihe.scrollLeft = reihe.scrollWidth; });   // neuestes Foto sichtbar
    return box;
  }));
}

$("verlauf-auf").addEventListener("click", oeffneVerlaufFormular);
$("verlauf-export").addEventListener("click", () => verlauf.exportiere(eintraege)
  .catch((err) => fehler(`Sichern fehlgeschlagen (${err.message || err}).`)));
$("verlauf-import-knopf").addEventListener("click", () => $("verlauf-import").click());
$("verlauf-import").addEventListener("change", async (ev) => {
  const f = ev.target.files?.[0];
  ev.target.value = "";
  if (!f) return;
  zeige("fehler", false);
  try {
    if (!session) await bereit;
    const n = await verlauf.importiere(f);
    await ladeVerlauf();   // rechnet die eingelesenen Fotos mit dem aktuellen Modell neu durch
    $("verlauf-meldung").textContent = `✓ ${n} ${n === 1 ? "Foto" : "Fotos"} eingelesen.`;
    zeige("verlauf-meldung");
  } catch (err) {
    fehler(`Die Sicherung konnte nicht eingelesen werden (${err?.message || err}).`);
  }
});
$("verlauf-stelle").addEventListener("change", (ev) => zeige("verlauf-neu-feld", ev.target.value === ""));
$("verlauf-abbruch").addEventListener("click", () => verlaufFormular(false));
$("verlauf-form").addEventListener("submit", speichereVerlauf);

$("befund-auf").addEventListener("click", () => {
  fuelleAuswahl();
  $("befund-notiz").value = "";
  $("befund-arzt").checked = false;
  befundFormular(true);
});
$("gesund-knopf").addEventListener("click", merkeGesund);
$("einrichten-knopf").addEventListener("click", () => $("einlernen").click());
$("einrichten-zu").addEventListener("click", () => {
  try { localStorage.setItem("einrichten-zu", "1"); } catch { /* privater Modus */ }
  zeige("einrichten-meldung", false);
  zeigeEinrichten();
});
$("einlernen-knopf").addEventListener("click", () => $("einlernen").click());
$("einlernen").addEventListener("change", (ev) => {
  const dateien = [...(ev.target.files || [])];
  ev.target.value = "";
  if (dateien.length) lerneGesundeHaut(dateien);
});
$("befund-abbruch").addEventListener("click", () => befundFormular(false));
$("befund-form").addEventListener("submit", speichereBefund);
$("export").addEventListener("click", () => lernen.exportiere(befunde, meta.model_version)
  .catch((e) => fehler(`Export fehlgeschlagen (${e.message || e}).`)));
$("alle-loeschen").addEventListener("click", async () => {
  if (!confirm(`Alle ${befunde.length} eingetragenen Befunde von diesem iPhone löschen?`)) return;
  try {
    await lernen.loescheAlle();
  } catch (e) {
    fehler(`Befunde konnten nicht gelöscht werden (${e?.message || e}).`);
    return;
  }
  befunde = [];
  zeigeBefunde();
});

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

$("kamera").addEventListener("change", (ev) => {
  const f = ev.target.files?.[0];
  ev.target.value = "";
  if (f) analysiere(f);
});
$("kamera-knopf").addEventListener("click", () => { anhaengen = false; $("kamera").click(); });

// iPhone/iPad in Safari (nicht vom Home-Bildschirm gestartet): Hinweis zum Installieren, einmal wegklickbar
try {
  const ios = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const alsApp = navigator.standalone === true || matchMedia("(display-mode: standalone)").matches;
  if (ios && !alsApp && localStorage.getItem("installieren-zu") !== "1") zeige("installieren");
  $("installieren-zu").addEventListener("click", () => {
    zeige("installieren", false);
    try { localStorage.setItem("installieren-zu", "1"); } catch { /* privater Modus */ }
  });
} catch { /* ohne Hinweis weiter */ }

// Normale Auswahl beginnt eine neue Stelle; der Knopf „Weiteres Foto“ hängt an die bisherige an.
$("pick").addEventListener("click", () => { anhaengen = false; });
$("weiteres-foto").addEventListener("click", () => {
  $("datei").click();   // löst auch den Klick-Handler oben aus (setzt zurück) – deshalb erst danach merken
  anhaengen = true;
});

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service Worker:", e));
}
