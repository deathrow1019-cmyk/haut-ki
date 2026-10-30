# Haut-KI

Web-App, die auf Fotos Hautkrebs, Muttermale und häufige Hautkrankheiten (25 Arten) einschätzt.
Das Modell läuft komplett im Browser des Geräts (onnxruntime-web, WebAssembly) – **Fotos werden nirgendwohin
hochgeladen**. Nach dem ersten Laden funktioniert die App offline.

**Lernprojekt, kein Medizinprodukt, keine Diagnose.** Die Trefferquote liegt bei Handyfotos deutlich unter der
bei Arzt-Aufnahmen. Bei Hautveränderungen immer einen Hautarzt fragen.

## Auf dem iPhone installieren

Seite in Safari öffnen → warten, bis „✓ Offline bereit“ unten steht → *Teilen* → *Zum Home-Bildschirm*.

## Modell und Daten

ConvNeXt-Tiny (timm, vortrainiert auf ImageNet-22k), nachtrainiert auf:

- ISIC 2019 (BCN_20000, HAM10000, MSK) – CC BY-NC 4.0
- Google SCIN (Skin Condition Image Network)
- Fitzpatrick17k (Groh et al. 2021), Bilder von atlasdermatologico.com.br – CC BY-NC-SA 3.0

Nur für nicht-kommerzielle Nutzung. Testwerte stehen in `modell/bericht.json` und in der App unter
„Wie zuverlässig ist das?“.
