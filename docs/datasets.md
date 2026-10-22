# Dataset

Elenco ricavato dal documento `dataset_video_furti.docx` (Drive › LOOKUP › 03_Prodotto › Librerie › video).

Regole:
- I file scaricati vanno in `data/<nome-dataset>/`, che è escluso da git.
- Ogni dataset va scaricato con uno script in `scripts/download_<nome>.py`, così il setup è ripetibile.
- **Prima di usare un dataset per addestrare un modello commerciale, verificane la licenza** e annotala in `docs/licenses.md`. Molti dataset accademici sono concessi solo per ricerca.

| Dataset | Contenuto | Uso previsto | Licenza |
|---|---|---|---|
| UCF-Crime | ~1.900 video CCTV reali, 13 categorie tra cui Shoplifting, Stealing, Robbery | Test su scene reali di taccheggio | da verificare |
| DCSASS | Comportamenti sospetti da videosorveglianza | Classificazione azioni in scene affollate | da verificare |
| CCTV Shoplifting Detection (Simuletic) | Video sintetici con pose a 17 keypoint | Occultamento senza dati biometrici reali | da verificare |
| Innovatiana Shoplifting | Clip di laboratorio, normale vs anomalo | Classificazione azioni | da verificare |
| MERL Shopping | 106 video zenitali in negozio, azioni mano-scaffale | Interazione "normale" con la merce | da verificare |
| CAVIAR (Shopping Center) | Corridoi di centri commerciali, grandangolo | Tracking e sosta | da verificare |
| ShanghaiTech Campus | Video anomaly detection, ~300k frame | Traiettorie e movimenti anomali | da verificare |
| CUHK Avenue | Loitering, corse, lanci di oggetti | Traiettorie insolite | da verificare |
| SXD | Scambi di oggetti tra persone | Furti in combutta | da verificare |
| Roboflow Universe / Kaggle | Dataset vari già etichettati (formato YOLO) | Detection di persona, mano, prodotto | una per dataset |

Per la milestone 1 (conteggio persone) basta un singolo video: va bene una clip di CAVIAR o una ripresa inscenata da noi.
