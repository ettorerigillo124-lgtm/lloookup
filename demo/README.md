# LookUp: demo live anti-occultamento

Il telefono fa da webcam e il sistema riconosce penna, occhiali, quaderno e sacchetto. Segue gli oggetti: visibile → in mano → vicino al sacchetto → occultato. Quando scatta l'occultamento, la persona diventa **rossa**. La persona è **sempre** oscurata (pixelation); gli oggetti restano visibili.

> ⚠️ Usa Ultralytics (AGPL-3.0): codice **solo per la demo**, non entra nel prodotto. Vedi `DECISIONS.md` D-01.

## 1. Installazione (una volta, ~10 minuti, ~5 GB)

Serve Python 3.10–3.13 (`python3 --version`) e git.

**Mac**
```bash
cd lloookup
python3 -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
```

**Windows (PowerShell)**
```powershell
cd lloookup
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r demo\requirements.txt
```
Se PowerShell blocca lo script di attivazione: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Al **primo avvio** servono internet e un paio di minuti: vengono scaricati i modelli in `models/` (~45 MB) e l'encoder CLIP in `~/.cache/clip` (~340 MB). Dopo, la demo funziona **offline**. Fai il primo avvio a casa, non davanti agli investitori.

Prova senza telefono, con la clip:
```bash
python demo/main.py --source data/samples/test_clip.mp4
```

## 2. Collegare il telefono come webcam

### Mac + iPhone: Continuity Camera (consigliato, nessuna app)
1. iPhone con iOS 16+ e Mac con macOS 13 Ventura+, **stesso Apple ID**, Wi-Fi e Bluetooth accesi su entrambi.
2. Su iPhone: Impostazioni → Generali → AirPlay e Continuità → **Fotocamera Continuità** attiva.
3. Metti l'iPhone fermo su un supporto, **schermo bloccato**, in orizzontale.
4. Dai il permesso fotocamera al terminale: Impostazioni di Sistema → Privacy e sicurezza → **Fotocamera** → attiva Terminale (o iTerm / VS Code). Dopo, chiudi e riapri il terminale.
5. Trova l'indice della camera:
   ```bash
   python demo/main.py --list-cameras
   ```
   Di solito l'iPhone è `0` o `1` (l'altro è la webcam del Mac). Se l'iPhone non compare, sbloccalo e riapri la camera: deve uscire il suono di Continuità.

### Android (Mac o Windows): DroidCam o IP Webcam, via Wi-Fi
Telefono e computer sulla **stessa rete Wi-Fi** (l'hotspot del telefono va bene ed evita il Wi-Fi della sala).
- **DroidCam** (Play Store): apri l'app e leggi l'IP, ad esempio `192.168.1.20`.
  ```bash
  python demo/main.py --source http://192.168.1.20:4747/video
  ```
- **IP Webcam** (Play Store): in fondo tocca "Avvia server" e leggi l'IP.
  ```bash
  python demo/main.py --source http://192.168.1.20:8080/video
  ```
Prima prova l'URL nel browser del computer: se lì non vedi il video, il problema è la rete, non la demo.

DroidCam esiste anche per **iPhone**, con lo stesso URL `http://IP:4747/video`. È utile su Windows, dove Continuity Camera non c'è.

### Windows + iPhone/Android: Camo (camera virtuale)
1. Installa **Camo** sul telefono e **Camo Studio** sul PC (reincubate.com/camo), poi collega il telefono via USB.
2. Camo diventa una webcam di Windows: `python demo/main.py --list-cameras`, poi `--source <indice>`.
3. Se Windows nega l'accesso: Impostazioni → Privacy e sicurezza → Fotocamera → "Consenti alle app desktop di accedere alla fotocamera".

## 3. Demo live

```bash
python demo/main.py --source 0          # oppure 1, o l'URL http://... del telefono
```
| Tasto | Azione |
|---|---|
| `f` | schermo intero on/off |
| `r` | reset allarme (dopo ogni dimostrazione) |
| `q` / `Esc` | esci |

**Copione consigliato:** penna, occhiali e quaderno sul tavolo, il sacchetto in mano o appoggiato.
1. Prendi il quaderno, mostralo e rimettilo giù: diventa giallo, poi torna verde. Nessun allarme.
2. Prendi la penna (giallo), avvicinala al sacchetto (arancione) e mettila dentro. Dopo ~1 s la persona diventa rossa e compare il banner.
3. Premi `r` e si riparte.

**Inquadratura (conta più di tutto):**
- Telefono a 1,5–2 m, alla stessa altezza del tavolo, luce frontale (niente finestra alle spalle).
- La penna è l'oggetto più difficile: usane una **grande e colorata**, non sottile e bianca.
- Tieni oggetti e sacchetto **ben dentro l'inquadratura**.

## 4. Registrare il video di backup

Salva solo il video **già anonimizzato** (con overlay e pannello), mai quello grezzo:
```bash
python demo/main.py --source 0 --record data/outputs/backup_demo.mp4
```
Da un file, senza finestra (più veloce):
```bash
python demo/main.py --source data/samples/test_clip.mp4 --headless --record data/outputs/debug_out.mp4
```
`data/` non finisce mai in git.

## 5. Taratura

```bash
python demo/diagnose.py --source data/samples/test_clip.mp4
python demo/diagnose.py --source data/samples/test_clip.mp4 --frames data/outputs/frames --every 2
```
Per ogni secondo stampa: privacy applicata, numero di mani, oggetti con confidence massima, stato di ogni traccia ed eventi. Tutte le soglie sono in `demo/config.yaml`, commentate.

Problemi tipici:
- **Penna mai vista:** abbassa `conf` della penna (es. 0.08) o avvicina il telefono.
- **Falsi oggetti sullo sfondo:** alza `conf` di quella classe.
- **Allarme troppo facile:** alza `state.missing_frames`.
- **Non scatta mai:** abbassa `state.missing_frames` o alza `state.bag_dist_px`.
- **FPS bassi:** `detection.detect_every: 2` e `detection.imgsz: 512`.

Test privacy: `python -m pytest demo/tests`.

## File

| File | Cosa fa |
|---|---|
| `main.py` | sorgente video, ciclo principale, tasti, registrazione |
| `detector.py` | YOLO-World (oggetti), YOLOv8-seg (persona), MediaPipe (mani, volto), scelta del device (CUDA / MPS / CPU) |
| `state_machine.py` | tracking e macchina a stati con isteresi |
| `privacy.py` | oscuramento della persona e fail-safe |
| `overlay.py` | bbox, persona rossa, banner, pannello eventi, FPS |
| `config.yaml` | classi, prompt e tutte le soglie |
| `diagnose.py` | report per secondo, per la taratura |
