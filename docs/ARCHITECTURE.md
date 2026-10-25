# Architettura LookUp

Stato: **approvata nell'impianto generale il 22/10/2026**. Le domande aperte sono in fondo. Il documento va aggiornato quando cambia qualcosa di strutturale (`CLAUDE.md`).

## 1. Scopo e perimetro del prototipo

Il cancello Tecnico (**15/11/2026**) chiede tre cose: il perimetro dell'MVP, il costo hardware per negozio e materiale per valutare i candidati tecnici. **Non chiede un prodotto funzionante.**

| | Contenuto |
|---|---|
| **Obbligatorio entro il 15/11** | M1 (conteggio persone su file, output JSON) · M2 (benchmark hardware → tabella box → telecamere → €/telecamera) |
| **Facoltativo, se avanza tempo** | Demo dell'occultamento con regole sui keypoint, su dati pubblici o inscenati |
| **Da tenere fin da M1 (costano poco)** | Interfacce `Protocol` tra i blocchi · tipo `RedactedFrame` più test/lint che vieta `imwrite`/`VideoWriter` fuori da `privacy/` · nessun `track_id` negli output di prodotto |
| **Produzione: progettato qui, non implementato ora** | Ring buffer di pacchetti compressi · cifratura AES delle clip · job di retention · encoder hardware · sincronizzazione offline-first |
| **Fuori dal prototipo** | Uscita senza cassa multi-camera · re-ID · backend cloud · MQTT · GStreamer/DeepStream · omografia · analisi delle casse e dei dipendenti |

## 2. Diagramma della pipeline

Legenda: `[P]` = implementato nel prototipo · `[F]` = facoltativo nel prototipo · `[PROD]` = solo produzione

```
 NVR / telecamera IP (H.264 o H.265)                file video (dataset pubblici, inscenati)
   ├─ substream (analisi) ───────────────┐                          │
   └─ mainstream (clip) ──┐              │                          │
                          │              ▼                          ▼
                          │   [1 INGEST]  [P] FileSource (OpenCV) · [P, da M2] RtspSource (PyAV)
                          │       │  scheduler orari di apertura · scarta l'audio · sampling fps
                          ▼       │
     [PROD] ring buffer RAM di    │ Frame (RAM)
            pacchetti compressi   ▼
                          │   [2 DETECT]  [P] YOLOX (ONNX Runtime) → Detections
                          │       ▼
                          │   [3 TRACK]   [P] ByteTrack → Tracks (ID effimeri, solo RAM)
                          │       ▼
                          │   [4 ZONE]    [P] M1: contatore · [PROD] poligoni, permanenza, linee
                          │       ▼
                          │   [5 BEHAVIOR][F] RTMPose sui track in zona scaffale + regole → BehaviorSignal
                          │       ▼
                          │   [6 EVENT]   Event (metadati, nessun track_id)
                          │       │
                          └──────►├──► PRIVACY GATE (privacy/) — UNICO punto che scrive pixel
                                  │      redact(): blur della testa da box e keypoint (fail-closed)
                                  │      frame senza detection → scartato
                                  │      → RedactedFrame → writer (clip ≤10 s / video demo)
                                  │      [PROD] encoder hardware → cifratura AES-GCM → TTL 72 h
                                  ▼
                       disco locale: SQLite eventi + clip oscurate (+ data/demo, data/eval in dev)
                                  │
                                  ▼
                          [7 ALERT/DASHBOARD] [P] FastAPI locale, testo neutro "evento da verificare"
                                  │
                                  ▼  [PROD] sync offline-first quando c'è rete
                       fuori dal negozio: solo Event (+ clip oscurata se configurata)
```

**Modello di esecuzione**
- **Prototipo:** un processo Python, un ciclo sincrono per sorgente.
- **Produzione:** decodifica hardware per telecamera, inferenza a batch su tutte le telecamere del box, logica per telecamera in thread o processi.

I contratti dati (§3) restano gli stessi nei due casi.

## 3. Contratti tra i blocchi

Sono dataclass Python in memoria, con interfacce `Protocol` per ogni blocco (`FrameSource`, `Detector`, `Tracker`, `Counter`/`ZoneEngine`, `Sink`). Si serializzano solo `Event`, i report JSON e, in sviluppo, i file di `data/eval/`.

| Tipo | Contenuto | Persistenza |
|---|---|---|
| `Frame` | `camera_id`, `frame_idx`, `ts_mono`, `ts_wall`, `image: np.ndarray (H×W×3 BGR)`, `source_size` | Solo RAM |
| `Detections` | `xyxy (N,4) float32` in pixel, `score (N,)`, `class_id (N,)`, riferimento al frame | Solo RAM |
| `Tracks` | `track_id` (int effimero, riparte a ogni avvio), `xyxy`, `score`, `age`, `hits`, `state` | Solo RAM; scritti in `data/eval/` **solo su dataset pubblici** (D9) |
| `ZoneEvent` | `track_id`, `zone_id`, `kind` (enter/exit/cross), `ts`, `dwell_s` | Solo RAM |
| `Pose` | `keypoints (17,3)` formato COCO (x, y, visibilità), `track_id` | Solo RAM |
| `BehaviorSignal` | `track_id`, `type`, `score`, `ts_start`, `ts_end`, `evidence` | Solo RAM |
| `RedactedFrame` | Immagine già passata da `privacy.redact()`; non si costruisce in altro modo | Unico tipo accettato dai writer |
| `Event` | `event_id` (uuid4), `store_id`, `camera_id`, `type` (nome neutro), `ts_start`, `ts_end`, `zone_ids`, `confidence`, `model_versions`, `clip_ref?`, `status` (`da_verificare`) | SQLite locale; in produzione sincronizzato. **Mai** `track_id`, immagini o coordinate per persona |

## 4. Blocchi: scelte e motivazioni

### 4.1 Ingest
- **Prototipo:**
  - M1: `FileSource` con OpenCV `VideoCapture`.
  - Da M2: `RtspSource` con PyAV (riconnessione, timestamp, accesso ai pacchetti).
  - Analisi sul **substream**, sampling configurabile (5/10/15 fps).
- **H.265:** obbligatorio. L'FFmpeg delle wheel OpenCV e PyAV decodifica HEVC su CPU. In M2 si misura il costo di H.264 rispetto a H.265; Jetson (NVDEC) e Intel (QuickSync) lo decodificano in hardware.
- **Orari di apertura:** configurazione per negozio (YAML). Fuori orario l'ingest non apre gli stream. Per i file l'orario viene ignorato.
- **Audio:** mai decodificato.
- **Produzione:** GStreamer con decodifica hardware; ring buffer in RAM dei **pacchetti compressi** del mainstream (pochi MB per telecamera), decodificati solo quando scatta un evento.

### 4.2 Detection persone
- **Prototipo:** YOLOX-S e YOLOX-Tiny, pesi COCO di Megvii filtrati sulla classe person, eseguiti con ONNX Runtime. D-FINE-S o RT-DETRv2-S come riferimento di accuratezza in M2.
- **Motivazione:**
  - Apache-2.0.
  - Export ONNX maturo.
  - È nel Hailo Model Zoo, quindi confrontabile tra Jetson e Hailo.
  - INT8 semplice (CNN).
  - È il detector nativo di ByteTrack.
- **Produzione:** si decide dopo M2, con fine-tuning su dati con licenza pulita (video inscenati).

### 4.3 Tracking
- **Prototipo e produzione:** ByteTrack (MIT), solo moto, senza aspetto. OC-SORT come piano B per le occlusioni.
- **Motivazione:** leggero, robusto alle detection con confidenza bassa, nessun embedding (regola 1).

### 4.4 Zone e regole
- **M1:** `Counter` con:
  - persone uniche (track distinti);
  - massimo di persone contemporanee;
  - timeline al secondo;
  - linea in/out opzionale.
- **Dopo M1:** poligoni e linee in coordinate normalizzate (YAML per telecamera) con `shapely`; ancoraggio al centro del lato inferiore del box; permanenza.
- **Fuori dal prototipo:** sequenze tra telecamere (uscita senza cassa), omografia.

### 4.5 Comportamenti (facoltativo)
- **Demo:** RTMPose-t/s via `rtmlib`, **solo** sui track in zona scaffale. Una macchina a stati sui keypoint: polso in zona scaffale → polso vicino a busto, tasca o borsa entro T s.
- **Pesi:** usare solo varianti con dataset noti. Le "coco" sono AIC+COCO: vedi `licenses.md`, Q-L2.
- **Produzione:** classificatore leggero su sequenze di scheletri, addestrato da noi su dati con licenza pulita, con le regole come pre-filtro.

### 4.6 Evento, privacy gate e clip
Il modulo `privacy/` è l'**unico** punto che scrive pixel su disco. Vale per:
- clip degli eventi;
- video demo con overlay;
- screenshot di debug.

**`redact(frame, tracks, poses) -> RedactedFrame`**
1. **Regione della testa** dai keypoint della testa (naso, occhi, orecchie) se ci sono; altrimenti dal 15–25% superiore del box della persona, allargato con un margine. Le percentuali vanno tarate.
2. **Pixelatura forte o riempimento pieno**, non un gaussiano leggero.
3. **Frame senza nessuna detection → scartato**: non arriva mai al writer (fail-closed).
5. **Soglia di detection più bassa per la privacy:** il gate oscura anche le detection incerte, con una soglia più bassa di quella di analisi (default 0,1 contro 0,4, entrambe configurabili). Nel dubbio si oscura.
4. **Punto di estensione `FaceDetector`:** interfaccia presente ma **disattivata**. Si attiva solo con il via libera del legale (Q-L1). In ogni caso aggiunge aree da oscurare, non ne toglie.

**Writer**
- Accettano solo `RedactedFrame`.
- La clip ha durata massima configurabile (default 10 s).
- **Test/lint:** un test fallisce se `cv2.imwrite`, `cv2.VideoWriter` o equivalenti compaiono fuori da `privacy/`. Altri test verificano la durata massima, lo scarto dei frame senza detection e l'assenza di `track_id` negli `Event`.

**Produzione**
- Encoder hardware (NVENC/QuickSync) oppure OpenH264, **non** libx264 (GPL).
- Cifratura AES-GCM.
- Retention **72 h** di default, configurabile.
- Accesso solo al responsabile del negozio.
- Richieste delle forze dell'ordine → si rimanda all'NVR del negozio: LookUp non ha video in chiaro (da confermare, Q-L10).

### 4.7 Alert e dashboard
- **Prototipo:** FastAPI locale, che mostra la lista eventi e, se c'è, la clip oscurata.
- **Art. 22 GDPR (decisione D11):**
  - L'evento nasce con `status = da_verificare`.
  - Il testo è neutro ("evento da verificare", mai "furto").
  - Nessuna azione automatica: decide sempre una persona.
  - I nomi dei tipi di evento sono descrittivi e non accusatori (es. `interazione_scaffale_da_verificare`).
- **Produzione:**
  - Offline-first: SQLite locale come coda, sincronizzazione al ritorno della rete.
  - Destinatario e canale sono aperti (Q-B6); ipotesi: responsabile o vigilante su smartphone, latenza < 10 s.
  - Nessuna notifica attraverso servizi extra-UE.

## 5. Dove si applicano le regole privacy

| Regola (`CLAUDE.md`) | Dove si applica | Come si verifica |
|---|---|---|
| 1. Niente biometria | Tracker solo moto; nessun embedding né face recognition; `track_id` effimeri | Test: gli `Event` e i JSON di prodotto non contengono `track_id`. Revisione delle dipendenze in `licenses.md` |
| 2. Volti oscurati | `privacy.redact()` prima di ogni scrittura; `RedactedFrame` unico tipo scrivibile | Test/lint su `imwrite`/`VideoWriter`; recall del blur su clip inscenate annotate |
| 3. Minimizzazione | Fuori dal negozio escono solo `Event` e clip oscurata facoltativa; nessuno stream | Il prototipo non ha componenti di rete in uscita |
| 4. Niente salvataggi silenziosi | Ring buffer solo in RAM; **swap disabilitato o cifrato e core dump disattivati sul box**; log senza frame né ID | Checklist di installazione del box; test sulla durata massima delle clip |
| 5. Dati di test | Solo dataset pubblici o inscenati; `data/` escluso da git; `data/eval/` (track con ID) solo per dataset pubblici | `.gitignore` e revisione |
| 6. Licenze | `docs/licenses.md` | Revisione a ogni nuova dipendenza |

**Video demo per il deck:** passa dallo stesso privacy gate e va in `data/demo/` (escluso da git). Va **rivisto a mano frame per frame** prima di metterlo nel deck: il blur fail-closed e la soglia privacy più bassa riducono il rischio ma non coprono le persone mai rilevate in un frame con altre detection. Nel deck vanno **solo** video inscenati con liberatoria firmata o clip di dataset pubblici, mai video di negozi (D21).

## 6. Prototipo e produzione

| Blocco | Prototipo (entro il 15/11) | Produzione (ipotesi) |
|---|---|---|
| Ingest | OpenCV (file), PyAV (RTSP), CPU, H.264 e H.265 | GStreamer con decodifica hardware, ring buffer di pacchetti compressi |
| Detector | YOLOX-S/Tiny COCO (Apache-2.0) | Modello scelto dopo M2, fine-tuned su dati inscenati |
| Runtime | ONNX Runtime; TensorRT su Jetson in M2 | TensorRT, HailoRT o OpenVINO, in base al box |
| Tracking | ByteTrack | ByteTrack / OC-SORT |
| Zone | Contatore (M1), linea in/out opzionale | Poligoni, permanenza, sequenze tra zone |
| Comportamento | Facoltativo: RTMPose + regole | Classificatore su scheletri addestrato da noi |
| Privacy | `redact()` dalla testa, scarto dei frame senza detection, test/lint | Più face detector (se approvato), cifratura, retention 72 h |
| Encoding | OpenCV (solo uso interno, nessuna distribuzione) | Encoder hardware o OpenH264 |
| Eventi | SQLite locale | SQLite + sync offline-first |
| Alert | Dashboard FastAPI locale | Notifica al responsabile o vigilante (canale UE), latenza < 10 s |

## 7. Hardware e benchmark (M2)

### 7.1 Piano
1. **Fase A, PC/portatile, ONNX Runtime CPU:**
   - fps per modello e risoluzione;
   - costo della decodifica di substream e mainstream, H.264 e H.265;
   - costo di tracking e post-processing.

   Serve a validare gli script di benchmark e a isolare il costo "non-inferenza" della pipeline.
2. **Fase B, Jetson Orin Nano Super developer kit:** FP16 e INT8 con TensorRT, decodifica NVDEC, potenza assorbita, temperatura e throttling, soak test di 24 h.
3. **Fase C, mini-PC Intel N100:** baseline solo CPU/iGPU con OpenVINO, misurata. Misura anche la parte host (decodifica, pre/post-processing, tracking) di un futuro box Hailo.
4. **Hailo-8:** stimato combinando i dati del Hailo Model Zoo con la parte host misurata sull'N100, segnato come **stimato** nella tabella finale.

### 7.2 Limiti del confronto (Jetson + N100 acquistati, Hailo stimato)
**I numeri Jetson no; il confronto con Hailo e CPU sì, in parte.** Tre motivi:
1. **Il Jetson esegue qualunque modello** (FP16, transformer). C'è il rischio di scegliere un modello che su Hailo non si compila o perde accuratezza in INT8.
   - *Mitigazione:* la matrice di confronto tra box usa solo modelli presenti nel Hailo Model Zoo (YOLOX-S/Tiny). La presenza delle varianti esatte va verificata. I DETR restano solo riferimento di accuratezza.
2. **I numeri Hailo pubblicati misurano solo l'acceleratore.** Escludono decodifica, pre/post-processing, NMS, tracking e overhead Python, che su un box Hailo girano sulla CPU host (es. N100) e possono essere il vero collo di bottiglia.
   - *Mitigazione:* misurare in fase A la parte CPU della pipeline per frame e riportarla separata dall'inferenza. Il portatile però non rappresenta un N100.
   - **Deciso (D17):** acquistiamo anche un mini-PC N100 (circa €200–300, da verificare), che misura la baseline solo CPU e la parte host di un futuro box Hailo.
3. **Il developer kit non è il box di produzione.** Carrier industriale, dissipazione e power mode cambiano. Misurare nel power mode che useremmo in negozio e dichiararlo nella tabella.

### 7.3 Output di M2
| Box | Prezzo (fonte, data) | Modello / risoluzione / precisione | fps di analisi per telecamera | Telecamere supportate | W a pieno carico | €/telecamera | €/telecamera/mese (36 mesi; 24 / 48) | Misurato o stimato |
|---|---|---|---|---|---|---|---|---|

**Formula:** €/telecamera = costo del box (più alimentatore, storage, case) / telecamere supportate. €/telecamera/mese = €/telecamera / mesi di ammortamento. Ipotesi di lavoro 36 mesi, con variante a 24 e 48 mesi (Q-B3).

"Telecamere supportate" = numero massimo di telecamere elaborate in modo sostenuto all'fps di analisi scelto per 24 h, **senza calo della recall** rispetto al riferimento e senza throttling.

## 8. Milestone 1 come scheletro della pipeline

```
python -m lookup.count --video data/samples/esempio.mp4 [--line x1,y1,x2,y2] [--analysis-fps 10] [--show]

ingest.FileSource → detect.YoloxDetector (ONNX Runtime) → track.ByteTracker → zones.Counter → events.JsonSink
```

- Le interfacce `Protocol` sono quelle del §3. In M2 basta aggiungere `RtspSource` e i backend TensorRT o OpenVINO; in M3 `Counter` diventa `ZoneEngine`.
- Già in M1 ci sono `privacy/` con `RedactedFrame`, `redact()` e il test/lint, anche se M1 non scrive frame. `--show` visualizza soltanto e non scrive.
- **Output di prodotto (JSON):** nessun `track_id`. Contiene `unique_people`, `max_concurrent`, `timeline` al secondo, `line_crossings` (se c'è `--line`), `id_switch_estimate`, `processing_fps`, `analysis_fps`, `models`.
- **Stima dei cambi di ID** (sempre riportata):
  - euristica senza ground truth: track brevi o frammentati, cioè nati e morti lontano dai bordi dell'inquadratura;
  - con ground truth: cambi di ID e IDF1 calcolati da `scripts/eval_count.py`.
- **Valutazione:** `scripts/eval_count.py` scrive i track con ID in `data/eval/` (solo dataset pubblici) e riporta errore sul conteggio, cambi di ID, falsi positivi e fps.

## 9. Alternative scartate

| Alternativa | Motivo |
|---|---|
| Ultralytics YOLOv5/8/11, RT-DETR e YOLO-pose di Ultralytics, YOLOv10 | AGPL-3.0: serve la licenza enterprise |
| YOLOv6, YOLOv7, YOLOv9 | GPL-3.0 |
| YOLO-NAS | Pesi con licenza Deci restrittiva |
| RF-DETR Atto/Femto/Pico/XL/2XL | PML 1.0 (le varianti N/S/M/L restano candidate, Apache-2.0) |
| Pesi YOLOX person-only di ByteTrack | Addestrati su CrowdHuman e MOT17, non commerciali |
| SORT, DeepSORT | GPL-3.0 |
| boxmot | AGPL-3.0 |
| BoT-SORT/StrongSORT con ReID, qualunque ReID di aspetto | Fuori perimetro fino al memo legale (Q-L4) |
| OpenPose | Licenza non commerciale |
| MediaPipe Pose, MoveNet | Single-person o frontali, inadatti alle inquadrature CCTV dall'alto |
| VideoMAE | CC BY-NC 4.0 |
| Action recognition video (X3D ecc.) nel prototipo | Costo e dati necessari non compatibili con il 15/11 |
| InsightFace (RetinaFace/SCRFD) | Pesi solo ricerca |
| Face detector nel prototipo | Default prudente fino al memo (Q-L1); si oscura la testa ricavata dal corpo |
| Gaussiano leggero per il blur | Parzialmente reversibile |
| libx264 per le clip di prodotto | GPL |
| Grafana nel prodotto | AGPL-3.0 |
| Google Coral | 4 TOPS, operatori limitati, software poco mantenuto |
| DeepStream/GStreamer, MQTT, backend cloud, omografia nel prototipo | Non servono per il cancello del 15/11 |

## 10. Rischi tecnici e metriche

I falsi allarmi sono la metrica principale (`CLAUDE.md`) e vanno riportati in ogni script di valutazione.

| Rischio | Metrica | Dove si misura |
|---|---|---|
| Angolazioni dall'alto, dome, fisheye | Recall del detector divisa per angolazione | MERL (zenitale), CAVIAR, video inscenati |
| Bassa risoluzione e compressione del substream, IR notturno | Recall in funzione dell'altezza della persona in pixel → altezza minima nei requisiti di installazione | Dataset ridimensionati e ricompressi, inscenati |
| H.265 e codec degli NVR consumer | CPU per stream con H.264 e H.265; errori di decodifica | M2 |
| Occlusioni (scaffali, folla) | Cambi di ID al minuto, IDF1, sovrastima delle persone uniche | M1, CAVIAR |
| fps di analisi bassi per limiti di calcolo | IDF1 e recall a 5, 10 e 15 fps | M1 e M2 |
| Falsi allarmi (commessi, telefono in tasca, manichini, poster, specchi) | **Falsi allarmi all'ora** su video "normale"; obiettivo ≤ 1-2 al giorno per negozio (da validare) | MERL, ore di video inscenato normale |
| Blur mancato | Recall del blur (frame con un volto visibile e non oscurato) su clip inscenate annotate; frame scartati in % | Test di `privacy/`, revisione dei video demo |
| Posa inaffidabile dall'alto o su corpi parziali | % di keypoint visibili; precision e recall dell'occultamento | Demo facoltativa |
| Instabilità RTSP | Tempo di riconnessione, deriva dei timestamp, soak test di 24 h | M2 (RtspSource) |
| Calore e consumi nell'armadio | Temperatura, throttling, W | M2 sul Jetson |
| Confronto hardware parziale (Hailo stimato) | Colonna "misurato o stimato" nella tabella di M2 | M2 |
| Distanza tra dati inscenati o pubblici e negozio reale | Dichiarata in tutti i report; pilota dopo il memo legale | — |
| Licenze dei pesi | `licenses.md` aggiornato, colonna "verificata" | Continuo |

## 11. Decisioni registrate

| # | Data | Decisione | Motivazione |
|---|---|---|---|
| D1 | 22/10/2026 | Perimetro del 15/11: M1 + M2 obbligatori, demo dell'occultamento facoltativa | Il cancello chiede perimetro, costi e materiale per i candidati, non il prodotto |
| D2 | 22/10/2026 | Detector del prototipo: YOLOX (Apache-2.0), pesi COCO | Licenza permissiva, ONNX, confrontabile su Jetson e Hailo |
| D3 | 22/10/2026 | Tracker: ByteTrack, solo moto | MIT, nessun embedding |
| D4 | 22/10/2026 | Interfacce `Protocol` e contratti dati del §3 fin da M1 | M1 è lo scheletro della pipeline |
| D5 | 22/10/2026 | `privacy/` è l'unico punto che scrive pixel; `RedactedFrame` e test/lint fin da M1 | Regola 2 verificabile automaticamente |
| D6 | 22/10/2026 | Blur solo sulla regione della testa ricavata da box e keypoint; nessun face detector; frame senza detection scartato; estensione `FaceDetector` disattivata | Default prudente fino al memo legale (Q-L1) |
| D7 | 22/10/2026 | Nessun `track_id` negli output di prodotto | Regola 1 |
| D8 | 22/10/2026 | Conteggio M1: persone uniche, massimo contemporanee, timeline al secondo, linea in/out opzionale, stima dei cambi di ID sempre presente | Risposta business 1 |
| D9 | 22/10/2026 | Sui dataset pubblici i track con ID si possono scrivere in `data/eval/` (escluso da git); restano vietati embedding e identificazione | Risposta legale 5 |
| D10 | 22/10/2026 | Dataset "solo ricerca": solo valutazione interna, mai training del modello commerciale | Risposta legale 3 |
| D11 | 22/10/2026 | Art. 22: alert "da verificare", testo neutro, nessuna azione automatica | Risposta legale 8 |
| D12 | 22/10/2026 | Nessuna analisi delle casse né dei dipendenti; sweethearting fuori perimetro | Art. 4 dello Statuto dei Lavoratori (Q-L6) |
| D13 | 22/10/2026 | Retention delle clip 72 h di default, accesso solo al responsabile del negozio | Risposta legale 10 (da confermare) |
| D14 | 22/10/2026 | Elaborazione solo negli orari di apertura configurati | Risposta business 8 |
| D15 | 22/10/2026 | Offline-first: il box funziona senza rete | Risposta business 9 |
| D16 | 22/10/2026 | Ingest compatibile con H.265 | Ipotesi sul profilo dei negozi (Hikvision/Dahua) |
| D17 | 22/10/2026 | M2: PC con ONNX Runtime → Jetson Orin Nano Super dev kit + mini-PC N100; Hailo-8 stimato combinando i dati del Model Zoo con la parte host misurata sull'N100 | Due box su tre misurati a basso costo; limiti in §7.2 |
| D18 | 22/10/2026 | Dashboard solo locale (FastAPI) nel prototipo; nessuna notifica extra-UE | Risposte business 7 e legale 12 |
| D19 | 22/10/2026 | Re-ID di aspetto e uscita senza cassa multi-camera fuori dal prototipo | Risposte legale 4 e business 5 |
| D20 | 22/10/2026 | Codice dei candidati libero (anche Ultralytics), ma non entra nel repo | Risposta business 11 |
| D21 | 22/10/2026 | Nel deck solo video inscenati con liberatoria firmata o clip di dataset pubblici, dopo revisione manuale frame per frame; mai video di negozi. Il privacy gate usa una soglia di detection più bassa di quella di analisi (default 0,1 vs 0,4) | Blur mancato sulle persone non rilevate |
| D22 | 25/10/2026 | Demo investitori in `demo/` sul branch `demo-mvp`: YOLO-World + YOLOv8-seg (Ultralytics, AGPL) come dipendenza, MediaPipe per mani e volto. Non entra nella pipeline di prodotto `lookup/` | Serve una detection open-vocabulary (penna, occhiali) per la demo di venerdì. Dettagli in `demo/DECISIONS.md` |

## 12. Domande aperte

Responsabile del memo legale: **Andrea**, scadenza **30/10/2026**. Le domande di business si chiudono con le visite del cancello Campo, entro il **15/11/2026**. "Da assegnare" = responsabile non ancora indicato.

### Legali
| # | Domanda | Responsabile | Scadenza | Default fino alla risposta |
|---|---|---|---|---|
| Q-L1 | Un face detector usato solo per oscurare è ammesso? | Andrea | 30/10 | No: solo testa da corpo (D6) |
| Q-L2 | Pesi addestrati su COCO, WIDER FACE e **AI Challenger** (RTMPose) usabili nel prodotto commerciale? | Andrea | 30/10 | Ok per uso interno nel prototipo per COCO e WIDER FACE: **default dei founder in attesa del memo, non parere legale**. AIC non ancora coperto |
| Q-L3 | Dataset "solo ricerca": la valutazione interna è ammessa? | Andrea | 30/10 | Solo valutazione, mai training (D10) |
| Q-L4 | Re-ID di aspetto dentro la stessa visita, solo in RAM? | Andrea | 30/10 | No, fuori perimetro |
| Q-L5 | Regola 1 sui dataset pubblici | — | — | **Chiusa** (D9) |
| Q-L6 | Art. 4 dello Statuto dei Lavoratori: cosa serve per installare il sistema dove ci sono dipendenti? Le casse si possono analizzare? | Andrea (con il giuslavorista, nel memo legale) | 30/10 | Nessuna analisi di casse o dipendenti (D12) |
| Q-L7 | Ruoli GDPR (negozio titolare, LookUp responsabile ex art. 28?) e DPIA | Andrea | 30/10 | Ipotesi: negozio titolare, LookUp responsabile; template DPIA per i clienti |
| Q-L8 | Art. 22 | — | — | **Chiusa nel prodotto** (D11) |
| Q-L9 | AI Act: conferma che il sistema è fuori da pratiche vietate e alto rischio | Andrea | 30/10 | Niente biometria, niente riconoscimento delle emozioni |
| Q-L10 | Retention 72 h, accesso e risposta alle forze dell'ordine (rinvio all'NVR) | Andrea | 30/10 | D13 |
| Q-L11 | Brevetti dei codec H.264/H.265 per le clip ricodificate | Andrea | 30/10 | Non bloccante: nessuna clip distribuita nel prototipo |
| Q-L12 | Notifiche attraverso servizi extra-UE | Andrea | 30/10 | Nessuna nel prototipo |
| Q-L13 | Modello di liberatoria per i video inscenati che copra valutazione, training commerciale e uso nel deck | Andrea | 30/10 | Video inscenati solo per uso interno finché non c'è la liberatoria |

### Business e prodotto
| # | Domanda | Responsabile | Scadenza | Ipotesi di lavoro |
|---|---|---|---|---|
| Q-B2 | Profilo del negozio: telecamere, NVR, risoluzioni, codec, dome e fisheye | Ettore (visite Campo) | 15/11 | 4-8 telecamere, Hikvision/Dahua, 1080p + substream, H.265 possibile |
| Q-B3 | Budget hardware, modello di vendita, **periodo di ammortamento** del box | Ettore e Andrea | 15/11 | Hardware incluso nel canone; canone sotto quello di Veesion; ammortamento 36 mesi (varianti 24/48) |
| Q-B4 | Soglia di falsi allarmi accettata dai negozianti | Ettore (visite Campo) | 15/11 | ≤ 1-2 al giorno per negozio; recall misurata, senza soglia |
| Q-B5 | Uscita senza cassa: esito della verifica dell'ipotesi 6 (numero di telecamere, integrazione con la cassa) | Andrea | — | Esito da inserire; fuori dal prototipo |
| Q-B6 | Destinatario, canale e latenza degli alert | Ettore (visite Campo) | 15/11 | Responsabile o vigilante su smartphone, < 10 s |
| Q-B9 | Connettività, spazio, alimentazione e aggiornamenti remoti | Ettore (visite Campo) | 15/11 | Offline-first (D15) |
| Q-B12 | Secondo acquisto per M2: mini-PC N100 | — | — | **Chiusa**: si acquista (D17) |

## 13. Licenze (sintesi)

Il registro completo, con la colonna "verificata", è in [`licenses.md`](licenses.md). Componenti scelti per il prototipo:

| Componente | Licenza | Verificata |
|---|---|---|
| OpenCV / opencv-python (con FFmpeg LGPL-2.1) | Apache-2.0 / MIT | sì |
| PyAV | BSD-3-Clause | sì |
| ONNX Runtime | MIT | sì |
| YOLOX (codice) | Apache-2.0 | sì |
| YOLOX pesi COCO | Apache-2.0 (da confermare per le release), dataset COCO | no |
| ByteTrack | MIT | sì |
| shapely | BSD-3-Clause | sì |
| FastAPI | MIT | sì |
| rtmlib / RTMPose (facoltativo) | Apache-2.0; pesi AIC+COCO da verificare | sì (codice) / no (pesi) |
| TensorRT / JetPack (M2) | Proprietaria NVIDIA | no |
