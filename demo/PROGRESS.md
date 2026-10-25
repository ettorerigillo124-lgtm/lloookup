# Stato della demo (25/10/2026)

## Cosa funziona (verificato sulla clip `data/samples/test_clip.mp4`, 480×640, 30 fps, 102 s)

- **Privacy:** la persona è oscurata in **tutti** i 3054 frame. 2988 con maschera, 58 con fail-safe a frame intero, 8 senza persona in scena (ultimi frame, stanza vuota). Ho controllato a vista i frame chiave del video registrato: nessun volto in chiaro. 8 test pytest su `privacy.py` passano.
- **Detection oggetti (YOLO-World a 640):** quaderno, occhiali, sacchetto e penna.
- **Macchina a stati:** VISIBILE → IN_MANO → VICINO_SACCHETTO → OCCULTATO, con isteresi, e la regola "la mano che teneva l'oggetto va al sacchetto".
- **Allarme:** persona rossa con contorno e banner. Resta attivo finché non si preme `r`.
- **UI:** bbox colorate per stato, pannello eventi con ora, FPS, `f` / `q` / `r`.
- **Altro:** registrazione del solo video anonimizzato (`--record`), `--headless`, `--list-cameras`, `diagnose.py`.

### Sequenza riconosciuta nella clip

Controllata guardando i frame. Ho verificato solo la parte bassa del corpo, senza volti.

| Evento reale (dai frame) | Rilevato | Esito |
|---|---|---|
| Quaderno messo nel sacchetto (~33–34 s) | `notebook occultata` 35,1 s | ✅ |
| Quaderno ripreso (~44 s) e rimesso nel sacchetto (~50 s) | `notebook di nuovo visibile` 45,1 s, `notebook occultata` 59,3 s (51,6 s con detection a ogni frame) | ✅, ma in ritardo con `detect_every: 2` |
| Occhiali tolti, tenuti in mano e messi nel sacchetto (~70–78 s) | `eyeglasses occultata` 79,7 s | ✅ |
| Penna presa (~86 s) e messa nel sacchetto (~94–96 s) | `pen presa in mano` 86,0 s, `pen occultata` 93,9 s | ✅ |
| Falsi allarmi | 0 sulla clip con la config attuale (prima della taratura: 1, il quaderno perso per 0,7 s a 30,8 s) | ✅ |

> ⚠️ La clip mostra **quattro** occultamenti (quaderno due volte, occhiali, penna), non solo la penna. Se nelle intenzioni solo la penna andava nel sacchetto, gli allarmi su quaderno e occhiali vanno considerati falsi: dimmelo e ritaro.

> ⚠️ Una sola clip, una sola stanza, una sola persona: **non** è una stima dei falsi allarmi per ora. È tarata su questa scena.

### Metriche

| | Valore |
|---|---|
| FPS sul container (CPU 4 core, nessuna GPU) | **8,7** con detection a frame alterni (5 con detection a ogni frame) |
| Tempo per frame (CPU) | YOLO-World 142 ms, YOLOv8-seg 40 ms, mani 19 ms, volto 4 ms, privacy + disegno 21 ms |
| FPS su Mac Apple Silicon (MPS) | **non misurato**: qui non c'è un Mac. Mi aspetto ≥15 FPS, da verificare |

Detection per classe sulla clip (confidence massima tipica):

| Classe | Prompt migliore | Confidence | Giudizio |
|---|---|---|---|
| sacchetto (shopper di tela) | `paper bag` / `shopping bag` | 0,6–0,95 | bene, con buchi di qualche frame |
| quaderno | `book` (molto meglio di `notebook`) | 0,4–0,9 quando è in mano e ben visibile | bene |
| occhiali | `glasses` | 0,35–0,9; meglio quando sono indossati | discreto; a 0,20 trovava falsi occhiali sullo scaffale, ora la soglia è 0,35 |
| penna | `pen` | **0,10–0,29** | **debole**: si vede solo a tratti. Dipende molto da distanza e risoluzione |

## Cosa non funziona o è fragile

1. **Penna:** a 480×640 e 2 m di distanza la penna è piccola, e la confidence sta poco sopra la soglia (0,10). Dal vivo: telefono più vicino (1,5 m), penna grande e colorata, 720p.
2. **Etichetta Left/Right di MediaPipe:** non è affidabile (spesso entrambe le mani risultano "Left"). Per questo la mano viene seguita per posizione.
3. **Ritardo dell'allarme:** 1,5 s dopo la sparizione (`missing_frames: 45`), di più se il sacchetto non viene rilevato per un po'. È voluto, per evitare falsi allarmi.
4. **Detection a frame alterni:** il secondo occultamento del quaderno arriva ~8 s dopo. Se il Mac regge ≥15 FPS anche con detection a ogni frame, metti `auto_skip_below_fps: 0`.
5. **Segmentazione:** a volte il letto in basso a sinistra viene preso per una persona. Peggiora solo l'estetica (zona oscurata in più), non la privacy.
6. **MPS + YOLO-World:** non provato. Se su Mac dà errori, metti `device: cpu` in config.
7. **Push su GitHub:** non ancora possibile, manca l'accesso. Il lavoro è nei commit locali del branch `demo-mvp`.

## COSE CHE DEVE FARE IL MARCO A MANO

1. **Dare accesso al repo:** installa la Claude GitHub App su `ettorerigillo124-lgtm/lloookup` (https://github.com/apps/claude/installations/select_target), così faccio push del branch `demo-mvp`. Senza questo il codice resta solo nel container, e il container viene cancellato.
2. **Installazione sul Mac:** segui README §1. Il **primo avvio va fatto con internet**: scarica ~45 MB di modelli e ~340 MB di CLIP. Fallo prima del giorno della demo.
3. **Permesso fotocamera:** Impostazioni di Sistema → Privacy e sicurezza → Fotocamera → attiva il Terminale (o iTerm / VS Code), poi riapri il terminale.
4. **Telefono come webcam:**
   - iPhone: attiva Continuity Camera, stesso Apple ID, Wi-Fi e Bluetooth accesi.
   - Android: DroidCam o IP Webcam, stessa rete.
   - Poi `python demo/main.py --list-cameras` per trovare l'indice. Dettagli in README §2.
5. **Misurare gli FPS sul Mac:** guarda il numero in basso a destra. Se è < 15, metti `detection.imgsz: 512` o `detect_every: 2`.
6. **Prova generale con la scena vera** (stesso tavolo, luce, distanza e oggetti), poi `python demo/diagnose.py --source <registrazione>`. Ritocca le soglie in `config.yaml` (README §5).
7. **Video di backup:** registralo con `--record` (README §4) e guardalo **tutto** prima di mostrarlo: deve essere anonimizzato in ogni frame.
8. **Licenze:** conferma con Andrea l'eccezione AGPL (DECISIONS.md D-01) e quella per `--record` (D-02).
9. **Consenso:** nella clip di test e nella demo compaiono solo persone che hanno dato il consenso (regola 5 del `CLAUDE.md`).

## Prossimi passi possibili

- **Penna più robusta:** un crop ad alta risoluzione attorno alle mani, con un secondo passaggio di YOLO-World sul crop.
- **Modalità "solo penna":** mettere in config che solo la penna è merce sorvegliata.
- **Più dati:** 2–3 clip di prova con sequenze diverse, inclusi movimenti che non devono dare allarme (prendo e rimetto giù), per misurare i falsi allarmi.
