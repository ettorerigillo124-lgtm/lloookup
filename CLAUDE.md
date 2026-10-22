# LookUp

Analisi video anti-taccheggio per il retail italiano. Gira in negozio (edge) sulle telecamere che il punto vendita ha già. Non usa biometria. La compliance GDPR fa parte del prodotto ed è il nostro vantaggio principale contro i concorrenti (Veesion, Blindzone).

Fase attuale: **prototipo per il cancello Tecnico** (scadenza 15/11/2026). Il codice serve a stimare perimetro MVP, prestazioni e costo hardware per negozio, e a fare da demo. Non è codice di produzione: privilegia chiarezza e misurabilità rispetto all'ottimizzazione.

## Regole privacy (non negoziabili)

Queste regole valgono per ogni riga di codice. Se un compito richiede di violarne una, fermati e chiedi.

1. **Niente biometria.** Non usare riconoscimento facciale, face embedding, gait recognition o qualsiasi identificazione di una persona specifica. La re-identificazione è consentita solo dentro la stessa visita, con ID anonimi e temporanei che non vengono mai salvati su disco.
2. **Volti oscurati.** Ogni fotogramma che lascia la pipeline in memoria (clip di evento, screenshot, debug) ha i volti sfocati. Il blur va applicato prima di qualsiasi scrittura.
3. **Minimizzazione.** Dal negozio escono solo eventi e metadati (tipo evento, timestamp, zona, confidenza) e, se serve, una clip breve con volti oscurati. Nessuno stream video continuo verso il cloud.
4. **Niente salvataggi silenziosi.** Nessun fotogramma o video viene scritto su disco salvo la clip dell'evento, con durata massima configurabile (default 10 s) e cancellazione automatica dopo la retention configurata.
5. **Dati di test.** Usa solo dataset pubblici o video inscenati con persone consenzienti (vedi `docs/datasets.md`). Mai filmati reali di negozi finché il memo legale non è validato.
6. **Licenze.** Prima di aggiungere un modello o una libreria, controlla la licenza e annotala in `docs/licenses.md`. Segnala le licenze AGPL/GPL (es. Ultralytics YOLO) e quelle "solo ricerca" (alcuni dataset), perché il prodotto sarà commerciale.

## Architettura (bozza)

```
telecamera/NVR (RTSP) o file video
  → ingest (decodifica, frame sampling)
  → detection persone
  → tracking (ByteTrack o simile)
  → zone (scaffale, casse, uscita) e regole
  → classificazione comportamenti (posa, occultamento)
  → evento: metadati + clip breve con volti oscurati
  → alert / dashboard
```

I dettagli e le decisioni vanno in `docs/ARCHITECTURE.md`. Aggiornalo quando cambia qualcosa di strutturale.

## Stack

- Python 3.11+
- OpenCV per I/O video
- PyTorch / ONNX Runtime per l'inferenza (export ONNX per poter girare su dispositivi edge)
- Detector persone: da scegliere tenendo conto della licenza
- Tracking: ByteTrack o equivalente
- Test: pytest

## Struttura del repo

```
lookup/
  ingest/      lettura RTSP e file video
  detect/      detection persone
  track/       tracking
  zones/       definizione zone e regole
  behavior/    classificazione comportamenti
  privacy/     blur volti, retention, minimizzazione
  events/      formato eventi, clip, alert
scripts/       download dataset, benchmark, valutazione
tests/
docs/          ARCHITECTURE.md, datasets.md, licenses.md
data/          video e dataset: MAI in git
```

## Come lavorare

- Un compito per sessione. Per ogni modifica non banale, prima proponi un piano.
- Scrivi i test prima o insieme al codice. Ogni modulo in `privacy/` ha test che verificano le regole sopra.
- Commit piccoli con messaggi chiari. Non committare mai file in `data/`, modelli pesanti o credenziali.
- Ogni funzione di rilevamento ha uno script di valutazione in `scripts/` che riporta almeno: detection corrette, falsi allarmi per ora di video, FPS. I falsi allarmi sono la metrica più importante: segnalali sempre.
- Quando non sei sicuro di un requisito di business o legale, chiedi invece di assumere.

## Milestone

1. **Conteggio persone** su un file video: detection + tracking + conteggio, output JSON. (Prima fetta, coincide con la prova pratica per i candidati tecnici.)
2. **Benchmark hardware**: FPS per telecamera sul dispositivo edge candidato → quante telecamere per box → costo hardware per negozio.
3. **Zone e permanenza**: definire zone su un'inquadratura, misurare tempo di sosta.
4. **Uscita senza passaggio in cassa**: percorso del track tra zone.
5. **Comportamenti**: occultamento della merce (pose estimation), con valutazione su clip etichettate.
6. **Eventi e alert**: clip con volti oscurati, notifica, dashboard minima.

## Comandi

<!-- Da completare quando il progetto è inizializzato -->
- Installazione: `pip install -e ".[dev]"`
- Test: `pytest`
- Demo conteggio: `python -m lookup.count --video data/samples/esempio.mp4`
