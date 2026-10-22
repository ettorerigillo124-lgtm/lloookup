# Licenze

Registro di ogni codice, peso pre-addestrato, dataset e tool che usiamo o abbiamo valutato. Regola in `CLAUDE.md` (privacy, punto 6): **prima di aggiungere un modello o una libreria, la sua riga deve essere qui**.

## Come leggere la tabella

- **Verificata = sì**: il file di licenza è stato letto sul repository il **22/10/2026** (`raw.githubusercontent.com`, file `LICENSE*` nella radice del branch di default, salvo diversa indicazione in "Fonte").
  - Copre solo il codice nella radice. Sottocartelle, binari distribuiti e **pesi** possono avere licenze diverse: per questo i pesi hanno una riga separata.
- **Verificata = no**: dato riportato da conoscenza pregressa o da una fonte secondaria. Va controllato prima di usarlo fuori dal prototipo.
- **Stato**:
  - `prototipo`: lo usiamo adesso.
  - `candidato`: lo valutiamo nei benchmark.
  - `produzione?`: ipotesi per dopo il prototipo.
  - `scartato`: non usare nel prodotto.
- ⚠️ = AGPL/GPL, non commerciale o "solo ricerca": non può entrare nel prodotto commerciale senza licenza separata o parere legale.

## Codice: inferenza, visione, utilità

| Componente | Licenza | Uso | Stato | Verificata | Fonte / note |
|---|---|---|---|---|---|
| Python 3.11+ | PSF-2.0 | Runtime | prototipo | no | |
| NumPy | BSD-3-Clause | Array | prototipo | sì | `numpy/numpy` LICENSE.txt |
| OpenCV | Apache-2.0 | I/O video, disegno overlay | prototipo | sì | `opencv/opencv` |
| opencv-python (wheel pip) | MIT (packaging) + **FFmpeg LGPL-2.1** incluso; Qt LGPL-3 nelle wheel non-headless | Distribuzione di OpenCV | prototipo | sì | README e LICENSE-3RD-PARTY di `opencv/opencv-python`. Usare `opencv-python-headless` sul box |
| PyAV | BSD-3-Clause | Ingest RTSP, accesso ai pacchetti (da M2) | prototipo | sì | `PyAV-Org/PyAV` LICENSE.txt. Configurazione dell'FFmpeg incluso nelle wheel **non verificata** |
| ONNX Runtime | MIT | Inferenza su PC/CPU | prototipo | sì | `microsoft/onnxruntime` |
| PyTorch | BSD-3-Clause (stile BSD) | Export ONNX, eventuale training | prototipo | sì | `pytorch/pytorch` |
| OpenVINO | Apache-2.0 | Inferenza su x86 (stima e benchmark CPU/iGPU/NPU) | candidato | sì | `openvinotoolkit/openvino` |
| shapely | BSD-3-Clause | Poligoni delle zone | prototipo | sì | `shapely/shapely` LICENSE.txt |
| supervision | MIT | Riferimento per ByteTrack e zone | candidato | sì | `roboflow/supervision` LICENSE.md |
| PyYAML | MIT | Configurazione zone e orari | prototipo | sì | `yaml/pyyaml` |
| pydantic | MIT | Validazione di configurazione ed eventi | candidato | sì | `pydantic/pydantic` |
| FastAPI | MIT | Dashboard locale | prototipo | sì | `fastapi/fastapi` |
| pytest | MIT | Test | prototipo | sì | `pytest-dev/pytest` |
| SQLite | Public domain | Archivio eventi locale | prototipo | no | |
| cryptography (pyca) | Apache-2.0 OR BSD-3-Clause | Cifratura delle clip | produzione? | sì | `pyca/cryptography` |

## Codice: detector

| Componente | Licenza | Uso | Stato | Verificata | Fonte / note |
|---|---|---|---|---|---|
| YOLOX (Megvii) | Apache-2.0 | Detector persone | **prototipo** | sì | `Megvii-BaseDetection/YOLOX` |
| MMDetection / RTMDet | Apache-2.0 | Detector alternativo | candidato | sì | `open-mmlab/mmdetection` |
| RT-DETR / RT-DETRv2 (repo ufficiale) | Apache-2.0 | Riferimento di accuratezza | candidato | sì | `lyuwenyu/RT-DETR`. RT-DETR **dentro Ultralytics** è AGPL |
| D-FINE | Apache-2.0 | Riferimento di accuratezza | candidato | sì | `Peterande/D-FINE` |
| RF-DETR (pacchetto `rfdetr`) | Apache-2.0 | Possibile alternativa | candidato | sì | `roboflow/rf-detr`. ⚠️ I modelli "Plus" hanno licenza diversa, vedi i pesi |
| ⚠️ Ultralytics (YOLOv8/11, RT-DETR, YOLO-pose) | **AGPL-3.0** | — | scartato (serve licenza enterprise) | sì | `ultralytics/ultralytics` |
| ⚠️ YOLOv5 | **AGPL-3.0** | — | scartato | sì | `ultralytics/yolov5` |
| ⚠️ YOLOv10 | **AGPL-3.0** | — | scartato | sì | `THU-MIG/yolov10` |
| ⚠️ YOLOv6 | **GPL-3.0** | — | scartato | sì | `meituan/YOLOv6` |
| ⚠️ YOLOv7 | **GPL-3.0** | — | scartato | sì | `WongKinYiu/yolov7` |
| ⚠️ YOLOv9 | **GPL-3.0** | — | scartato | sì | `WongKinYiu/yolov9` |
| super-gradients (codice YOLO-NAS) | Apache-2.0 | — | scartato (per i pesi) | sì | `Deci-AI/super-gradients` |

## Codice: tracking

| Componente | Licenza | Uso | Stato | Verificata | Fonte / note |
|---|---|---|---|---|---|
| ByteTrack | MIT | Tracker | **prototipo** | sì | `ifzhang/ByteTrack` |
| OC-SORT | MIT | Tracker di riserva | candidato | sì | `noahcao/OC_SORT` |
| Norfair | BSD-3-Clause | Alternativa | candidato | sì | `tryolabs/norfair` |
| BoT-SORT | MIT | — | scartato (ReID di aspetto fuori perimetro) | sì | `NirAharon/BoT-SORT` |
| ⚠️ boxmot | **AGPL-3.0** | — | scartato | sì | `mikel-brostrom/boxmot` |
| ⚠️ SORT | **GPL-3.0** | — | scartato | sì | `abewley/sort` |
| ⚠️ DeepSORT | **GPL-3.0** | — | scartato | sì | `nwojke/deep_sort` |

## Codice: posa e comportamento

| Componente | Licenza | Uso | Stato | Verificata | Fonte / note |
|---|---|---|---|---|---|
| rtmlib | Apache-2.0 | Inferenza RTMPose in ONNX | candidato (demo facoltativa) | sì | `Tau-J/rtmlib` |
| MMPose / RTMPose | Apache-2.0 | Origine dei modelli di posa | candidato | sì | `open-mmlab/mmpose` |
| ViTPose | Apache-2.0 | — | scartato (troppo pesante per l'edge) | sì | `ViTAE-Transformer/ViTPose` |
| MediaPipe | Apache-2.0 | — | scartato (single-person, frontale) | sì | `google-ai-edge/mediapipe` |
| MoveNet (tfjs-models) | Apache-2.0 | — | scartato (inquadrature frontali) | sì | `tensorflow/tfjs-models`. Licenza dei pesi TF Hub non verificata |
| ⚠️ OpenPose | **Non commerciale (ricerca accademica)** | — | scartato | sì | `CMU-Perceptual-Computing-Lab/openpose` |
| MMAction2 (ST-GCN, PoseC3D) | Apache-2.0 | Riferimento per il classificatore su scheletri | produzione? | sì | `open-mmlab/mmaction2` |
| SlowFast / X3D | Apache-2.0 | — | scartato per ora (pesante) | sì | `facebookresearch/SlowFast` |
| ⚠️ VideoMAE | **CC BY-NC 4.0** | — | scartato | sì | `MCG-NJU/VideoMAE` |

## Codice: privacy (blur)

| Componente | Licenza | Uso | Stato | Verificata | Fonte / note |
|---|---|---|---|---|---|
| Regione della testa da box e keypoint | Codice nostro | Blur fail-closed | **prototipo** | — | Nessun face detector (decisione D6) |
| YuNet (OpenCV Zoo) | MIT (file nella cartella del modello) | Punto di estensione, **disattivato** | in attesa del legale | sì | `opencv/opencv_zoo/models/face_detection_yunet/LICENSE`. Addestrato su WIDER FACE: vedi i dataset |
| InsightFace (codice) | MIT | — | scartato (per i pesi) | sì | README di `deepinsight/insightface` |

## Codice: runtime edge e video (produzione)

| Componente | Licenza | Uso | Stato | Verificata | Fonte / note |
|---|---|---|---|---|---|
| NVIDIA JetPack / TensorRT | Proprietaria NVIDIA (gratuita) | Benchmark M2 su Jetson | prototipo (M2) | no | |
| NVIDIA DeepStream | Proprietaria NVIDIA | — | fuori dal prototipo | no | |
| Savant | Apache-2.0 | — | fuori dal prototipo | no | |
| Hailo Model Zoo | MIT | Stime dei benchmark Hailo | candidato | sì | `hailo-ai/hailo_model_zoo` |
| Hailo Dataflow Compiler / HailoRT | Proprietaria / da verificare | Compilazione per Hailo | produzione? | no | |
| pycoral (Coral Edge TPU) | Apache-2.0 | — | scartato (software poco mantenuto, 4 TOPS) | sì | `google-coral/pycoral` |
| GStreamer | LGPL-2.1 | Ingest con decodifica hardware | produzione? | no | |
| FFmpeg | LGPL-2.1 o GPL, **dipende dalla build** | Decodifica ed encode | prototipo (via wheel) | no | Le build con `--enable-gpl` diventano GPL |
| ⚠️ x264 (libx264) | **GPL-2.0** | — | scartato per le clip di prodotto | no | Usare encoder hardware o OpenH264 |
| OpenH264 (Cisco) | BSD-2-Clause | Encode H.264 software | produzione? | sì | `cisco/openh264`. La copertura brevettuale vale solo per il binario distribuito da Cisco: domanda Q-L11 |
| Eclipse Mosquitto | EPL-2.0 OR BSD-3-Clause (EDL-1.0) | — | fuori dal prototipo | sì | `eclipse-mosquitto/mosquitto` LICENSE.txt |
| ⚠️ Grafana | **AGPL-3.0** | — | scartato per il prodotto | sì | `grafana/grafana` |

## Pesi pre-addestrati

La licenza del codice **non** copre automaticamente i pesi: contano anche i termini dei dataset di addestramento (domande legali Q-L2 e Q-L3).

| Pesi | Licenza dichiarata | Dataset di addestramento | Uso | Stato | Verificata | Note |
|---|---|---|---|---|---|---|
| YOLOX-Nano/Tiny/S (release Megvii `0.1.1rc0`) | Apache-2.0 (come il repo, da confermare per le release) | COCO | Detector M1 e M2 | **prototipo** | no | Filtrare la classe person. Uso interno ok fino al memo legale (Q-L2) |
| YOLOX "humanart" (OpenMMLab, linkati da rtmlib) | non dichiarata | HumanArt + COCO | — | non usare | no | Licenza di HumanArt non verificata |
| ⚠️ YOLOX person-only di ByteTrack | non dichiarata | **CrowdHuman, MOT17** (non commerciali) | — | scartato | no | Usare i pesi COCO |
| RTMDet (MMDetection) | Apache-2.0 (da confermare) | COCO | Candidato | candidato | no | |
| RT-DETR / RT-DETRv2 / D-FINE | da verificare | COCO (alcune varianti con Objects365) | Riferimento | candidato | no | Licenza di Objects365 da verificare |
| RF-DETR N/S/M/L | Apache-2.0 | COCO (+ pre-training) | Candidato | candidato | sì | Tabella nel README di `roboflow/rf-detr` |
| ⚠️ RF-DETR Atto/Femto/Pico/XL/2XL | **PML 1.0** | — | — | scartato | sì | Stesso README |
| ⚠️ YOLO-NAS | **Licenza Deci restrittiva** (no rivendita/distribuzione senza consenso) | COCO, Objects365 | — | scartato | sì | `LICENSE.YOLONAS.md` in super-gradients |
| RTMPose (configurazioni "coco") | da verificare | **AIC + COCO** (non solo COCO) | Demo facoltativa | candidato | sì (dataset, da README) | Il README di `mmpose/projects/rtmpose` raggruppa questi modelli sotto "AIC+COCO". Licenza di AI Challenger: domanda Q-L2 |
| RTMPose Body7/Body8 (default di rtmlib) | da verificare | COCO, AIC, CrowdPose, MPII, Halpe e altri | — | non usare finché non è chiarito | sì (dataset, da README) | Troppi dataset con termini diversi |
| YuNet | MIT | WIDER FACE | Estensione disattivata | in attesa del legale | sì (licenza) / no (dataset) | |
| ⚠️ InsightFace (RetinaFace, SCRFD, ecc.) | **Solo ricerca non commerciale** | WIDER FACE, ecc. | — | scartato | sì | README di InsightFace |
| ⚠️ VideoMAE | **CC BY-NC 4.0** | Kinetics, ecc. | — | scartato | sì | |

## Dataset

Nessuna licenza di dataset è stata verificata finora. Fino al memo legale valgono le regole del 22/10/2026:
- I dataset "solo ricerca" si usano **solo per valutazione interna**, mai per addestrare il modello commerciale.
- COCO e WIDER FACE sono ok per uso interno nel prototipo. È un **default dei founder in attesa del memo legale, non un parere legale**.

| Dataset | Licenza | Uso previsto | Training ammesso? | Verificata | Note |
|---|---|---|---|---|---|
| COCO | Annotazioni CC BY 4.0; immagini con licenze Flickr miste (dato da verificare) | Pre-training dei pesi di terzi | Prototipo sì (interno); prodotto: Q-L2 | no | |
| WIDER FACE | da verificare | Pesi di YuNet | Prototipo sì (interno); prodotto: Q-L2 | no | |
| AI Challenger (AIC) | da verificare | Pesi di RTMPose | Q-L2 | no | Non coperto dal default dei founder del 22/10 su COCO e WIDER FACE |
| ⚠️ CrowdHuman | non commerciale (da verificare) | — | no | no | |
| ⚠️ MOT17 | CC BY-NC-SA 3.0 (da verificare) | Solo valutazione del tracking | no | no | |
| UCF-Crime | da verificare | Valutazione interna | no finché non è verificata | no | Da `docs/datasets.md` |
| DCSASS | da verificare | Valutazione interna | no finché non è verificata | no | |
| CCTV Shoplifting Detection (Simuletic) | da verificare | Valutazione interna | no finché non è verificata | no | |
| Innovatiana Shoplifting | da verificare | Valutazione interna | no finché non è verificata | no | |
| MERL Shopping | da verificare | Valutazione dell'interazione "normale" e dei falsi allarmi | no finché non è verificata | no | |
| CAVIAR | da verificare | Clip per M1, tracking | no finché non è verificata | no | |
| ShanghaiTech Campus | da verificare | Valutazione interna | no finché non è verificata | no | |
| CUHK Avenue | da verificare | Valutazione interna | no finché non è verificata | no | |
| SXD | da verificare | Valutazione interna | no finché non è verificata | no | |
| Roboflow Universe / Kaggle | una per dataset | Da valutare caso per caso | caso per caso | no | Una riga per ogni dataset effettivamente scaricato |
| Video inscenati LookUp | Proprietà LookUp + liberatorie dei partecipanti | Valutazione, training, demo | sì (se le liberatorie coprono l'uso commerciale) | — | Serve un modello di liberatoria: domanda Q-L13 |
