"""Modelli: YOLO-World (oggetti), YOLO-seg (persona), MediaPipe (mani, volto)."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def pick_device(pref="auto"):
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Det:
    cls: str
    conf: float
    box: np.ndarray            # x1, y1, x2, y2 in pixel del frame
    prompt: str = ""


@dataclass
class Persons:
    mask: np.ndarray | None    # uint8 0/255, stessa dimensione del frame (None = nessuna persona)
    boxes: list = field(default_factory=list)


class ObjectDetector:
    """YOLO-World con piu' prompt per classe: per ogni classe tiene il prompt migliore."""

    def __init__(self, cfg, device):
        from ultralytics import YOLOWorld
        self.cfg = cfg
        self.device = device
        self.prompts, self.prompt_cls = [], []
        self.cls_conf, self.bag_classes = {}, set()
        for c in cfg["classes"]:
            self.cls_conf[c["name"]] = c.get("conf", 0.25)
            if c.get("bag"):
                self.bag_classes.add(c["name"])
            for p in c["prompts"]:
                self.prompts.append(p)
                self.prompt_cls.append(c["name"])
        self.model = YOLOWorld(cfg["models"]["world"])
        self.model.set_classes(self.prompts)
        self.min_conf = min(self.cls_conf.values())

    def __call__(self, frame):
        d = self.cfg["detection"]
        r = self.model.predict(frame, imgsz=d["imgsz"], conf=self.min_conf, iou=0.5,
                               max_det=d["max_det"], device=self.device, verbose=False)[0]
        dets = []
        for b in r.boxes:
            i = int(b.cls)
            cls, conf = self.prompt_cls[i], float(b.conf)
            if conf >= self.cls_conf[cls]:
                dets.append(Det(cls, conf, b.xyxy[0].cpu().numpy(), self.prompts[i]))
        return merge_synonyms(dets)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_synonyms(dets, thr=0.5):
    """NMS per classe canonica: prompt diversi della stessa classe sullo stesso oggetto -> una sola Det."""
    out = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(o.cls != d.cls or iou(o.box, d.box) < thr for o in out):
            out.append(d)
    return out


class PersonSegmenter:
    def __init__(self, cfg, device):
        from ultralytics import YOLO
        self.cfg = cfg["person"]
        self.device = device
        self.model = YOLO(cfg["models"]["person_seg"])

    def __call__(self, frame):
        h, w = frame.shape[:2]
        r = self.model.predict(frame, imgsz=self.cfg["imgsz"], conf=self.cfg["conf"], classes=[0],
                               retina_masks=True, device=self.device, verbose=False)[0]
        if len(r.boxes) == 0:
            return Persons(None, [])
        boxes = [b.cpu().numpy() for b in r.boxes.xyxy]
        mask = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            m = r.masks.data.cpu().numpy()
            for mi in m:
                if mi.shape != (h, w):
                    mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)
                mask[mi > 0.5] = 255
        else:  # nessuna maschera: usa le bbox
            for x1, y1, x2, y2 in boxes:
                mask[int(y1):int(y2), int(x1):int(x2)] = 255
        return Persons(mask, boxes)


class HandDetector:
    """MediaPipe HandLandmarker (Tasks API). Restituisce una lista di array (21, 2) in pixel."""

    def __init__(self, cfg):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        hc = cfg["hands"]
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg["models"]["hands"]),
            running_mode=vision.RunningMode.VIDEO, num_hands=hc["num_hands"],
            min_hand_detection_confidence=hc["min_conf"], min_hand_presence_confidence=hc["min_conf"],
            min_tracking_confidence=hc["min_conf"])
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.ts = 0

    def __call__(self, frame, ts_ms):
        h, w = frame.shape[:2]
        self.ts = max(self.ts + 1, int(ts_ms))   # timestamp strettamente crescente
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        res = self.lm.detect_for_video(img, self.ts)
        return [np.array([[p.x * w, p.y * h] for p in hand]) for hand in res.hand_landmarks]


class FaceDetector:
    """MediaPipe FaceDetector: serve SOLO a garantire che il volto resti oscurato. Nessun riconoscimento."""

    def __init__(self, cfg):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=cfg["models"]["face"]),
                                          min_detection_confidence=0.4)
        self.fd = vision.FaceDetector.create_from_options(opts)

    def __call__(self, frame):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        out = []
        for d in self.fd.detect(img).detections:
            b = d.bounding_box
            out.append(np.array([b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height], float))
        return out


def check_models(cfg):
    """Scarica i modelli MediaPipe se mancano (i pesi YOLO li scarica ultralytics da solo)."""
    import urllib.request
    urls = {
        cfg["models"]["hands"]: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        cfg["models"]["face"]: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    }
    for path, url in urls.items():
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            print(f"Scarico {p.name} ...")
            urllib.request.urlretrieve(url, p)
    Path(cfg["models"]["world"]).parent.mkdir(parents=True, exist_ok=True)
