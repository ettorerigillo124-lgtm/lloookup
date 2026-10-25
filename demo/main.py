"""Demo LookUp: detection oggetti + mani + persona oscurata + allarme occultamento.

Esempi:
  python demo/main.py --source 0
  python demo/main.py --source http://192.168.1.20:4747/video
  python demo/main.py --source data/samples/test_clip.mp4 --headless --record data/outputs/debug_out.mp4
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from detector import (FaceDetector, HandDetector, ObjectDetector, PersonSegmenter, check_models,
                      pick_device)
from overlay import FpsMeter, compose, draw_scene
from privacy import PrivacyGuard
from state_machine import StateMachine

ROOT = Path(__file__).resolve().parent.parent
WINDOW = "LookUp demo"


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in cfg["models"].items():  # percorsi relativi alla radice del repo
        if not Path(v).is_absolute():
            cfg["models"][k] = str(ROOT / v)
    return cfg


def open_source(src):
    src = str(src)
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Impossibile aprire la sorgente video: {src}")
    if src.isdigit():  # webcam / telefono: chiedi 720p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    is_file = not src.isdigit() and Path(src).exists()
    return cap, is_file


class Pipeline:
    def __init__(self, cfg):
        check_models(cfg)
        self.cfg = cfg
        self.device = pick_device(cfg["device"])
        print(f"Device: {self.device}")
        self.objects = ObjectDetector(cfg, self.device)
        self.persons = PersonSegmenter(cfg, self.device)
        self.hands = HandDetector(cfg)
        self.faces = FaceDetector(cfg)
        self.guard = PrivacyGuard(cfg)
        self.sm = StateMachine(cfg, self.objects.bag_classes)
        self.detect_every = cfg["detection"]["detect_every"]
        self.n = 0
        self.last_dets = []

    def step(self, frame, t):
        """Elabora un frame. Restituisce (frame_anonimizzato_con_overlay, info)."""
        h, w = frame.shape[:2]
        scale = max(h, w) / 640
        persons = self.persons(frame)
        hands, sides = self.hands(frame, t * 1000)
        faces = self.faces(frame)
        ran = self.n % self.detect_every == 0
        if ran:
            self.last_dets = self.objects(frame)
            self.sm.update(t, self.last_dets, hands, scale, sides)
        self.n += 1

        reveal = [(tr.cls, tr.box) for tr in self.sm.visible_tracks() if tr.missing == 0]
        out, mode = self.guard.apply(frame, persons, faces, hands, reveal)
        # da qui in poi si lavora SOLO sul frame anonimizzato
        draw_scene(out, self.sm.visible_tracks(), self.sm.bags, hands, persons.mask, self.sm.alert,
                   s=max(h, w) / 720)
        info = dict(mode=mode, persons=len(persons.boxes), hands=len(hands), faces=len(faces),
                    dets=self.last_dets if ran else None, ran=ran)
        return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="indice webcam (0), URL http/rtsp o file video")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--headless", action="store_true", help="nessuna finestra (test/registrazione)")
    ap.add_argument("--record", default=None,
                    help="salva il video GIA' ANONIMIZZATO (con overlay) in questo file")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["source"]
    pipe = Pipeline(cfg)
    cap, is_file = open_source(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 < src_fps < 121):
        src_fps = 30.0

    if is_file:
        pipe.sm.stamp = lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}"
    else:
        pipe.sm.stamp = lambda t: datetime.now().strftime("%H:%M:%S")

    writer = None
    fps = FpsMeter()
    fullscreen = False
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    t_start = time.monotonic()
    idx = 0
    auto = cfg["detection"]["auto_skip_below_fps"]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    break
                print("Frame non ricevuto, riprovo...")
                time.sleep(0.2)
                continue
            t = idx / src_fps if is_file else time.monotonic() - t_start
            out, info = pipe.step(frame, t)
            f = fps.tick()
            idx += 1
            if auto and idx == 60 and f < auto and pipe.detect_every == 1:
                pipe.detect_every = 2
                print(f"FPS {f:.1f} < {auto}: detection a frame alterni")
            canvas = compose(out, pipe.sm.events, f, cfg, pipe.sm.alert, info["mode"],
                             f"det/{pipe.detect_every}")

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    rec_fps = src_fps if is_file else max(10.0, min(30.0, f or 15.0))
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), rec_fps,
                                             (canvas.shape[1], canvas.shape[0]))
                    print(f"Registrazione (solo video anonimizzato) -> {args.record}")
                writer.write(canvas)

            if args.headless:
                if idx % 100 == 0:
                    print(f"frame {idx}  t={t:5.1f}s  {f:4.1f} FPS  privacy={info['mode']}  "
                          f"allarme={pipe.sm.alert}")
            else:
                cv2.imshow(WINDOW, canvas)
                if is_file:  # rispetta la velocita' del video se siamo piu' veloci
                    lag = (time.monotonic() - t_start) - t
                    wait = max(1, int((-lag) * 1000)) if lag < 0 else 1
                else:
                    wait = 1
                k = cv2.waitKey(wait) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    pipe.sm.reset(t)
                if k == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if args.max_seconds and t >= args.max_seconds:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print("Eventi:")
    for stamp, txt in pipe.sm.events:
        print(f"  {stamp}  {txt}")
    print(f"FPS medio finale: {fps.fps:.1f}")


if __name__ == "__main__":
    main()
