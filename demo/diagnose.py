"""Diagnostica per la taratura: per ogni secondo della clip stampa oggetti visti, confidence e stati.

  python demo/diagnose.py --source data/samples/test_clip.mp4
  python demo/diagnose.py --source data/samples/test_clip.mp4 --frames data/outputs/frames --every 2

Non scrive nulla su disco salvo --frames (frame GIA' anonimizzati, per controllo visivo).
"""
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from main import Pipeline, load_config, open_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--frames", default=None, help="cartella dove salvare frame anonimizzati")
    ap.add_argument("--every", type=float, default=1.0, help="secondi tra un frame salvato e l'altro")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe = Pipeline(cfg)
    cap, is_file = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe.sm.stamp = lambda t: f"{int(t // 60):02d}:{t % 60:04.1f}"
    if args.frames:
        Path(args.frames).mkdir(parents=True, exist_ok=True)

    sec = defaultdict(lambda: {"det": {}, "hands": 0, "modes": Counter(), "n": 0})
    modes = Counter()
    n_ev = 0
    idx, t0 = 0, time.perf_counter()
    next_save = 0.0
    print(f"{'sec':>4} | {'privacy':<12} | mani | oggetti (max conf)              | tracce (stato)")
    print("-" * 110)
    last_sec = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        out, info = pipe.step(frame, t)
        s = int(t)
        row = sec[s]
        row["n"] += 1
        row["modes"][info["mode"]] += 1
        modes[info["mode"]] += 1
        row["hands"] = max(row["hands"], info["hands"])
        for d in info["dets"] or []:
            row["det"][d.cls] = max(row["det"].get(d.cls, 0), d.conf)
        if args.frames and t >= next_save:
            cv2.imwrite(str(Path(args.frames) / f"f_{t:06.1f}.png"), out)
            next_save += args.every
        if s != last_sec and last_sec >= 0:
            _print_row(last_sec, sec[last_sec], pipe)
            for stamp, txt in pipe.sm.events[n_ev:]:
                print(f"     >>> {stamp} {txt}")
            n_ev = len(pipe.sm.events)
        last_sec = s
        idx += 1
    if last_sec >= 0:
        _print_row(last_sec, sec[last_sec], pipe)
    el = time.perf_counter() - t0
    print("-" * 110)
    print(f"Frame: {idx}  tempo: {el:.1f}s  FPS medi: {idx / el:.1f}  device: {pipe.device}")
    print(f"Privacy: {dict(modes)}  (none = nessuna persona rilevata e nessun fail-safe)")
    print(f"Allarme finale: {pipe.sm.alert}")
    print("Eventi:")
    for stamp, txt in pipe.sm.events:
        print(f"  {stamp}  {txt}")


def _print_row(s, row, pipe):
    mode = max(row["modes"], key=row["modes"].get)
    objs = " ".join(f"{k}:{v:.2f}" for k, v in sorted(row["det"].items(), key=lambda x: -x[1]))
    trs = " ".join(f"{tr.cls}#{tr.id}:{tr.state}" for tr in pipe.sm.visible_tracks())
    print(f"{s:4d} | {mode:<12} | {row['hands']:4d} | {objs:<32} | {trs}")


if __name__ == "__main__":
    main()
