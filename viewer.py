#!/usr/bin/env python3
"""viewer.py — bir bölümü kaydedip zaman kaydırmalı tahta arayüzü (viewer.html) üretir.

  python viewer.py                       # runs/base_imb5/pop.npy'nin en iyi 20 ağı
  python viewer.py --pop runs/x/pop.npy --seed 3 --set n_imb=0
  python viewer.py --random              # eğitilmemiş (rastgele) ağlar
"""
import argparse
import json
import os
import subprocess

import numpy as np

import blof_sim as bs

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default=os.path.join(HERE, "runs", "base_imb5", "pop.npy"))
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "viewer.html"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    cfg = bs.Config()
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    rng = np.random.default_rng(a.seed)
    if a.random or not os.path.exists(a.pop):
        thetas = bs.init_pop(cfg, rng)[:cfg.n_seats]
        src = "rastgele ağlar"
    else:
        thetas = np.load(a.pop)[:cfg.n_seats]
        src = os.path.relpath(a.pop, HERE)
    rec = []
    excess, info = bs.run_episode(thetas, cfg, rng, rec=rec)
    data = dict(src=src, seed=a.seed, T=cfg.T, V0=cfg.V0, n_imb=cfg.n_imb, n_zi=cfg.n_zi,
                cash0=info["cash0"], sh0=info["sh0"], steps=rec,
                excess=[float(x) for x in excess])
    with open(os.path.join(HERE, "viewer_template.html")) as f:
        html = f.read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"{src}, seed {a.seed}: {len(rec)} adım -> {a.out}")
    if not a.no_open:
        subprocess.Popen(["xdg-open", a.out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
