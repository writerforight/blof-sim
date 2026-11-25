#!/usr/bin/env python3
"""viewer.py — index.html'i (tahta arayüzü) üretir.

Sayfa simülasyonu tarayıcıda sim.js ile kendisi çalıştırır; seed, adım sayısı,
ZI/IMB sayısı vb. sayfadan değiştirilir. Buraya sadece Config varsayılanları ve
eğitilmiş popülasyonların ilk n_seats ağı gömülür.

  python viewer.py                                   # runs/*/pop.npy -> index.html
  python viewer.py --pop deneme=runs/deneme/pop.npy  # ek popülasyon
"""
import argparse
import glob
import json
import os
import subprocess
from dataclasses import asdict

import numpy as np

import blof_sim as bs

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", nargs="*", default=[], help="ad=yol/pop.npy (varsayılan: runs/*/pop.npy)")
    ap.add_argument("--out", default=os.path.join(HERE, "index.html"))
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    cfg = bs.Config()
    srcs = dict(kv.split("=", 1) for kv in a.pop) if a.pop else {
        os.path.basename(os.path.dirname(p)): p
        for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "pop.npy")))
        if "test" not in p}
    # base_imb5 önce gelsin (varsayılan seçim)
    names = sorted(srcs, key=lambda k: (k != "base_imb5", k))
    pops = {k: np.round(np.load(srcs[k])[:cfg.n_seats], 5).tolist() for k in names}
    with open(os.path.join(HERE, "viewer_template.html")) as f:
        html = f.read()
    html = html.replace("__CFG__", json.dumps(asdict(cfg))).replace(
        "__POPS__", json.dumps(pops, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"{', '.join(names)} -> {a.out} ({os.path.getsize(a.out) // 1024} KB)")
    if a.open:
        subprocess.Popen(["xdg-open", a.out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
