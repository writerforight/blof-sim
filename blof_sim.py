#!/usr/bin/env python3
"""blof_sim.py — küçük MLP ajanlı limit emir defteri + evrim (temel sürüm).

Yapı (iskelet):
  1. Config        : tüm ayarlar (deneyler --set key=value ile değişir)
  2. Book          : tek hisse, fiyat-zaman öncelikli limit emir defteri
  3. Arka plan     : ZI (temel değer etrafında rastgele emir) + IMB (derinlik
                     dengesizliğine göre piyasa emri veren basit ajan)
  4. MLP ajan      : ~390 parametre, 16 giriş -> 16 gizli -> 7 çıkış
  5. run_episode   : bir bölüm (T adım), ölçümleri toplar
  6. evolve        : popülasyon, seçilim, mutasyon, elitler, CSV kaydı

Çalıştırma:
  python blof_sim.py --gens 50
  python blof_sim.py --gens 50 --set n_imb=0 --out runs/imb0
"""
import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np


# ----------------------------------------------------------------------------
# 1. Ayarlar
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # piyasa
    T: int = 200                # bölüm uzunluğu (adım)
    warmup: int = 20            # sadece ZI'nın emir koyduğu ısınma adımı
    V0: float = 1000.0          # temel değer (tick)
    fund_kappa: float = 0.05    # temel değerin V0'a dönüş hızı
    fund_sigma: float = 1.0     # temel değer şoku (tick/adım)
    final_k: int = 10           # ödül fiyatı = son k adımın orta fiyat ortalaması
    # ZI likidite ajanları (dış, sınırsız bakiye)
    n_zi: int = 10
    zi_rate: float = 0.6
    zi_noise: float = 3.0
    zi_spread: float = 12.0
    zi_qty: int = 5
    zi_life: int = 30           # ZI emri bu kadar adım sonra iptal edilir
    # IMB ajanları: ilk L kademe dengesizliği eşiği aşınca piyasa emri
    n_imb: int = 5
    imb_rate: float = 0.5
    imb_thr: float = 0.3
    imb_levels: int = 3
    imb_qty: int = 4
    # MLP ajanları
    n_seats: int = 20
    hidden: int = 16
    max_offset: float = 10.0    # fiyat farkı sınırı (tick)
    max_qty: int = 20
    max_open: int = 5
    temp: float = 1.0           # eylem seçimi sıcaklığı (0 = argmax)
    cash_lo: float = 20000.0
    cash_hi: float = 100000.0
    sh_lo: int = 20
    sh_hi: int = 100
    # evrim
    pop: int = 60               # n_seats'in katı olmalı
    rounds: int = 3             # her ağ bir nesilde kaç bölüm oynar
    elites: int = 6
    parents: int = 20
    sigma: float = 0.05


N_IN, N_OUT = 16, 7
BUY, SELL = 1, -1
A_NAMES = ["al", "sat", "iptal_al", "iptal_sat", "bekle"]


def n_params(cfg):
    return N_IN * cfg.hidden + cfg.hidden + cfg.hidden * N_OUT + N_OUT


# ----------------------------------------------------------------------------
# 2. Emir defteri.  Emir = [oid, owner, side, price, qty, qty0, t0]
# ----------------------------------------------------------------------------
class Book:
    def __init__(self, on_fill):
        self.side = {BUY: {}, SELL: {}}   # fiyat -> deque[emir]
        self.orders = {}
        self.nid = 0
        self.on_fill = on_fill
        self.last = None
        self.volume = 0

    def best(self, s):
        d = self.side[s]
        if not d:
            return None
        return max(d) if s == BUY else min(d)

    def mid(self):
        b, a = self.best(BUY), self.best(SELL)
        return None if b is None or a is None else 0.5 * (a + b)

    def levels(self, s, L):
        d = self.side[s]
        ks = sorted(d, reverse=(s == BUY))[:L]
        v = [sum(o[4] for o in d[k]) for k in ks]
        return v + [0] * (L - len(v))

    def submit(self, owner, s, price, qty, t, rest=True):
        """Karşı tarafla eşleştir, kalanı deftere koy. Duran emri döndürür."""
        opp = self.side[-s]
        while qty > 0 and opp:
            bp = min(opp) if s == BUY else max(opp)
            if (s == BUY and bp > price) or (s == SELL and bp < price):
                break
            q = opp[bp]
            while qty > 0 and q:
                mo = q[0]
                x = min(qty, mo[4])
                mo[4] -= x
                qty -= x
                self.last = bp
                self.volume += x
                if mo[4] == 0:
                    q.popleft()
                    del self.orders[mo[0]]
                self.on_fill(mo, owner, s, bp, x, t)
            if not q:
                del opp[bp]
        if qty > 0 and rest:
            o = [self.nid, owner, s, price, qty, qty, t]
            self.nid += 1
            self.orders[o[0]] = o
            self.side[s].setdefault(price, deque()).append(o)
            return o
        return None

    def cancel(self, oid):
        o = self.orders.pop(oid)
        q = self.side[o[2]][o[3]]
        q.remove(o)
        if not q:
            del self.side[o[2]][o[3]]
        return o


# ----------------------------------------------------------------------------
# 4. MLP
# ----------------------------------------------------------------------------
def unpack(theta, cfg):
    h = cfg.hidden
    i = 0
    W1 = theta[i:i + N_IN * h].reshape(N_IN, h); i += N_IN * h
    b1 = theta[i:i + h]; i += h
    W2 = theta[i:i + h * N_OUT].reshape(h, N_OUT); i += h * N_OUT
    b2 = theta[i:i + N_OUT]
    return W1, b1, W2, b2


def init_pop(cfg, rng):
    h = cfg.hidden
    scale = np.concatenate([
        np.full(N_IN * h, 1 / np.sqrt(N_IN)), np.zeros(h),
        np.full(h * N_OUT, 1 / np.sqrt(h)), np.zeros(N_OUT)])
    return rng.normal(size=(cfg.pop, n_params(cfg))) * scale


# ----------------------------------------------------------------------------
# 3 + 5. Bölüm
# ----------------------------------------------------------------------------
def run_episode(thetas, cfg, rng, rec=None):
    """rec bir liste verilirse her adımın anlık görüntüsü ona eklenir (görüntüleyici için)."""
    S = len(thetas)
    nets = [unpack(th, cfg) for th in thetas]
    cash = rng.uniform(cfg.cash_lo, cfg.cash_hi, S)
    shares = rng.integers(cfg.sh_lo, cfg.sh_hi + 1, S).astype(float)
    cash0, sh0 = cash.copy(), shares.copy()
    res_cash = np.zeros(S)
    res_sh = np.zeros(S)
    open_ids = [set() for _ in range(S)]
    last_fill = np.full((S, 2), -10**9)   # [:,0] alış dolumu, [:,1] satış dolumu
    m = {k: np.zeros(S) for k in
         ("placed", "filled", "cancelled", "ncancel", "life", "flip")}
    acts = np.zeros(5)
    trust_I, trust_d = [], []
    cur = {"ev": [], "act": [-1] * S}

    def on_fill(mo, taker, ts, p, q, t):
        if rec is not None:
            b, s_ = (taker, mo[1]) if ts == BUY else (mo[1], taker)
            cur["ev"].append(["T", int(p), int(q), int(b), int(s_), int(ts)])
        for owner, s, maker in ((mo[1], mo[2], True), (taker, ts, False)):
            if owner < 0:
                continue
            if s == BUY:
                cash[owner] -= p * q
                shares[owner] += q
                if maker:
                    res_cash[owner] -= p * q
            else:
                cash[owner] += p * q
                shares[owner] -= q
                if maker:
                    res_sh[owner] -= q
            if maker and mo[4] == 0:
                open_ids[owner].discard(mo[0])
            m["filled"][owner] += q
            last_fill[owner, 0 if s == BUY else 1] = t

    book = Book(on_fill)
    zi_q = deque()
    V = cfg.V0
    mids = []

    def ref_price():
        mid = book.mid()
        if mid is not None:
            return mid
        return book.last if book.last is not None else cfg.V0

    def imbalance(L):
        b, a = sum(book.levels(BUY, L)), sum(book.levels(SELL, L))
        return 0.0 if a + b == 0 else (b - a) / (b + a)

    def zi_act(t):
        s = BUY if rng.random() < 0.5 else SELL
        est = V + rng.normal(0, cfg.zi_noise)
        price = int(round(est - s * rng.uniform(0, cfg.zi_spread)))
        o = book.submit(-1, s, price, cfg.zi_qty, t)
        if o is not None:
            zi_q.append(o)

    def imb_act(t):
        I = imbalance(cfg.imb_levels)
        if I > cfg.imb_thr:
            book.submit(-2, BUY, 10**9, cfg.imb_qty, t, rest=False)
        elif I < -cfg.imb_thr:
            book.submit(-2, SELL, -1, cfg.imb_qty, t, rest=False)

    def place(i, s, price, qty, t):
        if len(open_ids[i]) >= cfg.max_open or price <= 0:
            return
        if s == BUY:
            qty = min(qty, int((cash[i] - res_cash[i]) // price))
        else:
            qty = min(qty, int(shares[i] - res_sh[i]))
        if qty <= 0:
            return
        m["placed"][i] += qty
        if rec is not None:
            cur["ev"].append(["P", i, s, int(price), int(qty)])
        o = book.submit(i, s, price, qty, t)
        if o is not None:
            open_ids[i].add(o[0])
            if s == BUY:
                res_cash[i] += price * o[4]
            else:
                res_sh[i] += o[4]

    def cancel_side(i, s, t):
        for oid in [x for x in open_ids[i] if book.orders[x][2] == s]:
            o = book.cancel(oid)
            open_ids[i].discard(oid)
            rem = o[4]
            if s == BUY:
                res_cash[i] -= o[3] * rem
            else:
                res_sh[i] -= rem
            m["cancelled"][i] += rem
            m["ncancel"][i] += 1
            m["life"][i] += t - o[6]
            # blöf imzası: emir yaşarken karşı tarafta dolum almış, sonra iptal
            flip = last_fill[i, 1 if s == BUY else 0] >= o[6]
            if flip:
                m["flip"][i] += rem
            if rec is not None:
                cur["ev"].append(["C", i, s, int(o[3]), int(rem), int(flip), int(t - o[6])])

    def seat_act(i, t, ma):
        ref = ref_price()
        W = cash[i] + shares[i] * ref
        own_b = own_a = 0
        for x in open_ids[i]:
            o = book.orders[x]
            if o[2] == BUY:
                own_b += o[4]
            else:
                own_a += o[4]
        sp = (book.best(SELL) - book.best(BUY)) if book.mid() is not None else 20.0
        last = book.last if book.last is not None else ref
        prev = mids[-1] if mids else ref
        obs = np.array([
            2 * shares[i] * ref / W - 1, (cash[i] - res_cash[i]) / W,
            (ref - prev) / 10, (last - ref) / 10, sp / 10, (ref - ma) / 10,
            *(np.log1p(book.levels(BUY, 3)) / 3), *(np.log1p(book.levels(SELL, 3)) / 3),
            np.log1p(own_b) / 3, np.log1p(own_a) / 3, len(open_ids[i]) / cfg.max_open,
            t / cfg.T])
        W1, b1, W2, b2 = nets[i]
        out = np.tanh(obs @ W1 + b1) @ W2 + b2
        lg = out[:5]
        if cfg.temp > 0:
            p = np.exp((lg - lg.max()) / cfg.temp)
            a = rng.choice(5, p=p / p.sum())
        else:
            a = int(np.argmax(lg))
        acts[a] += 1
        cur["act"][i] = int(a)
        off = np.tanh(out[5]) * cfg.max_offset
        qty = 1 + int(round(1 / (1 + np.exp(-out[6])) * (cfg.max_qty - 1)))
        if a == 0:
            trust_I.append(imbalance(3)); trust_d.append(1)
            place(i, BUY, int(round(ref + off)), qty, t)
        elif a == 1:
            trust_I.append(imbalance(3)); trust_d.append(-1)
            place(i, SELL, int(round(ref - off)), qty, t)
        elif a == 2:
            cancel_side(i, BUY, t)
        elif a == 3:
            cancel_side(i, SELL, t)

    def ladder(s):
        out = []
        for k in sorted(book.side[s], reverse=(s == BUY))[:12]:
            q = book.side[s][k]
            out.append([int(k), int(sum(o[4] for o in q)),
                        int(sum(o[4] for o in q if o[1] >= 0)),
                        sorted({int(o[1]) for o in q if o[1] >= 0})])
        return out

    def snapshot(t):
        return dict(t=t, V=round(float(V), 2), mid=float(ref_price()),
                    last=None if book.last is None else int(book.last),
                    bids=ladder(BUY), asks=ladder(SELL),
                    cash=[round(float(x), 1) for x in cash],
                    sh=[int(x) for x in shares],
                    rc=[round(float(x), 1) for x in res_cash],
                    rs=[int(x) for x in res_sh],
                    nopen=[len(x) for x in open_ids],
                    flip=[int(x) for x in m["flip"]],
                    act=cur["act"], ev=cur["ev"])

    seats = list(range(S))
    for t in range(-cfg.warmup, cfg.T):
        V += cfg.fund_kappa * (cfg.V0 - V) + rng.normal(0, cfg.fund_sigma)
        while zi_q and zi_q[0][6] <= t - cfg.zi_life:
            o = zi_q.popleft()
            if o[0] in book.orders:
                book.cancel(o[0])
        actors = [("z", None)] * cfg.n_zi
        if t >= 0:
            actors = actors + [("i", None)] * cfg.n_imb + [("s", i) for i in seats]
        ma = np.mean(mids[-20:]) if mids else ref_price()
        for k in rng.permutation(len(actors)):
            kind, i = actors[k]
            if kind == "z":
                if rng.random() < cfg.zi_rate:
                    zi_act(t)
            elif kind == "i":
                if rng.random() < cfg.imb_rate:
                    imb_act(t)
            else:
                seat_act(i, t, ma)
        if t >= 0:
            mids.append(ref_price())
            if rec is not None:
                rec.append(snapshot(t))
        cur["ev"] = []
        cur["act"] = [-1] * S

    Pf = float(np.mean(mids[-cfg.final_k:]))
    W0 = cash0 + sh0 * cfg.V0
    # fitness = elde tutmaya (hiçbir şey yapmamaya) göre fazla getiri
    excess = ((cash + shares * Pf) - (cash0 + sh0 * Pf)) / W0
    mids = np.array(mids)
    info = dict(cash0=cash0.tolist(), sh0=sh0.tolist(), m=m, acts=acts, trust_I=trust_I, trust_d=trust_d,
                vol=float(np.std(np.diff(mids))), volume=book.volume,
                price_end=Pf)
    return excess, info


# ----------------------------------------------------------------------------
# 6. Evrim
# ----------------------------------------------------------------------------
def evolve(cfg, gens, seed, out):
    assert cfg.pop % cfg.n_seats == 0
    rng = np.random.default_rng(seed)
    pop = init_pop(cfg, rng)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    cols = ["gen", "fit_best", "fit_mean", "fill", "cancel", "life", "flip",
            "trust", "vol", "volume", "price"] + A_NAMES
    fcsv = open(os.path.join(out, "gens.csv"), "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(cols)
    print("gen  best%   mean%  dolum iptal ömür  blöf  güven  oyn   " +
          " ".join(f"{a[:6]:>6}" for a in A_NAMES))
    for g in range(gens):
        t0 = time.time()
        fit = np.zeros(cfg.pop)
        cnt = np.zeros(cfg.pop)
        agg = {k: 0.0 for k in ("placed", "filled", "cancelled", "ncancel", "life", "flip")}
        acts = np.zeros(5)
        tI, td, vols, volume, prices = [], [], [], 0, []
        for _ in range(cfg.rounds):
            perm = rng.permutation(cfg.pop)
            for k in range(cfg.pop // cfg.n_seats):
                idx = perm[k * cfg.n_seats:(k + 1) * cfg.n_seats]
                ex, info = run_episode(pop[idx], cfg, rng)
                fit[idx] += ex
                cnt[idx] += 1
                for key in agg:
                    agg[key] += info["m"][key].sum()
                acts += info["acts"]
                tI += info["trust_I"]; td += info["trust_d"]
                vols.append(info["vol"]); volume += info["volume"]
                prices.append(info["price_end"])
        fit /= cnt
        order = np.argsort(-fit)
        P = max(agg["placed"], 1)
        row = dict(
            gen=g, fit_best=fit[order[0]] * 100, fit_mean=fit.mean() * 100,
            fill=agg["filled"] / P, cancel=agg["cancelled"] / P,
            life=agg["life"] / max(agg["ncancel"], 1), flip=agg["flip"] / P,
            trust=float(np.corrcoef(tI, td)[0, 1]) if len(set(td)) > 1 else 0.0,
            vol=float(np.mean(vols)), volume=volume, price=float(np.mean(prices)))
        af = acts / acts.sum()
        wr.writerow([row[c] for c in cols[:11]] + list(af))
        fcsv.flush()
        print(f"{g:3d} {row['fit_best']:6.2f} {row['fit_mean']:7.2f}  "
              f"{row['fill']:.2f}  {row['cancel']:.2f} {row['life']:5.1f} "
              f"{row['flip']:.3f} {row['trust']:+.2f} {row['vol']:4.1f}  " +
              " ".join(f"{x:6.2f}" for x in af) + f"   ({time.time() - t0:.1f}s)")
        # seçilim: elitler aynen, kalanlar en iyi 'parents' içinden kopya + gürültü
        elite = pop[order[:cfg.elites]]
        par = pop[order[:cfg.parents]]
        kids = par[rng.integers(cfg.parents, size=cfg.pop - cfg.elites)]
        kids = kids + cfg.sigma * rng.normal(size=kids.shape)
        pop = np.vstack([elite, kids])
        np.save(os.path.join(out, "pop.npy"), pop)
    fcsv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value (Config alanları)")
    a = ap.parse_args()
    cfg = Config()
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    out = a.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    print(f"parametre/ajan: {n_params(cfg)}  ->  {out}")
    evolve(cfg, a.gens, a.seed, out)


if __name__ == "__main__":
    main()
