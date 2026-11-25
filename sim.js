// sim.js — blof_sim.py'deki run_episode'un tarayıcı (JS) kopyası.
// Aynı kurallar; rastgele sayı üreteci farklı olduğundan aynı seed Python ile aynı bölümü vermez.
(function (root) {
"use strict";
const BUY = 1, SELL = -1, N_IN = 16, N_OUT = 7;

function makeRng(seed) {
  let a = (seed >>> 0) ^ 0x9E3779B9;
  const u = () => {               // mulberry32
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = (mu, sd) => {
    if (spare !== null) { const z = spare; spare = null; return mu + sd * z; }
    let x, y, r;
    do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r === 0 || r >= 1);
    const f = Math.sqrt(-2 * Math.log(r) / r);
    spare = y * f;
    return mu + sd * x * f;
  };
  const int = (lo, hi) => lo + Math.floor(u() * (hi - lo + 1));
  const perm = n => {
    const p = Array.from({length: n}, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); const x = p[i]; p[i] = p[j]; p[j] = x; }
    return p;
  };
  return {u, normal, int, perm};
}

class Book {
  constructor(onFill) {
    this.side = {[BUY]: new Map(), [SELL]: new Map()};
    this.orders = new Map(); this.nid = 0; this.onFill = onFill; this.last = null; this.volume = 0;
  }
  best(s) {
    let b = null;
    for (const k of this.side[s].keys()) if (b === null || (s === BUY ? k > b : k < b)) b = k;
    return b;
  }
  mid() { const b = this.best(BUY), a = this.best(SELL); return b === null || a === null ? null : 0.5 * (a + b); }
  keys(s) { return [...this.side[s].keys()].sort((x, y) => s === BUY ? y - x : x - y); }
  levels(s, L) {
    const d = this.side[s], v = this.keys(s).slice(0, L).map(k => d.get(k).reduce((a, o) => a + o[4], 0));
    while (v.length < L) v.push(0);
    return v;
  }
  submit(owner, s, price, qty, t, rest = true) {
    const opp = this.side[-s];
    while (qty > 0 && opp.size) {
      const bp = this.best(-s);
      if ((s === BUY && bp > price) || (s === SELL && bp < price)) break;
      const q = opp.get(bp);
      while (qty > 0 && q.length) {
        const mo = q[0], x = Math.min(qty, mo[4]);
        mo[4] -= x; qty -= x; this.last = bp; this.volume += x;
        if (mo[4] === 0) { q.shift(); this.orders.delete(mo[0]); }
        this.onFill(mo, owner, s, bp, x, t);
      }
      if (!q.length) opp.delete(bp);
    }
    if (qty > 0 && rest) {
      const o = [this.nid++, owner, s, price, qty, qty, t];
      this.orders.set(o[0], o);
      const d = this.side[s];
      if (!d.has(price)) d.set(price, []);
      d.get(price).push(o);
      return o;
    }
    return null;
  }
  cancel(oid) {
    const o = this.orders.get(oid); this.orders.delete(oid);
    const d = this.side[o[2]], q = d.get(o[3]);
    q.splice(q.indexOf(o), 1);
    if (!q.length) d.delete(o[3]);
    return o;
  }
}

function randomPop(cfg, seed) {
  const rng = makeRng(seed), h = cfg.hidden, out = [];
  for (let k = 0; k < cfg.n_seats; k++) {
    const th = [];
    for (let i = 0; i < N_IN * h; i++) th.push(rng.normal(0, 1 / Math.sqrt(N_IN)));
    for (let i = 0; i < h; i++) th.push(0);
    for (let i = 0; i < h * N_OUT; i++) th.push(rng.normal(0, 1 / Math.sqrt(h)));
    for (let i = 0; i < N_OUT; i++) th.push(0);
    out.push(th);
  }
  return out;
}

// Üreteç: ilerlemeyi (0..1) verir, sonunda kaydı döndürür.
function* episode(thetas, cfg, seed, levelsShown = 10) {
  const rng = makeRng(seed), S = thetas.length, h = cfg.hidden;
  const cash = new Float64Array(S), shares = new Float64Array(S);
  for (let i = 0; i < S; i++) {
    cash[i] = cfg.cash_lo + rng.u() * (cfg.cash_hi - cfg.cash_lo);
    shares[i] = rng.int(cfg.sh_lo, cfg.sh_hi);
  }
  const cash0 = Array.from(cash), sh0 = Array.from(shares);
  const resCash = new Float64Array(S), resSh = new Float64Array(S);
  const openIds = Array.from({length: S}, () => new Set());
  const lastFill = Array.from({length: S}, () => [-1e9, -1e9]);
  const M = {placed: 0, filled: 0, cancelled: 0, flip: 0};
  let ev = [], act = new Array(S).fill(-1);
  const R = {V: [], mid: [], last: [], bids: [], asks: [], cash: [], sh: [], nopen: [], act: [], ev: []};

  const onFill = (mo, taker, ts, p, q, t) => {
    ev.push(["T", p, q, ts === BUY ? taker : mo[1], ts === BUY ? mo[1] : taker, ts]);
    for (const [owner, s, maker] of [[mo[1], mo[2], true], [taker, ts, false]]) {
      if (owner < 0) continue;
      if (s === BUY) { cash[owner] -= p * q; shares[owner] += q; if (maker) resCash[owner] -= p * q; }
      else { cash[owner] += p * q; shares[owner] -= q; if (maker) resSh[owner] -= q; }
      if (maker && mo[4] === 0) openIds[owner].delete(mo[0]);
      M.filled += q;
      lastFill[owner][s === BUY ? 0 : 1] = t;
    }
  };
  const book = new Book(onFill);
  const ziQ = [];
  let ziHead = 0, V = cfg.V0;
  const mids = [];
  const refPrice = () => { const m = book.mid(); return m !== null ? m : (book.last !== null ? book.last : cfg.V0); };
  const imbalance = L => {
    const b = book.levels(BUY, L).reduce((x, y) => x + y, 0), a = book.levels(SELL, L).reduce((x, y) => x + y, 0);
    return a + b === 0 ? 0 : (b - a) / (b + a);
  };
  const ziAct = t => {
    const s = rng.u() < 0.5 ? BUY : SELL;
    const est = V + rng.normal(0, cfg.zi_noise);
    const o = book.submit(-1, s, Math.round(est - s * rng.u() * cfg.zi_spread), cfg.zi_qty, t);
    if (o) ziQ.push(o);
  };
  const imbAct = t => {
    const I = imbalance(cfg.imb_levels);
    if (I > cfg.imb_thr) book.submit(-2, BUY, 1e9, cfg.imb_qty, t, false);
    else if (I < -cfg.imb_thr) book.submit(-2, SELL, -1, cfg.imb_qty, t, false);
  };
  const place = (i, s, price, qty, t) => {
    if (openIds[i].size >= cfg.max_open || price <= 0) return;
    qty = s === BUY ? Math.min(qty, Math.floor((cash[i] - resCash[i]) / price)) : Math.min(qty, Math.floor(shares[i] - resSh[i]));
    if (qty <= 0) return;
    M.placed += qty;
    ev.push(["P", i, s, price, qty]);
    const o = book.submit(i, s, price, qty, t);
    if (o) { openIds[i].add(o[0]); if (s === BUY) resCash[i] += price * o[4]; else resSh[i] += o[4]; }
  };
  const cancelSide = (i, s, t) => {
    for (const oid of [...openIds[i]].filter(x => book.orders.get(x)[2] === s)) {
      const o = book.cancel(oid); openIds[i].delete(oid);
      const rem = o[4];
      if (s === BUY) resCash[i] -= o[3] * rem; else resSh[i] -= rem;
      M.cancelled += rem;
      const flip = lastFill[i][s === BUY ? 1 : 0] >= o[6];
      if (flip) M.flip += rem;
      ev.push(["C", i, s, o[3], rem, flip ? 1 : 0, t - o[6]]);
    }
  };
  const obs = new Float64Array(N_IN), hid = new Float64Array(h), out = new Float64Array(N_OUT);
  const seatAct = (i, t, ma) => {
    const ref = refPrice(), W = cash[i] + shares[i] * ref;
    let ownB = 0, ownA = 0;
    for (const x of openIds[i]) { const o = book.orders.get(x); if (o[2] === BUY) ownB += o[4]; else ownA += o[4]; }
    const sp = book.mid() !== null ? book.best(SELL) - book.best(BUY) : 20;
    const last = book.last !== null ? book.last : ref, prev = mids.length ? mids[mids.length - 1] : ref;
    const lb = book.levels(BUY, 3), la = book.levels(SELL, 3);
    const v = [2 * shares[i] * ref / W - 1, (cash[i] - resCash[i]) / W, (ref - prev) / 10, (last - ref) / 10, sp / 10,
      (ref - ma) / 10, ...lb.map(x => Math.log1p(x) / 3), ...la.map(x => Math.log1p(x) / 3),
      Math.log1p(ownB) / 3, Math.log1p(ownA) / 3, openIds[i].size / cfg.max_open, t / cfg.T];
    for (let k = 0; k < N_IN; k++) obs[k] = v[k];
    const th = thetas[i], o2 = N_IN * h + h, ob = o2 + h * N_OUT;
    for (let j = 0; j < h; j++) { let s = th[N_IN * h + j]; for (let k = 0; k < N_IN; k++) s += obs[k] * th[k * h + j]; hid[j] = Math.tanh(s); }
    for (let k = 0; k < N_OUT; k++) { let s = th[ob + k]; for (let j = 0; j < h; j++) s += hid[j] * th[o2 + j * N_OUT + k]; out[k] = s; }
    let a;
    if (cfg.temp > 0) {
      const mx = Math.max(out[0], out[1], out[2], out[3], out[4]);
      const p = [0, 1, 2, 3, 4].map(k => Math.exp((out[k] - mx) / cfg.temp)), tot = p.reduce((x, y) => x + y, 0);
      let r = rng.u() * tot; a = 4;
      for (let k = 0; k < 5; k++) { r -= p[k]; if (r < 0) { a = k; break; } }
    } else { a = 0; for (let k = 1; k < 5; k++) if (out[k] > out[a]) a = k; }
    act[i] = a;
    const off = Math.tanh(out[5]) * cfg.max_offset;
    const qty = 1 + Math.round(1 / (1 + Math.exp(-out[6])) * (cfg.max_qty - 1));
    if (a === 0) place(i, BUY, Math.round(ref + off), qty, t);
    else if (a === 1) place(i, SELL, Math.round(ref - off), qty, t);
    else if (a === 2) cancelSide(i, BUY, t);
    else if (a === 3) cancelSide(i, SELL, t);
  };
  const ladder = s => book.keys(s).slice(0, levelsShown).map(k => {
    const q = book.side[s].get(k);
    let tot = 0, mlp = 0; const who = new Set();
    for (const o of q) { tot += o[4]; if (o[1] >= 0) { mlp += o[4]; who.add(o[1]); } }
    return mlp ? [k, tot, mlp, [...who].sort((x, y) => x - y)] : [k, tot];
  });

  for (let t = -cfg.warmup; t < cfg.T; t++) {
    V += cfg.fund_kappa * (cfg.V0 - V) + rng.normal(0, cfg.fund_sigma);
    while (ziHead < ziQ.length && ziQ[ziHead][6] <= t - cfg.zi_life) {
      const o = ziQ[ziHead++];
      if (book.orders.has(o[0])) book.cancel(o[0]);
    }
    const actors = [];
    for (let k = 0; k < cfg.n_zi; k++) actors.push(-1);
    if (t >= 0) { for (let k = 0; k < cfg.n_imb; k++) actors.push(-2); for (let i = 0; i < S; i++) actors.push(i); }
    const w = mids.slice(-20), ma = w.length ? w.reduce((x, y) => x + y, 0) / w.length : refPrice();
    for (const k of rng.perm(actors.length)) {
      const a = actors[k];
      if (a === -1) { if (rng.u() < cfg.zi_rate) ziAct(t); }
      else if (a === -2) { if (rng.u() < cfg.imb_rate) imbAct(t); }
      else seatAct(a, t, ma);
    }
    if (t >= 0) {
      const ref = refPrice();
      mids.push(ref);
      R.V.push(Math.round(V * 10) / 10); R.mid.push(ref); R.last.push(book.last);
      R.bids.push(ladder(BUY)); R.asks.push(ladder(SELL));
      R.cash.push(Array.from(cash, x => Math.round(x))); R.sh.push(Array.from(shares));
      R.nopen.push(openIds.map(x => x.size).join(""));
      R.act.push(act.map(x => x < 0 ? "-" : String(x)).join(""));
      R.ev.push(ev);
      if (t % 500 === 499) yield (t + 1) / cfg.T;
    }
    ev = []; act = new Array(S).fill(-1);
    if (ziHead > 5000) { ziQ.splice(0, ziHead); ziHead = 0; }
  }
  const k = Math.min(cfg.final_k, mids.length);
  const Pf = mids.slice(-k).reduce((x, y) => x + y, 0) / k;
  R.excess = cash0.map((c, i) => ((cash[i] + shares[i] * Pf) - (c + sh0[i] * Pf)) / (c + sh0[i] * cfg.V0));
  R.cash0 = cash0; R.sh0 = sh0; R.volume = book.volume;
  R.stats = {fill: M.filled / Math.max(M.placed, 1), cancel: M.cancelled / Math.max(M.placed, 1),
             flip: M.flip / Math.max(M.placed, 1), Pf};
  return R;
}

const api = {episode, randomPop, makeRng, Book};
if (typeof module !== "undefined" && module.exports) module.exports = api; else root.BlofSim = api;
})(typeof window !== "undefined" ? window : globalThis);
