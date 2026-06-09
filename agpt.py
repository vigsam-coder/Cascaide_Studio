"""AuroraCascadeGPT — with the structural scorecard wired into training.

Changes vs the original:
  * config: max_seq_len=9000, d_model=512, depth=12, n_heads=8, dropout=0.1
  * truncate=False (drop oversize cascades instead of fake-END truncation)
  * AMP (mixed precision) + gradient accumulation + optional gradient checkpointing
    so the bigger model / longer context fits and trains fast on a good GPU
  * the 7 weighted scorecard requirements (+ diagnostics) computed periodically by
    GENERATING samples and comparing to a held-out reference; best_model.pt is saved
    on the lowest scorecard SCORE (not val loss). best_loss_model.pt still tracks loss.

Timing note: scoring = autoregressive generation, which is the cost. It runs only every
`score_every` epochs, on `score_max_ref` held-out cascades, generating `score_n_per_energy`
per 25 keV band, with generation length capped to ~1.3x the real cascades in each band.
Each scoring pass prints its wall time. Lower `score_every` / `score_n_per_energy` if slow.
"""

import os, math, json, glob, time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TYPE_VAC, TYPE_SIA = 0, 1


# =========================================================================== #
# Normalizer / data loading  (UNCHANGED — global centroid + per-axis scale)
# =========================================================================== #
class CoordNormalizer:
    def __init__(self, percentile=99.0):
        self.percentile = percentile
        self.G = np.zeros(3, np.float32); self.s = np.ones(3, np.float32)
    def fit(self, all_coords):
        if len(all_coords):
            self.G = all_coords.mean(axis=0).astype(np.float32)
            dev = np.abs(all_coords - self.G)
            s = np.percentile(dev, self.percentile, axis=0).astype(np.float32)
            self.s = np.where(s < 1e-8, 1.0, s).astype(np.float32)
        return self
    def transform(self, c):  return ((c - self.G) / self.s).astype(np.float32)
    def inverse(self, n):    return (n * self.s + self.G).astype(np.float32)
    def to_dict(self):
        return {'percentile': self.percentile, 'G': self.G.tolist(), 's': self.s.tolist()}
    @classmethod
    def from_dict(cls, d):
        o = cls(d['percentile']); o.G = np.asarray(d['G'], np.float32)
        o.s = np.asarray(d['s'], np.float32); return o


def load_synthetic_raw(n_samples=1500, seed=0, e_lo=5.0, e_hi=250.0,
                       pairs_per_keV=1.0, max_pairs=1100):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_samples):
        e = float(rng.uniform(e_lo, e_hi))
        mean_pairs = max(1.0, e * pairs_per_keV)
        n = int(np.clip(round(rng.lognormal(np.log(mean_pairs), 0.5)), 0, max_pairs))
        center = rng.normal(0, 60, size=3); core = rng.normal(0, 8, size=3)
        vac = center + core + rng.normal(0, 5, size=(n, 3))
        d = rng.normal(0, 1, size=(n, 3)); d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
        sia = center + core + d * rng.uniform(8, 20, size=(n, 1)) + rng.normal(0, 2.5, size=(n, 3))
        out.append((vac.astype(np.float32), sia.astype(np.float32), e))
    return out


def load_real_raw(data_root):
    samples = []
    for ed in sorted(glob.glob(os.path.join(data_root, "*keV"))):
        meta = os.path.join(ed, "metadata.json")
        if not os.path.exists(meta): continue
        with open(meta) as f: data = json.load(f)
        emap = {c["cascade_id"]: c["pka_energy_eV"] for c in data.get("cascades", [])}
        for vf in sorted(glob.glob(os.path.join(ed, "*_min_vac.dump"))):
            try: cid = int(os.path.basename(vf).split("_")[0])
            except Exception: continue
            sf = vf.replace("_min_vac.dump", "_min_sia.dump")
            if os.path.exists(sf) and cid in emap:
                samples.append((vf, sf, emap[cid] / 1000.0))
    def _load(path):
        try:
            from ovito.io import import_file
            d = import_file(path).compute()
            if d.particles.count == 0: return np.zeros((0, 3), np.float32)
            return np.array(d.particles.positions, dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), np.float32)
    out = [(_load(vf), _load(sf), e) for vf, sf, e in samples]
    print(f"[load_real_raw] {len(out)} cascades from {data_root}")
    return out


PAD, BOS, END, VAC, SIA = 0, 1, 2, 3, 4
N_SPECIAL = 5
PH_TYPE, PH_X, PH_Y, PH_Z = 0, 1, 2, 3


class CascadeTokenizer:
    def __init__(self, n_energy_bins=64, n_coord_bins=512, order='pair'):
        self.NE = n_energy_bins; self.K = n_coord_bins; self.order = order
        self.e_min = 0.0; self.e_max = 1.0; self.clip = 1.0
        self.normalizer = CoordNormalizer()
        self.E_BASE = N_SPECIAL; self.C_BASE = N_SPECIAL + self.NE
        self.vocab_size = self.C_BASE + self.K

    def fit(self, raw, coord_percentile=99.9, energy_pad=0.02):
        es = np.array([e for _, _, e in raw], np.float32); span = float(np.ptp(es))
        self.e_min = float(es.min()) - energy_pad * (span + 1e-6)
        self.e_max = float(es.max()) + energy_pad * (span + 1e-6)
        chunks = [c for v, s, _ in raw for c in (v, s) if len(c)]
        allc = np.concatenate(chunks, 0) if chunks else np.zeros((0, 3), np.float32)
        self.normalizer.fit(allc)
        if len(allc):
            normed = self.normalizer.transform(allc)
            self.clip = max(float(np.percentile(np.abs(normed), coord_percentile)), 1e-3)
        return self

    def _e_to_id(self, e):
        f = (e - self.e_min) / (self.e_max - self.e_min)
        return self.E_BASE + int(np.clip(math.floor(f * self.NE), 0, self.NE - 1))
    def _id_to_e(self, tid):
        b = tid - self.E_BASE
        return self.e_min + (b + 0.5) * (self.e_max - self.e_min) / self.NE
    def _coord_to_id(self, v):
        f = (v + self.clip) / (2 * self.clip)
        return self.C_BASE + int(np.clip(math.floor(f * self.K), 0, self.K - 1))
    def _id_to_coord(self, tid):
        b = tid - self.C_BASE
        return -self.clip + (b + 0.5) * (2 * self.clip) / self.K

    def is_energy(self, tid): return self.E_BASE <= tid < self.C_BASE
    def is_coord(self, tid):  return self.C_BASE <= tid < self.vocab_size
    def is_type(self, tid):   return tid in (VAC, SIA)

    def _order_defects(self, vac, sia):
        nv, ns = self.normalizer.transform(vac), self.normalizer.transform(sia)
        events = []
        if self.order == 'pair' and len(nv) == len(ns) and len(nv) > 0:
            remaining = list(range(len(ns))); pairs = []
            for i in range(len(nv)):
                if not remaining: break
                d = ((ns[remaining] - nv[i]) ** 2).sum(1)
                j = remaining[int(np.argmin(d))]; remaining.remove(j); pairs.append((i, j))
            pairs.sort(key=lambda p: float((nv[p[0]] ** 2).sum()))
            for i, j in pairs:
                events.append((VAC, nv[i])); events.append((SIA, ns[j]))
        else:
            def srt(arr):
                idx = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0])) if len(arr) else []
                return [arr[k] for k in idx]
            events += [(VAC, c) for c in srt(nv)]; events += [(SIA, c) for c in srt(ns)]
        return events

    def encode(self, vac, sia, energy, max_len=None):
        toks = [BOS, self._e_to_id(energy)]
        for typ, xyz in self._order_defects(np.asarray(vac, np.float32),
                                            np.asarray(sia, np.float32)):
            toks.append(typ); toks += [self._coord_to_id(float(v)) for v in xyz]
        toks.append(END)
        if max_len is not None and len(toks) > max_len:
            cut = max_len - 1; cut -= (cut - 2) % 4
            toks = toks[:max(2, cut)] + [END]
        return toks

    def decode(self, toks):
        energy = None; vac, sia = [], []; i = 0
        while i < len(toks):
            t = toks[i]
            if t in (BOS, PAD): i += 1; continue
            if t == END: break
            if self.is_energy(t) and energy is None:
                energy = self._id_to_e(t); i += 1; continue
            if self.is_type(t):
                if i + 3 >= len(toks): break
                cs = toks[i + 1:i + 4]
                if not all(self.is_coord(c) for c in cs): i += 1; continue
                xyz = self.normalizer.inverse(
                    np.array([self._id_to_coord(c) for c in cs], np.float32))
                (vac if t == VAC else sia).append(xyz); i += 4; continue
            i += 1
        vac = np.array(vac, np.float32) if vac else np.zeros((0, 3), np.float32)
        sia = np.array(sia, np.float32) if sia else np.zeros((0, 3), np.float32)
        return vac, sia, energy

    def phase_masks(self, device):
        m = torch.zeros(4, self.vocab_size, dtype=torch.bool, device=device)
        m[PH_TYPE, VAC] = m[PH_TYPE, SIA] = m[PH_TYPE, END] = True
        m[PH_X, self.C_BASE:] = True; m[PH_Y, self.C_BASE:] = True; m[PH_Z, self.C_BASE:] = True
        return m

    def to_dict(self):
        return {'NE': self.NE, 'K': self.K, 'order': self.order,
                'e_min': self.e_min, 'e_max': self.e_max, 'clip': self.clip,
                'normalizer': self.normalizer.to_dict()}
    @classmethod
    def from_dict(cls, d):
        o = cls(d['NE'], d['K'], d.get('order', 'pair'))
        o.e_min, o.e_max, o.clip = d['e_min'], d['e_max'], d['clip']
        o.normalizer = CoordNormalizer.from_dict(d['normalizer']); return o


class CascadeSequenceDataset(Dataset):
    def __init__(self, raw, tokenizer, max_seq_len=9000, truncate=False):
        self.tok = tokenizer; self.seqs = []; skipped = 0
        for vac, sia, e in raw:
            toks = tokenizer.encode(vac, sia, e, max_len=max_seq_len if truncate else None)
            if len(toks) > max_seq_len: skipped += 1; continue
            self.seqs.append(torch.tensor(toks, dtype=torch.long))
        lens = [len(s) for s in self.seqs] or [0]
        print(f"[CascadeSequenceDataset] {len(self.seqs)} seqs (skipped {skipped}); "
              f"len min/med/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}; "
              f"vocab={tokenizer.vocab_size}")
    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]


def collate(batch):
    T = max(len(s) for s in batch); B = len(batch)
    x = torch.full((B, T), PAD, dtype=torch.long)
    for i, s in enumerate(batch): x[i, :len(s)] = s
    return x


# =========================================================================== #
# Model  (+ optional gradient checkpointing)
# =========================================================================== #
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps); return n * self.w


def _rope_tables(T, dh, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    t = torch.arange(T, device=device).float(); f = torch.outer(t, inv)
    emb = torch.cat([f, f], -1); return emb.cos(), emb.sin()


def _apply_rope(x, cos, sin):
    d = x.shape[-1]; x1, x2 = x[..., :d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], -1); return x * cos[None, None] + rot * sin[None, None]


class Attention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__(); self.h = n_heads; self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.drop = dropout
    def forward(self, x, cos, sin, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if cache is not None and cache[0] is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2); v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=(q.size(2) == k.size(2)),
            dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(o), new_cache


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__(); h = int(2 * mult * d / 3); h = 32 * ((h + 31) // 32)
        self.w1 = nn.Linear(d, h, bias=False); self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False); self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, d, n_heads, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attention(d, n_heads, dropout)
        self.n2 = RMSNorm(d); self.mlp = SwiGLU(d, mlp_mult, dropout)
    def forward(self, x, cos, sin, cache=None):
        a, new_cache = self.attn(self.n1(x), cos, sin, cache)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, new_cache


class CascadeGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=12, n_heads=8,
                 mlp_mult=4, dropout=0.0, max_seq_len=9000, use_checkpoint=False):
        super().__init__(); self.max_seq_len = max_seq_len; self.use_checkpoint = use_checkpoint
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_mult, dropout) for _ in range(depth)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.dh = d_model // n_heads
        self.apply(self._init); self._cos = self._sin = None

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def _rope(self, start_pos, T, device):
        need = start_pos + T
        if self._cos is None or self._cos.shape[0] < need or self._cos.device != device:
            self._cos, self._sin = _rope_tables(max(need, self.max_seq_len), self.dh, device)
        return self._cos[start_pos:start_pos + T], self._sin[start_pos:start_pos + T]

    def forward(self, idx):                          # training / scoring path
        B, T = idx.shape
        cos, sin = self._rope(0, T, idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x, _ = ckpt_util.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x, _ = blk(x, cos, sin)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def decode_forward(self, idx, caches, start_pos):  # incremental generation
        B, T = idx.shape
        cos, sin = self._rope(start_pos, T, idx.device)
        x = self.tok_emb(idx)
        new_caches = []
        for li, blk in enumerate(self.blocks):
            c = caches[li] if caches is not None else None
            x, nc = blk(x, cos, sin, c)
            new_caches.append(nc)
        return self.head(self.norm_f(x)), new_caches


# =========================================================================== #
# STRUCTURAL SCORECARD  (the 7 weighted requirements -> one `score`, + diagnostics)
# pure numpy; point-subsampled for speed during training
# =========================================================================== #
DEFAULT_LINK_SCALES = (4., 6., 8., 10., 13., 16., 20., 25.)
REQUIREMENTS = {  # value <= target passes; weight sets contribution to the score
    "count_mape":      {"label": "Count error (MAPE)",        "target": 0.15, "weight": 1.0},
    "conservation":    {"label": "Frenkel violation |nv-ns|", "target": 1.0,  "weight": 1.0},
    "radial_js":       {"label": "Radial profile (JS)",       "target": 0.05, "weight": 1.5},
    "rdf_l1":          {"label": "g(r) shape (L1)",           "target": 0.20, "weight": 2.0},
    "cluster_l1":      {"label": "Sub-cascade spectrum (L1)", "target": 0.15, "weight": 3.0},
    "nn_w1":           {"label": "NN-distance W1",            "target": 1.0,  "weight": 1.0},
    "vac_sia_sep_err": {"label": "vac-SIA separation err",    "target": 3.0,  "weight": 1.0},
}


def _pdist(a):
    d = a[:, None, :] - a[None, :, :]
    return np.sqrt(np.maximum((d * d).sum(-1), 0.0))

def _nn(a):
    a = np.asarray(a, np.float32).reshape(-1, 3)
    if len(a) < 2: return np.zeros(0, np.float32)
    D = _pdist(a); np.fill_diagonal(D, np.inf); return D.min(1).astype(np.float32)

def _radial_pdf(a, rmax, nbins=24):
    a = np.asarray(a, np.float32).reshape(-1, 3)
    if len(a) == 0: return np.zeros(nbins)
    rad = np.linalg.norm(a - a.mean(0), axis=1)
    c, _ = np.histogram(rad, bins=np.linspace(0, rmax, nbins + 1)); return c / max(c.sum(), 1)

def _rdf(a, rmax, nbins=24):
    a = np.asarray(a, np.float32).reshape(-1, 3)
    if len(a) < 2: return np.zeros(nbins)
    dists = _pdist(a)[np.triu_indices(len(a), 1)]
    edges = np.linspace(0, rmax, nbins + 1); c, _ = np.histogram(dists, bins=edges)
    shell = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = len(a) / ((4 / 3) * np.pi * rmax ** 3)
    return c / np.maximum(0.5 * len(a) * rho * shell, 1e-12)

def _cluster_spectrum(a, scales=DEFAULT_LINK_SCALES):
    a = np.asarray(a, np.float32).reshape(-1, 3); n = len(a)
    if n == 0: return np.zeros(len(scales))
    D = _pdist(a); ncl = []
    for eps in scales:
        parent = list(range(n))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        ii, jj = np.where((D <= eps) & np.triu(np.ones_like(D, bool), 1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj: parent[ri] = rj
        ncl.append(len(np.unique([find(i) for i in range(n)])))
    return np.array(ncl, float)

def _largest_cluster_frac(a, link=10.0):
    a = np.asarray(a, np.float32).reshape(-1, 3); n = len(a)
    if n == 0: return 0.0
    D = _pdist(a); parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    ii, jj = np.where((D <= link) & np.triu(np.ones_like(D, bool), 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    _, lab = np.unique([find(i) for i in range(n)], return_inverse=True)
    sizes = np.bincount(lab); return float(sizes.max() / sizes.sum())

def _rg(a):
    a = np.asarray(a, np.float32).reshape(-1, 3)
    if len(a) == 0: return 0.0
    return float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))

def _w1(a, b):
    a = np.sort(np.asarray(a, np.float64).ravel()); b = np.sort(np.asarray(b, np.float64).ravel())
    if len(a) == 0 or len(b) == 0: return float("nan")
    q = np.linspace(0, 1, max(len(a), len(b)) * 2)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))

def _js(p, q, eps=1e-12):
    p = np.asarray(p, np.float64) + eps; q = np.asarray(q, np.float64) + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(x * np.log(x / y)); return float(0.5 * kl(p, m) + 0.5 * kl(q, m))

def _l1(yr, yg):
    yr = np.asarray(yr, np.float64); yg = np.asarray(yg, np.float64)
    return float(np.abs(yr - yg).sum() / (np.abs(yr).sum() + 1e-12))

def _union(s):
    parts = [p for p in (s["vac"], s["sia"]) if len(p)]
    return np.concatenate(parts, 0) if parts else np.zeros((0, 3), np.float32)


def score_cards(generated, reference, energy_bin=25.0, nbins=24,
                scales=DEFAULT_LINK_SCALES, max_points=600, seed=0):
    """Return dict with overall `score` (lower=better) + per-requirement values + diagnostics.
    Distribution-vs-distribution within each energy bin; point-subsampled for speed."""
    rng = np.random.default_rng(seed)
    def cap(u): return u[rng.choice(len(u), max_points, replace=False)] if len(u) > max_points else u
    key = lambda e: round(e / energy_bin) * energy_bin
    gen_g, ref_g = {}, {}
    for s in generated: gen_g.setdefault(key(s["energy"]), []).append(s)
    for s in reference: ref_g.setdefault(key(s["energy"]), []).append(s)
    energies = sorted(set(gen_g) & set(ref_g))

    acc = {k: [] for k in REQUIREMENTS}
    diag = {"rg_w1": [], "lcf_real": [], "lcf_gen": []}
    for e in energies:
        R, G = ref_g[e], gen_g[e]
        Ru = [cap(_union(s)) for s in R if len(_union(s)) >= 2]
        Gu = [cap(_union(s)) for s in G if len(_union(s)) >= 2]
        if not Ru or not Gu: continue
        rad = [np.linalg.norm(c - c.mean(0), axis=1).max() for c in Ru]
        rmax_rad = float(np.percentile(rad, 95)) * 1.05 if rad else 1.0
        pair = [np.percentile(_pdist(c)[np.triu_indices(len(c), 1)], 90) for c in Ru if len(c) >= 2]
        rmax_rdf = float(np.mean(pair)) if pair else 1.0

        def mean_curve(clouds, fn):
            acc2 = None; n = 0
            for c in clouds:
                y = fn(c); acc2 = y if acc2 is None else acc2 + y; n += 1
            return acc2 / max(n, 1)
        radial_r = mean_curve(Ru, lambda c: _radial_pdf(c, rmax_rad, nbins))
        radial_g = mean_curve(Gu, lambda c: _radial_pdf(c, rmax_rad, nbins))
        rdf_r = mean_curve(Ru, lambda c: _rdf(c, rmax_rdf, nbins))
        rdf_g_ = mean_curve(Gu, lambda c: _rdf(c, rmax_rdf, nbins))
        spec_r = mean_curve(Ru, lambda c: _cluster_spectrum(c, scales))
        spec_g = mean_curve(Gu, lambda c: _cluster_spectrum(c, scales))

        r_tot = np.array([len(s["vac"]) + len(s["sia"]) for s in R], float)
        g_tot = np.array([len(s["vac"]) + len(s["sia"]) for s in G], float)
        acc["count_mape"].append(abs(g_tot.mean() - r_tot.mean()) / max(r_tot.mean(), 1e-9))
        acc["conservation"].append(np.mean([abs(len(s["vac"]) - len(s["sia"])) for s in G]))
        acc["radial_js"].append(_js(radial_r, radial_g))
        acc["rdf_l1"].append(_l1(rdf_r, rdf_g_))
        acc["cluster_l1"].append(_l1(spec_r, spec_g))
        nn_r = np.concatenate([_nn(c) for c in Ru] or [np.zeros(0)])
        nn_g = np.concatenate([_nn(c) for c in Gu] or [np.zeros(0)])
        acc["nn_w1"].append(_w1(nn_r, nn_g))
        def sep(s):
            v, si = s["vac"], s["sia"]
            return np.linalg.norm(v.mean(0) - si.mean(0)) if len(v) and len(si) else 0.0
        acc["vac_sia_sep_err"].append(abs(np.mean([sep(s) for s in R]) - np.mean([sep(s) for s in G])))
        diag["rg_w1"].append(_w1([_rg(c) for c in Ru], [_rg(c) for c in Gu]))
        diag["lcf_real"].append(np.mean([_largest_cluster_frac(c) for c in Ru]))
        diag["lcf_gen"].append(np.mean([_largest_cluster_frac(c) for c in Gu]))

    requirements, num, den = {}, 0.0, 0.0
    for k, spec in REQUIREMENTS.items():
        vals = [v for v in acc[k] if np.isfinite(v)]
        value = float(np.mean(vals)) if vals else float("nan")
        requirements[k] = {"value": value, "target": spec["target"],
                           "weight": spec["weight"], "label": spec["label"],
                           "pass": bool(np.isfinite(value) and value <= spec["target"])}
        if np.isfinite(value):
            num += spec["weight"] * (value / spec["target"]); den += spec["weight"]
    score = float(num / den) if den else float("nan")
    diag = {k: (float(np.mean([x for x in v if np.isfinite(x)])) if v else float("nan"))
            for k, v in diag.items()}
    return {"score": score, "n_pass": sum(r["pass"] for r in requirements.values()),
            "n_req": len(requirements), "requirements": requirements, "diagnostics": diag,
            "energies": [float(e) for e in energies]}


def print_scorecard(sc, prefix="[score]"):
    print(f"{prefix} SCORE {sc['score']:.3f}  ({sc['n_pass']}/{sc['n_req']} pass)  "
          f"energies={sc['energies']}")
    for k, r in sc["requirements"].items():
        v = r["value"]; vs = "nan" if not np.isfinite(v) else f"{v:.4f}"
        print(f"{prefix}   {r['label']:26s} {vs:>9s} / {r['target']:<5g} "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    d = sc["diagnostics"]
    print(f"{prefix}   diag: Rg-W1 {d['rg_w1']:.2f} | LCF real {d['lcf_real']:.2f} "
          f"gen {d['lcf_gen']:.2f}")


# =========================================================================== #
# Sampling / scoring helpers
# =========================================================================== #
def _filter_logits(logits, temperature, top_k, top_p):
    if temperature != 1.0: logits = logits / max(temperature, 1e-6)
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
    if top_p and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cdf = torch.cumsum(F.softmax(sl, -1), -1)
        rm = cdf > top_p; rm[:, 1:] = rm[:, :-1].clone(); rm[:, 0] = False
        sl[rm] = -float('inf')
        logits = torch.full_like(logits, -float('inf')).scatter(-1, si, sl)
    return logits


@torch.no_grad()
def sample_tokens(model, tok, energies_keV, *, max_new=8192, temperature=1.0,
                  top_k=0, top_p=1.0, prefix=None, device='cpu'):
    model.eval()
    B = len(energies_keV)
    pm = tok.phase_masks(device)
    if prefix is None:
        seqs = [[BOS, tok._e_to_id(float(e))] for e in energies_keV]
    else:
        seqs = [list(p) for p in prefix]
    L0 = len(seqs[0])
    assert all(len(s) == L0 for s in seqs), "cached sampler assumes equal-length prefixes"

    def phase_of(seq):
        return {0: PH_TYPE, 1: PH_X, 2: PH_Y, 3: PH_Z}[len(seq[2:]) % 4]
    phase = [phase_of(s) for s in seqs]
    out_seqs = [list(s) for s in seqs]
    done = [False] * B
    idx = torch.tensor(seqs, dtype=torch.long, device=device)
    logits, caches = model.decode_forward(idx, None, 0)
    last = logits[:, -1, :]; pos = L0
    for _ in range(max_new):
        if all(done): break
        allow = pm[torch.tensor(phase, device=device)]
        last = last.masked_fill(~allow, -float('inf'))
        last = _filter_logits(last, temperature, top_k, top_p)
        nxt = torch.multinomial(F.softmax(last, -1), 1)
        for i in range(B):
            if done[i]: continue
            t = int(nxt[i, 0]); out_seqs[i].append(t)
            if phase[i] == PH_TYPE:
                done[i] = (t == END)
                if not done[i]: phase[i] = PH_X
            elif phase[i] == PH_X: phase[i] = PH_Y
            elif phase[i] == PH_Y: phase[i] = PH_Z
            else:                  phase[i] = PH_TYPE
        logits, caches = model.decode_forward(nxt, caches, pos)
        last = logits[:, -1, :]; pos += 1
    return out_seqs


@torch.no_grad()
def generate(model, tok, energy_keV, n_samples=1, *, temperature=1.0,
             top_k=0, top_p=0.95, device='cpu', max_new=9000):
    toks = sample_tokens(model, tok, [energy_keV] * n_samples, max_new=max_new,
                         temperature=temperature, top_k=top_k, top_p=top_p, device=device)
    return [tok.decode(t)[:2] for t in toks]


@torch.no_grad()
def score_model(model, tok, ref_dicts, device, *, energy_bin=25.0, n_per_energy=16,
                gen_batch=32, max_points=600, temperature=1.0, top_p=0.95, seed=0):
    """Generate samples matched to the reference energy distribution and score them.
    Generation length is capped per band to ~1.3x the real cascades there (CPU/GPU-friendly)."""
    model.eval(); rng = np.random.default_rng(seed)
    bins = {}
    for s in ref_dicts:
        bins.setdefault(round(s["energy"] / energy_bin) * energy_bin, []).append(s)
    gen = []
    for center, samples in sorted(bins.items()):
        rlens = [len(tok.encode(s["vac"], s["sia"], s["energy"], max_len=None)) for s in samples]
        mn = int(min(model.max_seq_len, 1.3 * max(rlens) + 8))
        n = min(n_per_energy, len(samples))
        ens = [float(samples[i]["energy"]) for i in rng.integers(0, len(samples), size=n)]
        for i in range(0, len(ens), gen_batch):
            chunk = ens[i:i + gen_batch]
            seqs = sample_tokens(model, tok, chunk, max_new=mn, temperature=temperature,
                                 top_k=0, top_p=top_p, device=device)
            for seq, e in zip(seqs, chunk):
                v, s, _ = tok.decode(seq)
                gen.append({"vac": v, "sia": s, "energy": float(e)})
    return score_cards(gen, ref_dicts, energy_bin=energy_bin, max_points=max_points)


# =========================================================================== #
# Training  (AMP + grad-accum + scorecard-in-the-loop, best on SCORE)
# =========================================================================== #
def _save(model, tok, cfg, path, epoch, extra=None):
    d = {'epoch': epoch, 'model_state_dict': model.state_dict(),
         'tokenizer': tok.to_dict(), 'config': cfg}
    if extra: d.update(extra)
    torch.save(d, path)


def load_model(path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=device, weights_only=False)
    tok = CascadeTokenizer.from_dict(ck['tokenizer']); cfg = ck['config']
    model = CascadeGPT(tok.vocab_size, cfg['d_model'], cfg['depth'], cfg['n_heads'],
                       max_seq_len=cfg['max_seq_len']).to(device)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    return model, tok


def train(raw, tokenizer, *, output_dir='./checkpoints_cascadegpt',
          d_model=512, depth=12, n_heads=8, dropout=0.1,
          max_seq_len=9000, batch_size=2, grad_accum=2, lr=3e-4, weight_decay=0.1,
          num_epochs=30, warmup_frac=0.03, grad_clip=1.0, val_frac=0.1,
          log_every=1, save_every=10,
          use_amp=True, use_checkpoint=True,
          # ---- scorecard-in-the-loop ----
          score_every=5, score_start_epoch=3, score_n_per_energy=16,
          score_energy_bin=25.0, score_max_ref=120, score_max_points=600,
          score_top_p=0.95, device=None, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    cfg = {'d_model': d_model, 'depth': depth, 'n_heads': n_heads, 'max_seq_len': max_seq_len}

    # split RAW (so we keep val coords for scoring), then tokenize
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(raw))
    n_val = max(1, int(val_frac * len(raw)))
    val_raw = [raw[i] for i in perm[:n_val]]
    train_raw = [raw[i] for i in perm[n_val:]]
    train_ds = CascadeSequenceDataset(train_raw, tokenizer, max_seq_len=max_seq_len, truncate=False)
    val_ds   = CascadeSequenceDataset(val_raw,   tokenizer, max_seq_len=max_seq_len, truncate=False)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    vl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    # held-out reference for scoring (cap for speed)
    ref_dicts = [{"vac": v, "sia": s, "energy": e} for (v, s, e) in val_raw]
    if len(ref_dicts) > score_max_ref:
        sel = rng.permutation(len(ref_dicts))[:score_max_ref]
        ref_dicts = [ref_dicts[i] for i in sel]

    model = CascadeGPT(tokenizer.vocab_size, d_model, depth, n_heads,
                       dropout=dropout, max_seq_len=max_seq_len, use_checkpoint=use_checkpoint).to(device)
    print(f"device={device} | params: {sum(p.numel() for p in model.parameters()):,} | "
          f"train/val seqs: {len(train_ds)}/{len(val_ds)} | ref for scoring: {len(ref_dicts)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    opt_steps_per_epoch = max(1, len(tl) // grad_accum)
    total_opt_steps = opt_steps_per_epoch * num_epochs
    warm = max(1, int(warmup_frac * total_opt_steps))
    def lr_mult(s):
        if s < warm: return s / warm
        p = (s - warm) / max(1, total_opt_steps - warm); return 0.5 * (1 + math.cos(math.pi * p))

    cuda = ('cuda' in str(device))
    amp_on = use_amp and cuda
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = amp_on and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    autocast_ctx = (lambda: torch.autocast('cuda', dtype=amp_dtype)) if amp_on else nullcontext
    print(f"AMP={'on('+str(amp_dtype).split('.')[-1]+')' if amp_on else 'off'} | "
          f"grad_accum={grad_accum} (eff batch {batch_size*grad_accum}) | checkpoint={use_checkpoint}")

    best_score = float('inf'); best_loss = float('inf')
    tr_hist, va_hist, va_ep = [], [], []; opt_step = 0
    for ep in range(num_epochs):
        model.train(); losses = []; t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for i, x in enumerate(tl):
            x = x.to(device)
            with autocast_ctx():
                logits = model(x[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       x[:, 1:].reshape(-1), ignore_index=PAD) / grad_accum
            (scaler.scale(loss) if use_scaler else loss).backward()
            losses.append(loss.item() * grad_accum)
            if (i + 1) % grad_accum == 0:
                for g in opt.param_groups: g['lr'] = lr * lr_mult(opt_step)
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                opt.zero_grad(set_to_none=True); opt_step += 1
        tr_hist.append(float(np.mean(losses)) if losses else float('nan'))

        # ---- validation loss ----
        do_log = (ep + 1) % log_every == 0 or ep == 0
        vmean = float('nan')
        if do_log:
            model.eval(); vls = []
            with torch.no_grad():
                for x in vl:
                    x = x.to(device)
                    with autocast_ctx():
                        lg = model(x[:, :-1])
                        vls.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                   x[:, 1:].reshape(-1), ignore_index=PAD).item())
            vmean = float(np.mean(vls)) if vls else float('nan')
            va_hist.append(vmean); va_ep.append(ep)
            print(f"epoch {ep+1:4d}/{num_epochs} | train {tr_hist[-1]:.4f} | val {vmean:.4f} | "
                  f"ppl {math.exp(min(vmean,20)):.2f} | lr {opt.param_groups[0]['lr']:.2e} | "
                  f"{time.time()-t0:.1f}s")
            if np.isfinite(vmean) and vmean < best_loss:
                best_loss = vmean
                _save(model, tokenizer, cfg, os.path.join(output_dir, 'best_loss_model.pt'),
                      ep, {'val_loss': vmean})

        # ---- scorecard (generation-based) -> save best on SCORE ----
        if (ep + 1) >= score_start_epoch and (ep + 1) % score_every == 0:
            ts = time.time()
            sc = score_model(model, tokenizer, ref_dicts, device,
                             energy_bin=score_energy_bin, n_per_energy=score_n_per_energy,
                             max_points=score_max_points, top_p=score_top_p)
            print_scorecard(sc, prefix=f"[score ep{ep+1}]")
            print(f"[score ep{ep+1}] scoring took {time.time()-ts:.0f}s")
            if np.isfinite(sc['score']) and sc['score'] < best_score:
                best_score = sc['score']
                _save(model, tokenizer, cfg, os.path.join(output_dir, 'best_model.pt'),
                      ep, {'score': sc['score'], 'scorecard': {k: v['value']
                           for k, v in sc['requirements'].items()}})
                print(f"[score ep{ep+1}] *** new best SCORE {best_score:.3f} -> best_model.pt ***")

        if (ep + 1) % save_every == 0:
            _save(model, tokenizer, cfg, os.path.join(output_dir, f'ckpt_{ep+1}.pt'), ep)

    _save(model, tokenizer, cfg, os.path.join(output_dir, 'final_model.pt'), num_epochs - 1)
    _plot_curves(tr_hist, va_hist, va_ep, output_dir)
    print(f"done. best val {best_loss:.4f} | best score {best_score:.3f}")
    return model, tokenizer


def _plot_curves(tr, va, va_ep, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(tr) + 1), tr, label='Train', alpha=.8)
    if va: ax.plot([e + 1 for e in va_ep], va, label='Val', marker='o', ms=3, alpha=.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('cross-entropy (nats/token)')
    ax.set_title('CascadeGPT training'); ax.grid(True, alpha=.3); ax.legend()
    plt.tight_layout(); out = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(out, dpi=150); plt.close(fig); print(f"saved {out}")


def main():
    DATA_ROOT = "/lcrc/project/battdat/vignesh/projects/defect/defect_files"
    OUTPUT_DIR = "/lcrc/project/battdat/vignesh/projects/defect/agpt"

    if os.path.isdir(DATA_ROOT):
        raw = load_real_raw(DATA_ROOT)
    else:
        print(f"DATA_ROOT not found ({DATA_ROOT}); using synthetic cascades.")
        raw = load_synthetic_raw(n_samples=1500)

    tok = CascadeTokenizer(n_energy_bins=64, n_coord_bins=1024, order='pair').fit(raw)

    model, tok = train(
        raw, tok, output_dir=OUTPUT_DIR,
        d_model=512, depth=12, n_heads=8, dropout=0.1,
        max_seq_len=9000, batch_size=2, grad_accum=2,
        lr=3e-4, weight_decay=0.1, num_epochs=30,
        use_amp=True, use_checkpoint=True,
        score_every=5, score_start_epoch=3, score_n_per_energy=16,
        score_energy_bin=25.0, score_max_ref=120,
    )
    print("\nDone! best_model.pt = best scorecard SCORE; best_loss_model.pt = best val loss.")


if __name__ == '__main__':
    main()