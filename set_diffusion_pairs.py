import os
import math
import json
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TYPE_VAC, TYPE_SIA = 0, 1

class CoordNormalizer:
    def __init__(self, percentile=99.0):
        self.percentile = percentile
        self.G = np.zeros(3, dtype=np.float32)
        self.s = np.ones(3, dtype=np.float32)

    def fit(self, all_coords):
        if len(all_coords):
            self.G = all_coords.mean(axis=0).astype(np.float32)
            dev = np.abs(all_coords - self.G)
            s = np.percentile(dev, self.percentile, axis=0).astype(np.float32)
            self.s = np.where(s < 1e-8, 1.0, s).astype(np.float32)
        return self

    def transform(self, c):
        return ((c - self.G) / self.s).astype(np.float32)

    def inverse(self, n):
        return (n * self.s + self.G).astype(np.float32)

    def to_dict(self):
        return {'percentile': self.percentile, 'G': self.G.tolist(),
                's': self.s.tolist()}

    @classmethod
    def from_dict(cls, d):
        o = cls(d['percentile'])
        o.G = np.asarray(d['G'], np.float32)
        o.s = np.asarray(d['s'], np.float32)
        return o



def _make_item(vac, sia, normalizer):
    """vac[Nv,3], sia[Ns,3] raw -> (coords[N,3] normed, types[N], n_pairs)."""
    parts_c, parts_t = [], []
    if len(vac):
        parts_c.append(normalizer.transform(np.asarray(vac, np.float32)))
        parts_t.append(np.full(len(vac), TYPE_VAC, np.int64))
    if len(sia):
        parts_c.append(normalizer.transform(np.asarray(sia, np.float32)))
        parts_t.append(np.full(len(sia), TYPE_SIA, np.int64))
    if parts_c:
        coords = np.concatenate(parts_c, 0).astype(np.float32)
        types = np.concatenate(parts_t, 0)
    else:
        coords = np.zeros((0, 3), np.float32)
        types = np.zeros((0,), np.int64)
    n_pairs = max(len(vac), len(sia))     # == n_vac == n_sia under conservation
    return coords, types, n_pairs


class _PairDatasetBase(Dataset):
    """Holds variable-length coord/type items plus full (energy, n_pairs)
    arrays for count-head training (the latter includes empty cascades)."""
    def _finalize(self, raw, energies, normalizer, energy_divisor):
        if normalizer is None:
            chunks = [c for vs in raw for c in vs if len(c)]
            allc = (np.concatenate(chunks, 0) if chunks
                    else np.zeros((0, 3), np.float32))
            normalizer = CoordNormalizer().fit(allc)
        self.normalizer = normalizer
        self.energy_divisor = energy_divisor

        self.items = []            # coord-bearing cascades only (n_total > 0)
        e_all, n_all = [], []      # ALL cascades (for the count head)
        for (vac, sia), e in zip(raw, energies):
            coords, types, n_pairs = _make_item(vac, sia, normalizer)
            e_all.append(e / energy_divisor)
            n_all.append(n_pairs)
            if len(coords) > 0:
                self.items.append({
                    'coords': torch.from_numpy(coords),
                    'types': torch.from_numpy(types),
                    'energy': torch.tensor(e / energy_divisor, dtype=torch.float32),
                    'n_pairs': n_pairs})
        self.count_energy = torch.tensor(e_all, dtype=torch.float32)
        self.count_npairs = torch.tensor(n_all, dtype=torch.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return it['coords'], it['types'], it['energy'], it['n_pairs']


class SyntheticCascadePairDataset(_PairDatasetBase):
    """Mimics the real stats: n_vac == n_sia, heavy-tailed count rising with E."""
    def __init__(self, n_samples=1500, energy_divisor=300.0, seed=0,
                 normalizer=None):
        rng = np.random.default_rng(seed)
        raw, energies = [], []
        for _ in range(n_samples):
            e = float(rng.uniform(5, 250))
            # mean ~ 2*e, log-normal spread -> heavy tail like the real data
            mean_pairs = max(1.0, e * 1.0)
            n_pairs = int(np.clip(
                np.round(rng.lognormal(np.log(mean_pairs), 0.5)), 0, 1100))
            center = rng.normal(0, 60, size=3)
            core = rng.normal(0, 8, size=3)
            vac = center + core + rng.normal(0, 5, size=(n_pairs, 3))
            d = rng.normal(0, 1, size=(n_pairs, 3))
            d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
            sia = (center + core + d * rng.uniform(8, 20, size=(n_pairs, 1))
                   + rng.normal(0, 2.5, size=(n_pairs, 3)))
            raw.append((vac.astype(np.float32), sia.astype(np.float32)))
            energies.append(e)
        self._finalize(raw, energies, normalizer, energy_divisor)


class RealCascadePairDataset(_PairDatasetBase):
    """Drop-in for the real dumps (same layout as the image pipeline)."""
    def __init__(self, data_root, energy_divisor=300.0, normalizer=None,
                 percentile=99.0):
        samples = []
        for ed in sorted(glob.glob(os.path.join(data_root, "*keV"))):
            meta = os.path.join(ed, "metadata.json")
            if not os.path.exists(meta):
                continue
            with open(meta) as f:
                data = json.load(f)
            emap = {c["cascade_id"]: c["pka_energy_eV"]
                    for c in data.get("cascades", [])}
            for vf in sorted(glob.glob(os.path.join(ed, "*_min_vac.dump"))):
                try:
                    cid = int(os.path.basename(vf).split("_")[0])
                except Exception:
                    continue
                sf = vf.replace("_min_vac.dump", "_min_sia.dump")
                if os.path.exists(sf) and cid in emap:
                    samples.append((vf, sf, emap[cid] / 1000.0))
        raw, energies = [], []
        for vf, sf, e in samples:
            raw.append((self._load(vf), self._load(sf)))
            energies.append(e)
        if normalizer is None:
            chunks = [c for vs in raw for c in vs if len(c)]
            allc = (np.concatenate(chunks, 0) if chunks
                    else np.zeros((0, 3), np.float32))
            normalizer = CoordNormalizer(percentile).fit(allc)
        self._finalize(raw, energies, normalizer, energy_divisor)
        print(f"[RealCascadePairDataset] {len(samples)} cascades, "
              f"{len(self.items)} non-empty, G={normalizer.G.round(2)}, "
              f"s={normalizer.s.round(2)}")

    @staticmethod
    def _load(path):
        try:
            from ovito.io import import_file
            data = import_file(path).compute()
            if data.particles.count == 0:
                return np.zeros((0, 3), np.float32)
            return np.array(data.particles.positions, dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), np.float32)


def collate_dynamic(batch):
    """Pad each batch to its own max length; build a real-token mask."""
    coords, types, energies, npairs = zip(*batch)
    B = len(batch)
    Nmax = max(c.shape[0] for c in coords)
    coord_pad = torch.zeros(B, Nmax, 3)
    type_pad = torch.zeros(B, Nmax, dtype=torch.long)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)      # True = real token
    for i, (c, t) in enumerate(zip(coords, types)):
        n = c.shape[0]
        coord_pad[i, :n] = c
        type_pad[i, :n] = t
        mask[i, :n] = True
    energies = torch.stack(energies)
    npairs = torch.tensor(npairs, dtype=torch.float32)
    return coord_pad, type_pad, mask, energies, npairs


class CountHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, energy):                 # energy: [B]
        out = self.net(energy[:, None])
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-3
        return mu, sigma                        # of y = log1p(N_pairs)

    def nll(self, energy, n_pairs):
        mu, sigma = self(energy)
        y = torch.log1p(n_pairs.clamp(min=0))
        return (0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)).mean()

    @torch.no_grad()
    def sample(self, energy, cap=1300):
        mu, sigma = self(energy)
        y = mu + sigma * torch.randn_like(mu)
        n = torch.expm1(y).round().clamp(0, cap).long()
        return n

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, d))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, key_padding_mask=None):
        sa, ca, ga, sm, cm, gm = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), sa, ca)
        a, _ = self.attn(h, h, h, need_weights=False,
                         key_padding_mask=key_padding_mask)
        x = x + ga.unsqueeze(1) * a
        h = modulate(self.norm2(x), sm, cm)
        x = x + gm.unsqueeze(1) * self.mlp(h)
        return x


class SetDenoiser(nn.Module):
    def __init__(self, d_model=256, n_heads=8, depth=6, mlp_ratio=4.0,
                 dropout=0.0):
        super().__init__()
        self.d = d_model
        self.in_proj = nn.Linear(3, d_model)            # coords only
        self.type_emb = nn.Embedding(2, d_model)        # vac / sia
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.e_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_f = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.proj_out = nn.Linear(d_model, 3)
        for m in (self.adaLN_f[-1], self.proj_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, coords, t, energy, types, key_padding_mask=None):
        c = self.t_mlp(timestep_embedding(t, self.d)) + self.e_mlp(energy[:, None])
        h = self.in_proj(coords) + self.type_emb(types)
        for blk in self.blocks:
            h = blk(h, c, key_padding_mask=key_padding_mask)
        sh, sc = self.adaLN_f(c).chunk(2, dim=-1)
        h = modulate(self.norm_f(h), sh, sc)
        return self.proj_out(h)


class CoordDiffusion:
    def __init__(self, T=1000, schedule='cosine', device='cpu'):
        self.T = T
        self.device = device
        betas = (self._cosine(T) if schedule == 'cosine'
                 else torch.linspace(1e-4, 0.02, T, dtype=torch.float64))
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), ab[:-1]])
        to = lambda x: x.float().to(device)
        self.betas = to(betas)
        self.sqrt_ab = to(torch.sqrt(ab))
        self.sqrt_1m_ab = to(torch.sqrt(1.0 - ab))
        self.sqrt_recip_ab = to(torch.sqrt(1.0 / ab))
        self.sqrt_recipm1_ab = to(torch.sqrt(1.0 / ab - 1.0))
        self.post_var = to(betas * (1.0 - ab_prev) / (1.0 - ab))
        self.post_c1 = to(betas * torch.sqrt(ab_prev) / (1.0 - ab))
        self.post_c2 = to((1.0 - ab_prev) * torch.sqrt(alphas) / (1.0 - ab))

    @staticmethod
    def _cosine(T, s=0.008):
        t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        ab = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        ab = ab / ab[0]
        return torch.clamp(1 - ab[1:] / ab[:-1], 1e-4, 0.999)

    def _ext(self, a, t, shape):
        return a.gather(0, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        return (self._ext(self.sqrt_ab, t, x0.shape) * x0 +
                self._ext(self.sqrt_1m_ab, t, x0.shape) * noise)

    def loss(self, model, coords, types, mask, energy):
        B = coords.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device).long()
        noise = torch.randn_like(coords)
        x_t = self.q_sample(coords, t, noise)
        kpm = ~mask                                    # True = ignore (pad)
        pred = model(x_t, t, energy, types, key_padding_mask=kpm)
        per = ((pred - noise) ** 2).mean(dim=-1)       # [B, N]
        m = mask.float()
        return (per * m).sum() / m.sum().clamp(min=1.0)

    @torch.no_grad()
    def predict_x0(self, x_t, t, eps):
        return (self._ext(self.sqrt_recip_ab, t, x_t.shape) * x_t -
                self._ext(self.sqrt_recipm1_ab, t, x_t.shape) * eps)

    @torch.no_grad()
    def sample(self, model, energy, types, clip_x0=4.0):
        """energy:[B]  types:[B,N] (known)  -> coords [B,N,3]. No padding."""
        model.eval()
        B, N = types.shape
        x = torch.randn(B, N, 3, device=self.device)
        for tv in reversed(range(self.T)):
            t = torch.full((B,), tv, device=self.device, dtype=torch.long)
            eps = model(x, t, energy, types, key_padding_mask=None)
            x0 = self.predict_x0(x, t, eps).clamp(-clip_x0, clip_x0)
            mean = (self._ext(self.post_c1, t, x.shape) * x0 +
                    self._ext(self.post_c2, t, x.shape) * x)
            if tv > 0:
                x = mean + torch.sqrt(self._ext(self.post_var, t, x.shape)) \
                    * torch.randn_like(x)
            else:
                x = mean
        return x


@torch.no_grad()
def generate(denoiser, diff, counthead, norm, energy_keV, energy_divisor=300.0,
             n_samples=1, cap=1300, device='cpu'):
    """Full generation: energy -> N_pairs -> coords -> (vac, sia) absolute."""
    e = torch.full((n_samples,), energy_keV / energy_divisor, device=device)
    npairs = counthead.sample(e, cap=cap)                       # [n_samples]
    out = []
    for i in range(n_samples):
        np_i = int(npairs[i].item())
        if np_i == 0:
            out.append((np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)))
            continue
        types = torch.cat([
            torch.full((np_i,), TYPE_VAC, dtype=torch.long),
            torch.full((np_i,), TYPE_SIA, dtype=torch.long)])[None].to(device)
        ei = e[i:i + 1]
        coords = diff.sample(denoiser, ei, types)[0].cpu().numpy()
        coords = norm.inverse(coords)
        out.append((coords[:np_i], coords[np_i:]))              # vac, sia
    return out

def train(dataset, *, output_dir='./checkpoints_pairs', d_model=256, depth=6,
          n_heads=8, T=1000, schedule='cosine', batch_size=32, lr=2e-4,
          weight_decay=1e-2, num_epochs=40, count_lr=1e-3, count_weight=1.0,
          val_frac=0.1, device=None, log_every=5, save_every=20,
          sample_every=25, sample_energies=(10., 50., 100., 200.),
          energy_divisor=300.0, count_cap=1300, count_steps_per_epoch=50,
          seed=42):
    os.makedirs(output_dir, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  non-empty cascades={len(dataset)}")

    n_val = max(1, int(val_frac * len(dataset)))
    tr, va = random_split(dataset, [len(dataset) - n_val, n_val],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True,
                    collate_fn=collate_dynamic)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_dynamic)

    denoiser = SetDenoiser(d_model=d_model, depth=depth, n_heads=n_heads).to(device)
    counthead = CountHead().to(device)
    print(f"denoiser params: {sum(p.numel() for p in denoiser.parameters()):,} | "
          f"counthead params: {sum(p.numel() for p in counthead.parameters()):,}")
    diff = CoordDiffusion(T=T, schedule=schedule, device=device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs, 1e-6)
    opt_c = torch.optim.Adam(counthead.parameters(), lr=count_lr)

    norm = dataset.normalizer if hasattr(dataset, 'normalizer') \
        else dataset.dataset.normalizer
    # full (energy, n_pairs) arrays for the count head (includes empties)
    base = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    ce = base.count_energy.to(device)
    cn = base.count_npairs.to(device)

    best = float('inf')
    tr_hist, va_hist, va_ep = [], [], []
    for epoch in range(num_epochs):
        denoiser.train(); losses = []; t0 = time.time()
        for coords, types, mask, energy, _ in tl:
            coords, types = coords.to(device), types.to(device)
            mask, energy = mask.to(device), energy.to(device)
            opt.zero_grad(set_to_none=True)
            loss = diff.loss(denoiser, coords, types, mask, energy)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sched.step()

        # ---- count head: a few full-data SGD steps per epoch ----
        counthead.train()
        for _ in range(count_steps_per_epoch):
            idx = torch.randint(0, len(ce), (min(512, len(ce)),), device=device)
            opt_c.zero_grad(set_to_none=True)
            closs = counthead.nll(ce[idx], cn[idx])
            closs.backward(); opt_c.step()

        tr_hist.append(float(np.mean(losses)) if losses else float('nan'))

        if (epoch + 1) % log_every == 0 or epoch == 0:
            denoiser.eval(); vls = []
            with torch.no_grad():
                for coords, types, mask, energy, _ in vl:
                    vl_ = diff.loss(denoiser, coords.to(device), types.to(device),
                                    mask.to(device), energy.to(device))
                    if torch.isfinite(vl_):
                        vls.append(vl_.item())
            va_l = float(np.mean(vls)) if vls else float('nan')
            va_hist.append(va_l); va_ep.append(epoch)
            with torch.no_grad():
                cnll = counthead.nll(ce, cn).item()
            print(f"epoch {epoch+1:4d}/{num_epochs} | coord {tr_hist[-1]:.5f} | "
                  f"val {va_l:.5f} | countNLL {cnll:.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")
            if np.isfinite(va_l) and va_l < best:
                best = va_l
                _save(denoiser, counthead, norm, T, schedule, d_model, depth,
                      n_heads, count_cap, energy_divisor,
                      os.path.join(output_dir, 'best_model.pt'), epoch)
        if (epoch + 1) % save_every == 0:
            _save(denoiser, counthead, norm, T, schedule, d_model, depth,
                  n_heads, count_cap, energy_divisor,
                  os.path.join(output_dir, f'ckpt_{epoch+1}.pt'), epoch)
        if sample_every and (epoch + 1) % sample_every == 0:
            print(f"  -> generating samples at epoch {epoch+1}...")
            _visualize(denoiser, diff, counthead, norm, epoch + 1, output_dir,
                       sample_energies, energy_divisor, count_cap, device)

    _save(denoiser, counthead, norm, T, schedule, d_model, depth, n_heads,
          count_cap, energy_divisor,
          os.path.join(output_dir, 'final_model.pt'), num_epochs - 1)
    _plot_curves(tr_hist, va_hist, va_ep, output_dir)
    print(f"done. best val {best:.5f}")
    return denoiser, diff, counthead, norm


def _save(denoiser, counthead, norm, T, schedule, d_model, depth, n_heads,
          count_cap, energy_divisor, path, epoch):
    torch.save({'epoch': epoch,
                'denoiser_state_dict': denoiser.state_dict(),
                'counthead_state_dict': counthead.state_dict(),
                'normalizer': norm.to_dict(),
                'config': {'T': T, 'schedule': schedule, 'd_model': d_model,
                           'depth': depth, 'n_heads': n_heads,
                           'count_cap': count_cap,
                           'energy_divisor': energy_divisor}}, path)


def load_model(path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck['config']
    den = SetDenoiser(d_model=cfg['d_model'], depth=cfg['depth'],
                      n_heads=cfg['n_heads']).to(device)
    den.load_state_dict(ck['denoiser_state_dict'])
    ch = CountHead().to(device)
    ch.load_state_dict(ck['counthead_state_dict'])
    diff = CoordDiffusion(T=cfg['T'], schedule=cfg['schedule'], device=device)
    norm = CoordNormalizer.from_dict(ck['normalizer'])
    return den, diff, ch, norm, cfg

@torch.no_grad()
def _visualize(denoiser, diff, counthead, norm, epoch, output_dir,
               test_energies, energy_divisor, cap, device):
    test_energies = list(test_energies)
    n = len(test_energies)
    fig = plt.figure(figsize=(4 * n, 8))
    for i, e in enumerate(test_energies):
        (vac, sia), = generate(denoiser, diff, counthead, norm, e,
                               energy_divisor, n_samples=1, cap=cap, device=device)
        ax1 = fig.add_subplot(2, n, i + 1, projection='3d')
        if len(vac):
            ax1.scatter(*vac.T, c='blue', s=8, alpha=0.6, label=f'Vac ({len(vac)})')
        if len(sia):
            ax1.scatter(*sia.T, c='red', s=8, alpha=0.6, label=f'SIA ({len(sia)})')
        ax1.set_title(f'{e} keV'); ax1.legend(fontsize=6)
        for s in ('x', 'y', 'z'):
            getattr(ax1, f'set_{s}label')(s.upper(), fontsize=6)
        ax2 = fig.add_subplot(2, n, n + i + 1)
        if len(vac):
            ax2.scatter(vac[:, 0], vac[:, 1], c='blue', s=8, alpha=0.6, label='Vac')
        if len(sia):
            ax2.scatter(sia[:, 0], sia[:, 1], c='red', s=8, alpha=0.6, label='SIA')
        ax2.set_title(f'{e} keV - XY'); ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.axis('equal'); ax2.legend(fontsize=6)
    plt.suptitle(f'Generated Defects (count-head set diffusion) - Epoch {epoch}')
    plt.tight_layout()
    out = os.path.join(output_dir, f'samples_epoch_{epoch}.png')
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"     saved {out}")


def _plot_curves(tr, va, va_ep, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(tr) + 1), tr, label='Train', alpha=0.8)
    if va:
        ax.plot([e + 1 for e in va_ep], va, label='Val', marker='o', ms=3, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('coord eps MSE'); ax.set_yscale('log')
    ax.set_title('Training Progress'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}")


@torch.no_grad()
def evaluate(denoiser, diff, counthead, norm, energy_keV, energy_divisor=300.0,
             n_samples=64, cap=1300, device='cpu'):
    samples = generate(denoiser, diff, counthead, norm, energy_keV,
                       energy_divisor, n_samples=n_samples, cap=cap, device=device)
    nv = [len(v) for v, s in samples]
    ns = [len(s) for v, s in samples]
    tot = [a + b for a, b in zip(nv, ns)]
    icd = [np.linalg.norm(v.mean(0) - s.mean(0))
           for v, s in samples if len(v) and len(s)]
    print(f"\n--- {energy_keV} keV ({n_samples} samples) ---")
    print(f"  total defects : {np.mean(tot):6.1f} +/- {np.std(tot):.1f}")
    print(f"  vac / sia     : {np.mean(nv):.1f} / {np.mean(ns):.1f}")
    if icd:
        print(f"  vac-SIA centroid dist : {np.mean(icd):.2f} A")
    return {'total': tot, 'n_vac': nv, 'n_sia': ns, 'icd': icd}


def main():

    DATA_ROOT = "/lcrc/project/battdat/vignesh/projects/defect/defect_files"
    OUTPUT_DIR = "/lcrc/project/battdat/vignesh/projects/defect/cascade_set_diffusion_pairs"

    ENERGY_DIVISOR = 300.0
    COUNT_CAP = 1300

    if os.path.isdir(DATA_ROOT):
        ds = RealCascadePairDataset(DATA_ROOT, energy_divisor=ENERGY_DIVISOR)
    else:
        print(f"DATA_ROOT not found ({DATA_ROOT}); using synthetic cascades.")
        ds = SyntheticCascadePairDataset(n_samples=1500,
                                         energy_divisor=ENERGY_DIVISOR)

    den, diff, ch, norm = train(
        ds,
        output_dir=OUTPUT_DIR,
        d_model=256,
        depth=6,
        n_heads=8,
        T=1000,
        schedule='cosine',
        batch_size=16,                # dynamic padding -> usually fits; lower to
                                      # 8 if the rare >150 keV batches OOM
        lr=2e-4,
        weight_decay=1e-2,
        num_epochs=600,
        count_lr=1e-3,
        count_weight=1.0,
        count_steps_per_epoch=50,
        log_every=5,
        save_every=50,
        sample_every=25,
        sample_energies=(10.0, 20.0, 50.0, 100.0, 150.0, 200.0),
        energy_divisor=ENERGY_DIVISOR,
        count_cap=COUNT_CAP,
    )

    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)
    for e in (10.0, 50.0, 100.0, 200.0):
        evaluate(den, diff, ch, norm, e, energy_divisor=ENERGY_DIVISOR,
                 n_samples=48, cap=COUNT_CAP)
    print("\nDone!")


if __name__ == '__main__':
    main()