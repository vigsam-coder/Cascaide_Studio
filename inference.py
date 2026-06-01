"""
USAGE
-----
  python inference.py --ckpt checkpoints_pairs/best_model.pt \
                      --sweep 10,50,100,200 --ensemble 16 --out exports

  # No checkpoint yet? Generate a physically-plausible stand-in ensemble
  # (numpy only -- no torch) so the export+viewer loop works end-to-end:
  python inference.py --demo --energy 100 --ensemble 12 --out exports

  # Single energy, choose formats:
  python inference.py --ckpt best_model.pt --energy 75 --ensemble 8 \
                      --formats json,xyz,dump,csv --out exports
"""

import os
import json
import time
import argparse
import datetime
import numpy as np

SCHEMA = "cascade_cloud_v1"
TYPE_NAMES = {0: "vac", 1: "sia"}



def _bbox(points):
    if len(points) == 0:
        z = [0.0, 0.0, 0.0]
        return z, z
    return points.min(0).tolist(), points.max(0).tolist()


def member_stats(vac, sia):
    """Geometric summary of one cascade, all in absolute Angstrom."""
    allp = np.concatenate([p for p in (vac, sia) if len(p)], 0) \
        if (len(vac) or len(sia)) else np.zeros((0, 3), np.float32)
    if len(allp) == 0:
        return {"n_vac": 0, "n_sia": 0, "n_total": 0,
                "centroid": [0, 0, 0], "rg": 0.0,
                "bbox_min": [0, 0, 0], "bbox_max": [0, 0, 0],
                "extent": [0, 0, 0], "vac_sia_centroid_dist": 0.0}
    c = allp.mean(0)
    rg = float(np.sqrt(((allp - c) ** 2).sum(1).mean()))
    bmin, bmax = _bbox(allp)
    icd = (float(np.linalg.norm(vac.mean(0) - sia.mean(0)))
           if (len(vac) and len(sia)) else 0.0)
    return {
        "n_vac": int(len(vac)), "n_sia": int(len(sia)),
        "n_total": int(len(vac) + len(sia)),
        "centroid": [round(float(v), 3) for v in c],
        "rg": round(rg, 3),
        "bbox_min": [round(float(v), 3) for v in bmin],
        "bbox_max": [round(float(v), 3) for v in bmax],
        "extent": [round(float(bmax[i] - bmin[i]), 3) for i in range(3)],
        "vac_sia_centroid_dist": round(icd, 3),
    }


def ensemble_stats(members):
    tot = np.array([m["stats"]["n_total"] for m in members], float)
    nv = np.array([m["stats"]["n_vac"] for m in members], float)
    ns = np.array([m["stats"]["n_sia"] for m in members], float)
    rg = np.array([m["stats"]["rg"] for m in members], float)
    icd = np.array([m["stats"]["vac_sia_centroid_dist"] for m in members], float)

    def pct(a, p):
        return round(float(np.percentile(a, p)), 2) if len(a) else 0.0

    hist_counts, hist_edges = (np.histogram(tot, bins=12)
                               if len(tot) else (np.array([]), np.array([])))
    return {
        "n_members": len(members),
        "total":  {"mean": round(float(tot.mean()), 1) if len(tot) else 0,
                   "std": round(float(tot.std()), 1) if len(tot) else 0,
                   "min": int(tot.min()) if len(tot) else 0,
                   "max": int(tot.max()) if len(tot) else 0,
                   "p50": pct(tot, 50), "p99": pct(tot, 99)},
        "n_vac_mean": round(float(nv.mean()), 1) if len(nv) else 0,
        "n_sia_mean": round(float(ns.mean()), 1) if len(ns) else 0,
        "rg_mean": round(float(rg.mean()), 2) if len(rg) else 0,
        "vac_sia_centroid_dist_mean": round(float(icd.mean()), 2) if len(icd) else 0,
        "count_hist": {"counts": hist_counts.tolist(),
                       "edges": [round(float(e), 1) for e in hist_edges]},
    }


def _round_coords(arr, nd=3):
    return np.round(np.asarray(arr, np.float64), nd).tolist()


def synth_ensemble(energy_keV, n_members, cap, seed=0):
    rng = np.random.default_rng(seed + int(energy_keV * 1000))
    members = []
    for _ in range(n_members):
        mean_pairs = max(1.0, energy_keV * 1.0)
        n_pairs = int(np.clip(np.round(rng.lognormal(np.log(mean_pairs), 0.5)),
                              0, cap))
        # absolute box origin: a global centroid far from 0 to prove we report
        # TRUE absolute coordinates (not recentred)
        G = np.array([512.0, 512.0, 512.0])
        n_clusters = max(1, int(round(2 + energy_keV / 30)))
        vac_parts, sia_parts = [], []
        for c in range(n_clusters):
            frac = n_pairs // n_clusters + (1 if c < n_pairs % n_clusters else 0)
            if frac <= 0:
                continue
            center = G + rng.normal(0, 0.18 * (30 * (energy_keV / 20) ** 0.4), 3)
            core = rng.normal(0, 6, 3)
            vac = center + core + rng.normal(0, 5, (frac, 3))
            d = rng.normal(0, 1, (frac, 3))
            d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
            sia = (center + core + d * rng.uniform(8, 20, (frac, 1))
                   + rng.normal(0, 2.5, (frac, 3)))
            vac_parts.append(vac)
            sia_parts.append(sia)
        vac = (np.concatenate(vac_parts, 0) if vac_parts
               else np.zeros((0, 3))).astype(np.float32)
        sia = (np.concatenate(sia_parts, 0) if sia_parts
               else np.zeros((0, 3))).astype(np.float32)
        members.append((vac, sia))
    return members

def model_ensemble(ckpt, energy_keV, n_members, cap, device):
    import torch  # noqa
    try:
        from set_diffusion_pairs import load_model, generate
    except Exception as e:
        raise SystemExit(
            "Could not import set_diffusion_pairs.py. Place inference.py next to "
            "your training file (the module that defines load_model/generate).\n"
            f"Import error: {e}")
    den, diff, ch, norm, cfg = load_model(ckpt, device=device)
    ed = cfg.get("energy_divisor", 300.0)
    out = generate(den, diff, ch, norm, energy_keV, energy_divisor=ed,
                   n_samples=n_members, cap=cap, device=device)
    return out, norm, ed



def build_payload(raw_members, energy_keV, material, energy_divisor,
                  source, normalizer=None):
    members = []
    for i, (vac, sia) in enumerate(raw_members):
        vac = np.asarray(vac, np.float32).reshape(-1, 3)
        sia = np.asarray(sia, np.float32).reshape(-1, 3)
        members.append({
            "id": i,
            "vac": _round_coords(vac),
            "sia": _round_coords(sia),
            "stats": member_stats(vac, sia),
        })
    meta = {
        "energy_keV": energy_keV,
        "material": material,
        "energy_divisor": energy_divisor,
        "units": "angstrom",
        "coordinate_frame": "absolute_box",
        "source": source,
        "model": "set_diffusion_pairs (count-head DiT)",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if normalizer is not None:
        try:
            meta["normalizer"] = normalizer.to_dict()
        except Exception:
            pass
    return {"format": SCHEMA, "meta": meta, "ensemble": members,
            "ensemble_stats": ensemble_stats(members)}

def write_json(payload, path):
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def write_stats(payload, path):
    with open(path, "w") as f:
        json.dump({"meta": payload["meta"],
                   "ensemble_stats": payload["ensemble_stats"]}, f, indent=2)
    return path


def write_xyz(vac, sia, path, energy_keV):
    n = len(vac) + len(sia)
    with open(path, "w") as f:
        f.write(f"{n}\n")
        f.write(f'Properties=species:S:1:pos:R:3 energy_keV={energy_keV} '
                f'frame=absolute_box units=angstrom\n')
        for p in vac:
            f.write(f"V {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        for p in sia:
            f.write(f"I {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    return path


def write_dump(vac, sia, path, pad=5.0):
    allp = np.concatenate([p for p in (vac, sia) if len(p)], 0) \
        if (len(vac) or len(sia)) else np.zeros((1, 3))
    lo = allp.min(0) - pad
    hi = allp.max(0) + pad
    n = len(vac) + len(sia)
    with open(path, "w") as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{n}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for a in range(3):
            f.write(f"{lo[a]:.4f} {hi[a]:.4f}\n")
        f.write("ITEM: ATOMS id type x y z\n")
        i = 1
        for p in vac:
            f.write(f"{i} 1 {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n"); i += 1
        for p in sia:
            f.write(f"{i} 2 {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n"); i += 1
    return path


def write_csv(vac, sia, path):
    with open(path, "w") as f:
        f.write("type,x,y,z\n")
        for p in vac:
            f.write(f"vac,{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")
        for p in sia:
            f.write(f"sia,{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")
    return path

def run_energy(energy_keV, args, manifest):
    print(f"\n=== {energy_keV} keV  |  ensemble={args.ensemble}  "
          f"|  source={'demo' if args.demo else 'model'} ===")
    t0 = time.time()
    if args.demo:
        raw = synth_ensemble(energy_keV, args.ensemble, args.cap, seed=args.seed)
        norm, ed = None, 300.0
        source = "demo_synthetic"
    else:
        raw, norm, ed = model_ensemble(args.ckpt, energy_keV, args.ensemble,
                                       args.cap, args.device)
        source = f"checkpoint:{os.path.basename(args.ckpt)}"

    payload = build_payload(raw, energy_keV, args.material, ed, source, norm)
    es = payload["ensemble_stats"]
    print(f"  total/cascade: {es['total']['mean']} +/- {es['total']['std']} "
          f"(min {es['total']['min']}, max {es['total']['max']}, "
          f"p99 {es['total']['p99']})")
    print(f"  vac/sia mean : {es['n_vac_mean']} / {es['n_sia_mean']} | "
          f"Rg {es['rg_mean']} A | vac-sia dist {es['vac_sia_centroid_dist_mean']} A")

    tag = f"{energy_keV:g}keV"
    files = {}
    fmts = [s.strip() for s in args.formats.split(",") if s.strip()]

    if "json" in fmts:
        p = write_json(payload, os.path.join(args.out, f"cascade_{tag}_ensemble.json"))
        files["json"] = p
        print(f"  wrote {p}")
    write_stats(payload, os.path.join(args.out, f"ensemble_stats_{tag}.json"))

    per_file = {k: [] for k in ("xyz", "dump", "csv") if k in fmts}
    if per_file:
        mdir = os.path.join(args.out, "members")
        os.makedirs(mdir, exist_ok=True)
        for m in payload["ensemble"]:
            vac = np.asarray(m["vac"], np.float32).reshape(-1, 3)
            sia = np.asarray(m["sia"], np.float32).reshape(-1, 3)
            base = os.path.join(mdir, f"{tag}_m{m['id']:03d}")
            if "xyz" in fmts:
                per_file["xyz"].append(write_xyz(vac, sia, base + ".xyz", energy_keV))
            if "dump" in fmts:
                per_file["dump"].append(write_dump(vac, sia, base + ".dump"))
            if "csv" in fmts:
                per_file["csv"].append(write_csv(vac, sia, base + ".csv"))
        for k, v in per_file.items():
            print(f"  wrote {len(v)} .{k} member files -> {mdir}")
        files.update({k: f"members/ ({len(v)} files)" for k, v in per_file.items()})

    manifest["energies"].append({
        "energy_keV": energy_keV, "ensemble": args.ensemble,
        "files": files, "ensemble_stats": es})
    print(f"  done in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Inference + export for cascade diffusion.")
    ap.add_argument("--ckpt", default=None, help="path to trained checkpoint (.pt)")
    ap.add_argument("--demo", action="store_true",
                    help="synthesize a plausible ensemble with numpy (no torch/ckpt)")
    ap.add_argument("--energy", type=float, default=None, help="single PKA energy (keV)")
    ap.add_argument("--sweep", type=str, default=None,
                    help="comma list of energies, e.g. 10,50,100,200")
    ap.add_argument("--ensemble", type=int, default=12, help="samples per energy")
    ap.add_argument("--cap", type=int, default=1300, help="max N_pairs (count clamp)")
    ap.add_argument("--material", default="W", choices=["W", "Fe", "Ni"])
    ap.add_argument("--formats", default="json,xyz,dump,csv",
                    help="subset of json,xyz,dump,csv")
    ap.add_argument("--out", default="exports")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.demo and not args.ckpt:
        ap.error("provide --ckpt PATH (trained model) or --demo (numpy stand-in).")
    if args.device is None and not args.demo:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    energies = ([float(e) for e in args.sweep.split(",")] if args.sweep
                else [args.energy if args.energy is not None else 100.0])
    os.makedirs(args.out, exist_ok=True)
    manifest = {"format": SCHEMA, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "source": "demo" if args.demo else args.ckpt,
                "material": args.material, "energies": []}

    for e in energies:
        run_energy(e, args, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {os.path.join(args.out, 'manifest.json')}")
    print("Open the viewer and load any cascade_*_ensemble.json file.")


if __name__ == "__main__":
    main()