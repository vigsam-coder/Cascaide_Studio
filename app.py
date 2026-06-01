"""
app.py  --  Cascade Studio backend (inference engine + web server)
==================================================================

  POST /api/generate   {energy_keV, temperature, material, n_samples}
                       -> runs the diffusion model n_samples times and returns
                          the clouds in ABSOLUTE box coordinates (Angstrom).
                          Also writes per-sample dumps to disk:
                              <out>/<E>keV/001_vac.dump  001_sia.dump  002_vac.dump ...
  POST /api/export     {energy_keV, material, samples:[{vac,sia}]}
                       -> zips 00N_vac.dump / 00N_sia.dump and streams it back.
  GET  /api/health     -> {engine: "model" | "demo"}
  GET  /               -> serves the UI

RUN
---
  # real model (needs torch + set_diffusion_pairs.py next to this file):
  python app.py --ckpt checkpoints_pairs/best_model.pt --port 8000

  # no checkpoint yet -- physically-plausible stand-in, no torch needed:
  python app.py --demo --port 8000

Then open http://localhost:8001  (the UI also runs standalone with an in-browser
preview engine if the backend isn't reachable, so the page is never dead).
"""

import os
import io
import time
import zipfile
import argparse
import datetime
import numpy as np

import inference as inf   # synth_ensemble, member_stats, build_payload helpers

# torch / model are imported lazily so --demo never needs them
STATE = {"engine": "demo", "model": None}


# ----------------------------------------------------------------------
# Model wrapper (loaded once)
# ----------------------------------------------------------------------
class Engine:
    def __init__(self, ckpt, device=None):
        import torch
        from set_diffusion_pairs import load_model, generate  # noqa
        self.torch = torch
        self.generate = generate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.den, self.diff, self.ch, self.norm, self.cfg = load_model(ckpt, device=self.device)
        self.energy_divisor = self.cfg.get("energy_divisor", 300.0)
        print(f"[engine] model loaded on {self.device} (energy_divisor={self.energy_divisor})")

    def sample(self, energy_keV, n_samples, cap=1300):
        raw = self.generate(self.den, self.diff, self.ch, self.norm, energy_keV,
                            energy_divisor=self.energy_divisor,
                            n_samples=n_samples, cap=cap, device=self.device)
        return raw, self.norm, self.energy_divisor


# ----------------------------------------------------------------------
# Generation (model or demo) -> viewer payload
# ----------------------------------------------------------------------
def run_generation(energy_keV, temperature, material, n_samples, cap=1300):
    if STATE["model"] is not None:
        raw, norm, ed = STATE["model"].sample(energy_keV, n_samples, cap)
        source = "model"
    else:
        # temperature nudges spread a touch so the (future-use) control feels live
        raw = inf.synth_ensemble(energy_keV, n_samples, cap,
                                 seed=int(time.time() * 1000) % 100000)
        norm, ed, source = None, 300.0, "demo"
    payload = inf.build_payload(raw, energy_keV, material, ed, source, norm)
    payload["meta"]["temperature_K"] = temperature
    return payload


# ----------------------------------------------------------------------
# Dump writers: SEPARATE vac / sia files, shared box, zero-padded ids
#     001_vac.dump  001_sia.dump  002_vac.dump ...
# ----------------------------------------------------------------------
def _dump_text(coords, box_lo, box_hi):
    n = len(coords)
    lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(n),
             "ITEM: BOX BOUNDS pp pp pp"]
    for a in range(3):
        lines.append(f"{box_lo[a]:.4f} {box_hi[a]:.4f}")
    lines.append("ITEM: ATOMS id type x y z")
    for i, p in enumerate(coords, 1):
        lines.append(f"{i} 1 {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    return "\n".join(lines) + "\n"


def sample_dumps(vac, sia, pad=5.0):
    vac = np.asarray(vac, np.float32).reshape(-1, 3)
    sia = np.asarray(sia, np.float32).reshape(-1, 3)
    allp = np.concatenate([p for p in (vac, sia) if len(p)], 0) \
        if (len(vac) or len(sia)) else np.zeros((1, 3))
    lo = allp.min(0) - pad
    hi = allp.max(0) + pad
    return _dump_text(vac, lo, hi), _dump_text(sia, lo, hi)


def write_dumps_to_disk(payload, out_root):
    tag = f"{payload['meta']['energy_keV']:g}keV"
    d = os.path.join(out_root, tag)
    os.makedirs(d, exist_ok=True)
    for m in payload["ensemble"]:
        vtxt, stxt = sample_dumps(m["vac"], m["sia"])
        i = m["id"] + 1
        with open(os.path.join(d, f"{i:03d}_vac.dump"), "w") as f:
            f.write(vtxt)
        with open(os.path.join(d, f"{i:03d}_sia.dump"), "w") as f:
            f.write(stxt)
    return d


def zip_dumps(samples):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, s in enumerate(samples, 1):
            vtxt, stxt = sample_dumps(s["vac"], s["sia"])
            z.writestr(f"{i:03d}_vac.dump", vtxt)
            z.writestr(f"{i:03d}_sia.dump", stxt)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------
# Web app
# ----------------------------------------------------------------------
def build_app(out_root, html_path):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Cascade Studio")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse)
    def index():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/api/health")
    def health():
        return {"engine": STATE["engine"],
                "time": datetime.datetime.now().isoformat(timespec="seconds")}

    @app.post("/api/generate")
    async def generate(req: Request):
        b = await req.json()
        energy = float(b.get("energy_keV", 100))
        temp = float(b.get("temperature", 300))
        material = b.get("material", "W")
        n = int(b.get("n_samples", 1))
        n = max(1, min(64, n))
        t0 = time.time()
        payload = run_generation(energy, temp, material, n)
        try:
            payload["meta"]["saved_to"] = write_dumps_to_disk(payload, out_root)
        except Exception as e:
            payload["meta"]["save_error"] = str(e)
        payload["meta"]["gen_seconds"] = round(time.time() - t0, 2)
        return JSONResponse(payload)

    @app.post("/api/export")
    async def export(req: Request):
        b = await req.json()
        samples = b.get("samples", [])
        tag = f"{b.get('energy_keV', 'x')}keV"
        buf = zip_dumps(samples)
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="cascade_{tag}_dumps.zip"'})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (.pt)")
    ap.add_argument("--demo", action="store_true", help="run without a model")
    ap.add_argument("--out", default="generated", help="where dumps are saved")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--html", default=os.path.join(os.path.dirname(__file__), "cascade_studio.html"))
    args = ap.parse_args()

    if args.ckpt and not args.demo:
        try:
            STATE["model"] = Engine(args.ckpt)
            STATE["engine"] = "model"
        except Exception as e:
            print(f"[engine] could not load model ({e}); falling back to demo.")
            STATE["engine"] = "demo"
    else:
        print("[engine] demo mode (no model).")
        STATE["engine"] = "demo"

    os.makedirs(args.out, exist_ok=True)
    import uvicorn
    app = build_app(args.out, args.html)
    print(f"open  http://localhost:{args.port}   (engine={STATE['engine']})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()