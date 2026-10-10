"""Train fm and hybrid on both datasets and write the data contract to site/data."""
import json, os, sys
sys.path.insert(0, "/home/josh/repos/fm-hybrid")
import torch, fm
torch.set_num_threads(8)

OUT = os.path.join(os.path.dirname(__file__), "..", "site", "data")
RES = "/home/josh/repos/fm-hybrid/results.jsonl"
DS, MS, NOTE = ["moons", "gaussians8"], ["fm", "hybrid"], "joint16_expect"
CFG = {**fm.HYBRID, "bits": 4, "R": 6, "w_bce": 1, "w_off": 1, "decode": "expect",
       "offset_target": "expect", "head": "joint", "smooth": 0.0, "width": 128}
N, STEPS, B, R = 1000, 48, 4, 6


def cfg_of(m): return CFG if m == "hybrid" else {}


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f: json.dump(obj, f, separators=(",", ":"))


@torch.no_grad()
def trajectory(model, m, ds):
    x = torch.randn(N, 2, generator=torch.Generator().manual_seed(1000))
    h, frames = 1.0 / STEPS, [x.clone()]
    for i in range(STEPS):
        t = torch.full((N, 1), i * h)
        xm = x + 0.5 * h * fm.predict_v(model, m, cfg_of(m), x, t)[0]
        x = x + h * fm.predict_v(model, m, cfg_of(m), xm, t + 0.5 * h)[0]
        frames.append(x.clone())
    return torch.stack(frames).float().contiguous().numpy().tobytes()


@torch.no_grad()
def quant(model, ds):
    x, t, v = fm.batch(ds, 512, torch.Generator().manual_seed(7))
    vh, (lg, off, ex) = fm.predict_v(model, "hybrid", CFG, x, t)
    sm = lg.softmax(-1)
    ug = fm.ungray(torch.arange(16), B)
    probs = torch.zeros_like(sm)
    probs[:, :, ug] = sm
    centres = fm.centre(torch.arange(16), B, R)
    expect = (probs * centres).sum(-1)
    return dict(bits=B, R=R, n=512, gray=[fm.gray(i) for i in range(16)], centres=centres.tolist(),
                xt=x.tolist(), t=t[:, 0].tolist(), v=v.tolist(), bucket=fm.bucketize(v, B, R).tolist(),
                logits=lg.tolist(), probs=probs.tolist(), expect=expect.tolist(), offset=off.tolist(),
                vhat=(expect + off).tolist())


@torch.no_grad()
def fixture():
    torch.manual_seed(5)
    x, t, v = fm.batch("moons", 16, torch.Generator().manual_seed(3))
    fx = dict(hidden=8, n=16, x=x.tolist(), t=t[:, 0].tolist(), v=v.tolist())
    for m, out in (("fm", 2), ("hybrid", 34)):
        net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8),
                                  torch.nn.SiLU(), torch.nn.Linear(8, out))
        c = cfg_of(m)
        d = {f"{k}{i}": getattr(net[j], k).tolist() for i, j in ((1, 0), (2, 2), (3, 4)) for k in ("weight", "bias")}
        d = {k.replace("weight", "W").replace("bias", "b"): w for k, w in d.items()}
        d["vhat"] = fm.predict_v(net, m, c, x, t)[0].tolist()
        d["loss"] = fm.loss_fn(net, m, c, x, t, v).item()
        if m == "hybrid":
            raw = fm.predict_v(net, m, c, x, t)[1]
            d["ce"] = torch.nn.functional.cross_entropy(raw[0].reshape(-1, 16), fm.gray(fm.bucketize(v, B, R)).reshape(-1)).item()
            d["off"] = torch.nn.functional.mse_loss(raw[1], v - raw[2]).item()
        fx[m] = d
    return fx


def results():
    rows = [json.loads(l) for l in open(RES)]
    def seeds(note, ds): return [r["energy_distance"] for r in sorted(rows, key=lambda r: r["seed"]) if r["note"] == note and r["data"] == ds]
    def stat(s):
        mu = sum(s) / len(s)
        return dict(mean=mu, std=(sum((a - mu) ** 2 for a in s) / len(s)) ** 0.5, seeds=s)
    notes = list(dict.fromkeys(r["note"] for r in rows))
    return dict(best_note=NOTE, cfg=next(r["cfg"] for r in rows if r["note"] == NOTE),
                datasets={ds: dict(baseline=stat(seeds("baseline", ds)), hybrid=stat(seeds(NOTE, ds))) for ds in DS},
                log=[dict(note=n, cfg=next(r["cfg"] for r in rows if r["note"] == n),
                          **{ds: sum(seeds(n, ds)) / len(seeds(n, ds)) for ds in DS}) for n in notes])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    man = dict(frames=STEPS + 1, n=N, bounds=[-3.5, 3.5], bits=B, R=R, datasets=DS, methods=MS, traj={}, data={})
    for ds in DS:
        man["traj"][ds] = {}
        for m in MS:
            model, _ = fm.train(m, cfg_of(m), ds, steps=3000, seed=0)
            fn = f"traj_{ds}_{m}.bin"
            open(os.path.join(OUT, fn), "wb").write(trajectory(model, m, ds))
            man["traj"][ds][m] = fn
            if m == "hybrid": dump(f"quant_{ds}.json", quant(model, ds))
            print(ds, m, "done", flush=True)
        man["data"][ds] = f"data_{ds}.bin"
        open(os.path.join(OUT, man["data"][ds]), "wb").write(fm.data(ds, N, torch.Generator().manual_seed(2000)).float().numpy().tobytes())
    dump("manifest.json", man); dump("fixture.json", fixture()); dump("results.json", results())
