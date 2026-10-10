#!/usr/bin/env node
// check_trainer.js - site/js/trainer.js matches the PyTorch fixture, has correct gradients and learns.
"use strict";
const fs = require("fs");
const path = require("path");
if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log("usage: node tools/check_trainer.js");
  process.exit(0);
}
const root = path.join(__dirname, "..");
const tpath = path.join(root, "site", "js", "trainer.js");
const fpath = path.join(root, "site", "data", "fixture.json");
for (const p of [tpath, fpath]) {
  if (!fs.existsSync(p)) { console.log(`FAIL: ${p} missing`); process.exit(1); }
}
const fails = [];
try {
  const FMTrainer = require(tpath);
  const fx = JSON.parse(fs.readFileSync(fpath, "utf8"));
  const n = fx.n;
  const x = Float64Array.from(fx.x.flat()), t = Float64Array.from(fx.t), v = Float64Array.from(fx.v.flat());
  const mean = (a) => a.reduce((s, q) => s + q, 0) / a.length;
  const names = ["W1", "b1", "W2", "b2", "W3", "b3"];
  for (const m of ["fm", "hybrid"]) {
    const f = fx[m];
    const tr = new FMTrainer(m, "moons", { hidden: fx.hidden, seed: 1 });
    tr.setParams(f);
    let r = tr.lossGrad(x, t, v, n);
    const dl = Math.abs(r.loss - f.loss);
    let dv = 0;
    const ref = f.vhat.flat();
    for (let i = 0; i < ref.length; i++) dv = Math.max(dv, Math.abs(r.vhat[i] - ref[i]));
    if (!(dl < 1e-5)) fails.push(`${m}: loss differs by ${dl}`);
    if (!(dv < 1e-5)) fails.push(`${m}: vhat differs by ${dv}`);
    const an = tr.grads.map((g) => Float64Array.from(g));
    const fixed = m === "hybrid" ? v.map((q, i) => q - r.expect[i]) : undefined;   // expect is detached in fm.py
    let worst = 0;
    const h = 1e-5;
    tr.params.forEach((p, k) => {
      for (let i = 0; i < p.length; i++) {
        const o = p[i];
        p[i] = o + h; const lp = tr.lossGrad(x, t, v, n, fixed).loss;
        p[i] = o - h; const lm = tr.lossGrad(x, t, v, n, fixed).loss;
        p[i] = o;
        const num = (lp - lm) / (2 * h);
        worst = Math.max(worst, Math.abs(an[k][i] - num) / Math.max(1e-6, Math.abs(an[k][i]) + Math.abs(num)));
      }
    });
    if (!(worst < 1e-4)) fails.push(`${m}: gradient relative error ${worst.toExponential(2)}`);
    console.log(`${m}: dloss ${dl.toExponential(1)} dvhat ${dv.toExponential(1)} gradrel ${worst.toExponential(1)}`);
  }
  const trs = ["fm", "hybrid"].map((m) => new FMTrainer(m, "moons", { hidden: 64, seed: 1 }));
  const ref = FMTrainer.getDataset("moons", 500, 99);
  const e0 = trs.map((tr) => FMTrainer.energyDistance(tr.sample(500, 20, 3), ref));
  const losses = [[], []];
  for (let i = 0; i < 400; i++) trs.forEach((tr, k) => losses[k].push(tr.step()));
  trs.forEach((tr, k) => {
    const m = ["fm", "hybrid"][k];
    const a = mean(losses[k].slice(0, 50)), b = mean(losses[k].slice(-50));
    if (!(b < 0.9 * a)) fails.push(`${m}: last-50 loss ${b.toFixed(4)} not < 0.9 * first-50 ${a.toFixed(4)}`);
    const s = tr.sample(500, 20, 3);
    if (!(s instanceof Float32Array) || s.length !== 1000) fails.push(`${m}: sample is not a Float32Array of 1000`);
    else if (s.some((q) => !Number.isFinite(q) || q < -4 || q > 4)) fails.push(`${m}: sample value not finite or outside [-4, 4]`);
    const e1 = FMTrainer.energyDistance(s, ref);
    if (!(e1 < e0[k])) fails.push(`${m}: energy distance ${e1} not below step 0 value ${e0[k]}`);
    console.log(`${m}: loss ${a.toFixed(3)} -> ${b.toFixed(3)}, energy ${e0[k].toFixed(4)} -> ${e1.toFixed(4)}`);
  });
} catch (e) {
  fails.push(`exception: ${e && e.stack ? e.stack : e}`);
}
if (fails.length) { fails.forEach((f) => console.log("FAIL: " + f)); process.exit(1); }
console.log("OK: trainer matches fixture, gradients agree, both methods learn");
