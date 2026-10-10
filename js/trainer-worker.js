// trainer-worker.js - worker for background training
"use strict";
importScripts("trainer.js");

var trainers = null;
var state = { step: 0, running: false, done: false };
var maxSteps = 3000;
var data = null;
var dataRefresh = null;
var timer = null, CHUNK = 10, METRIC_EVERY = 100;

function schedule() { if (timer === null) timer = setTimeout(loop, 0); }

// One chunk of lockstep steps per macrotask, so pause and reset messages get through.
function loop() {
  timer = null;
  if (!state.running || !trainers) return;
  var lf = NaN, lh = NaN;
  for (var i = 0; i < CHUNK && state.step < maxSteps; i++) {
    lf = trainers.fm.step(); lh = trainers.hybrid.step(); state.step++;
  }
  var done = state.step >= maxSteps;
  self.postMessage({ type: "progress", step: state.step, loss: { fm: lf, hybrid: lh }, done: done, epoch: state.epoch });
  if (state.step % METRIC_EVERY === 0 || done) {
    var fmS = trainers.fm.sample(500, 20, state.step), hyS = trainers.hybrid.sample(500, 20, state.step);
    var ref = FMTrainer.getDataset(dataRefresh.dataset, 500, dataRefresh.seed + state.step);
    self.postMessage({ type: "metric", step: state.step, epoch: state.epoch,
      ed: { fm: FMTrainer.energyDistance(fmS, ref), hybrid: FMTrainer.energyDistance(hyS, ref) },
      samples: { fm: fmS, hybrid: hyS } }, [fmS.buffer, hyS.buffer]);
  }
  if (done) { state.running = false; state.done = true;
    self.postMessage({ type: "state", step: state.step, running: false, done: true, epoch: state.epoch }); return; }
  schedule();
}

self.onmessage = function (e) {
  var msg = e.data;
  try {
    if (msg.type === "init") {
      var dataset = msg.dataset || "moons";
      var seed = msg.seed === undefined ? 0 : msg.seed;
      var epoch = msg.epoch;
      trainers = {
        fm: new FMTrainer("fm", dataset, { seed: seed, hidden: 64, batch: 128, lr: 2e-3 }),
        hybrid: new FMTrainer("hybrid", dataset, { seed: seed, hidden: 64, batch: 128, lr: 2e-3 })
      };
      state = { step: 0, running: false, done: false, epoch: epoch };
      data = FMTrainer.getDataset(dataset, 1000, seed + 10000);
      dataRefresh = { dataset: dataset, seed: seed };
      self.postMessage({ type: "data", buf: data });
      self.postMessage({ type: "state", step: 0, running: false, done: false, epoch: epoch });
    } else if (msg.type === "run") {
      if (!trainers) { self.postMessage({ type: "error", message: "not initialized" }); return; }
      if (!state.running && !state.done) { state.running = true; schedule(); }
      self.postMessage({ type: "state", step: state.step, running: state.running, done: state.done, epoch: state.epoch });
    } else if (msg.type === "pause") {
      state.running = false;
      if (timer !== null) { clearTimeout(timer); timer = null; }
      self.postMessage({ type: "state", step: state.step, running: false, done: false, epoch: state.epoch });
    } else if (msg.type === "reset") {
      if (timer !== null) { clearTimeout(timer); timer = null; }
      state = { step: 0, running: false, done: false, epoch: msg.epoch };
      trainers = null;
      self.postMessage({ type: "state", step: 0, running: false, done: false, epoch: state.epoch });
    } else {
      self.postMessage({ type: "error", message: "unknown message type: " + msg.type });
    }
  } catch (e) {
    self.postMessage({ type: "error", message: e && e.stack ? e.stack : String(e) });
  }
};
