/* train-widget.js - trains the baseline and the hybrid side by side in a worker. Nothing runs at load. */
(function () {
  "use strict";
  var MAX = 3000, B = 3.5, READOUT_MS = 250;
  var here = document.currentScript && document.currentScript.src;
  var WORKER_URL = here ? here.replace(/[^\/]*$/, "trainer-worker.js") : "js/trainer-worker.js";
  var METHODS = ["fm", "hybrid"], NAMES = { fm: "Flow matching", hybrid: "Hybrid" };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }
  function css(name, fb) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fb; }
  function fit(c) {
    var d = window.devicePixelRatio || 1, w = c.clientWidth || 300, h = c.clientHeight || 200;
    if (c.width !== Math.round(w * d) || c.height !== Math.round(h * d)) { c.width = Math.round(w * d); c.height = Math.round(h * d); }
    var x = c.getContext("2d");
    x.setTransform(d, 0, 0, d, 0, 0);
    return { x: x, w: w, h: h };
  }

  function Widget(root) {
    var self = this;
    this.root = root;
    root.appendChild(el("p", "lede train-intro", "Both networks start from the same seed and train on the same batches. The dots are what each one generates from noise, and the grey cloud is the data."));
    var box = el("div", "widget");
    root.appendChild(box);
    root = box;
    var bar = el("div", "controls");
    this.select = el("select", "train-dataset");
    ["moons", "gaussians8"].forEach(function (v) { var o = el("option", "", v); o.value = v; self.select.appendChild(o); });
    this.select.setAttribute("aria-label", "Dataset");
    this.startBtn = el("button", "btn train-start", "Start");
    this.pauseBtn = el("button", "btn train-pause", "Pause");
    this.resetBtn = el("button", "btn train-reset", "Reset");
    this.stepEl = el("span", "readout step-readout");
    this.metricEl = el("span", "readout metric-readout");
    [this.select, this.startBtn, this.pauseBtn, this.resetBtn, this.stepEl, this.metricEl].forEach(function (n) { bar.appendChild(n); });
    root.appendChild(bar);
    var row = el("div", "train-row");
    this.dist = {};
    METHODS.forEach(function (m) {
      var f = el("figure", "panel train-fig");
      f.dataset.method = m;
      var c = el("canvas", "train-dist");
      c.dataset.method = m;
      c.setAttribute("role", "img");
      c.setAttribute("aria-label", NAMES[m] + " samples");
      f.appendChild(c);
      f.appendChild(el("figcaption", "panel-label", NAMES[m]));
      row.appendChild(f);
      self.dist[m] = c;
    });
    root.appendChild(row);
    this.metric = el("canvas", "train-metric");
    this.metric.setAttribute("role", "img");
    this.metric.setAttribute("aria-label", "Energy distance against training step, flow matching and hybrid");
    root.appendChild(this.metric);
    this.worker = null; this.epoch = 0; this.running = false; this.done = false; this.paused = false;
    this.clear();
    this.select.addEventListener("change", function () { self.reset(); });
    this.startBtn.addEventListener("click", function () { self.start(); });
    this.pauseBtn.addEventListener("click", function () { self.pause(); });
    this.resetBtn.addEventListener("click", function () { self.reset(); });
    window.addEventListener("resize", function () { self.draw(); });
    this.sync(true);
    this.draw();
  }

  Widget.prototype.clear = function () {
    this.step = 0; this.data = null; this.samples = {}; this.ed = { fm: [], hybrid: [] };
    this.lastRO = 0; this.roTimer = 0; this.failed = false;
  };

  Widget.prototype.buttons = function () {
    this.startBtn.disabled = this.running || this.done;
    this.pauseBtn.disabled = !this.running;
    this.resetBtn.disabled = false;
    this.startBtn.textContent = this.step > 0 && !this.done ? "Resume" : "Start";
  };

  Widget.prototype.start = function () {
    var self = this;
    if (this.running || this.done) return;
    if (this.paused) { this.paused = false; this.running = true; this.buttons(); this.sync(true); this.draw(); return; }
    if (!this.worker) {
      try { this.worker = new Worker(WORKER_URL); } catch (e) { return this.fail(e); }
      var w = this.worker, ep = ++this.epoch;
      w.onmessage = function (e) { if (w === self.worker && ep === self.epoch) self.onMessage(e.data); };
      w.onerror = function (e) { if (e.preventDefault) e.preventDefault(); self.fail(e.message || "worker error"); };
      w.postMessage({ type: "init", dataset: this.select.value, seed: 1, epoch: ep });
    }
    this.running = true;
    this.worker.postMessage({ type: "run", epoch: this.epoch });
    this.buttons();
  };

  // The worker trains in one blocking loop, so it cannot hear a pause; the widget freezes its display instead.
  Widget.prototype.pause = function () {
    if (!this.running) return;
    this.running = false; this.paused = true;
    this.buttons(); this.sync(true);
  };

  Widget.prototype.reset = function () {
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    this.epoch++; this.running = this.done = this.paused = false;
    this.clear(); this.buttons(); this.sync(true); this.draw();
  };

  Widget.prototype.fail = function (msg) {
    if (this.worker) { this.worker.terminate(); this.worker = null; }
    this.running = false; this.failed = true;
    this.buttons();
    this.metricEl.textContent = "trainer error: " + msg + ". Serve the page over http and reload.";
  };

  Widget.prototype.onMessage = function (m) {
    if (!m) return;
    if (m.type === "error") return this.fail(m.message);
    if (m.type === "data") this.data = m.buf;
    else if (m.type === "progress") { this.step = m.step; if (m.done) this.finish(); }
    else if (m.type === "metric") {
      this.step = Math.max(this.step, m.step);
      METHODS.forEach(function (k) { this.ed[k].push([m.step, m.ed[k]]); this.samples[k] = m.samples[k]; }, this);
    } else if (m.type === "state" && m.done) { this.step = Math.max(this.step, m.step); this.finish(); }
    if (!this.paused) { this.sync(m.type === "state"); this.draw(); }
  };

  Widget.prototype.finish = function () { this.done = true; this.running = false; this.buttons(); };

  Widget.prototype.sync = function (now) {
    var self = this, t = Date.now();
    if (!now && t - this.lastRO < READOUT_MS) {
      if (!this.roTimer) this.roTimer = setTimeout(function () { self.roTimer = 0; self.sync(true); }, READOUT_MS);
      return;
    }
    this.lastRO = t;
    this.stepEl.textContent = this.step;
    this.stepEl.title = "training steps, of " + MAX;
    var f = function (k) { var a = self.ed[k]; return a.length ? a[a.length - 1][1].toFixed(4) : "-"; };
    if (!this.failed) this.metricEl.textContent = "energy distance  fm " + f("fm") + "  hybrid " + f("hybrid");
  };

  Widget.prototype.draw = function () {
    if (this.raf) return;
    var self = this;
    this.raf = requestAnimationFrame(function () { self.raf = 0; self.drawDist(); self.drawMetric(); });
  };

  Widget.prototype.drawDist = function () {
    var self = this;
    METHODS.forEach(function (m) {
      var c = fit(self.dist[m]), x = c.x, s = Math.min(c.w, c.h), ox = (c.w - s) / 2, oy = (c.h - s) / 2;
      x.clearRect(0, 0, c.w, c.h);
      var plot = function (arr, r, col) {
        if (!arr) return;
        x.fillStyle = col; x.beginPath();
        for (var i = 0; i < arr.length; i += 2) {
          var px = ox + (arr[i] + B) / (2 * B) * s, py = oy + (B - arr[i + 1]) / (2 * B) * s;
          x.moveTo(px + r, py); x.arc(px, py, r, 0, 6.2832);
        }
        x.fill();
      };
      x.globalAlpha = 0.22; plot(self.data, 1.8, css("--muted", "#888")); x.globalAlpha = 1;
      plot(self.samples[m], 2.6, css("--" + m, "#888"));
    });
  };

  Widget.prototype.drawMetric = function () {
    var c = fit(this.metric), x = c.x, w = c.w, h = c.h, padL = 52, padR = 14, padT = 14, padB = 36;
    var pw = w - padL - padR, ph = h - padT - padB, muted = css("--muted", "#888"), grid = css("--border", "#333");
    var all = this.ed.fm.concat(this.ed.hybrid), lo = 0.005, hi = 1, i;
    for (i = 0; i < all.length; i++) { var v = Math.max(all[i][1], 1e-4); if (v < lo) lo = v; if (v > hi) hi = v; }
    var y0 = Math.log(lo * 0.9), y1 = Math.log(hi * 1.1);
    var X = function (s) { return padL + s / MAX * pw; }, Y = function (v) { return padT + (y1 - Math.log(Math.max(v, 1e-4))) / (y1 - y0) * ph; };
    x.clearRect(0, 0, w, h);
    x.font = "11px " + css("--mono", "monospace"); x.fillStyle = muted; x.strokeStyle = grid; x.lineWidth = 1; x.setLineDash([2, 4]);
    x.textAlign = "right"; x.textBaseline = "middle";
    for (var d = Math.floor(Math.log10(lo * 0.9)); d <= Math.ceil(Math.log10(hi * 1.1)); d++) {
      [1, 3].forEach(function (k) {
        var v = k * Math.pow(10, d);
        if (v < lo * 0.9 || v > hi * 1.1) return;
        x.beginPath(); x.moveTo(padL, Y(v)); x.lineTo(padL + pw, Y(v)); x.stroke();
        x.fillText(String(+v.toPrecision(1)), padL - 6, Y(v));
      });
    }
    x.textAlign = "center"; x.textBaseline = "top";
    for (i = 0; i <= MAX; i += 500) x.fillText(String(i), X(i), padT + ph + 5);
    x.fillText("training step", padL + pw / 2, padT + ph + 20);
    x.save(); x.translate(12, padT + ph / 2); x.rotate(-Math.PI / 2); x.textBaseline = "middle";
    x.fillText("energy distance (log)", 0, 0); x.restore();
    x.setLineDash([]);
    x.beginPath(); x.moveTo(padL, padT + ph + 0.5); x.lineTo(padL + pw, padT + ph + 0.5); x.stroke();
    var self = this;
    METHODS.forEach(function (m, j) {
      var a = self.ed[m], col = css("--" + m, "#888");
      x.strokeStyle = col; x.lineWidth = 2.2; x.lineJoin = "round"; x.shadowColor = col; x.shadowBlur = 10; x.beginPath();
      a.forEach(function (p, k) { if (k) x.lineTo(X(p[0]), Y(p[1])); else x.moveTo(X(p[0]), Y(p[1])); });
      x.stroke(); x.shadowBlur = 0;
      if (a.length) { var q = a[a.length - 1]; x.fillStyle = col; x.beginPath(); x.arc(X(q[0]), Y(q[1]), 3.5, 0, 6.2832); x.fill(); }
      x.fillStyle = col; x.textAlign = "right"; x.textBaseline = "middle";
      x.fillRect(padL + pw - 24, padT + 12 + j * 16 - 1.5, 14, 3);
      x.fillText(NAMES[m], padL + pw - 32, padT + 12 + j * 16);
    });
  };

  function go() {
    var root = document.getElementById("train");
    if (root && !root._w) root._w = new Widget(root);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go); else go();
})();
