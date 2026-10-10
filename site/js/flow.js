/* flow.js - prior-to-data animation player for flow-matching section.
 * Builds DOM inside section#flow and manages side-by-side baseline (fm) and
 * hybrid trajectory players with shared controls.
 * Classic script: no modules, no external requests, no window assumptions.
 */
(function () {
  "use strict";

  var HAS_WINDOW = typeof window !== "undefined";
  var HAS_DOCUMENT = typeof document !== "undefined";

  // Colour rules copied from playground.js: hue by final position, blended to START_HSL at t=0.
  var SATURATION = 70, LIGHT_LO = 45, LIGHT_HI = 70;
  var MOONS_HUES = [160, 25];
  var MOONS_CENTRES = [[-0.8, -0.4], [0.8, 0.4]];
  var EIGHT_RADIUS = 2.2, EIGHT_SPREAD = 0.6;
  var DEFAULT_BOUNDS = [-3.5, 3.5, -3.5, 3.5];
  var START_HSL = [215, 70, 58];

  function colourFor(label, x, y, dataset, bounds) {
    x = Number(x);
    y = Number(y);
    var lab = (label === null || label === undefined) ? null : Math.trunc(Number(label));
    var dx, dy;
    if (dataset === "moons" && (lab === 0 || lab === 1)) {
      dx = x - MOONS_CENTRES[lab][0];
      dy = y - MOONS_CENTRES[lab][1];
      if (lab === 1) { dx = -dx; dy = -dy; }
      var ang = Math.atan2(dy, dx);
      if (ang < -Math.PI / 2) ang += 2 * Math.PI;
      var f = Math.min(1, Math.max(0, ang / Math.PI));
      return [MOONS_HUES[lab], SATURATION, LIGHT_LO + (LIGHT_HI - LIGHT_LO) * f];
    }
    if (dataset === "gaussians8" && lab !== null && lab >= 0 && lab < 8) {
      var a = 2 * Math.PI * lab / 8;
      var d = Math.hypot(x - EIGHT_RADIUS * Math.cos(a), y - EIGHT_RADIUS * Math.sin(a));
      var g = Math.min(1, d / EIGHT_SPREAD);
      return [lab * 45, SATURATION, LIGHT_LO + (LIGHT_HI - LIGHT_LO) * g];
    }
    var b = bounds || DEFAULT_BOUNDS;
    var cx = (b[0] + b[1]) / 2, cy = (b[2] + b[3]) / 2;
    var rmax = Math.min(b[1] - b[0], b[3] - b[2]) / 2;
    dx = x - cx;
    dy = y - cy;
    var hue = ((Math.atan2(dy, dx) * 180 / Math.PI) + 360) % 360;
    var r = Math.min(1, Math.hypot(dx, dy) / rmax);
    return [hue, 72, 62 - 22 * r];
  }

  function hslToRgb(h, s, l) {
    h = (((h / 360) % 1) + 1) % 1;
    s = s / 100;
    l = l / 100;
    if (s === 0) return [l, l, l];
    var m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s;
    var m1 = 2 * l - m2;
    function v(hue) {
      hue = ((hue % 1) + 1) % 1;
      if (hue < 1 / 6) return m1 + (m2 - m1) * hue * 6;
      if (hue < 0.5) return m2;
      if (hue < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - hue) * 6;
      return m1;
    }
    return [v(h + 1 / 3), v(h), v(h - 1 / 3)];
  }

  function toLinear(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function toSrgb(c) {
    c = c < 0 ? 0 : c > 1 ? 1 : c;
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  }

  function rgbToOklab(rgb) {
    var r = toLinear(rgb[0]), g = toLinear(rgb[1]), b = toLinear(rgb[2]);
    var l_ = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    var m_ = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    var s_ = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [
      0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
      1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
      0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    ];
  }

  function oklabToRgb(lab) {
    var L = lab[0], a = lab[1], b = lab[2];
    var l_ = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
    var m_ = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
    var s_ = Math.pow(L - 0.0894841775 * a - 1.2914855480 * b, 3);
    return [
      toSrgb(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_),
      toSrgb(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_),
      toSrgb(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
    ];
  }

  var START_RGB = hslToRgb(START_HSL[0], START_HSL[1], START_HSL[2]);
  var START_LAB = rgbToOklab(START_RGB);

  function blendRgb(finalRgb, t, finalLab) {
    t = Number(t);
    t = !(t > 0) ? 0 : t > 1 ? 1 : t;
    if (t <= 0) return START_RGB.slice();
    if (t >= 1) return finalRgb.slice();
    var f = finalLab || rgbToOklab(finalRgb);
    return oklabToRgb([
      START_LAB[0] + t * (f[0] - START_LAB[0]),
      START_LAB[1] + t * (f[1] - START_LAB[1]),
      START_LAB[2] + t * (f[2] - START_LAB[2])
    ]);
  }

  function cssRgb(c) {
    return "rgb(" + Math.round(255 * c[0]) + ", " + Math.round(255 * c[1]) + ", " + Math.round(255 * c[2]) + ")";
  }

  function prefersReducedMotion() {
    try {
      return !!(HAS_WINDOW && window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    } catch (e) {
      return false;
    }
  }

  function clamp01(t) {
    t = Number(t);
    if (!isFinite(t)) return 0;
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  function fetchOk(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status + " for " + url);
      return r;
    });
  }

  function FlowCanvas(canvasEl, manifest, data, dataset, method, pointRadius) {
    if (!(this instanceof FlowCanvas)) return new FlowCanvas(canvasEl, manifest, data, dataset, method, pointRadius);
    this.canvas = canvasEl;
    this.ctx = canvasEl && canvasEl.getContext ? canvasEl.getContext("2d") : null;
    this.manifest = manifest;
    this.data = data;
    this.dataset = dataset;
    this.method = method;
    this.pointRadius = pointRadius || 2;
    this._t = 0;
    this._dpr = 1;
    this._cssW = 300;
    this._cssH = 300;
    this._colors = null;
    this._finalRgb = null;
    this._finalLab = null;
    this._computeColors();
  }

  FlowCanvas.prototype._computeColors = function () {
    if (!this.manifest || !this.data) return;
    var nPoints = this.manifest.n;
    var nSteps = this.manifest.frames;
    var dataset = this.dataset;
    var bounds = this.manifest.bounds;
    var boundsExpanded = bounds.length === 2 ? [bounds[0], bounds[1], bounds[0], bounds[1]] : bounds;
    var labels = this.manifest.labels && this.manifest.labels[dataset] ? this.manifest.labels[dataset] : null;
    var base = (nSteps - 1) * nPoints * 2;
    var d = this.data;
    var finalRgb = new Array(nPoints);
    var finalLab = new Array(nPoints);
    for (var j = 0; j < nPoints; j++) {
      var c = colourFor(labels ? labels[j] : null, d[base + 2 * j], d[base + 2 * j + 1], dataset, boundsExpanded);
      finalRgb[j] = hslToRgb(c[0], c[1], c[2]);
      finalLab[j] = rgbToOklab(finalRgb[j]);
    }
    this._finalRgb = finalRgb;
    this._finalLab = finalLab;
  };

  FlowCanvas.prototype._colorsAt = function (t) {
    if (!this._finalRgb) return [];
    var n = this._finalRgb.length;
    var out = new Array(n);
    for (var j = 0; j < n; j++) {
      out[j] = cssRgb(blendRgb(this._finalRgb[j], t, this._finalLab[j]));
    }
    return out;
  };

  FlowCanvas.prototype._frameBelow = function (t) {
    var nSteps = this.manifest.frames;
    var lo = 0, hi = nSteps - 1;
    if (t <= 0) return 0;
    if (t >= 1) return hi;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (mid / (nSteps - 1) <= t) lo = mid; else hi = mid;
    }
    return lo;
  };

  FlowCanvas.prototype._resize = function () {
    var c = this.canvas;
    var dpr = (HAS_WINDOW && window.devicePixelRatio) || 1;
    var rect = c.getBoundingClientRect && c.getBoundingClientRect();
    var w = (rect && rect.width) || c.clientWidth || 300;
    var h = (rect && rect.height) || c.clientHeight || 400;
    if (w <= 0) w = 300;
    if (h <= 0) h = 400;
    var bw = Math.max(1, Math.round(w * dpr));
    var bh = Math.max(1, Math.round(h * dpr));
    if (c.width !== bw || c.height !== bh) {
      c.width = bw;
      c.height = bh;
    }
    this._cssW = w;
    this._cssH = h;
    this._dpr = dpr;
  };

  FlowCanvas.prototype.setT = function (t) {
    this._t = clamp01(t);
    this._draw();
  };

  FlowCanvas.prototype._draw = function () {
    if (!this.data || !this.ctx) return;
    var ctx = this.ctx;
    var c = this.canvas;
    var dpr = this._dpr || 1;
    var w = this._cssW, h = this._cssH;
    var bounds = this.manifest.bounds;
    var b = bounds.length === 2 ? [bounds[0], bounds[1], bounds[0], bounds[1]] : bounds;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var sx = w / (b[1] - b[0]);
    var sy = h / (b[3] - b[2]);
    var s = Math.min(sx, sy);
    var ox = (w - s * (b[1] - b[0])) / 2;
    var oy = (h - s * (b[3] - b[2])) / 2;

    var t = this._t;
    var nSteps = this.manifest.frames;
    var i0 = this._frameBelow(t);
    var i1 = Math.min(i0 + 1, nSteps - 1);
    var t0 = i0 / (nSteps - 1), t1 = i1 / (nSteps - 1);
    var a = (i1 === i0 || t1 <= t0) ? 0 : (t - t0) / (t1 - t0);
    if (a < 0) a = 0; else if (a > 1) a = 1;

    var nPoints = this.manifest.n;
    var d = this.data;
    var base0 = i0 * nPoints * 2, base1 = i1 * nPoints * 2;
    var r = this.pointRadius;
    var colors = this._colorsAt(t);
    var TAU = Math.PI * 2;

    for (var j = 0; j < nPoints; j++) {
      var k = 2 * j;
      var x0 = d[base0 + k], y0 = d[base0 + k + 1];
      var x = x0 + a * (d[base1 + k] - x0);
      var y = y0 + a * (d[base1 + k + 1] - y0);
      var px = ox + (x - b[0]) * s;
      var py = oy + (b[3] - y) * s;
      ctx.fillStyle = colors[j];
      ctx.beginPath();
      ctx.arc(px, py, r, 0, TAU);
      ctx.fill();
    }
  };

  function FlowPlayer(sectionEl) {
    if (!(this instanceof FlowPlayer)) return new FlowPlayer(sectionEl);
    this.section = sectionEl;
    this.manifest = null;
    this.dataBins = {};
    this.players = {};
    this.canvasEls = {};
    this._t = 0;
    this._playing = false;
    this._raf = 0;
    this._lastTick = 0;
    this._destroyed = false;
    this._listeners = [];
    this._resizeObserver = null;
    this._datasetSelect = null;
    this._sliderEl = null;
    this._readoutEl = null;
    this._playButtonEl = null;

    this._buildDom();
    this._bindControls();
    this.ready = this._load();
  }

  FlowPlayer.prototype._buildDom = function () {
    var self = this;
    var box = document.createElement("div");
    box.className = "widget";

    var controls = document.createElement("div");
    controls.className = "controls";

    var select = document.createElement("select");
    select.className = "flow-dataset";
    var opt1 = document.createElement("option");
    opt1.value = "moons";
    opt1.textContent = "Moons";
    select.appendChild(opt1);
    var opt2 = document.createElement("option");
    opt2.value = "gaussians8";
    opt2.textContent = "Gaussians 8";
    select.appendChild(opt2);
    this._datasetSelect = select;
    controls.appendChild(select);

    var slider = document.createElement("input");
    slider.type = "range";
    slider.className = "t-slider range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.001";
    slider.value = "0";
    slider.setAttribute("aria-label", "Time t");
    this._sliderEl = slider;
    controls.appendChild(slider);

    var readout = document.createElement("span");
    readout.className = "t-readout readout";
    readout.setAttribute("aria-live", "off");
    readout.textContent = "t = 0.00";
    this._readoutEl = readout;
    controls.appendChild(readout);

    var playBtn = document.createElement("button");
    playBtn.className = "play btn";
    playBtn.setAttribute("aria-pressed", "false");
    playBtn.textContent = "Play";
    this._playButtonEl = playBtn;
    controls.appendChild(playBtn);

    var canvasRow = document.createElement("div");
    canvasRow.className = "duo";

    for (var i = 0; i < 2; i++) {
      var method = i === 0 ? "fm" : "hybrid";
      var wrapper = document.createElement("figure");
      wrapper.className = "panel";
      wrapper.setAttribute("data-method", method);

      var label = document.createElement("figcaption");
      label.className = "panel-label";
      label.textContent = method === "fm" ? "Baseline flow matching" : "Hybrid, quantised head";

      var canvas = document.createElement("canvas");
      canvas.className = "flow-canvas";
      canvas.setAttribute("data-method", method);
      wrapper.appendChild(canvas);
      wrapper.appendChild(label);

      canvasRow.appendChild(wrapper);
      this.canvasEls[method] = canvas;
    }

    box.appendChild(canvasRow);
    box.appendChild(controls);
    this.section.appendChild(box);
  };

  FlowPlayer.prototype._bindControls = function () {
    var self = this;
    if (this._datasetSelect) {
      this._on(this._datasetSelect, "change", function () {
        self._onDatasetChange();
      });
    }
    if (this._sliderEl) {
      var onInput = function () {
        if (self._playing) self.pause();
        self.setT(clamp01(self._sliderEl.value));
      };
      this._on(this._sliderEl, "input", onInput);
      this._on(this._sliderEl, "change", onInput);
      this._on(this._sliderEl, "pointerdown", function () { if (self._playing) self.pause(); });
    }
    if (this._playButtonEl) {
      this._on(this._playButtonEl, "click", function () { self.toggle(); });
    }
  };

  FlowPlayer.prototype._on = function (el, type, fn, options) {
    if (!el) return;
    el.addEventListener(type, fn, options);
    this._listeners.push([el, type, fn, options]);
  };

  FlowPlayer.prototype._load = function () {
    var self = this;
    return fetchOk("data/manifest.json").then(function (r) { return r.json(); }).then(function (manifest) {
      self.manifest = manifest;
      var promises = [];
      var datasets = manifest.datasets || [];
      var methods = manifest.methods || [];
      for (var i = 0; i < datasets.length; i++) {
        for (var j = 0; j < methods.length; j++) {
          (function (ds, method) {
            var binFile = manifest.traj && manifest.traj[ds] && manifest.traj[ds][method];
            if (binFile) {
              promises.push(
                fetchOk("data/" + binFile).then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
                  self.dataBins[ds + "_" + method] = new Float32Array(buf);
                })
              );
            }
          })(datasets[i], methods[j]);
        }
      }
      return Promise.all(promises);
    }).then(function () {
      self._initCanvases();
      self._observeResize();
      var methods = ["fm", "hybrid"];
      for (var i = 0; i < methods.length; i++) {
        self._resizeCanvas(methods[i]);
      }
      self.setT(0);
      return self;
    });
  };

  FlowPlayer.prototype._initCanvases = function () {
    var dataset = this._datasetSelect ? this._datasetSelect.value : "moons";
    var methods = ["fm", "hybrid"];
    for (var i = 0; i < methods.length; i++) {
      var method = methods[i];
      var canvas = this.canvasEls[method];
      if (!canvas) continue;
      var data = this.dataBins[dataset + "_" + method];
      if (!data) continue;
      this.players[method] = new FlowCanvas(canvas, this.manifest, data, dataset, method, 2.5);
    }
  };

  FlowPlayer.prototype._resizeCanvas = function (method) {
    var player = this.players[method];
    if (player) player._resize();
  };

  FlowPlayer.prototype._onDatasetChange = function () {
    this._initCanvases();
    this.setT(this._t);
  };

  FlowPlayer.prototype._observeResize = function () {
    var self = this;
    var methods = ["fm", "hybrid"];
    var onResize = function () {
      for (var i = 0; i < methods.length; i++) {
        self._resizeCanvas(methods[i]);
      }
      self._draw();
    };
    if (typeof ResizeObserver === "function") {
      this._resizeObserver = new ResizeObserver(onResize);
      for (var i = 0; i < methods.length; i++) {
        var canvas = this.canvasEls[methods[i]];
        if (canvas) {
          this._resizeObserver.observe(canvas);
        }
      }
    } else {
      this._on(window, "resize", onResize);
    }
  };

  FlowPlayer.prototype.setT = function (t) {
    this._t = clamp01(t);
    this._draw();
    this._syncUi();
  };

  FlowPlayer.prototype._draw = function () {
    var methods = ["fm", "hybrid"];
    for (var i = 0; i < methods.length; i++) {
      var player = this.players[methods[i]];
      if (player) player.setT(this._t);
    }
  };

  FlowPlayer.prototype._syncUi = function () {
    if (this._sliderEl) {
      this._sliderEl.value = String(this._t);
      this._sliderEl.style.setProperty("--p", (this._t * 100).toFixed(2) + "%");
    }
    if (this._readoutEl) this._readoutEl.textContent = "t = " + this._t.toFixed(2);
    if (this._playButtonEl) {
      this._playButtonEl.setAttribute("aria-pressed", this._playing ? "true" : "false");
      this._playButtonEl.textContent = this._playing ? "Pause" : "Play";
    }
  };

  FlowPlayer.prototype._tick = function (now) {
    if (!this._playing || this._destroyed) return;
    var self = this;
    var interval = 1000 / 24;
    if (!this._lastTick) this._lastTick = now;
    if (now - this._lastTick >= interval) {
      this._lastTick = now - ((now - this._lastTick) % interval);
      var newT = this._t + (1 / 6) / 24;
      if (newT >= 1) {
        if (prefersReducedMotion()) {
          this.pause();
          return;
        }
        newT = 0;
      }
      this.setT(newT);
    }
    this._raf = requestAnimationFrame(function (ts) { self._tick(ts); });
  };

  FlowPlayer.prototype.play = function () {
    if (this._destroyed || this._playing || !this.manifest) return;
    var self = this;
    if (this._t >= 1) this._t = 0;
    this._playing = true;
    this._lastTick = 0;
    this._syncUi();
    this._raf = requestAnimationFrame(function (ts) { self._tick(ts); });
  };

  FlowPlayer.prototype.pause = function () {
    this._playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
    this._syncUi();
  };

  FlowPlayer.prototype.toggle = function () {
    if (this._playing) this.pause(); else this.play();
  };

  FlowPlayer.prototype.destroy = function () {
    this._destroyed = true;
    this.pause();
    for (var i = 0; i < this._listeners.length; i++) {
      var l = this._listeners[i];
      l[0].removeEventListener(l[1], l[2], l[3]);
    }
    this._listeners = [];
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
  };

  function autoInit() {
    if (!HAS_DOCUMENT) return;
    var section = document.getElementById("flow");
    if (!section) return;
    try {
      var player = new FlowPlayer(section);
      section.__flowPlayer = player;
      player.ready.catch(function (err) {
        console.warn("FlowPlayer failed to load:", err && err.message ? err.message : err);
      });
    } catch (e) {
      console.warn("FlowPlayer init failed:", e);
    }
  }

  if (HAS_WINDOW) window.FlowPlayer = FlowPlayer;

  if (HAS_DOCUMENT) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", autoInit);
    } else {
      autoInit();
    }
  }
})();
