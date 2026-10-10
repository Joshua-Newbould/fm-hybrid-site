/* site.js - hero particles and scroll reveal. Decorative only. */
(function () {
  "use strict";
  var doc = document.documentElement;
  doc.classList.add("js");
  var still = window.matchMedia("(prefers-reduced-motion: reduce)");

  function css(n) { return getComputedStyle(doc).getPropertyValue(n).trim(); }
  function gauss() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(6.2832 * Math.random()); }
  function moons() {
    var a = Math.random() * Math.PI, up = Math.random() < 0.5;
    var j = 0.07 * gauss(), k = 0.07 * gauss();
    return up ? [Math.cos(a) - 0.5 + j, Math.sin(a) - 0.25 + k] : [0.5 - Math.cos(a) + j, 0.25 - Math.sin(a) + k];
  }
  function ring() {
    var k = (Math.random() * 8) | 0, a = k * Math.PI / 4;
    return [1.6 * Math.cos(a) + 0.12 * gauss(), 1.6 * Math.sin(a) + 0.12 * gauss()];
  }
  function hexRgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.replace(/./g, "$&$&");
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function ease(u) { return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }

  function hero() {
    var top = document.getElementById("top"), cv = top && top.querySelector(".hero-canvas");
    if (!cv) return;
    var g = cv.getContext("2d"), W = 0, H = 0, dpr = 1, N = 0, P = [], shape = 0, t0 = 0, raf = 0, on = true;
    var mx = -1e4, my = -1e4, cvx = 0, cvy = 0, lastMove = 0, lastNow = 0, snap = false, cols = [];
    var HOLD0 = 900, FLOW = 3200, HOLD1 = 2200, CYCLE = HOLD0 + FLOW + HOLD1;

    function seed() {
      var f = shape % 2 ? ring : moons;
      for (var i = 0; i < N; i++) {
        var p = P[i] || (P[i] = {});
        var z = [gauss(), gauss()], d = f();
        p.ax = z[0]; p.ay = z[1]; p.bx = d[0]; p.by = d[1];
        p.lag = Math.random() * 0.18;
        p.c = (d[0] + 2) / 4;
      }
    }
    function palette() {
      var a = hexRgb(css("--fm") || "#5ec8ff"), b = hexRgb(css("--hybrid") || "#ff8a4c");
      cols = [];
      for (var k = 0; k < 16; k++) {
        var u = k / 15;
        cols.push("rgb(" + [0, 1, 2].map(function (j) { return Math.round(a[j] + (b[j] - a[j]) * u); }).join(",") + ")");
      }
    }
    function size() {
      var r = cv.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = r.width; H = r.height;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.round(Math.min(2200, W * H / 520));
      if (n !== N) { N = n; P.length = Math.min(P.length, N); seed(); }
      palette();
    }
    function frame(now, fade) {
      if (!t0) t0 = now;
      var e = now - t0;
      if (e > CYCLE) { t0 = now; shape++; seed(); e = 0; }
      var u = Math.min(Math.max((e - HOLD0) / FLOW, 0), 1);
      var wide = W > 820, cx = wide ? W * 0.7 : W * 0.5, cy = wide ? H * 0.5 : H * 0.36;
      var sc = Math.min(wide ? W * 0.13 : W * 0.2, H * 0.17);
      g.globalCompositeOperation = "destination-out";
      g.fillStyle = "rgba(0,0,0," + fade + ")";
      g.fillRect(0, 0, W, H);
      g.globalCompositeOperation = "source-over";
      // Each dot carries a velocity. The flow pulls it towards where the schedule
      // says it should be (a spring), the pointer knocks it away like a small ball
      // with a push that grows with pointer speed, and damping brings it back.
      var dt = lastNow ? Math.min((now - lastNow) / 1000, 0.05) : 1 / 60;
      lastNow = now;
      var damp = Math.exp(-4.5 * dt), R = 80, speed = Math.sqrt(cvx * cvx + cvy * cvy);
      cvx *= 0.82; cvy *= 0.82;
      for (var i = 0; i < N; i++) {
        var p = P[i], s = ease(Math.min(Math.max((u - p.lag) / (1 - 0.18), 0), 1));
        var tx = cx + ((1 - s) * p.ax + s * p.bx) * sc, ty = cy - ((1 - s) * p.ay + s * p.by) * sc;
        if (p.x === undefined || snap) { p.x = tx; p.y = ty; p.vx = 0; p.vy = 0; }
        p.vx += (tx - p.x) * 70 * dt; p.vy += (ty - p.y) * 70 * dt;
        var dx = p.x - mx, dy = p.y - my, d2 = dx * dx + dy * dy;
        if (d2 < R * R) {
          var d = Math.sqrt(d2) + 1e-3, k = (1 - d / R) * (700 + 2.5 * speed) * dt;
          p.vx += (dx / d) * k + cvx * 0.35 * (1 - d / R); p.vy += (dy / d) * k + cvy * 0.35 * (1 - d / R);
        }
        p.vx *= damp; p.vy *= damp;
        var v2 = p.vx * p.vx + p.vy * p.vy;
        if (v2 > 2250000) { var q = 1500 / Math.sqrt(v2); p.vx *= q; p.vy *= q; }
        p.x += p.vx * dt; p.y += p.vy * dt;
        g.fillStyle = cols[Math.max(0, Math.min(15, (p.c * 15) | 0))];
        g.fillRect(p.x - 0.9, p.y - 0.9, 1.8, 1.8);
      }
      snap = false;
    }
    function loop(now) { frame(now, 0.16); raf = on ? requestAnimationFrame(loop) : 0; }
    function start() {
      if (raf || still.matches || !on || document.hidden) return;
      raf = requestAnimationFrame(loop);
    }
    function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; }
    function staticFrame() { stop(); t0 = 1; snap = true; g.clearRect(0, 0, W, H); frame(1 + HOLD0 + FLOW, 1); }

    size();
    if (still.matches) staticFrame(); else start();
    window.addEventListener("resize", function () { size(); if (still.matches) staticFrame(); });
    top.addEventListener("pointermove", function (ev) {
      var r = cv.getBoundingClientRect(), nx = ev.clientX - r.left, ny = ev.clientY - r.top, t = ev.timeStamp || performance.now();
      var h = Math.min(Math.max((t - lastMove) / 1000, 0.004), 0.1);
      if (mx > -1e3) { cvx = 0.5 * cvx + 0.5 * (nx - mx) / h; cvy = 0.5 * cvy + 0.5 * (ny - my) / h; }
      mx = nx; my = ny; lastMove = t;
    });
    top.addEventListener("pointerleave", function () { mx = my = -1e4; cvx = cvy = 0; });
    document.addEventListener("visibilitychange", function () { if (document.hidden) stop(); else start(); });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) { on = es[0].isIntersecting; if (on) start(); else stop(); }).observe(top);
    }
    still.addEventListener && still.addEventListener("change", function () { if (still.matches) staticFrame(); else start(); });
    var mq = window.matchMedia("(prefers-color-scheme: light)");
    mq.addEventListener && mq.addEventListener("change", palette);
  }

  function reveal() {
    var els = document.querySelectorAll("main > section, footer");
    if (still.matches || !("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });
    els.forEach(function (el) { el.classList.add("reveal"); io.observe(el); });
  }

  function navSpy() {
    var links = document.querySelectorAll("#top nav a");
    if (!("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle("here", a.getAttribute("href") === "#" + e.target.id); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main > section[id]").forEach(function (s) { io.observe(s); });
  }

  function go() { hero(); reveal(); navSpy(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go); else go();
})();
