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
    var HOLD0 = 2250, FLOW = 6400, HOLD1 = 5500, CYCLE = HOLD0 + FLOW + HOLD1;

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
      // Each dot is a small ball. The pointer is a cue ball: a dot it touches takes the
      // pointer's velocity and bounces off it, then rolls, bounces off the walls and slows
      // under friction. The flow pulls a dot back to its place only once it is slow, so a
      // struck dot travels first and is gathered back as its speed dies away.
      var dt = lastNow ? Math.min((now - lastNow) / 1000, 0.05) : 1 / 60;
      lastNow = now;
      var RC = 30, REST = 0.8, tooFast = 1500;
      cvx *= 0.82; cvy *= 0.82;
      for (var i = 0; i < N; i++) {
        var p = P[i], s = ease(Math.min(Math.max((u - p.lag) / (1 - 0.18), 0), 1));
        var tx = cx + ((1 - s) * p.ax + s * p.bx) * sc, ty = cy - ((1 - s) * p.ay + s * p.by) * sc;
        if (p.x === undefined || snap) { p.x = tx; p.y = ty; p.vx = 0; p.vy = 0; }
        var sp = Math.sqrt(p.vx * p.vx + p.vy * p.vy), w = 1 / (1 + (sp / 110) * (sp / 110));
        p.vx += (tx - p.x) * 70 * w * dt; p.vy += (ty - p.y) * 70 * w * dt;
        var dmp = Math.exp(-(0.7 + 5 * w) * dt);
        p.vx *= dmp; p.vy *= dmp;
        var dx = p.x - mx, dy = p.y - my, d2 = dx * dx + dy * dy;
        if (d2 < RC * RC) {
          var d = Math.sqrt(d2) + 1e-6, nx = dx / d, ny = dy / d;
          p.x = mx + nx * RC; p.y = my + ny * RC;
          if (!p.hit) {   // one strike per contact: a random direction (away from the cue) and a random speed
            var ang = Math.random() * 6.2832, spd = 220 + Math.random() * 700 + 0.4 * Math.sqrt(cvx * cvx + cvy * cvy);
            p.vx = Math.cos(ang) * spd; p.vy = Math.sin(ang) * spd;
            if (p.vx * nx + p.vy * ny < 0) { p.vx = -p.vx; p.vy = -p.vy; }
            p.hit = true;
          }
        } else p.hit = false;
        var v2 = p.vx * p.vx + p.vy * p.vy;
        if (v2 > tooFast * tooFast) { var q = tooFast / Math.sqrt(v2); p.vx *= q; p.vy *= q; }
        p.x += p.vx * dt; p.y += p.vy * dt;
        if (p.x < 1) { p.x = 1; p.vx = -p.vx * REST; } else if (p.x > W - 1) { p.x = W - 1; p.vx = -p.vx * REST; }
        if (p.y < 1) { p.y = 1; p.vy = -p.vy * REST; } else if (p.y > H - 1) { p.y = H - 1; p.vy = -p.vy * REST; }
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
    window.addEventListener("themechange", palette);
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
    var links = document.querySelectorAll(".site-nav a");
    if (!("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle("here", a.getAttribute("href") === "#" + e.target.id); });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main > section[id]").forEach(function (s) { io.observe(s); });
  }

  function theme() {
    var root = document.documentElement, btn = document.querySelector(".theme-toggle");
    var saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) {}
    if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
    if (!btn) return;
    btn.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") ? root.getAttribute("data-theme") === "dark" : !window.matchMedia("(prefers-color-scheme: light)").matches;
      var next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      window.dispatchEvent(new Event("themechange"));
      window.dispatchEvent(new Event("resize"));   // canvases read their colours on redraw
    });
  }
  function go() { theme(); hero(); reveal(); navSpy(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go); else go();
})();
