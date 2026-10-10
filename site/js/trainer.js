/* trainer.js - flow matching and hybrid training in the browser.
 * UMD guard: require("./trainer.js") under node, self.FMTrainer in workers.
 */
(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  } else if (root) {
    root.FMTrainer = api;
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var LIMIT = 3.0;

  /* ---------- seeded random numbers ---------- */

  function Rng(seed) {
    this.s = (Number(seed) >>> 0) || 0x9e3779b9;
    this._spare = null;
  }
  Rng.prototype.random = function () {
    var t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  Rng.prototype.normal = function () {
    if (this._spare !== null) { var s = this._spare; this._spare = null; return s; }
    var u = 0, v = 0;
    while (u <= 1e-12) u = this.random();
    v = this.random();
    var r = Math.sqrt(-2 * Math.log(u));
    this._spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  Rng.prototype.int = function (n) { return Math.floor(this.random() * n); };

  function clip(v) { return v < -LIMIT ? -LIMIT : v > LIMIT ? LIMIT : v; }

  /* ---------- Gray code ---------- */

  function gray(i) { return i ^ (i >> 1); }

  function ungray(g, bits) {
    var x = g;
    for (var k = 1; k < bits; k++) x = x ^ (g >> k);
    return x;
  }

  /* ---------- datasets (match fm.py data()) ---------- */

  function twoMoons(n, seed, noise) {
    noise = noise === undefined ? 0.08 : noise;
    var rng = new Rng(seed === undefined ? 0 : seed);
    var nUp = Math.floor(n / 2);
    var pts = new Float32Array(n * 2);
    var i;
    for (i = 0; i < n; i++) {
      var a = Math.PI * rng.random(), x, y;
      if (i < nUp) { x = Math.cos(a); y = Math.sin(a); }
      else { x = 1 - Math.cos(a); y = 0.5 - Math.sin(a); }
      pts[2 * i] = clip((x - 0.5) * 2.4 + noise * rng.normal());
      pts[2 * i + 1] = clip((y - 0.25) * 2.4 + noise * rng.normal());
    }
    for (i = n - 1; i > 0; i--) {
      var j = rng.int(i + 1);
      var tx = pts[2 * i], ty = pts[2 * i + 1];
      pts[2 * i] = pts[2 * j]; pts[2 * i + 1] = pts[2 * j + 1];
      pts[2 * j] = tx; pts[2 * j + 1] = ty;
    }
    return pts;
  }

  function eightGaussians(n, seed, radius, std) {
    radius = radius === undefined ? 2 : radius;
    std = std === undefined ? 0.2 : std;
    var rng = new Rng(seed === undefined ? 0 : seed);
    var pts = new Float32Array(n * 2);
    for (var i = 0; i < n; i++) {
      var k = rng.int(8), ang = 2 * Math.PI * k / 8;
      pts[2 * i] = clip(radius * Math.cos(ang) + std * rng.normal());
      pts[2 * i + 1] = clip(radius * Math.sin(ang) + std * rng.normal());
    }
    return pts;
  }

  var DATASETS = { moons: twoMoons, eightgauss: eightGaussians };
  function getDataset(name, n, seed) {
    var key = name === "twoMoons" ? "moons" : name === "eightGaussians" ? "eightgauss" : name;
    var fn = DATASETS[key];
    if (!fn) throw new Error("unknown dataset " + name + "; choose moons or eightgauss");
    return fn(n, seed);
  }

  /* ---------- Utility functions ---------- */

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function softmax(logits) {
    var max = Math.max.apply(null, Array.from(logits));
    var exp = new Float64Array(logits.length);
    var sum = 0;
    for (var i = 0; i < logits.length; i++) {
      exp[i] = Math.exp(logits[i] - max);
      sum += exp[i];
    }
    for (var i = 0; i < exp.length; i++) exp[i] /= sum;
    return exp;
  }

  function centre(i, bits, R) {
    return -R + (i + 0.5) * (2 * R / (1 << bits));
  }

  function bucketize(v, bits, R) {
    var B = 1 << bits;
    return Math.max(0, Math.min(B - 1, Math.floor((v + R) / (2 * R) * B)));
  }

  /* ---------- MLP 3 -> H -> H -> out with SiLU ---------- */

  function makeParams(outDim, H, rng) {
    function lin(out, inp) {
      var b = 1 / Math.sqrt(inp), W = new Float64Array(out * inp), c = new Float64Array(out), i;
      for (i = 0; i < W.length; i++) W[i] = (2 * rng.random() - 1) * b;
      for (i = 0; i < out; i++) c[i] = (2 * rng.random() - 1) * b;
      return [W, c];
    }
    var l1 = lin(H, 3), l2 = lin(H, H), l3 = lin(outDim, H);
    return [l1[0], l1[1], l2[0], l2[1], l3[0], l3[1]];
  }

  function FMTrainer(method, datasetName, opts) {
    if (!(this instanceof FMTrainer)) return new FMTrainer(method, datasetName, opts);
    opts = opts || {};
    this.method = method === "hybrid" ? "hybrid" : "fm";
    this.datasetName = datasetName || "moons";
    this.seed = opts.seed === undefined ? 0 : Number(opts.seed);
    this.lr = opts.lr > 0 ? Number(opts.lr) : 2e-3;
    this.batch = opts.batch > 0 ? Math.floor(opts.batch) : 128;
    this.hidden = opts.hidden > 0 ? Math.floor(opts.hidden) : 64;
    this.pool = opts.pool > 0 ? Math.floor(opts.pool) : 20000;

    // Hybrid config
    this.bits = 4;
    this.R = 6;
    this.outDim = this.method === "hybrid" ? 2 * (1 << this.bits) + 2 : 2;

    this.data = getDataset(this.datasetName, this.pool, this.seed);
    this.reset();
  }

  FMTrainer.prototype.reset = function () {
    var H = this.hidden, B = this.batch;
    this.rng = new Rng((this.seed * 2654435761 + 12345) >>> 0);
    this.params = makeParams(this.outDim, H, this.rng);
    this.m = this.params.map(function (p) { return new Float64Array(p.length); });
    this.v = this.params.map(function (p) { return new Float64Array(p.length); });
    this.grads = this.params.map(function (p) { return new Float64Array(p.length); });
    this.steps = 0;
    this.losses = [];
    // scratch buffers
    this._X = new Float64Array(B * 3);
    this._T = new Float64Array(B * 2);
    this._z1 = new Float64Array(B * H); this._h1 = new Float64Array(B * H);
    this._z2 = new Float64Array(B * H); this._h2 = new Float64Array(B * H);
    this._out = new Float64Array(B * this.outDim);
    this._d1 = new Float64Array(B * H); this._d2 = new Float64Array(B * H);
    this._dout = new Float64Array(B * this.outDim);
  };

  FMTrainer.prototype._forward = function (X, n) {
    var H = this.hidden, outDim = this.outDim;
    var p = this.params;
    var W1 = p[0], b1 = p[1], W2 = p[2], b2 = p[3], W3 = p[4], b3 = p[5];
    var z1 = this._z1, h1 = this._h1, z2 = this._z2, h2 = this._h2, out = this._out;

    for (var r = 0; r < n; r++) {
      var x0 = X[3 * r], x1 = X[3 * r + 1], x2 = X[3 * r + 2], o = r * H, j, k, s, z, sg;
      for (j = 0; j < H; j++) {
        z = b1[j] + W1[3 * j] * x0 + W1[3 * j + 1] * x1 + W1[3 * j + 2] * x2;
        z1[o + j] = z; h1[o + j] = z * sigmoid(z);
      }
      for (j = 0; j < H; j++) {
        s = b2[j]; var w = j * H;
        for (k = 0; k < H; k++) s += W2[w + k] * h1[o + k];
        z2[o + j] = s; sg = sigmoid(s); h2[o + j] = s * sg;
      }
      for (k = 0; k < outDim; k++) {
        s = b3[k];
        for (var l = 0; l < H; l++) s += W3[k * H + l] * h2[o + l];
        out[r * outDim + k] = s;
      }
    }
  };

  FMTrainer.prototype.setParams = function (params) {
    for (var i = 0; i < this.params.length; i++) {
      var src = params[["W1", "b1", "W2", "b2", "W3", "b3"][i]];
      if (!src) throw new Error("missing param " + ["W1", "b1", "W2", "b2", "W3", "b3"][i]);
      var flat = src.flat ? src.flat() : (Array.isArray(src[0]) ? src.flat() : src);
      for (var j = 0; j < this.params[i].length; j++) {
        this.params[i][j] = flat[j];
      }
    }
  };

  FMTrainer.prototype.lossGrad = function (xy, t, v, n, offTarget) {
    // Construct X array from xy (n*2) and t (n)
    var X = new Float64Array(n * 3);
    for (var i = 0; i < n; i++) {
      X[3 * i] = xy[2 * i];
      X[3 * i + 1] = xy[2 * i + 1];
      X[3 * i + 2] = t[i];
    }

    this._forward(X, n);
    var H = this.hidden, outDim = this.outDim, bits = this.bits, R = this.R;
    var out = this._out, dout = this._dout, grads = this.grads;
    for (var i = 0; i < grads.length; i++) grads[i].fill(0);

    var loss = 0, vhat = new Float64Array(n * 2), expects = new Float64Array(n * 2);

    if (this.method === "fm") {
      // FM: vhat = out, loss = mse
      for (var i = 0; i < n * 2; i++) {
        var e = out[i] - v[i];
        loss += e * e;
        vhat[i] = out[i];
        dout[i] = (2 / (2 * n)) * e;
      }
      loss /= (2 * n);
    } else {
      // Hybrid: separate logits and offset, compute vhat from softmax and offset
      var numBuckets = 1 << bits;
      for (var i = 0; i < n; i++) {
        for (var c = 0; c < 2; c++) {
          var logitsStart = c * numBuckets;
          var logits = new Float64Array(numBuckets);
          for (var j = 0; j < numBuckets; j++) {
            logits[j] = out[i * outDim + logitsStart + j];
          }
          var probs = softmax(logits);
          var expect = 0;
          for (var j = 0; j < numBuckets; j++) {
            var bucketIdx = ungray(j, bits);
            expect += probs[j] * centre(bucketIdx, bits, R);
          }
          var offset = out[i * outDim + 2 * numBuckets + c];
          vhat[i * 2 + c] = expect + offset;

          // Loss: cross-entropy on logits + mse on offset
          var bucketIdx = bucketize(v[i * 2 + c], bits, R);
          var grayIdx = gray(bucketIdx);
          for (var j = 0; j < numBuckets; j++) {
            var target = (j === grayIdx) ? 1 : 0;
            var prob = probs[j];
            loss -= (target * Math.log(Math.max(1e-10, prob))) / (2 * n);
            dout[i * outDim + logitsStart + j] = (prob - target) / (2 * n);
          }

          // Offset MSE loss
          var offsetError = offset - (offTarget ? offTarget[i * 2 + c] : v[i * 2 + c] - expect);
          expects[i * 2 + c] = expect;
          loss += offsetError * offsetError / (2 * n);
          dout[i * outDim + 2 * numBuckets + c] = (2 / (2 * n)) * offsetError;

        }
      }
    }

    // Backward pass
    var p = this.params, g = grads;
    var W2 = p[2], W3 = p[4];
    var gW1 = g[0], gb1 = g[1], gW2 = g[2], gb2 = g[3], gW3 = g[4], gb3 = g[5];
    var dz2 = this._d2, dz1 = this._d1;
    var z1 = this._z1, h1 = this._h1, z2 = this._z2, h2 = this._h2;

    for (var r = 0; r < n; r++) {
      var o = r * H;
      var outOff = r * outDim;

      // Gradient through output layer (linear, no activation)
      for (var k = 0; k < outDim; k++) {
        gb3[k] += dout[outOff + k];
        for (var l = 0; l < H; l++) {
          gW3[k * H + l] += dout[outOff + k] * h2[o + l];
        }
      }

      // Gradient through h2 (hidden layer 2, with SiLU)
      for (var j = 0; j < H; j++) {
        var dh = 0;
        for (var k = 0; k < outDim; k++) {
          dh += dout[outOff + k] * W3[k * H + j];
        }
        var z = z2[o + j], sg = sigmoid(z);
        dz2[o + j] = dh * sg * (1 + z * (1 - sg));
      }

      // Gradient through hidden layer 2 (linear)
      for (var j = 0; j < H; j++) {
        gb2[j] += dz2[o + j];
        var w = j * H;
        for (var k = 0; k < H; k++) {
          gW2[w + k] += dz2[o + j] * h1[o + k];
        }
      }

      // Gradient through h1 (hidden layer 1, with SiLU)
      for (var k = 0; k < H; k++) {
        var ds = 0;
        for (var j = 0; j < H; j++) {
          ds += dz2[o + j] * W2[j * H + k];
        }
        z = z1[o + k]; sg = sigmoid(z);
        dz1[o + k] = ds * sg * (1 + z * (1 - sg));
      }

      // Gradient through input layer (linear)
      var x0 = X[3 * r], x1 = X[3 * r + 1], x2 = X[3 * r + 2];
      for (var j = 0; j < H; j++) {
        var d = dz1[o + j];
        gb1[j] += d;
        gW1[3 * j] += d * x0;
        gW1[3 * j + 1] += d * x1;
        gW1[3 * j + 2] += d * x2;
      }
    }

    return { loss: loss, vhat: vhat, expect: expects };
  };

  FMTrainer.prototype.step = function () {
    var B = this.batch, rng = this.rng, data = this.data, pool = this.pool;
    var xy = new Float64Array(B * 2);
    var times = new Float64Array(B);
    var v = new Float64Array(B * 2);
    var r, i, j;

    for (r = 0; r < B; r++) {
      var idx = rng.int(pool);
      var d0 = data[2 * idx], d1 = data[2 * idx + 1];
      var n0 = rng.normal(), n1 = rng.normal();
      var t = rng.random();
      xy[2 * r] = (1 - t) * n0 + t * d0;
      xy[2 * r + 1] = (1 - t) * n1 + t * d1;
      times[r] = t;
      v[2 * r] = d0 - n0;
      v[2 * r + 1] = d1 - n1;
    }

    var result = this.lossGrad(xy, times, v, B);
    var loss = result.loss;

    // Adam
    this.steps += 1;
    var b1c = 1 - Math.pow(0.9, this.steps), b2c = 1 - Math.pow(0.999, this.steps);
    var lr = this.lr, stepSize = lr / b1c, sqrtB2c = Math.sqrt(b2c);
    var p = this.params;
    for (i = 0; i < p.length; i++) {
      var P = p[i], G = this.grads[i], M = this.m[i], V_m = this.v[i];
      for (j = 0; j < P.length; j++) {
        var gr = G[j];
        M[j] = 0.9 * M[j] + 0.1 * gr;
        V_m[j] = 0.999 * V_m[j] + 0.001 * gr * gr;
        P[j] -= stepSize * M[j] / (Math.sqrt(V_m[j]) / sqrtB2c + 1e-8);
      }
    }

    this.losses.push(loss);
    return loss;
  };

  FMTrainer.prototype._velocity = function (xy, n, t, vel) {
    var H = this.hidden, chunk = 256;
    var X = new Float64Array(chunk * 3);
    for (var s = 0; s < n; s += chunk) {
      var m = Math.min(chunk, n - s);
      var oldOutDim = this.outDim;
      for (var r = 0; r < m; r++) {
        X[3 * r] = xy[2 * (s + r)];
        X[3 * r + 1] = xy[2 * (s + r) + 1];
        X[3 * r + 2] = t;
      }
      var oldOut = this._out;
      this._out = new Float64Array(m * this.outDim);
      this._X = X;
      var p = this.params;
      var W1 = p[0], b1 = p[1], W2 = p[2], b2 = p[3], W3 = p[4], b3 = p[5];
      var z1 = new Float64Array(m * H), h1 = new Float64Array(m * H);
      var z2 = new Float64Array(m * H), h2 = new Float64Array(m * H);
      var out = this._out;

      for (var r = 0; r < m; r++) {
        var x0 = X[3 * r], x1 = X[3 * r + 1], x2 = X[3 * r + 2], o = r * H, j, k, s_inner, z, sg;
        for (j = 0; j < H; j++) {
          z = b1[j] + W1[3 * j] * x0 + W1[3 * j + 1] * x1 + W1[3 * j + 2] * x2;
          z1[o + j] = z; h1[o + j] = z * sigmoid(z);
        }
        for (j = 0; j < H; j++) {
          s_inner = b2[j]; var w = j * H;
          for (k = 0; k < H; k++) s_inner += W2[w + k] * h1[o + k];
          z2[o + j] = s_inner; sg = sigmoid(s_inner); h2[o + j] = s_inner * sg;
        }
        for (k = 0; k < this.outDim; k++) {
          s_inner = b3[k];
          for (var l = 0; l < H; l++) s_inner += W3[k * H + l] * h2[o + l];
          out[r * this.outDim + k] = s_inner;
        }
      }

      if (this.method === "hybrid") {
        var numBuckets = 1 << this.bits;
        for (var r = 0; r < m; r++) {
          for (var c = 0; c < 2; c++) {
            var logitsStart = c * numBuckets;
            var logits = new Float64Array(numBuckets);
            for (var j = logitsStart; j < logitsStart + numBuckets; j++) {
              logits[j - logitsStart] = out[r * this.outDim + j];
            }
            var probs = softmax(logits);
            var expect = 0;
            for (var j = 0; j < numBuckets; j++) {
              var bucketIdx = ungray(j, this.bits);
              expect += probs[j] * centre(bucketIdx, this.bits, this.R);
            }
            var offset = out[r * this.outDim + 2 * numBuckets + c];
            vel[2 * (s + r) + c] = expect + offset;
          }
        }
      } else {
        for (var r = 0; r < m; r++) {
          vel[2 * (s + r)] = out[r * 2];
          vel[2 * (s + r) + 1] = out[r * 2 + 1];
        }
      }

      this._out = oldOut;
    }
  };

  FMTrainer.prototype.sample = function (nPoints, nSteps, seed) {
    nPoints = Math.max(0, Math.floor(nPoints || 1000));
    nSteps = Math.max(1, Math.floor(nSteps || 40));
    var rng = new Rng(seed === undefined ? 7 : seed);
    var x = new Float64Array(nPoints * 2), xm = new Float64Array(nPoints * 2);
    var k = new Float64Array(nPoints * 2), i, s;
    for (i = 0; i < 2 * nPoints; i++) x[i] = rng.normal();
    var dt = 1 / nSteps;
    for (s = 0; s < nSteps; s++) {
      var t = s * dt;
      this._velocity(x, nPoints, t, k);
      for (i = 0; i < 2 * nPoints; i++) xm[i] = x[i] + 0.5 * dt * k[i];
      this._velocity(xm, nPoints, t + 0.5 * dt, k);
      for (i = 0; i < 2 * nPoints; i++) x[i] += dt * k[i];
    }
    return Float32Array.from(x);
  };

  FMTrainer.energyDistance = function (a, b) {
    var n = a.length / 2, m = b.length / 2, i, j;
    var sum_ab = 0, sum_aa = 0, sum_bb = 0;
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        var dx = a[2 * i] - b[2 * j], dy = a[2 * i + 1] - b[2 * j + 1];
        sum_ab += Math.sqrt(dx * dx + dy * dy);
      }
    }
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        var dx = a[2 * i] - a[2 * j], dy = a[2 * i + 1] - a[2 * j + 1];
        sum_aa += Math.sqrt(dx * dx + dy * dy);
      }
    }
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
        var dx = b[2 * i] - b[2 * j], dy = b[2 * i + 1] - b[2 * j + 1];
        sum_bb += Math.sqrt(dx * dx + dy * dy);
      }
    }
    return (2 * sum_ab / (n * m) - sum_aa / (n * n) - sum_bb / (m * m));
  };

  FMTrainer.Rng = Rng;
  FMTrainer.twoMoons = twoMoons;
  FMTrainer.eightGaussians = eightGaussians;
  FMTrainer.getDataset = getDataset;
  FMTrainer.gray = gray;
  FMTrainer.ungray = ungray;
  FMTrainer.centre = centre;
  FMTrainer.bucketize = bucketize;

  return FMTrainer;
});
