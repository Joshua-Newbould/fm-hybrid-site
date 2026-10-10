(function() {
  'use strict';

  let data = null;
  let currentParticle = 0;
  let currentDataset = 'moons';

  const section = document.getElementById('quant');
  if (!section) return;
  const mk = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  };

  const widget = mk('div', 'widget');
  const controls = mk('div', 'controls');
  const select = mk('select', 'quant-dataset');
  select.setAttribute('aria-label', 'Dataset');
  select.innerHTML = '<option value="moons">Moons</option><option value="gaussians8">Gaussians 8</option>';
  select.addEventListener('change', (e) => {
    currentDataset = e.target.value;
    loadData();
  });
  const nextBtn = mk('button', 'btn next-particle', 'Next particle');
  nextBtn.addEventListener('click', () => {
    if (!data) return;
    currentParticle = (currentParticle + 1) % data.n;
    render();
  });
  const offsetReadout = mk('div', 'offset-readout readout');
  offsetReadout.setAttribute('aria-live', 'polite');
  controls.append(select, nextBtn, offsetReadout);

  const canvas = mk('canvas', 'quant-canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Predicted bucket probabilities, expected centre and offset for one particle');
  const strips = [0, 1].map((c) => {
    const strip = mk('div', 'bin-strip');
    strip.setAttribute('data-component', String(c));
    strip.setAttribute('data-name', c ? 'vy' : 'vx');
    for (let i = 0; i < 16; i++) strip.appendChild(mk('div', 'bin-label', (i ^ (i >> 1)).toString(2).padStart(4, '0')));
    return strip;
  });
  const legend = mk('div', 'legend');
  legend.innerHTML = '<span class="k k-bar">predicted mass</span><span class="k k-true">true velocity</span>' +
    '<span class="k k-exp">expected centre</span><span class="k k-off">offset to prediction</span>';
  widget.append(canvas, strips[0], strips[1], legend, controls);
  section.appendChild(widget);

  async function loadData() {
    offsetReadout.textContent = 'loading';
    let response;
    try {
      response = await fetch(`data/buckets_${currentDataset}.json`);
      if (!response.ok) throw new Error(response.status + ' ' + response.statusText);
      data = await response.json();
    } catch (err) {
      offsetReadout.textContent = 'could not load the bucket data (' + err.message + '); a content blocker may be stopping the request';
      return;
    }
    strips.forEach((strip) => strip.querySelectorAll('.bin-label').forEach((label, i) => {
      label.textContent = data.gray[i].toString(2).padStart(4, '0');
    }));
    currentParticle = 0;
    render();
  }

  const fmt = (a) => `(${a[0].toFixed(2)}, ${a[1].toFixed(2)})`;

  function render() {
    if (!data) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height, gap = 34, top = 8;
    ctx.clearRect(0, 0, w, h);
    const ph = (h - top - gap - 22) / 2;
    for (let comp = 0; comp < 2; comp++) drawComponent(ctx, comp, 0, top + comp * (ph + gap), w, ph);

    const p = currentParticle;
    strips.forEach((strip, c) => strip.querySelectorAll('.bin-label').forEach((l, i) => {
      l.classList.toggle('on', data.bucket[p][c] === i);
    }));
    offsetReadout.textContent = `particle ${p}  t ${data.t[p].toFixed(2)}  v ${fmt(data.v[p])}  offset ${fmt(data.offset[p])}`;
  }

  function drawComponent(ctx, comp, x, y, w, h) {
    const p = currentParticle;
    const probs = data.probs[p][comp];
    const v = data.v[p][comp];
    const expect = data.expect[p][comp];
    const vhat = data.vhat[p][comp];
    const bucket = data.bucket[p][comp];
    const cs = getComputedStyle(document.documentElement);
    const col = (n) => cs.getPropertyValue(n).trim();
    const text = col('--text'), muted = col('--muted'), line = col('--border'), hy = col('--hybrid');
    const R = data.R, binW = w / 16;
    const X = (val) => x + (val + R) / (2 * R) * w;
    const base = y + h;

    const counts = new Array(16).fill(0);
    for (let i = 0; i < data.n; i++) counts[data.bucket[i][comp]]++;
    const maxC = Math.max(...counts);
    ctx.fillStyle = muted;
    for (let i = 0; i < 16; i++) {
      ctx.globalAlpha = 0.03 + 0.11 * counts[i] / maxC;
      ctx.fillRect(x + i * binW + 1, y, binW - 2, h);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = hy;
    ctx.globalAlpha = 0.16;
    ctx.fillRect(x + bucket * binW + 1, y, binW - 2, h);
    ctx.globalAlpha = 1;

    const maxProb = Math.max(...probs);
    const grad = ctx.createLinearGradient(0, base - h * 0.78, 0, base);
    grad.addColorStop(0, hy);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    for (let i = 0; i < 16; i++) {
      const barH = (probs[i] / maxProb) * h * 0.78;
      ctx.fillStyle = grad;
      ctx.fillRect(x + i * binW + binW * 0.2, base - barH, binW * 0.6, barH);
      ctx.fillStyle = hy;
      ctx.fillRect(x + i * binW + binW * 0.2, base - barH, binW * 0.6, 2);
    }

    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, base + 0.5);
    ctx.lineTo(x + w, base + 0.5);
    ctx.stroke();
    ctx.fillStyle = muted;
    ctx.font = `11px ${cs.getPropertyValue('--mono')}`;
    ctx.textBaseline = 'top';
    for (const tick of [-R, -R / 2, 0, R / 2, R]) {
      ctx.textAlign = tick === -R ? 'left' : tick === R ? 'right' : 'center';
      ctx.fillText(String(tick), X(tick), base + 6);
    }
    ctx.textAlign = 'left';
    ctx.fillText(comp ? 'vy' : 'vx', x + 6, y + 6);

    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(X(v), y + 4);
    ctx.lineTo(X(v), base);
    ctx.stroke();
    ctx.setLineDash([]);

    const ex = X(expect), vx = X(vhat), ay = y + h * 0.14;
    ctx.strokeStyle = hy;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ex, ay - 8);
    ctx.lineTo(ex, base);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ex, ay);
    ctx.lineTo(vx, ay);
    ctx.stroke();
    const dir = vx >= ex ? 1 : -1;
    ctx.fillStyle = hy;
    ctx.beginPath();
    ctx.moveTo(vx + dir * 2, ay);
    ctx.lineTo(vx - dir * 6, ay - 4);
    ctx.lineTo(vx - dir * 6, ay + 4);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(vx, ay, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  let rt = 0;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(render, 80); });
  loadData();
})();
