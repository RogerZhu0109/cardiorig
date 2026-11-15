/* Interactive figures of the CardioRig project page.
   The decoder, the integration and the warp below follow src/rig.py and src/utils/svf.py:
   fields are displacements in pixels, and warp(img, u)(p) = img(p + u(p)). */
(function () {
  'use strict';
  const R = window.RIG, N = R.n, NN = N * N, TAU = 2 * Math.PI, BINS = R.endo.length;
  const COL = [0, 2, 1, 3]; // displayed curve j (1-cos 2pi, sin 2pi, 1-cos 4pi, sin 4pi) -> column of the parameters
  const SEG_C = ['#B64342', '#E9A6A1', '#E0B43C', '#8BCF8B', '#42949E', '#3775BA'];
  const CURVE_C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];
  const INK = '#111111', INK2 = '#55555a', RULE = '#e5e5e8', ACCENT = '#0F4D92', MARK = '#ff7a1a';
  const FONT = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif';
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function bytes(s) {
    const b = atob(s), a = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i);
    return a;
  }
  const FRAMES = bytes(R.frames), F = R.tau.length, MASK = bytes(R.mask);
  const ED = FRAMES.subarray(0, NN);
  const B8 = new Int8Array(bytes(R.basis).buffer), BASIS = new Float32Array(B8.length);
  for (let i = 0; i < B8.length; i++) BASIS[i] = B8[i] / 127;
  const NF = R.slots.length; // 28 fields act on this slice
  const HATS = bytes(R.hats), CAV = bytes(R.cavity), DEPTH = bytes(R.depth), DECAY = bytes(R.decay_map);
  const ACTIVE = new Set(R.slots);
  const VIEW = 68, V0 = (N - VIEW) / 2; // part of the slice shown, in pixels

  // ---------------------------------------------------------------- the rig
  function timeBasis(tau) {
    const w = TAU * tau;
    return [1 - Math.cos(w), 1 - Math.cos(2 * w), Math.sin(w), Math.sin(2 * w)];
  }
  function velocity(theta, tau) { // v = sum_k c_k(tau) B_k
    const tb = timeBasis(tau), v = new Float32Array(2 * NN);
    for (let f = 0; f < NF; f++) {
      const row = theta[R.slots[f]];
      const c = row[0] * tb[0] + row[1] * tb[1] + row[2] * tb[2] + row[3] * tb[3];
      if (c === 0) continue;
      const o = f * 2 * NN;
      for (let i = 0; i < 2 * NN; i++) v[i] += c * BASIS[o + i];
    }
    return v;
  }
  function sample(a, o, n, x, y) { // bilinear, border padding
    x = x < 0 ? 0 : x > n - 1 ? n - 1 : x;
    y = y < 0 ? 0 : y > n - 1 ? n - 1 : y;
    const x0 = x | 0, y0 = y | 0, x1 = x0 < n - 1 ? x0 + 1 : x0, y1 = y0 < n - 1 ? y0 + 1 : y0;
    const fx = x - x0, fy = y - y0;
    return (a[o + y0 * n + x0] * (1 - fx) + a[o + y0 * n + x1] * fx) * (1 - fy) +
           (a[o + y1 * n + x0] * (1 - fx) + a[o + y1 * n + x1] * fx) * fy;
  }
  function integrate(v, n, steps) { // exp(v) by scaling and squaring
    const nn = n * n, s = 1 / Math.pow(2, steps);
    let u = new Float32Array(2 * nn), t = new Float32Array(2 * nn);
    for (let i = 0; i < 2 * nn; i++) u[i] = v[i] * s;
    for (let k = 0; k < steps; k++) {
      for (let y = 0, i = 0; y < n; y++) for (let x = 0; x < n; x++, i++) {
        const sx = x + u[i], sy = y + u[nn + i];
        t[i] = u[i] + sample(u, 0, n, sx, sy);
        t[nn + i] = u[nn + i] + sample(u, nn, n, sx, sy);
      }
      const swap = u; u = t; t = swap;
    }
    return u;
  }
  function warpImage(img, u) {
    const out = new Float32Array(NN);
    for (let y = 0, i = 0; y < N; y++) for (let x = 0; x < N; x++, i++) out[i] = sample(img, 0, N, x + u[i], y + u[NN + i]);
    return out;
  }
  function generate(theta, tau) {
    const u = integrate(velocity(theta, tau), N, 7);
    return { u: u, img: warpImage(ED, u) };
  }
  function zeros() { return Array.from({ length: 58 }, function () { return [0, 0, 0, 0]; }); }
  function radius(radii, th) { // periodic linear interpolation over the angular bins
    const x = ((th % TAU) + TAU) % TAU / (TAU / BINS) - 0.5, i = Math.floor(x), f = x - i;
    return radii[(i + BINS) % BINS] * (1 - f) + radii[(i + 1) % BINS] * f;
  }
  function motionName(k) {
    if (k === 0) return 'Shift of the whole heart, left and right';
    if (k === 1) return 'Shift of the whole heart, up and down';
    if (k < 50) {
      const s = Math.floor((k - 2) / 3), ring = s < 6 ? 'Basal' : s < 12 ? 'Mid' : 'Apical';
      const n = s < 6 ? s + 1 : s < 12 ? s - 5 : s - 11;
      return ring + ' segment ' + n + ', ' + ['inner boundary', 'outer boundary', 'tangential motion'][(k - 2) % 3];
    }
    return 'RV sector ' + (Math.floor((k - 50) / 2) + 1) + ', ' + ((k - 50) % 2 ? 'tangential motion' : 'radial motion');
  }

  // ---------------------------------------------------------------- drawing
  function setup(canvas, w, h) {
    canvas.width = Math.round(w * DPR); canvas.height = Math.round(h * DPR);
    canvas.style.maxWidth = w + 'px'; canvas.style.aspectRatio = w + ' / ' + h;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    return ctx;
  }
  const off = document.createElement('canvas'); off.width = off.height = N;
  const octx = off.getContext('2d'), idata = octx.createImageData(N, N);
  function blit(ctx, S) {
    octx.putImageData(idata, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, V0, V0, VIEW, VIEW, 0, 0, S, S);
  }
  function drawGray(ctx, arr, S) {
    const d = idata.data;
    for (let i = 0; i < NN; i++) { const v = arr[i]; d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = v; d[4 * i + 3] = 255; }
    blit(ctx, S);
  }
  function drawHeat(ctx, value, S) { // value(i) in [0, 1]: white to the accent colour
    const d = idata.data;
    for (let i = 0; i < NN; i++) {
      const a = value(i);
      d[4 * i] = 255 - a * (255 - 15); d[4 * i + 1] = 255 - a * (255 - 77); d[4 * i + 2] = 255 - a * (255 - 146); d[4 * i + 3] = 255;
    }
    blit(ctx, S);
  }
  function P(x, S) { return (x + 0.5 - V0) * S / VIEW; }
  function unP(q, S) { return { x: q.x * VIEW / S + V0 - 0.5, y: q.y * VIEW / S + V0 - 0.5 }; }
  function ring(ctx, radii, S) {
    ctx.beginPath();
    for (let b = 0; b <= BINS; b++) {
      const a = (b % BINS + 0.5) * TAU / BINS, r = radii[b % BINS];
      const x = P(R.cx + r * Math.cos(a), S), y = P(R.cy + r * Math.sin(a), S);
      if (b) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.closePath();
  }
  function outlines(ctx, S, colour, dash) {
    ctx.save(); ctx.strokeStyle = colour; ctx.lineWidth = 1.2; ctx.setLineDash(dash || []);
    ring(ctx, R.endo, S); ctx.stroke(); ring(ctx, R.epi, S); ctx.stroke(); ctx.restore();
  }
  function arrow(ctx, x0, y0, x1, y1, head) {
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
    if (L < 0.6) return;
    const ux = dx / L, uy = dy / L, h = Math.min(head, L * 0.7);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - 0.6 * h * ux, y1 - 0.6 * h * uy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - h * ux + 0.45 * h * uy, y1 - h * uy - 0.45 * h * ux);
    ctx.lineTo(x1 - h * ux - 0.45 * h * uy, y1 - h * uy + 0.45 * h * ux);
    ctx.closePath(); ctx.fill();
  }
  function circle(ctx, x, y, r, colour) {
    ctx.save(); ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = colour; ctx.stroke(); ctx.restore();
  }
  function pointer(canvas, w, on) { // pointer position in drawing units, null when it leaves
    canvas.addEventListener('pointermove', function (e) {
      const r = canvas.getBoundingClientRect();
      on({ x: (e.clientX - r.left) * w / r.width, y: (e.clientY - r.top) * w / r.width });
    });
    canvas.addEventListener('pointerleave', function () { on(null); });
  }

  // ---------------------------------------------------------------- widgets
  const widgets = [];
  function register(id, w) {
    w.el = document.getElementById(id); w.visible = false; w.dirty = true;
    widgets.push(w);
    return w;
  }
  function makePlayer(w, period) {
    const btn = w.el.querySelector('.play'), sl = w.el.querySelector('.tau'), out = w.el.querySelector('.tauval');
    const p = {
      tau: 0, playing: !REDUCED,
      advance: function (dt) { p.tau = (p.tau + dt / period) % 1; sl.value = p.tau; show(); },
    };
    function show() {
      out.textContent = p.tau.toFixed(2);
      btn.textContent = p.playing ? '❚❚' : '▶';
      btn.setAttribute('aria-label', p.playing ? 'Pause' : 'Play');
    }
    btn.addEventListener('click', function () { p.playing = !p.playing; show(); w.dirty = true; });
    sl.addEventListener('input', function () { p.playing = false; p.tau = +sl.value; show(); w.dirty = true; });
    show();
    w.player = p;
    return p;
  }

  // 1. warp ------------------------------------------------------------------
  (function () {
    const w = register('warpDemo', {}), S = 320;
    const cl = w.el.querySelector('canvas.left'), cr = w.el.querySelector('canvas.right');
    const L = setup(cl, S, S), Rr = setup(cr, S, S), p = makePlayer(w, 3.2);
    const box = w.el.querySelector('.arrows'), read = w.el.querySelector('.readout');
    const a0 = R.septum + Math.PI, r0 = radius(R.endo, a0) + 1;
    const pinned = { x: R.cx + r0 * Math.cos(a0), y: R.cy + r0 * Math.sin(a0) };
    let hover = null;
    pointer(cr, S, function (q) { hover = q && unP(q, S); w.dirty = true; });
    box.addEventListener('change', function () { w.dirty = true; });
    w.draw = function () {
      const g = generate(R.predicted, p.tau);
      drawGray(L, ED, S); drawGray(Rr, g.img, S);
      if (box.checked) {
        Rr.strokeStyle = Rr.fillStyle = 'rgba(255,214,64,0.95)'; Rr.lineWidth = 1.3;
        for (let y = 3; y < N; y += 5) for (let x = 3; x < N; x += 5) {
          const i = y * N + x, ux = g.u[i], uy = g.u[NN + i];
          if (Math.hypot(ux, uy) > 0.5) arrow(Rr, P(x + ux, S), P(y + uy, S), P(x, S), P(y, S), 5);
        }
      }
      const q = hover || pinned;
      const ux = sample(g.u, 0, N, q.x, q.y), uy = sample(g.u, NN, N, q.x, q.y);
      circle(Rr, P(q.x, S), P(q.y, S), 6, MARK);
      L.save(); L.strokeStyle = 'rgba(255,255,255,0.8)'; L.setLineDash([3, 3]); L.lineWidth = 1.2;
      L.beginPath(); L.moveTo(P(q.x, S), P(q.y, S)); L.lineTo(P(q.x + ux, S), P(q.y + uy, S)); L.stroke();
      L.beginPath(); L.arc(P(q.x, S), P(q.y, S), 3, 0, TAU); L.stroke(); L.restore();
      circle(L, P(q.x + ux, S), P(q.y + uy, S), 6, MARK);
      read.innerHTML = 'At τ = <b>' + p.tau.toFixed(2) + '</b>, the marked pixel copies a point <b>' +
        (Math.hypot(ux, uy) * R.spacing).toFixed(1) + ' mm</b> away in the first frame.';
    };
  })();

  // 2. scaling and squaring ----------------------------------------------------
  (function () {
    const w = register('expDemo', {}), S = 320, n = 49, nn = n * n, s = S / (n - 1) * 0.86, o = S * 0.07;
    const L = setup(w.el.querySelector('canvas.left'), S, S), Rr = setup(w.el.querySelector('canvas.right'), S, S);
    const ks = w.el.querySelector('.k'), kv = w.el.querySelector('.kval'), st = w.el.querySelector('.strength');
    const read = w.el.querySelector('.readout');
    [ks, st].forEach(function (e) { e.addEventListener('input', function () { w.dirty = true; }); });
    function X(x) { return o + x * s; }
    w.draw = function () {
      const K = +ks.value, a = +st.value, c = (n - 1) / 2, sig = n / 5.5, v = new Float32Array(2 * nn);
      for (let y = 0, i = 0; y < n; y++) for (let x = 0; x < n; x++, i++) {
        const dx = x - c, dy = y - c, g = Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig));
        v[i] = (-a * dx - 0.2 * a * dy) * g; v[nn + i] = (-a * dy + 0.2 * a * dx) * g;
      }
      const u = integrate(v, n, K);
      kv.textContent = K;
      // left: the field, each arrow is one whole step
      L.fillStyle = '#fff'; L.fillRect(0, 0, S, S);
      L.strokeStyle = L.fillStyle = ACCENT; L.lineWidth = 1.1;
      for (let y = 0; y < n; y += 4) for (let x = 0; x < n; x += 4) {
        const i = y * n + x;
        if (Math.hypot(v[i], v[nn + i]) * s > 1.5) arrow(L, X(x), X(y), X(x + v[i]), X(y + v[nn + i]), 5);
        else { L.beginPath(); L.arc(X(x), X(y), 0.9, 0, TAU); L.fill(); }
      }
      // right: the grid moved by the deformation, folded cells in red
      Rr.fillStyle = '#fff'; Rr.fillRect(0, 0, S, S);
      let folded = 0;
      Rr.fillStyle = 'rgba(214,39,40,0.55)';
      for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
        const q = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]].map(function (pt) {
          const i = pt[1] * n + pt[0];
          return [X(pt[0] + u[i]), X(pt[1] + u[nn + i])];
        });
        let area = 0;
        for (let j = 0; j < 4; j++) { const b = q[(j + 1) % 4]; area += q[j][0] * b[1] - b[0] * q[j][1]; }
        if (area <= 0) {
          folded++;
          Rr.beginPath(); Rr.moveTo(q[0][0], q[0][1]); for (let j = 1; j < 4; j++) Rr.lineTo(q[j][0], q[j][1]);
          Rr.closePath(); Rr.fill();
        }
      }
      Rr.strokeStyle = INK; Rr.lineWidth = 0.9;
      for (let k = 0; k < n; k += 3) {
        Rr.beginPath();
        for (let t = 0; t < n; t++) { const i = k * n + t; if (t) Rr.lineTo(X(t + u[i]), X(k + u[nn + i])); else Rr.moveTo(X(t + u[i]), X(k + u[nn + i])); }
        Rr.stroke();
        Rr.beginPath();
        for (let t = 0; t < n; t++) { const i = t * n + k; if (t) Rr.lineTo(X(k + u[i]), X(t + u[nn + i])); else Rr.moveTo(X(k + u[i]), X(t + u[nn + i])); }
        Rr.stroke();
      }
      const steps = Math.pow(2, K);
      read.innerHTML = '<b>' + steps + '</b> ' + (steps === 1 ? 'step of size 1' : 'small steps of size 1/' + steps) +
        ', computed with <b>' + K + '</b> doubling' + (K === 1 ? '' : 's') + '. ' +
        (folded ? 'The grid is folded in <b>' + (100 * folded / ((n - 1) * (n - 1))).toFixed(1) + '%</b> of its area.' : 'The grid is not folded.');
    };
  })();

  // 3. coordinates -----------------------------------------------------------
  (function () {
    const w = register('coordsDemo', {}), S = 400;
    const canvas = w.el.querySelector('canvas'), ctx = setup(canvas, S, S), read = w.el.querySelector('.readout');
    const a0 = R.first_sector + 2.5 * TAU / 6, rm = 0.5 * (radius(R.endo, a0) + radius(R.epi, a0));
    const pinned = { x: R.cx + rm * Math.cos(a0), y: R.cy + rm * Math.sin(a0) };
    let hover = null;
    pointer(canvas, S, function (q) { hover = q && unP(q, S); w.dirty = true; });
    w.draw = function () {
      drawGray(ctx, ED, S);
      const cx = P(R.cx, S), cy = P(R.cy, S), k = S / VIEW;
      // segments as a band along the middle of the wall
      for (let s = 0; s < 6; s++) {
        ctx.beginPath(); ctx.strokeStyle = SEG_C[s]; ctx.lineWidth = 5; ctx.globalAlpha = 0.85;
        for (let j = 0; j <= 20; j++) {
          const a = R.first_sector + (s - 0.5 + j / 20) * TAU / 6, r = 0.5 * (radius(R.endo, a) + radius(R.epi, a));
          const x = P(R.cx + r * Math.cos(a), S), y = P(R.cy + r * Math.sin(a), S);
          if (j) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
        const am = R.first_sector + s * TAU / 6, rl = radius(R.epi, am) + 7;
        ctx.font = 'bold ' + FONT; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.7)'; ctx.fillStyle = '#fff';
        ctx.strokeText(s + 1, P(R.cx + rl * Math.cos(am), S), P(R.cy + rl * Math.sin(am), S));
        ctx.fillText(s + 1, P(R.cx + rl * Math.cos(am), S), P(R.cy + rl * Math.sin(am), S));
      }
      outlines(ctx, S, '#ffffff');
      // the direction theta = 0
      const rs = radius(R.epi, R.septum) + 9;
      ctx.save(); ctx.strokeStyle = '#fff'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + rs * k * Math.cos(R.septum), cy + rs * k * Math.sin(R.septum)); ctx.stroke(); ctx.restore();
      ctx.font = FONT; ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('θ = 0', cx + (rs + 5) * k * Math.cos(R.septum), cy + (rs + 5) * k * Math.sin(R.septum));
      // the point
      const q = hover || pinned, dx = q.x - R.cx, dy = q.y - R.cy, r = Math.hypot(dx, dy), th = Math.atan2(dy, dx);
      const rel = ((th - R.septum) % TAU + TAU) % TAU, rin = radius(R.endo, th), rout = radius(R.epi, th);
      ctx.strokeStyle = MARK; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(P(q.x, S), P(q.y, S)); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, 16, R.septum, R.septum + rel); ctx.stroke();
      ctx.fillStyle = MARK; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, TAU); ctx.fill();
      circle(ctx, P(q.x, S), P(q.y, S), 6, MARK);
      const d = Math.min(1, Math.max(0, (r - rin) / (rout - rin)));
      const seg = Math.floor((((th - R.first_sector + TAU / 12) % TAU + TAU) % TAU) / (TAU / 6)) + 1;
      const where = r < rin ? 'inside the cavity' : r > rout ? 'outside the wall' : 'in the wall, segment ' + seg;
      read.innerHTML = 'r = <b>' + (r * R.spacing).toFixed(1) + ' mm</b> &nbsp; θ = <b>' + (rel * 360 / TAU).toFixed(0) +
        '°</b> &nbsp; r<sub>in</sub> = <b>' + (rin * R.spacing).toFixed(1) + ' mm</b> &nbsp; r<sub>out</sub> = <b>' +
        (rout * R.spacing).toFixed(1) + ' mm</b> &nbsp; d = <b>' + d.toFixed(2) + '</b> &nbsp; (' + where + ')';
    };
  })();

  // 4. basis fields ----------------------------------------------------------
  (function () {
    const w = register('basisDemo', {}), S = 340, FS = 112;
    const main = setup(w.el.querySelector('canvas.main'), S, S);
    const fc = [0, 1, 2].map(function (j) { return setup(w.el.querySelector('canvas.f' + j), FS, FS); });
    const caps = [0, 1, 2].map(function (j) { return w.el.querySelector('.c' + j); });
    const fname = w.el.querySelector('.fname');
    let seg = 3, mot = 0;
    const MOTIONS = ['Inner boundary', 'Outer boundary', 'Tangential'];
    function chips(root, labels, get, set, colours) {
      labels.forEach(function (label, j) {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'chip';
        b.innerHTML = (colours ? '<span class="dot" style="background:' + colours[j] + '"></span>' : '') + label;
        b.addEventListener('click', function () { set(j); refresh(); w.dirty = true; });
        root.appendChild(b);
      });
      function refresh() { root.querySelectorAll('button').forEach(function (b, j) { b.setAttribute('aria-pressed', j === get()); }); }
      refresh();
    }
    chips(w.el.querySelector('.seg'), ['1', '2', '3', '4', '5', '6'], function () { return seg; }, function (j) { seg = j; }, SEG_C);
    chips(w.el.querySelector('.mot'), MOTIONS, function () { return mot; }, function (j) { mot = j; });
    const hat = function (i) { return HATS[seg * NN + i] / 255; };
    const q = function (i) { return CAV[i] / 255; }, d = function (i) { return DEPTH[i] / 255; }, g = function (i) { return DECAY[i] / 255; };
    const FACTORS = [
      [[hat, 'h_s', 'selects the segment'], [q, 'q', 'zero at the centre of the cavity'], [function (i) { return 1 - d(i); }, '1-d', 'one on the inner boundary, zero on the outer'], 'B_s^\\mathrm{in} = h_s\\,q\\,(1-d)\\,e_r'],
      [[hat, 'h_s', 'selects the segment'], [d, 'd', 'zero on the inner boundary, one on the outer'], [g, 'g', 'fades outside the heart'], 'B_s^\\mathrm{out} = h_s\\,d\\,g\\,e_r'],
      [[hat, 'h_s', 'selects the segment'], [q, 'q', 'zero at the centre of the cavity'], [g, 'g', 'fades outside the heart'], 'B_s^\\mathrm{t} = h_s\\,q\\,g\\,e_\\theta'],
    ];
    w.draw = function () {
      const o = (2 + 3 * seg + mot) * 2 * NN, spec = FACTORS[mot];
      drawGray(main, ED, S);
      outlines(main, S, 'rgba(255,255,255,0.9)');
      main.strokeStyle = 'rgba(0,0,0,0.75)'; main.fillStyle = 'rgba(0,0,0,0.75)'; main.lineWidth = 3.2;
      for (let pass = 0; pass < 2; pass++) {
        for (let y = 1; y < N; y += 3) for (let x = 1; x < N; x += 3) {
          const i = y * N + x, bx = -BASIS[o + i], by = -BASIS[o + NN + i]; // tissue moves against the sampling direction
          if (Math.hypot(bx, by) > 0.06) arrow(main, P(x, S), P(y, S), P(x + 4 * bx, S), P(y + 4 * by, S), 6);
        }
        main.strokeStyle = main.fillStyle = '#ffd640'; main.lineWidth = 1.6;
      }
      for (let j = 0; j < 3; j++) {
        drawHeat(fc[j], spec[j][0], FS);
        outlines(fc[j], FS, 'rgba(17,17,17,0.55)');
        caps[j].innerHTML = '<span class="m"></span>: ' + spec[j][2];
        window.katex.render(spec[j][1], caps[j].querySelector('.m'));
      }
      window.katex.render(spec[3], fname);
    };
  })();

  // 5. time curves -----------------------------------------------------------
  (function () {
    const w = register('curvesDemo', {}), CW = 440, CH = 250, S = 280;
    const chart = setup(w.el.querySelector('canvas.chart'), CW, CH), heart = setup(w.el.querySelector('canvas.heart'), S, S);
    const p = makePlayer(w, 3.2), sliders = Array.from(w.el.querySelectorAll('input.w'));
    const selM = w.el.querySelector('.motion'), selS = w.el.querySelector('.segment');
    let th = [2, 0, 0, 0]; // in displayed order
    function rows() { // rows of the 58 x 4 parameters driven by the sliders (mid ring: segments 6..11)
      const segs = +selS.value < 0 ? [0, 1, 2, 3, 4, 5] : [+selS.value];
      return segs.map(function (s) { return 2 + 3 * (6 + s) + (+selM.value); });
    }
    function show() {
      sliders.forEach(function (sl, j) { sl.value = th[j]; sl.nextElementSibling.textContent = (th[j] >= 0 ? '+' : '−') + Math.abs(th[j]).toFixed(2); });
      w.dirty = true;
    }
    sliders.forEach(function (sl, j) { sl.addEventListener('input', function () { th[j] = +sl.value; show(); }); });
    [selM, selS].forEach(function (e) { e.addEventListener('change', function () { w.dirty = true; }); });
    w.el.querySelectorAll('.preset').forEach(function (b) {
      b.addEventListener('click', function () {
        if (b.dataset.p === 'one') th = [2, 0, 0, 0];
        else if (b.dataset.p === 'zero') th = [0, 0, 0, 0];
        else {
          const rs = rows();
          th = COL.map(function (c) {
            const m = rs.reduce(function (a, k) { return a + R.predicted[k][c]; }, 0) / rs.length;
            return Math.max(-3, Math.min(3, Math.round(m * 20) / 20));
          });
        }
        show();
      });
    });
    const curve = [
      function (t) { return 1 - Math.cos(TAU * t); }, function (t) { return Math.sin(TAU * t); },
      function (t) { return 1 - Math.cos(2 * TAU * t); }, function (t) { return Math.sin(2 * TAU * t); },
    ];
    w.draw = function () {
      // chart
      const m = { l: 34, r: 12, t: 10, b: 36 }, pw = CW - m.l - m.r, ph = CH - m.t - m.b, M = 100;
      const ys = [0, 1, 2, 3].map(function (j) { return Array.from({ length: M + 1 }, function (_, i) { return th[j] * curve[j](i / M); }); });
      const sum = ys[0].map(function (_, i) { return ys[0][i] + ys[1][i] + ys[2][i] + ys[3][i]; });
      let lo = -1, hi = 4;
      ys.concat([sum]).forEach(function (a) { a.forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); }); });
      const step = hi - lo > 12 ? 4 : hi - lo > 6 ? 2 : 1;
      lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
      const X = function (t) { return m.l + t * pw; }, Y = function (v) { return m.t + (hi - v) / (hi - lo) * ph; };
      chart.fillStyle = '#fff'; chart.fillRect(0, 0, CW, CH);
      chart.font = FONT; chart.lineWidth = 1;
      for (let v = lo; v <= hi + 1e-9; v += step) {
        chart.strokeStyle = v === 0 ? '#9a9aa0' : RULE;
        chart.beginPath(); chart.moveTo(m.l, Y(v)); chart.lineTo(m.l + pw, Y(v)); chart.stroke();
        chart.fillStyle = INK2; chart.textAlign = 'right'; chart.textBaseline = 'middle'; chart.fillText(v, m.l - 6, Y(v));
      }
      chart.textAlign = 'center'; chart.textBaseline = 'top'; chart.fillStyle = INK2;
      [0, 0.25, 0.5, 0.75, 1].forEach(function (t) { chart.fillText(t === 0 ? 'ED' : t, X(t), m.t + ph + 6); });
      chart.textAlign = 'right'; chart.fillText('phase \u03C4', m.l + pw, m.t + ph + 18);
      function line(a, colour, width) {
        chart.strokeStyle = colour; chart.lineWidth = width; chart.lineJoin = 'round';
        chart.beginPath(); a.forEach(function (v, i) { if (i) chart.lineTo(X(i / M), Y(v)); else chart.moveTo(X(0), Y(v)); }); chart.stroke();
      }
      ys.forEach(function (a, j) { line(a, CURVE_C[j], 1.6); });
      line(sum, INK, 2.6);
      const c = th.reduce(function (a, v, j) { return a + v * curve[j](p.tau); }, 0);
      chart.strokeStyle = MARK; chart.lineWidth = 1.2;
      chart.beginPath(); chart.moveTo(X(p.tau), m.t); chart.lineTo(X(p.tau), m.t + ph); chart.stroke();
      chart.fillStyle = '#fff'; chart.beginPath(); chart.arc(X(p.tau), Y(c), 6, 0, TAU); chart.fill();
      chart.fillStyle = MARK; chart.beginPath(); chart.arc(X(p.tau), Y(c), 4, 0, TAU); chart.fill();
      chart.fillStyle = INK; chart.textBaseline = 'bottom'; chart.textAlign = p.tau > 0.8 ? 'right' : 'left';
      chart.fillText('c(τ) = ' + c.toFixed(2), X(p.tau) + (p.tau > 0.8 ? -9 : 9), Math.max(Y(c) - 6, m.t + 14));
      // heart
      const theta = zeros();
      rows().forEach(function (k) { COL.forEach(function (col, j) { theta[k][col] = th[j]; }); });
      drawGray(heart, generate(theta, p.tau).img, S);
      outlines(heart, S, 'rgba(255,255,255,0.85)', [4, 4]);
    };
    show();
  })();

  // 6. the whole rig ---------------------------------------------------------
  (function () {
    const w = register('rigDemo', {}), CELL = 12, ROW = 16, LEFT = 16, SW = LEFT + 58 * CELL + 2, SH = 4 * ROW + 36, S = 270;
    const stripC = w.el.querySelector('canvas.strip'), strip = setup(stripC, SW, SH);
    const real = setup(w.el.querySelector('canvas.real'), S, S), gen = setup(w.el.querySelector('canvas.gen'), S, S);
    const p = makePlayer(w, 3.2), read = w.el.querySelector('.strip-read');
    const editSeg = w.el.querySelector('.editseg'), alpha = w.el.querySelector('.alpha'), alphaVal = w.el.querySelector('.alphaval');
    let source = 'predicted', hover = -1;
    w.el.querySelectorAll('.src').forEach(function (b) {
      b.addEventListener('click', function () {
        source = b.dataset.s;
        w.el.querySelectorAll('.src').forEach(function (o) { o.setAttribute('aria-pressed', o === b); });
        w.dirty = true;
      });
    });
    [editSeg, alpha].forEach(function (e) { e.addEventListener('input', function () { w.dirty = true; }); });
    pointer(stripC, SW, function (q) {
      hover = q && q.y < 4 * ROW && q.x >= LEFT ? Math.min(57, Math.floor((q.x - LEFT) / CELL)) : -1;
      w.dirty = true;
    });
    function colour(v) { // blue, light grey, red
      const a = Math.max(-1, Math.min(1, v / 2.5)), lo = a < 0 ? [55, 117, 186] : [182, 67, 66], t = Math.abs(a);
      return 'rgb(' + [0, 1, 2].map(function (i) { return Math.round(247 + t * (lo[i] - 247)); }).join(',') + ')';
    }
    function theta() { // the edit of the paper: scale the two radial rows of one segment
      const src = R[source], a = +alpha.value, k = 2 + 3 * (6 + (+editSeg.value));
      return src.map(function (row, i) { return i === k || i === k + 1 ? row.map(function (v) { return v * a; }) : row; });
    }
    w.draw = function () {
      const th = theta();
      alphaVal.textContent = '×' + (+alpha.value).toFixed(2);
      strip.fillStyle = '#fafafb'; strip.fillRect(0, 0, SW, SH);
      for (let j = 0; j < 4; j++) { strip.fillStyle = CURVE_C[j]; strip.fillRect(2, j * ROW + 4, 8, 8); }
      for (let k = 0; k < 58; k++) {
        strip.globalAlpha = ACTIVE.has(k) ? 1 : 0.3;
        for (let j = 0; j < 4; j++) { strip.fillStyle = colour(th[k][COL[j]]); strip.fillRect(LEFT + k * CELL, j * ROW, CELL - 1, ROW - 1); }
      }
      strip.globalAlpha = 1;
      if (hover >= 0) { strip.strokeStyle = INK; strip.lineWidth = 1.5; strip.strokeRect(LEFT + hover * CELL - 1, -1 + 1, CELL + 1, 4 * ROW); }
      strip.font = FONT; strip.textBaseline = 'top'; strip.textAlign = 'center'; strip.lineWidth = 1;
      [[0, 2, 'shift'], [2, 50, 'LV wall: 16 segments × 3 motions'], [50, 58, 'RV: 4 sectors × 2']].forEach(function (g) {
        const x0 = LEFT + g[0] * CELL, x1 = LEFT + g[1] * CELL - 1, y = 4 * ROW + 5;
        strip.strokeStyle = INK2; strip.beginPath(); strip.moveTo(x0, y); strip.lineTo(x0, y + 4); strip.lineTo(x1, y + 4); strip.lineTo(x1, y); strip.stroke();
        strip.fillStyle = INK2; strip.fillText(g[2], Math.max(18, (x0 + x1) / 2), y + 10);
      });
      if (hover >= 0) {
        read.innerHTML = '<b>' + motionName(hover) + '</b>, weights \u03B8<sub>1</sub> to \u03B8<sub>4</sub>: ' + COL.map(function (c) { return th[hover][c].toFixed(2); }).join(', ') +
          (ACTIVE.has(hover) ? '' : ' (acts on other slices)');
      } else read.innerHTML = 'Point at a column to see its motion.';
      const f = Math.round(p.tau * F) % F;
      drawGray(real, FRAMES.subarray(f * NN, (f + 1) * NN), S);
      const g = generate(th, p.tau);
      drawGray(gen, g.img, S);
      [real, gen].forEach(function (c) { outlines(c, S, 'rgba(255,255,255,0.7)', [4, 4]); });
      const fi = R.slots.indexOf(hover);
      if (fi >= 0) { // the hovered field, in the direction tissue moves for a positive value
        const o = fi * 2 * NN;
        gen.strokeStyle = gen.fillStyle = MARK; gen.lineWidth = 1.5;
        for (let y = 1; y < N; y += 5) for (let x = 1; x < N; x += 5) {
          const i = y * N + x, bx = -BASIS[o + i], by = -BASIS[o + NN + i];
          if (Math.hypot(bx, by) > 0.08) arrow(gen, P(x, S), P(y, S), P(x + 5 * bx, S), P(y + 5 * by, S), 5);
        }
      }
    };
  })();

  // ---------------------------------------------------------------- start
  window.renderMathInElement(document.body, {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
  });
  const seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      const w = widgets.find(function (x) { return x.el === e.target; });
      w.visible = e.isIntersecting; w.dirty = true;
    });
  }, { rootMargin: '200px' });
  widgets.forEach(function (w) { seen.observe(w.el); });
  let last = performance.now();
  function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    widgets.forEach(function (w) {
      if (!w.visible) return;
      if (w.player && w.player.playing) { w.player.advance(dt); w.dirty = true; }
      if (w.dirty) { w.dirty = false; w.draw(); }
    });
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  window.RIG_WIDGETS = widgets;
})();
