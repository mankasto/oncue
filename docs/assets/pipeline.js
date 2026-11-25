// "How one test is built": a schematic pedestrian scene drawn as a soft 3D illustration, nine steps.
// Scene data (route, visibility, candidate starts) comes from pipeline-scene.js.
(() => {
  const S = window.PIPELINE_SCENE;
  const canvas = document.getElementById("pipe-anim");
  const ctx = canvas.getContext("2d");
  const W = 1000, H = 562;
  const SANS = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  const C = {
    bg: "#efe9e1", lawn: "#e1e7d3", walk: "#eee7dc", road: "#d5cfc6", curb: "#c1b8ab", lane: "#fbf9f6",
    tree: "#a9c093", trunk: "#9a8670", building: "#e7dfd3",
    parked: ["#b4a893", "#97a3b0", "#a8b3a4"], glass: "#33465a", ego: "#3b79d9",
    target: "#e5562e", exposure: "#1f9d6b", contact: "#d63a4a", hidden: "#8c97a3",
    ok: "#1f9d6b", bad: "#d63a4a", text: "#1c2e3e", muted: "#6b7682",
    shade: "rgba(40,52,68,0.24)", light: "255,212,110",
  };
  const TE = S.t_e, TC = S.t_c, DT = S.dt, ANCHOR = S.anchor;
  const EGO_SPEED = (S.ego.x[ANCHOR + 1] - S.ego.x[ANCHOR]) / DT;
  const SENSOR_H = 2.3, PERSON_H = 1.7;
  const LIGHT_DIR = norm([-0.45, -0.55, 0.7]);

  // ---------------------------------------------------------------- helpers
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, p) => a + (b - a) * p;
  const ease = (p) => (p = clamp(p), p < 0.5 ? 4 * p ** 3 : 1 - (-2 * p + 2) ** 3 / 2);
  const pop = (p) => { p = clamp(p); const c = 1.6; return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2; };
  const span = (lt, a, b) => clamp((lt - a) / (b - a));
  const frameOf = (t) => clamp((t + 1) / DT, 0, S.frames - 1);
  const egoX = (t) => S.ego.x[ANCHOR] + EGO_SPEED * t;
  function sample(series, t) {
    const f = frameOf(t), i = Math.floor(f), j = Math.min(i + 1, series.length - 1);
    return lerp(series[i], series[j], f - i);
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], (a[2] || 0) - (b[2] || 0)]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function crossV(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm(a) { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; }
  function rgb(hex) { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function shadeHex(hex, k, alpha = 1) {
    const [r, g, b] = rgb(hex).map((v) => Math.round(clamp(k >= 1 ? v + (255 - v) * (k - 1) : v * k, 0, 255)));
    return `rgba(${r},${g},${b},${alpha})`;
  }
  const ROUTE = S.route, CUM = [0];
  for (let i = 1; i < ROUTE.length; i++) CUM.push(CUM[i - 1] + Math.hypot(ROUTE[i][0] - ROUTE[i - 1][0], ROUTE[i][1] - ROUTE[i - 1][1]));
  const ROUTE_LEN = CUM[CUM.length - 1];
  function routeAt(s) {
    s = clamp(s, 0, ROUTE_LEN);
    let i = 1;
    while (i < CUM.length - 1 && CUM[i] < s) i++;
    const p = (s - CUM[i - 1]) / (CUM[i] - CUM[i - 1]);
    const [x0, y0] = ROUTE[i - 1], [x1, y1] = ROUTE[i];
    const len = Math.hypot(x1 - x0, y1 - y0);
    return { x: lerp(x0, x1, p), y: lerp(y0, y1, p), dx: (x1 - x0) / len, dy: (y1 - y0) / len };
  }
  function besideRoute(s, side) { const p = routeAt(s); return [p.x - p.dy * side, p.y + p.dx * side]; }
  const isVisible = (t, s) => S.visible[Math.round(frameOf(t))].some(([a, b]) => s >= a - 0.03 && s <= b + 0.03);

  // ---------------------------------------------------------------- camera and projection
  const CAMS = {
    wide: { tx: -9, ty: -0.8, yaw: 66, pitch: 38, dist: 58 },
    contact: { tx: 4, ty: -1.8, yaw: 70, pitch: 40, dist: 38 },
    exposure: { tx: 5, ty: -2.4, yaw: 74, pitch: 46, dist: 34 },
    zoom: { tx: 12.8, ty: -4.0, yaw: 82, pitch: 52, dist: 17.5 },
    rank: { tx: 15.6, ty: -3.6, yaw: 82, pitch: 52, dist: 19 },
    generate: { tx: 11.8, ty: -3.3, yaw: 78, pitch: 50, dist: 20 },
    replay: { tx: 2.5, ty: -1.8, yaw: 70, pitch: 42, dist: 36 },
    label: { tx: 5, ty: -1.0, yaw: 72, pitch: 44, dist: 32 },
    split: { tx: 8.5, ty: -1.4, yaw: 76, pitch: 50, dist: 31 },
  };
  const FOV = 28, NEAR = 0.6;
  let cam;
  function mixCam(a, b, p) { p = ease(p); const out = {}; for (const k of Object.keys(a)) out[k] = lerp(a[k], b[k], p); return out; }
  function drift(c, lt, amount = 1) { return { ...c, yaw: c.yaw + Math.sin(lt * 0.33) * 2.4 * amount, pitch: c.pitch + Math.sin(lt * 0.21) * 0.9 * amount }; }
  function setCam(c, view = { x: 0, y: 0, w: W, h: H }) {
    const yaw = (c.yaw * Math.PI) / 180, pitch = (c.pitch * Math.PI) / 180;
    const target = [c.tx, c.ty, 0];
    const pos = [target[0] - c.dist * Math.cos(pitch) * Math.cos(yaw), target[1] - c.dist * Math.cos(pitch) * Math.sin(yaw), c.dist * Math.sin(pitch)];
    const f = norm(sub(target, pos)), r = norm(crossV(f, [0, 0, 1])), u = crossV(r, f);
    cam = { pos, f, r, u, tx: c.tx, focal: view.h / 2 / Math.tan((FOV * Math.PI) / 360), cx: view.x + view.w / 2, cy: view.y + view.h / 2 + 18, view };
  }
  function toCam(p) { const v = sub(p, cam.pos); return [dot(v, cam.r), dot(v, cam.u), dot(v, cam.f)]; }
  const scr = (c) => [cam.cx + (c[0] / c[2]) * cam.focal, cam.cy - (c[1] / c[2]) * cam.focal];
  function P(p) { const c = toCam(p); return c[2] < NEAR ? null : scr(c); }
  const depth = (p) => toCam(p)[2];
  const pxPerM = (p) => cam.focal / Math.max(depth(p), NEAR);
  function clipPoly(cs) {
    const out = [];
    for (let i = 0; i < cs.length; i++) {
      const a = cs[i], b = cs[(i + 1) % cs.length], ina = a[2] >= NEAR, inb = b[2] >= NEAR;
      if (ina) out.push(a);
      if (ina !== inb) { const t = (NEAR - a[2]) / (b[2] - a[2]); out.push([lerp(a[0], b[0], t), lerp(a[1], b[1], t), NEAR]); }
    }
    return out;
  }
  function polyPath(points) {
    const cs = clipPoly(points.map(toCam));
    if (cs.length < 3) return false;
    ctx.beginPath();
    cs.forEach((c, i) => { const [x, y] = scr(c); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
    return true;
  }
  function fillPoly(points, fill, strokeColor = null) {
    if (!polyPath(points)) return;
    ctx.fillStyle = fill;
    ctx.fill();
    if (strokeColor) { ctx.strokeStyle = strokeColor; ctx.lineWidth = 0.8; ctx.stroke(); }
  }
  function stroke(points, { color, width = 1.5, dash = [], alpha = 1, cap = "round" }) {
    if (alpha <= 0 || points.length < 2) return;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = cap; ctx.lineJoin = "round"; ctx.setLineDash(dash);
    ctx.beginPath();
    let open = false;
    for (let i = 0; i < points.length - 1; i++) {
      let a = toCam(points[i]), b = toCam(points[i + 1]);
      if (a[2] < NEAR && b[2] < NEAR) { open = false; continue; }
      if (a[2] < NEAR) { const t = (NEAR - a[2]) / (b[2] - a[2]); a = [lerp(a[0], b[0], t), lerp(a[1], b[1], t), NEAR]; open = false; }
      let cut = false;
      if (b[2] < NEAR) { const t = (NEAR - a[2]) / (b[2] - a[2]); b = [lerp(a[0], b[0], t), lerp(a[1], b[1], t), NEAR]; cut = true; }
      const [ax, ay] = scr(a), [bx, by] = scr(b);
      if (!open) ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      open = !cut;
    }
    ctx.stroke();
    ctx.restore();
  }
  function circlePoints(x, y, z, r, n = 32) {
    const pts = [];
    for (let k = 0; k <= n; k++) { const a = (k / n) * 2 * Math.PI; pts.push([x + r * Math.cos(a), y + r * Math.sin(a), z]); }
    return pts;
  }

  // objects are queued and drawn far to near; overlays (labels, HUD) are drawn last
  let OBJECTS = [], OVERLAYS = [];
  const queue = (at, draw) => OBJECTS.push({ d: depth(at), draw });
  const overlay = (draw) => OVERLAYS.push(draw);
  function flush() {
    OBJECTS.sort((a, b) => b.d - a.d).forEach((o) => o.draw());
    OVERLAYS.forEach((draw) => draw());
    OBJECTS = []; OVERLAYS = [];
  }

  // ---------------------------------------------------------------- text and labels
  function runs(spec) { return (typeof spec === "string" ? [spec] : spec).map((r) => (typeof r === "string" ? { t: r } : r)); }
  const tc = (rest = "") => [{ t: "t", italic: true }, { t: "c", sub: true }, { t: "*", sup: true }, rest];
  const te = (rest = "") => [{ t: "t", italic: true }, { t: "e", sub: true }, { t: "*", sup: true }, rest];
  const wstar = (rest = "") => [{ t: "W", italic: true }, { t: "*", sup: true }, rest];
  function runFont(r, size, weight) {
    const s = r.sub || r.sup ? size * 0.72 : size;
    return r.italic ? `italic ${weight} ${s * 1.08}px Georgia, "Times New Roman", serif` : `${weight} ${s}px ${SANS}`;
  }
  function textWidth(spec, size, weight = 600) {
    let w = 0;
    for (const r of runs(spec)) { ctx.font = runFont(r, size, weight); w += ctx.measureText(r.t).width; }
    return w;
  }
  function text(spec, x, y, { size = 12, color = C.text, weight = 600, align = "left", alpha = 1 } = {}) {
    if (alpha <= 0) return;
    const w = textWidth(spec, size, weight);
    let cx = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = color; ctx.textBaseline = "middle";
    for (const r of runs(spec)) {
      ctx.font = runFont(r, size, weight);
      ctx.fillText(r.t, cx, y + (r.sub ? size * 0.3 : r.sup ? -size * 0.34 : 0));
      cx += ctx.measureText(r.t).width;
    }
    ctx.restore();
  }
  function card(x, y, w, h, { alpha = 1, r = 12, border = null } = {}) {
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = "rgba(255,255,255,0.96)";
    ctx.shadowColor = "rgba(28,46,62,0.16)"; ctx.shadowBlur = 16; ctx.shadowOffsetY = 4;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fill();
    ctx.shadowColor = "transparent";
    if (border) { ctx.strokeStyle = border; ctx.lineWidth = 1.5; ctx.stroke(); }
    ctx.restore();
  }
  // a label: a white pill with a coloured dot, tied to a point by a short leader
  function label(world, spec, color, { dx = 16, dy = -30, alpha = 1, size = 12 } = {}) {
    if (alpha <= 0) return;
    overlay(() => {
      const p = P(world);
      if (!p) return;
      const w = textWidth(spec, size) + 30, h = size + 14;
      const bx = p[0] + dx - (dx < 0 ? w : 0), by = p[1] + dy - h / 2;
      ctx.save();
      ctx.globalAlpha *= alpha;
      ctx.strokeStyle = "rgba(28,46,62,0.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(dx < 0 ? bx + w - 6 : bx + 6, by + h / 2); ctx.stroke();
      ctx.restore();
      card(bx, by, w, h, { alpha, r: h / 2 });
      ctx.save();
      ctx.globalAlpha *= alpha;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(bx + 12, by + h / 2, 4, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
      text(spec, bx + 21, by + h / 2 + 0.5, { size, alpha });
    });
  }

  // ---------------------------------------------------------------- the world
  function ground(t, { light = 1 } = {}) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(cam.view.x, cam.view.y, cam.view.w, cam.view.h);
    const x0 = cam.tx - 70, x1 = cam.tx + 70;
    fillPoly([[x0, -40], [x1, -40], [x1, 40], [x0, 40]], C.lawn);
    fillPoly([[x0, -6.6], [x1, -6.6], [x1, -4.6], [x0, -4.6]], C.walk);
    fillPoly([[x0, 5.25], [x1, 5.25], [x1, 7.25], [x0, 7.25]], C.walk);
    fillPoly([[x0, -4.6], [x1, -4.6], [x1, 5.25], [x0, 5.25]], C.road);
    for (const y of [-4.6, 5.25]) stroke([[x0, y], [x1, y]], { color: C.curb, width: 2 });
    for (let x = Math.floor(x0 / 6) * 6; x < x1; x += 6) fillPoly([[x, 1.62], [x + 3, 1.62], [x + 3, 1.88], [x, 1.88]], C.lane);
    fillPoly([[x0, -1.82], [x1, -1.82], [x1, -1.68], [x0, -1.68]], "rgba(251,249,246,0.75)");
    if (light > 0) sightLight(t, light);
  }
  // the ego's view as a warm pool of light ahead of it
  function sightLight(t, alpha) {
    const ex = egoX(t), pts = [[ex, 0, 0.02]];
    for (let a = -75; a <= 75; a += 5) { const r = (a * Math.PI) / 180; pts.push([ex + 46 * Math.cos(r), 46 * Math.sin(r), 0.02]); }
    const c = P([ex, 0, 0]);
    if (!c || !polyPath(pts)) return;
    const far = P([ex + 40, 0, 0]);
    const R = far ? Math.hypot(far[0] - c[0], far[1] - c[1]) : 400;
    const g = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], R);
    g.addColorStop(0, `rgba(${C.light},${0.42 * alpha})`);
    g.addColorStop(1, `rgba(${C.light},0)`);
    ctx.fillStyle = g;
    ctx.fill();
  }
  function shadows(t, alpha = 1) {
    if (alpha <= 0) return;
    for (const poly of S.shadows[Math.round(frameOf(t))]) fillPoly(poly.map(([x, y]) => [x, y, 0.03]), `rgba(30,44,62,${0.32 * alpha})`);
  }
  function scenery() {  // trees on the near lawn and low blocks across the road, for context only
    for (let k = -6; k <= 4; k++) {
      const x = k * 11 + 3;
      solidBox(x, 10, x + 8, 15, 0, 3.2, C.building, C.building);
      tree(x + 2.5, -9.6, 0.85);
      tree(x + 8.0, -9.3, 0.7);
    }
  }
  function tree(x, y, size = 1) {
    groundShadow([[x + 0.4, y - 0.3], 1.4 * size]);
    queue([x, y, 0], () => {
      const base = P([x, y, 0]), top = P([x, y, 2.6 * size]);
      if (!base || !top) return;
      ctx.strokeStyle = C.trunk; ctx.lineWidth = Math.max(1.5, 0.18 * pxPerM([x, y, 0])); ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(...base); ctx.lineTo(...top); ctx.stroke();
      const r = 1.35 * size * pxPerM([x, y, 2.6]);
      const g = ctx.createRadialGradient(top[0] - r * 0.35, top[1] - r * 0.4, r * 0.1, top[0], top[1], r);
      g.addColorStop(0, shadeHex(C.tree, 1.18)); g.addColorStop(1, shadeHex(C.tree, 0.86));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(top[0], top[1], r, 0, 2 * Math.PI); ctx.fill();
    });
  }
  function groundShadow([[x, y], r]) {
    const p = P([x, y, 0]);
    if (!p) return;
    const k = pxPerM([x, y, 0]);
    ctx.save();
    ctx.translate(p[0], p[1]);
    ctx.scale(1, 0.55);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r * k);
    g.addColorStop(0, "rgba(40,52,68,0.22)"); g.addColorStop(1, "rgba(40,52,68,0)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(0, 0, r * k, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }

  // a shaded box: visible faces only, lit from one direction
  function solidBox(x0, y0, x1, y1, z0, z1, sideColor, topColor, { alpha = 1, glass = null } = {}) {
    const corners = (z) => [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]];
    const b = corners(z0), t = corners(z1);
    const faces = [
      { pts: [b[0], b[1], t[1], t[0]], n: [0, -1, 0] }, { pts: [b[1], b[2], t[2], t[1]], n: [1, 0, 0] },
      { pts: [b[2], b[3], t[3], t[2]], n: [0, 1, 0] }, { pts: [b[3], b[0], t[0], t[3]], n: [-1, 0, 0] },
    ];
    queue([(x0 + x1) / 2, (y0 + y1) / 2, z0], () => {
      ctx.save();
      ctx.globalAlpha *= alpha;
      for (const f of faces) {
        const mid = f.pts.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4, s[2] + p[2] / 4], [0, 0, 0]);
        if (dot(f.n, sub(mid, cam.pos)) >= 0) continue;
        const k = 0.72 + 0.34 * Math.max(0, dot(f.n, LIGHT_DIR));
        fillPoly(f.pts, shadeHex(glass && Math.abs(f.n[0]) + Math.abs(f.n[1]) ? glass : sideColor, k), "rgba(28,46,62,0.10)");
      }
      fillPoly(t, shadeHex(topColor, 1.08), "rgba(28,46,62,0.10)");
      ctx.restore();
    });
  }
  // a car: a body and a glass cabin, heading +x, with a soft drop shadow
  function car(x0, y0, x1, y1, { body, alpha = 1 } = {}) {
    if (alpha <= 0) return;
    const L = x1 - x0;
    ctx.save(); ctx.globalAlpha *= alpha;
    fillPoly([[x0 + 0.35, y0 - 0.35, 0.01], [x1 + 0.45, y0 - 0.35, 0.01], [x1 + 0.45, y1 - 0.15, 0.01], [x0 + 0.35, y1 - 0.15, 0.01]], "rgba(40,52,68,0.16)");
    ctx.restore();
    solidBox(x0, y0, x1, y1, 0.2, 0.85, body, body, { alpha });
    solidBox(x0 + 0.24 * L, y0 + 0.15, x0 + 0.8 * L, y1 - 0.15, 0.85, 1.42, body, body, { alpha, glass: C.glass });
  }
  function parkedCars({ alpha = 1 } = {}) {
    Object.values(S.occluders).forEach(([x0, y0, x1, y1], k) => car(x0, y0, x1, y1, { body: C.parked[k % 3], alpha }));
  }
  // the outline of a car, for transparent occluders and the oracle's trial stops
  function carOutline(x0, y0, x1, y1, color, { alpha = 1, dash = [5, 4] } = {}) {
    if (alpha <= 0) return;
    const pts = (z) => [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]];
    queue([(x0 + x1) / 2, (y0 + y1) / 2, 0], () => {
      stroke(pts(0.02), { color, width: 1.4, dash, alpha });
      stroke(pts(1.2), { color, width: 1.4, dash, alpha });
      for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) stroke([[x, y, 0], [x, y, 1.2]], { color, width: 1.2, dash, alpha });
    });
  }
  function ego(t, { x = egoX(t), alpha = 1 } = {}) {
    const L = S.ego.length / 2, Wd = S.ego.width / 2;
    car(x - L, -Wd, x + L, Wd, { body: C.ego, alpha });
    queue([x, 0, 1.0], () => {  // the roof sensor at 2.3 m
      const roof = P([x, 0, 1.42]), top = P([x, 0, SENSOR_H]);
      if (!roof || !top) return;
      ctx.save(); ctx.globalAlpha *= alpha;
      ctx.strokeStyle = "#2a3d52"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(...roof); ctx.lineTo(...top); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.strokeStyle = C.ego; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(top[0], top[1], 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.restore();
    });
  }
  const sensor = (t, x = egoX(t)) => [x, 0, SENSOR_H];

  // a person: a rounded body and a head; hidden people are pale and dashed
  function person(x, y, { color = C.target, hidden = false, alpha = 1, scale = 1, glyph = null, dir = [1, 0] } = {}) {
    if (alpha <= 0 || scale <= 0.02) return;
    ctx.save(); ctx.globalAlpha *= alpha * (hidden ? 0.6 : 1);
    groundShadow([[x + 0.18, y - 0.12], 0.42 * scale]);
    if (glyph === "moving") {  // a small arrow on the ground ahead of a moving start
      const r = 0.3 * scale, a = [x + dir[0] * r * 1.6, y + dir[1] * r * 1.6], b = [x + dir[0] * r * 3.6, y + dir[1] * r * 3.6];
      const n = [-dir[1] * r * 0.8, dir[0] * r * 0.8];
      fillPoly([[b[0] + dir[0] * r * 0.6, b[1] + dir[1] * r * 0.6, 0.02], [b[0] + n[0], b[1] + n[1], 0.02], [b[0] - n[0], b[1] - n[1], 0.02]], color);
      stroke([[...a, 0.02], [...b, 0.02]], { color, width: Math.max(1.5, 0.09 * scale * pxPerM([x, y, 0])) });
    }
    ctx.restore();
    queue([x, y, 0], () => {
      const feet = P([x, y, 0.05]), shoulder = P([x, y, 1.25 * scale]), head = P([x, y, 1.55 * scale]);
      if (!feet || !shoulder || !head) return;
      const k = pxPerM([x, y, 0.8]);
      const bw = 0.46 * scale * k, hr = 0.17 * scale * k;
      ctx.save();
      ctx.globalAlpha *= alpha;
      const top = shoulder[1] - bw * 0.25, h = feet[1] - top;
      ctx.beginPath(); ctx.roundRect(feet[0] - bw / 2, top, bw, Math.max(h, bw), bw / 2);
      if (hidden) {
        ctx.fillStyle = shadeHex(color, 1.6, 0.35); ctx.fill();
        ctx.setLineDash([3, 3]); ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke();
      } else {
        const g = ctx.createLinearGradient(feet[0] - bw / 2, 0, feet[0] + bw / 2, 0);
        g.addColorStop(0, shadeHex(color, 1.15)); g.addColorStop(1, shadeHex(color, 0.82));
        ctx.fillStyle = g; ctx.fill();
      }
      ctx.beginPath(); ctx.arc(head[0], head[1] - hr * 0.2, hr, 0, 2 * Math.PI);
      if (hidden) { ctx.fillStyle = shadeHex(color, 1.6, 0.35); ctx.fill(); ctx.stroke(); }
      else { ctx.fillStyle = shadeHex("#f2c7a5", 1); ctx.fill(); }
      ctx.restore();
    });
  }
  // a map pin for the contact and exposure points: a ring on the ground and a lettered head
  function pin(x, y, color, letter, { alpha = 1, rise = 1, pulse = 0, h = 3.0 } = {}) {
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha *= alpha;
    fillPoly(circlePoints(x, y, 0.03, 0.6), shadeHex(color, 1.5, 0.35));
    stroke(circlePoints(x, y, 0.03, 0.6), { color, width: 2.2 });
    if (pulse > 0 && pulse < 1) stroke(circlePoints(x, y, 0.03, 0.6 + 1.8 * pulse), { color, width: 1.6, alpha: 1 - pulse });
    ctx.restore();
    const hh = h * ease(rise);
    queue([x, y, 0], () => {
      const base = P([x, y, 0]), top = P([x, y, hh]);
      if (!base || !top || hh <= 0.05) return;
      ctx.save(); ctx.globalAlpha *= alpha;
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(...base); ctx.lineTo(...top); ctx.stroke();
      ctx.fillStyle = color; ctx.shadowColor = "rgba(28,46,62,0.25)"; ctx.shadowBlur = 6; ctx.shadowOffsetY = 2;
      ctx.beginPath(); ctx.arc(top[0], top[1], 11, 0, 2 * Math.PI); ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.restore();
      text(letter, top[0], top[1] + 0.5, { size: 12, color: "#fff", weight: 700, align: "center", alpha });
    });
  }
  function routeLine({ upTo = 1, alpha = 1, split = 0, t = TE, dim = 1 }) {
    if (alpha <= 0) return;
    const end = ROUTE_LEN * upTo;
    for (let s = 0.2; s < end; s += 0.4) {
      const p = routeAt(s), q = P([p.x, p.y, 0.04]);
      if (!q) continue;
      const vis = split > 0 && isVisible(t, s);
      const k = pxPerM([p.x, p.y, 0]);
      ctx.globalAlpha = alpha * dim * (split > 0 && !vis ? 0.5 * split + (1 - split) : 1);
      ctx.fillStyle = split > 0 ? (vis ? C.target : C.hidden) : C.target;
      ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(1.6, 0.085 * k), 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- HUD
  function chapter(n, title, lt) {
    const a = span(lt, 0.1, 0.6);
    overlay(() => {
      ctx.save(); ctx.globalAlpha = a;
      ctx.fillStyle = C.target;
      ctx.beginPath(); ctx.arc(38, 36, 13, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
      text(String(n), 38, 36.5, { size: 13, color: "#fff", weight: 700, align: "center", alpha: a });
      text(title, 60, 36, { size: 19, weight: 700, alpha: a });
    });
  }
  const RX0 = 600, RX1 = 966, RY = 36, RT0 = -1, RT1 = 6;
  const tx = (t) => RX0 + ((t - RT0) / (RT1 - RT0)) * (RX1 - RX0);
  function ruler({ t, contact = 0, exposure = 0, window = 0, label: wl = 0, measured = 0 }) {
    overlay(() => {
      card(RX0 - 18, RY - 24, RX1 - RX0 + 36, 52, { r: 14 });
      ctx.save();
      ctx.strokeStyle = "rgba(28,46,62,0.25)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(RX0, RY + 2); ctx.lineTo(RX1, RY + 2); ctx.stroke();
      ctx.fillStyle = "rgba(28,46,62,0.06)";
      ctx.fillRect(tx(-1), RY - 3, tx(0) - tx(-1), 10);
      for (let k = RT0; k <= RT1; k++) {
        ctx.beginPath(); ctx.moveTo(tx(k), RY - 1); ctx.lineTo(tx(k), RY + 5); ctx.stroke();
        text(k === RT1 ? "6 s" : `${k}`.replace("-", "−"), tx(k), RY + 15, { size: 9.5, color: C.muted, weight: 500, align: "center" });
      }
      if (window > 0) {
        ctx.fillStyle = "rgba(229,86,46,0.32)";
        ctx.beginPath(); ctx.roundRect(tx(TC - S.W * window), RY - 2, tx(TC) - tx(TC - S.W * window), 8, 4); ctx.fill();
      }
      const mark = (time, color, spec, a) => {
        if (a <= 0) return;
        ctx.globalAlpha = a;
        ctx.strokeStyle = color; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(tx(time), RY - 5); ctx.lineTo(tx(time), RY + 9); ctx.stroke();
        ctx.globalAlpha = 1;
        text(spec, tx(time), RY - 13, { size: 11.5, color, align: "center", alpha: a });
      };
      mark(TE, C.exposure, te(), exposure);
      mark(TC, C.contact, tc(), contact);
      text(wstar(` = ${S.W.toFixed(1)} s`), (tx(TE) + tx(TC)) / 2, RY - 13, { size: 11, color: C.target, align: "center", alpha: wl * (1 - Math.max(exposure, contact) * 0) * 0 });
      if (wl > 0) text(wstar(` = ${S.W.toFixed(1)} s`), (tx(TE) + tx(TC)) / 2 - 2, RY - 13, { size: 11, color: C.target, align: "center", alpha: wl * (exposure > 0 ? 0 : 1) });
      if (measured > 0) text([{ t: "W", italic: true }, ` = ${S.W.toFixed(1)} s ✓`], tx(TC) - 8, RY - 13, { size: 11.5, color: C.ok, align: "right", alpha: measured });
      const x = tx(clamp(t, RT0, RT1));
      ctx.fillStyle = C.text;
      ctx.beginPath(); ctx.arc(x, RY + 2, 4.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.beginPath(); ctx.arc(x, RY + 2, 1.8, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
    });
  }

  // ---------------------------------------------------------------- candidates
  const CANDS = S.candidates.map((c, i) => ({ ...c, index: i }));
  const ACCELS = [...new Set(CANDS.filter((c) => c.kind === "standing").map((c) => c.accel))].sort();
  const sideOf = (c) => (c.kind === "moving" ? -0.9 : [0.45, 0, -0.45][ACCELS.indexOf(c.accel)]);
  function candidateAt(c, t, merge = 0) {
    const s = sample(c.arc, t), [x, y] = besideRoute(s, sideOf(c) * (1 - merge)), p = routeAt(s);
    return { x, y, s, dir: [p.dx, p.dy] };
  }
  const isMoving = (c) => c.kind === "moving";
  const KEPT = CANDS.filter((c) => c.rank).sort((a, b) => a.rank - b.rank);
  const CHOSEN = CANDS[S.chosen];

  // ---------------------------------------------------------------- segments
  function world(c, t, { light = 1, trees = true } = {}) {
    setCam(c);
    ground(t, { light });
    if (trees) scenery();
  }

  function seg1(lt, d) {
    const t = lerp(-1, 6, ease(span(lt, 0.4, d - 0.2)));
    world(drift({ ...CAMS.wide, tx: lerp(-16, 4, ease(span(lt, 0, d))), dist: 48 }, lt), t, { light: 0.6 });
    parkedCars();
    ego(t);
    label([egoX(t), 0, 2.4], "reference ego", C.ego, { alpha: span(lt, 0.8, 1.3), dy: -34 });
    chapter(1, "A recorded scene", lt);
    ruler({ t });
  }

  function seg2(lt, d) {
    const t = lerp(0, TC, ease(span(lt, 1.8, d - 0.6)));
    world(drift(mixCam({ ...CAMS.wide, tx: 4, dist: 48 }, CAMS.contact, span(lt, 0, 1.4)), lt), t, { light: 0.6 });
    routeLine({ upTo: ease(span(lt, 0.3, 1.8)) });
    parkedCars();
    const arrive = span(lt, d - 0.7, d - 0.4);
    pin(...S.contact, C.contact, "C", { alpha: span(lt, 1.5, 1.8), rise: span(lt, 1.5, 2.2), pulse: span(lt, 1.6, 2.6) });
    ego(t);
    const p = routeAt(2);
    label([p.x, p.y, 0], "target route", C.target, { alpha: span(lt, 0.8, 1.2) * (1 - span(lt, 2.6, 3.0)), dy: 30 });
    label([...S.contact, 3.0], ["contact point", ...(arrive > 0 ? ["   ", ...tc(` = ${TC.toFixed(1)} s`)] : [])], C.contact, { alpha: span(lt, 1.9, 2.3), dx: 20, dy: -6 });
    chapter(2, "Pick where they meet", lt);
    ruler({ t, contact: arrive });
  }

  function seg3(lt, d) {
    const back = ease(span(lt, 0.5, 2.4)), t = lerp(TC, TE, back);
    const sight = span(lt, 2.6, 3.6), split = span(lt, 3.8, 4.7), expo = span(lt, 4.8, 5.2);
    world(drift(mixCam(CAMS.contact, CAMS.exposure, span(lt, 0.4, 2.4)), lt), t, { light: 0.35 + 0.65 * sight });
    shadows(t, sight);
    routeLine({ split, t });
    parkedCars();
    pin(...S.contact, C.contact, "C");
    pin(...S.exposure, C.exposure, "E", { alpha: expo, rise: expo, pulse: span(lt, 4.9, 5.9), h: 2.2 });
    ego(t);
    if (sight > 0) {
      const eye = sensor(t);
      overlay(() => {
        stroke([eye, [...S.exposure, 0.6]], { color: C.exposure, width: 1.8, alpha: sight * (0.35 + 0.65 * expo) });
        stroke([eye, [13.6, -2.45, 1.42]], { color: "rgba(28,46,62,0.55)", width: 1.1, dash: [4, 4], alpha: sight });
      });
    }
    label([10.5, -5.4, 0], "hidden", C.hidden, { alpha: split, dy: 30, dx: -16 });
    label([...S.exposure, 2.2], ["exposure point", "   ", ...te(` = ${TE.toFixed(1)} s`)], C.exposure, { alpha: expo, dx: 22, dy: 0 });
    label([...S.contact, 3.0], ["contact point", "   ", ...tc(` = ${TC.toFixed(1)} s`)], C.contact, { dx: 20, dy: -6 });
    chapter(3, "Work back to the first sighting", lt);
    ruler({ t, contact: 1, exposure: span(lt, 2.1, 2.4), window: back, label: span(lt, 1.4, 2.0) * (1 - span(lt, 2.0, 2.3)) });
  }

  function hidingArea(c, t) {
    world(c, t, { light: 0.5, trees: false });
    shadows(t, 1);
    routeLine({ split: 1, t, dim: 0.6 });
    parkedCars();
    pin(...S.contact, C.contact, "C", { h: 2.6 });
    pin(...S.exposure, C.exposure, "E", { h: 2.0 });
  }
  function egoArrow(t, alpha) {
    if (alpha <= 0) return;
    overlay(() => {
      const p = P(sensor(t));
      if (p && p[0] > 40) return;
      const y = p ? clamp(p[1], 110, H - 60) : 220;
      card(14, y - 15, 132, 30, { alpha, r: 15 });
      ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = C.ego;
      ctx.beginPath(); ctx.moveTo(24, y); ctx.lineTo(33, y - 6); ctx.lineTo(33, y + 6); ctx.fill();
      ctx.restore();
      text("reference ego", 40, y + 0.5, { size: 12, alpha });
    });
  }

  function seg4(lt, d) {
    const t = lerp(TE, -1, ease(span(lt, 0.2, 1.3)));
    hidingArea(drift(mixCam(CAMS.exposure, CAMS.zoom, span(lt, 0, 1.4)), lt, 0.6), t);
    CANDS.forEach((c, k) => {
      const it = candidateAt(c, -1);
      const delay = isMoving(c) ? 1.7 + k * 0.12 : 2.6 + k * 0.06;
      person(it.x, it.y, { hidden: true, scale: 0.6 * pop(span(lt, delay, delay + 0.45)), glyph: c.kind, dir: it.dir });
    });
    egoArrow(t, span(lt, 1.0, 1.4));
    const mv = candidateAt(CANDS.find(isMoving), -1);
    label([mv.x, mv.y, 0], "walking ×4", C.target, { alpha: span(lt, 2.2, 2.6), dx: -18, dy: 36 });
    label([14.0, -4.7, 1.0], "standing ×15", C.target, { alpha: span(lt, 3.6, 4.0), dx: 34, dy: 26 });
    chapter(4, "Try hidden starts", lt);
    ruler({ t, contact: 1, exposure: 1, window: 1 });
  }

  function seg5(lt, d) {
    const early = span(lt, 0.3, 0.8), earlyGone = span(lt, 1.6, 2.2);
    const seen = span(lt, 2.4, 2.8), seenGone = span(lt, 3.6, 4.2);
    const t = lerp(-1, TE, ease(span(lt, 4.3, d - 1.3)));
    const merge = clamp((t - (TE - 1.3)) / 1.3), arrived = span(lt, d - 1.2, d - 0.8);
    hidingArea(drift(CAMS.zoom, lt, 0.6), t);
    for (const c of CANDS) {
      const it = candidateAt(c, Math.max(t, -1), merge);
      if (c.drop === "leaves_before_start") {
        person(it.x, it.y, { hidden: early === 0, color: early > 0 ? C.bad : C.target, scale: 0.6 * (1 - ease(earlyGone)), glyph: c.kind, dir: it.dir });
      } else if (c.drop === "seen_early") {
        if (seen > 0) overlay(() => stroke([sensor(t), [it.x, it.y, 0.9]], { color: C.bad, width: 2, alpha: seen * (1 - seenGone) }));
        person(it.x, it.y, { hidden: seen === 0, color: seen > 0 ? C.bad : C.target, scale: 0.6 * (1 - ease(seenGone)), glyph: c.kind, dir: it.dir });
      } else {
        person(it.x, it.y, { hidden: arrived === 0, color: arrived > 0 ? C.ok : C.target, scale: 0.6, glyph: c.kind, dir: it.dir });
      }
    }
    egoArrow(t, 1);
    label([12.9, -5.4, 0], "would leave before the start ×9", C.bad, { alpha: early * (1 - earlyGone), dx: -10, dy: 46 });
    const sc = candidateAt(CANDS.find((c) => c.drop === "seen_early"), -1);
    label([sc.x, sc.y, 1.0], "seen too early", C.bad, { alpha: seen * (1 - seenGone), dx: -14, dy: 42 });
    label([...S.exposure, 0.8], `${KEPT.length} reach E unseen`, C.ok, { alpha: arrived, dx: 26, dy: 14 });
    chapter(5, "Keep starts that stay hidden", lt);
    ruler({ t, contact: 1, exposure: 1, window: 1 });
  }

  const PANEL = { x: 646, y: 96, w: 322, row: 38 };
  function seg6(lt, d) {
    const fly = ease(span(lt, 0.4, 1.6)), sort = ease(span(lt, 2.0, 3.1)), cut = span(lt, 3.5, 4.1);
    hidingArea(drift(mixCam(CAMS.zoom, CAMS.rank, span(lt, 0, 1.2)), lt, 0.4), TE);
    const pa = span(lt, 0, 0.5);
    overlay(() => {
      const ex = P([...S.exposure, 0.8]);
      card(PANEL.x, PANEL.y, PANEL.w, PANEL.row * KEPT.length + 46, { alpha: pa, r: 16 });
      text("closer to logged pedestrians  →", PANEL.x + 84, PANEL.y + 22, { size: 11, color: C.muted, weight: 500, alpha: pa });
      const made = [...KEPT].sort((a, b) => a.index - b.index);
      for (const c of KEPT) {
        const row = lerp(made.indexOf(c), c.rank - 1, sort);
        const y = PANEL.y + 56 + row * PANEL.row;
        const x = lerp(ex[0], PANEL.x + 34, fly), yy = lerp(ex[1], y, fly);
        const out = c.rank > 5 ? cut : 0;
        const col = out ? "#c7c0b6" : C.ok;
        ctx.save();
        ctx.globalAlpha = pa;
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(x, yy, 9, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#fff";
        if (isMoving(c)) { ctx.beginPath(); ctx.moveTo(x - 3, yy - 5); ctx.lineTo(x + 5, yy); ctx.lineTo(x - 3, yy + 5); ctx.fill(); }
        else { ctx.fillRect(x - 4, yy - 4.5, 2.6, 9); ctx.fillRect(x + 1.4, yy - 4.5, 2.6, 9); }
        if (fly >= 1) {
          const w = 160 * Math.exp(-1.6 * c.typical) * ease(span(lt, 1.6, 2.2));
          ctx.fillStyle = out ? "#e2ddd5" : "rgba(31,157,107,0.8)";
          ctx.beginPath(); ctx.roundRect(PANEL.x + 62, y - 5, Math.max(w, 4), 10, 5); ctx.fill();
        }
        ctx.restore();
        if (sort >= 1) text(`#${c.rank}`, PANEL.x + PANEL.w - 20, y, { size: 12, color: out ? C.muted : C.text, align: "right" });
      }
      if (cut > 0) {
        const y = PANEL.y + 56 + 4.5 * PANEL.row;
        ctx.save(); ctx.globalAlpha = cut;
        ctx.strokeStyle = C.target; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(PANEL.x + 16, y); ctx.lineTo(PANEL.x + PANEL.w - 16, y); ctx.stroke();
        ctx.restore();
        text("keep the top 5", PANEL.x + PANEL.w - 50, y - 10, { size: 11, color: C.target, align: "right", alpha: cut });
      }
    });
    chapter(6, "Rank by realism", lt);
    ruler({ t: TE, contact: 1, exposure: 1, window: 1 });
  }

  function seg7(lt, d) {
    const inputs = span(lt, 0.9, 1.6), grow = ease(span(lt, 1.9, 3.5)), judge = span(lt, 3.7, 4.4), pick = span(lt, 4.6, 5.2);
    world(drift(mixCam(CAMS.rank, CAMS.generate, span(lt, 0, 1.2)), lt, 0.5), 0, { light: 0.3, trees: false });
    shadows(0, 0.6);
    for (const f of S.fan) {
      if (f.chosen) continue;
      const n = Math.max(2, Math.round(f.points.length * grow));
      const fail = judge > 0 && !f.ok;
      stroke(f.points.slice(0, n).map(([x, y]) => [x, y, 0.05]), { color: fail ? "#b9b1a6" : C.target, width: 1.1, alpha: grow > 0 ? (fail ? 0.55 - 0.4 * judge : 0.5) * (1 - pick * 0.5) : 0 });
    }
    const chosen = S.fan.find((f) => f.chosen).points.map(([x, y]) => [x, y, 0.06]);
    stroke(chosen.slice(0, Math.max(2, Math.round(chosen.length * ease(pick)))), { color: C.target, width: 4.5, alpha: pick });
    parkedCars();
    pin(...S.contact, C.contact, "C", { h: 2.6 });
    pin(...S.exposure, C.exposure, "E", { h: 2.0 });
    const start = candidateAt(CHOSEN, 0, 1);
    person(start.x, start.y, { hidden: true, scale: 0.75, glyph: "moving", dir: start.dir, color: C.ok });
    const box = { x: 30, y: 74, w: 230, h: 54 };
    overlay(() => {
      const ca = span(lt, 0.4, 0.9);
      card(box.x, box.y, box.w, box.h, { alpha: ca, r: 14, border: C.target });
      text("trajectory generator", box.x + box.w / 2, box.y + box.h / 2, { size: 14, weight: 700, align: "center", alpha: ca });
      const flow = (world3, color, k) => {
        const p = P(world3);
        if (!p || inputs <= 0) return;
        const end = [box.x + box.w * (0.3 + 0.2 * k), box.y + box.h];
        const u = clamp(inputs * 1.3 - k * 0.15);
        ctx.save(); ctx.globalAlpha = (1 - pick) * 0.9;
        ctx.strokeStyle = color; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(p[0], p[1]);
        ctx.quadraticCurveTo(p[0], end[1] + 40, lerp(p[0], end[0], u), lerp(p[1], end[1], u));
        ctx.stroke(); ctx.restore();
      };
      flow([start.x, start.y, 1.0], C.ok, 0);
      flow([...S.exposure, 2.0], C.exposure, 1);
      flow([...S.contact, 2.6], C.contact, 2);
    });
    label([start.x, start.y, 0], "rank 1 start", C.ok, { alpha: inputs * (1 - pick), dx: -18, dy: 38 });
    label([16.8, -3.2, 0], "64 candidates", C.target, { alpha: span(lt, 2.5, 3.0) * (1 - pick), dx: 22, dy: 28 });
    label([13.4, -5.1, 0], "selected", C.target, { alpha: span(lt, 5.0, 5.4), dx: -18, dy: 36 });
    chapter(7, "Generate the motion", lt);
    ruler({ t: 0, contact: 1, exposure: 1, window: 1 });
  }

  function seg8(lt, d) {
    const t = lerp(-1, TC, span(lt, 0.4, d - 0.9));
    const c = { ...CAMS.replay, tx: lerp(CAMS.replay.tx - 4, CAMS.replay.tx + 3, span(lt, 0, d)) };
    world(drift(mixCam(CAMS.generate, c, span(lt, 0, 1.0)), lt, 0.6), t, { light: 0.8 });
    shadows(t, 1);
    const s = sample(CHOSEN.arc, t), p = routeAt(s), seen = t >= TE - 1e-6;
    const walked = [];
    for (let u = sample(CHOSEN.arc, -1); u <= s; u += 0.2) { const q = routeAt(u); walked.push([q.x, q.y, 0.05, u]); }
    stroke(walked.filter((q) => q[3] <= S.s_reveal), { color: C.hidden, width: 1.8, dash: [3, 5] });
    stroke(walked.filter((q) => q[3] >= S.s_reveal), { color: C.target, width: 3 });
    parkedCars();
    pin(...S.contact, C.contact, "C", { h: 2.8, alpha: 0.9 });
    pin(...S.exposure, C.exposure, "E", { h: 2.0, alpha: 0.9 });
    ego(t);
    const eyeP = P(sensor(t));
    if (eyeP && eyeP[0] > -60) overlay(() => stroke([sensor(t), [p.x, p.y, 1.0]], { color: seen ? C.ok : "rgba(28,46,62,0.45)", width: seen ? 2 : 1.2, dash: seen ? [] : [4, 4] }));
    person(p.x, p.y, { hidden: !seen, glyph: "moving", dir: [p.dx, p.dy], scale: 0.9 });
    egoArrow(t, 1);
    label([p.x, p.y, PERSON_H * 0.9], seen ? "visible" : "hidden", seen ? C.ok : C.hidden, { alpha: span(lt, 0.6, 1.0), dx: 20, dy: -16 });
    chapter(8, "Replay the generated test", lt);
    ruler({ t, contact: 1, exposure: 1, measured: span(lt, d - 0.9, d - 0.5) });
  }

  const BRAKES = [2, 3, 4.5, 6, 8], LATENCY_S = 0.2, CONTACT_FRONT_X = S.contact[0] - 0.3;
  const MASKED_BRAKE_T = TE + 0.8, MASKED_DECEL = 4.0, OPEN_BRAKE_T = 2.0, OPEN_DECEL = 2.0;
  function braked(t, from, decel) {
    const x0 = egoX(Math.min(t, from));
    if (t <= from) return x0;
    const tau = Math.min(t - from, EGO_SPEED / decel);
    return x0 + EGO_SPEED * tau - 0.5 * decel * tau * tau;
  }
  const MASKED_HIT_T = (() => {
    for (let t = TE; t < 7; t += 0.01) if (braked(t, MASKED_BRAKE_T, MASKED_DECEL) + S.ego.length / 2 >= CONTACT_FRONT_X) return t;
    return Infinity;
  })();

  function seg9(lt, d) {
    const LABEL_END = 4.8;
    if (lt < LABEL_END) {
      world(drift(mixCam(CAMS.replay, CAMS.label, span(lt, 0, 0.9)), lt, 0.5), TE, { light: 0.6 });
      shadows(TE, 1);
      parkedCars();
      pin(...S.contact, C.contact, "C", { h: 2.6 });
      const front = egoX(TE) + S.ego.length / 2;
      BRAKES.forEach((a, k) => {
        const appear = span(lt, 1.0 + k * 0.35, 1.35 + k * 0.35);
        const stop = EGO_SPEED * LATENCY_S + EGO_SPEED ** 2 / (2 * a);
        const safe = front + stop < CONTACT_FRONT_X;
        const x = egoX(TE) + stop * ease(appear), L = S.ego.length / 2, Wd = S.ego.width / 2;
        carOutline(x - L, -Wd, x + L, Wd, safe ? C.ok : C.bad, { alpha: appear * 0.9 });
      });
      ego(TE);
      const pe = routeAt(S.s_reveal);
      person(pe.x, pe.y, { glyph: "moving", dir: [pe.dx, pe.dy], scale: 0.9 });
      label([egoX(TE), 0, 2.4], "brake after exposure", C.text, { alpha: span(lt, 0.9, 1.3), dy: -36 });
      const badge = span(lt, 3.2, 3.6);
      overlay(() => {
        card(W / 2 - 120, 72, 240, 54, { alpha: badge, r: 16, border: C.ok });
        text("Reactive", W / 2, 91, { size: 17, color: C.ok, align: "center", weight: 800, alpha: badge });
        text("a safe action exists after exposure", W / 2, 110, { size: 11, color: C.muted, weight: 500, align: "center", alpha: badge });
      });
      chapter(9, "Label, then test planners", lt);
      ruler({ t: TE, contact: 1, exposure: 1, window: 1 });
      return;
    }
    const lt2 = lt - LABEL_END, d2 = d - LABEL_END;
    const t = lerp(1.6, 5.6, span(lt2, 0.6, d2 - 0.6));
    const half = W / 2;
    for (const [view, masked] of [[{ x: 0, y: 0, w: half - 2, h: H }, true], [{ x: half + 2, y: 0, w: half - 2, h: H }, false]]) {
      ctx.save();
      ctx.beginPath(); ctx.rect(view.x, view.y, view.w, view.h); ctx.clip();
      setCam(drift(CAMS.split, lt, 0.4), view);
      ground(t, { light: masked ? 0.8 : 0.4 });
      scenery();
      const x = masked ? braked(Math.min(t, MASKED_HIT_T), MASKED_BRAKE_T, MASKED_DECEL) : braked(t, OPEN_BRAKE_T, OPEN_DECEL);
      const s = sample(CHOSEN.arc, t), p = routeAt(s);
      if (masked) { shadows(t, 1); parkedCars(); }
      else Object.values(S.occluders).forEach(([x0, y0, x1, y1]) => carOutline(x0, y0, x1, y1, "#7d8996", { alpha: 0.8 }));
      pin(...S.contact, C.contact, "C", { h: 2.4, alpha: 0.85 });
      ego(t, { x });
      person(p.x, p.y, { hidden: masked && t < TE, glyph: "moving", dir: [p.dx, p.dy], scale: 0.9 });
      overlay(() => {
        card(view.x + 16, 64, masked ? 128 : 150, 30, { r: 15 });
        text(masked ? "masked" : "unoccluded", view.x + 30, 79.5, { size: 13, weight: 700 });
      });
      if (masked) label([CONTACT_FRONT_X, 0, 1.6], "collision", C.bad, { alpha: t >= MASKED_HIT_T ? 1 : 0, dx: -16, dy: -40 });
      else label([x, 0, 1.6], "stops in time", C.ok, { alpha: span(t, 5.0, 5.3), dy: -40 });
      flush();
      ctx.restore();
    }
    ctx.fillStyle = "#fffdf9";
    ctx.fillRect(half - 2, 0, 4, H);
    chapter(9, "Label, then test planners", lt);
    ruler({ t, contact: 1, exposure: 1 });
  }

  const SEGMENTS = [
    { name: "Scene", dur: 4.8, draw: seg1 },
    { name: "Contact", dur: 5.6, draw: seg2 },
    { name: "Exposure", dur: 7.4, draw: seg3 },
    { name: "Starts", dur: 5.4, draw: seg4 },
    { name: "Hidden check", dur: 11.0, draw: seg5 },
    { name: "Rank", dur: 5.4, draw: seg6 },
    { name: "Generate", dur: 6.4, draw: seg7 },
    { name: "Replay", dur: 7.6, draw: seg8 },
    { name: "Label and test", dur: 10.8, draw: seg9 },
  ];
  const STARTS = [];
  let total = 0;
  for (const s of SEGMENTS) { STARTS.push(total); total += s.dur; }

  // ---------------------------------------------------------------- player
  const playButton = document.getElementById("pipe-play");
  const chips = document.getElementById("pipe-chips");
  const progress = document.getElementById("pipe-progress");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let now = 0, playing = !reduced, last = null;
  SEGMENTS.forEach((s, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="n">${String(i + 1).padStart(2, "0")}</span>${s.name}`;
    b.addEventListener("click", () => { now = reduced ? STARTS[i] + s.dur - 0.01 : STARTS[i] + 0.001; render(); });
    chips.appendChild(b);
  });
  const toggle = () => { playing = !playing; last = null; render(); };
  playButton.addEventListener("click", toggle);
  canvas.addEventListener("click", toggle);
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
  }
  function render() {
    let i = SEGMENTS.length - 1;
    while (i > 0 && STARTS[i] > now) i--;
    const seg = SEGMENTS[i], lt = now - STARTS[i];
    ctx.save();
    ctx.clearRect(0, 0, W, H);
    OBJECTS = []; OVERLAYS = [];
    seg.draw(Math.min(lt, seg.dur), seg.dur);
    flush();
    ctx.restore();
    const edge = Math.min(lt, seg.dur - lt);  // a short fade between segments
    if (edge < 0.25) { ctx.fillStyle = `rgba(250,248,245,${(1 - edge / 0.25) * 0.6})`; ctx.fillRect(0, 0, W, H); }
    [...chips.children].forEach((b, k) => b.classList.toggle("on", k === i));
    progress.style.width = `${(now / total) * 100}%`;
    playButton.textContent = playing ? "❚❚ Pause" : "▶ Play";
  }
  let onScreen = true;  // draw only while the canvas is on screen
  new IntersectionObserver((entries) => { onScreen = entries[0].isIntersecting; last = null; }).observe(canvas);
  function tick(ts) {
    if (onScreen) {
      if (playing && last !== null) now = (now + (ts - last) / 1000) % total;
      last = ts;
      render();
    }
    requestAnimationFrame(tick);
  }
  window.addEventListener("resize", resize);
  window.pipelineAnimation = { seek: (t) => { now = t; playing = false; render(); }, starts: STARTS };
  if (reduced) now = SEGMENTS[0].dur - 0.01;
  resize();
  requestAnimationFrame(tick);
})();
