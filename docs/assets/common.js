// OnCue site: shared page helpers (from the scenario library page). A page defines `L` (strings
// per language), `state` (with scene, frame, playing, view, box), `draw()` and `drawBev()`.
"use strict";
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $ = id => document.getElementById(id);
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

// ---------- language ----------
let LANG = "en";
function t(key, ...args) {
  const v = (L[LANG] && L[LANG][key] !== undefined) ? L[LANG][key] : (L.en[key] !== undefined ? L.en[key] : key);
  return typeof v === "string" ? v.replace(/\{(\d)\}/g, (_, i) => args[+i]) : v;
}
// the viewer's saved choice, else English whatever the browser's language (user 11-02)
function savedLanguage() {
  try { const saved = localStorage.getItem("oncue.lang"); if (saved) return saved; } catch (_) {}
  return "en";
}
function applyLang() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  if (L[LANG].title) document.title = t("title");
  for (const el of document.querySelectorAll("[data-i18n]")) el.innerHTML = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of document.querySelectorAll("[data-i18n-ph]")) el.placeholder = t(el.dataset.i18nPh);
}
function storeLanguage(lang) { LANG = lang; try { localStorage.setItem("oncue.lang", lang); } catch (_) {} applyLang(); }

// ---------- map geometry ----------
// The part of the map canvas no floating card covers, in canvas pixels.
function usable(canvas) {
  const dpr = window.devicePixelRatio || 1, c = canvas.getBoundingClientRect();
  let x0 = 0, x1 = c.width, y0 = 0, y1 = c.height;
  const over = el => {
    if (!el || el.hidden) return null;
    const st = getComputedStyle(el);
    return st.display === "none" || st.position !== "absolute" ? null : el.getBoundingClientRect();
  };
  const Lc = over($("left")), R = over($("right")), D = over($("dock")), Tb = over($("toolbar"));
  if (Lc) x0 = Math.max(x0, Lc.right - c.left);
  if (R) x1 = Math.min(x1, R.left - c.left);
  if (D) y1 = Math.min(y1, D.top - c.top);
  if (Tb) y0 = Math.max(y0, Tb.bottom - c.top);
  if (x1 - x0 < 120) { x0 = 0; x1 = c.width; }
  if (y1 - y0 < 120) { y0 = 0; y1 = c.height; }
  return [x0 * dpr, y0 * dpr, x1 * dpr, y1 * dpr];
}
function transform(canvas) {
  const [ux0, uy0, ux1, uy1] = usable(canvas);
  if (!state.view) {
    const [x0, y0, x1, y1] = state.box, pad = 40;
    const scale = Math.min((ux1 - ux0 - 2 * pad) / Math.max(x1 - x0, 10), (uy1 - uy0 - 2 * pad) / Math.max(y1 - y0, 10));
    state.view = { scale, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
  }
  const v = state.view, ox = (ux0 + ux1) / 2, oy = (uy0 + uy1) / 2;
  state.usable = [ux0, uy0, ux1, uy1];
  return [p => [ox + (p[0] - v.cx) * v.scale, oy - (p[1] - v.cy) * v.scale], v.scale];
}
function boundsOf(points) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

// ---------- drawing ----------
function tag(ctx, x, y, text, color, dpr) {  // a small rounded label next to a road user
  ctx.font = `600 ${11 * dpr}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 12 * dpr, h = 18 * dpr, bx = x + 10 * dpr;
  let by = y - h - 8 * dpr;
  if (state.usable && by < state.usable[1] + 4 * dpr) by = y + 8 * dpr;
  ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(bx, by, w, h, 9 * dpr); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(text, bx + 6 * dpr, by + h / 2 + 0.5 * dpr);
  ctx.textBaseline = "alphabetic";
}
function polyline(ctx, T, pts) {
  ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = T(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
}
function box(ctx, T, xyh, size, fill, stroke, dashed, width = 1.5) {
  const [l, w] = size, c = Math.cos(xyh[2]), s = Math.sin(xyh[2]);
  const corners = [[l, w], [-l, w], [-l, -w], [l, -w]].map(([a, b]) => [xyh[0] + (a * c - b * s) / 2, xyh[1] + (a * s + b * c) / 2]);
  polyline(ctx, T, corners); ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.setLineDash(dashed ? [4, 3] : []); ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke(); ctx.setLineDash([]); }
  const [hx, hy] = T([xyh[0] + l / 2 * c, xyh[1] + l / 2 * s]);
  ctx.beginPath(); ctx.arc(hx, hy, 1.8, 0, 7); ctx.fillStyle = stroke || fill; ctx.fill();
}
// ---------- scene style (11-12, after the reference figure the user chose): a light map of dotted
// lanes and road edges, soft road users without outlines, each with a fading trail of its last
// TRAIL_S seconds; the egos outlined in their run's colour ----------
const TRAIL_S = 1.0;  // how much of each road user's past the trail shows
const TRAIL_ALPHA = 0.32;  // the trail's newest step; older steps fade towards zero
// three tiers, faintest to boldest: lane centre lines, painted road lines, road edges (user 11-12:
// the road's structure must read clearly)
const DOT_GAP = { lane: 4.5, road_line: 3, edge: 3.6 };  // CSS pixels between the dots of a line
const DOT_WIDTH = { lane: 1.8, road_line: 2, edge: 3.2 };  // dot diameter, CSS pixels
const MIN_DOT_RADIUS = 2.5;  // a pedestrian stays visible at any zoom, CSS pixels

function dotted(ctx, color, kind, dpr) {  // round caps turn a near-zero dash into a dot
  ctx.strokeStyle = color; ctx.lineWidth = DOT_WIDTH[kind] * dpr; ctx.lineCap = "round";
  ctx.setLineDash([0.001, DOT_GAP[kind] * dpr]);
}
function drawMap(ctx, T, map, dpr) {
  for (const f of map) {
    if (f.kind === "crosswalk") { polyline(ctx, T, f.points); ctx.closePath(); ctx.fillStyle = css("--cross"); ctx.fill(); }
  }
  for (const f of map) {
    if (f.kind === "crosswalk") continue;
    polyline(ctx, T, f.points);
    if (f.kind === "lane") dotted(ctx, css("--lane"), "lane", dpr);
    // yellow lines (they divide opposing traffic) in yellow, when the data names the paint
    else if (f.kind === "road_line") dotted(ctx, f.paint === "yellow" ? css("--yellow-line") : css("--road-line"), "road_line", dpr);
    else dotted(ctx, f.median ? css("--median") : css("--edge"), "edge", dpr);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}
// a road user as a soft rounded box (vehicles, cyclists) or a dot (pedestrians), no outline
function softShape(ctx, T, scale, dpr, xyh, size, type, fill) {
  const [x, y] = T(xyh);
  ctx.fillStyle = fill; ctx.beginPath();
  if (type === "pedestrian") { ctx.arc(x, y, Math.max(MIN_DOT_RADIUS * dpr, size[0] * scale / 2), 0, 7); ctx.fill(); return; }
  const [l, w] = [size[0] * scale, size[1] * scale];
  ctx.save(); ctx.translate(x, y); ctx.rotate(-xyh[2]);  // the canvas's y axis points down
  ctx.roundRect(-l / 2, -w / 2, l, w, Math.min(l, w) * 0.3); ctx.fill(); ctx.restore();
}
// the positions of the last TRAIL_S seconds before frame f, oldest faintest, then (with `present`)
// the position at f
function withTrail(ctx, xyhs, f, dt, draw, present = true) {
  const steps = Math.round(TRAIL_S / dt);
  for (let k = steps; k >= 1; k--) {
    const p = xyhs[f - k]; if (!p) continue;
    ctx.globalAlpha = TRAIL_ALPHA * (1 - k / (steps + 1)); draw(p);
  }
  ctx.globalAlpha = 1;
  if (present && xyhs[f]) draw(xyhs[f]);
}
// every recorded road user at frame f: vehicles light blue, pedestrians and cyclists tan, the
// occluding vehicle darker
function drawRoadUsers(ctx, T, scale, dpr, scene, f) {
  for (const [id, tr] of Object.entries(scene.tracks)) {
    if (!tr.xyh[f]) continue;
    const fill = id === scene.occluder ? css("--occluder") : tr.type === "vehicle" ? css("--other") : css("--walker");
    withTrail(ctx, tr.xyh, f, scene.dt, p => softShape(ctx, T, scale, dpr, p, tr.size, tr.type, fill));
  }
}
// an ego: a trail in its colour, then its box outlined in that colour on the map's colour
function drawEgo(ctx, T, scale, dpr, xyhs, f, dt, size, color, dashed = false) {
  withTrail(ctx, xyhs, f, dt, p => softShape(ctx, T, scale, dpr, p, size, "vehicle", color), false);
  if (xyhs[f]) box(ctx, T, xyhs[f], size, css("--map"), color, dashed, 2.2 * dpr);
}
// the visibility model's static occluders (buildings, walls, vegetation approximated from the map),
// when a scene's data has them: polygons as [outer ring, hole rings...]
function drawStatic(ctx, T, polygons) {
  if (!polygons) return;
  ctx.fillStyle = css("--static");
  for (const rings of polygons) {
    ctx.beginPath();
    for (const ring of rings) { ring.forEach((p, i) => { const [x, y] = T(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath(); }
    ctx.fill("evenodd");
  }
}
function drawScale(ctx, scale, dpr) {  // scale bar, in the uncovered part of the map
  const [, , sx, sy] = state.usable;
  const meters = [5, 10, 20, 50, 100].find(m => m * scale > 60 * dpr) || 100;
  ctx.fillStyle = css("--muted"); ctx.fillRect(sx - 20 * dpr - meters * scale, sy - 18 * dpr, meters * scale, 2 * dpr);
  ctx.font = `${11 * dpr}px system-ui`; ctx.textAlign = "right"; ctx.fillText(`${meters} m`, sx - 20 * dpr, sy - 24 * dpr);
}
function canvasOf(id) {
  const canvas = $(id), dpr = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  return [canvas, ctx, dpr];
}
const signed = x => `${x >= 0 ? "+" : ""}${x.toFixed(1)}`;

// ---------- playback, pan and zoom, help ----------
function seek(i) { if (!state.scene) return; state.frame = Math.max(0, Math.min(state.scene.times.length - 1, i)); draw(); }
let lastTick = 0;
function tick(now) {
  if (state.playing && state.scene && now - lastTick > state.scene.dt * 1000) {
    lastTick = now; state.frame = state.frame + 1 >= state.scene.times.length ? state.scene.current_idx : state.frame + 1; draw();
  }
  requestAnimationFrame(tick);
}
function togglePlay() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶"; $("play").setAttribute("aria-label", state.playing ? "Pause" : "Play");
}
function bindMap(canvas) {
  canvas.addEventListener("wheel", e => {
    if (!state.view) return; e.preventDefault();
    state.view.scale *= Math.exp(-e.deltaY * 0.0015); drawBev();
  }, { passive: false });
  let drag = null;
  canvas.addEventListener("pointerdown", e => { if (state.view) { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; } });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return; const dpr = window.devicePixelRatio || 1;
    state.view.cx -= (e.clientX - drag[0]) * dpr / state.view.scale; state.view.cy += (e.clientY - drag[1]) * dpr / state.view.scale;
    drag = [e.clientX, e.clientY]; drawBev();
  });
  canvas.addEventListener("pointerup", () => { drag = null; canvas.style.cursor = "grab"; });
  canvas.addEventListener("dblclick", () => { state.view = null; drawBev(); });
  window.addEventListener("resize", () => { state.view = null; draw(); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}
function showHelp(on) { $("help").hidden = !on; if (on) $("help").querySelector(".close").focus(); }
function renderHelp(html, onLang) {
  const pick = l => `<button class="lang" data-lang="${l}" aria-pressed="${l === LANG}">${l === "zh" ? "中文" : "English"}</button>`;
  $("help-body").innerHTML = `<div class="help-top"><span class="langs">${pick("en")}${pick("zh")}</span>` +
    `<button class="close" aria-label="Close">${LANG === "zh" ? "关闭" : "Close"}</button></div>` + html;
  for (const el of $("help-body").querySelectorAll(".lang")) el.onclick = () => onLang(el.dataset.lang);
}
// statistics helpers
function tally(values) { const m = new Map(); for (const v of values) m.set(v, (m.get(v) || 0) + 1); return [...m.entries()].sort((a, b) => b[1] - a[1]); }
function bars(entries, total, label = x => x, color = () => "var(--accent)") {
  if (!entries.length) return `<p class="muted">–</p>`;
  const top = Math.max(...entries.map(e => e[1]));
  return entries.map(([k, n]) => `<div class="brow"><span class="lab" title="${esc(label(k))}">${esc(label(k))}</span>` +
    `<span class="bar"><i style="width:${(100 * n / top).toFixed(1)}%;background:${color(k)}"></i></span>` +
    `<span class="num">${n}${total ? ` (${Math.round(100 * n / total)}%)` : ""}</span></div>`).join("");
}
function hist(values, lo, hi, bins, unit) {
  const vals = values.filter(v => v !== null && v !== undefined && Number.isFinite(v));
  if (!vals.length) return `<p class="muted">–</p>`;
  const counts = new Array(bins).fill(0), w = (hi - lo) / bins;
  for (const v of vals) counts[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)))]++;
  const top = Math.max(...counts);
  return `<div class="hist">${counts.map((c, i) => `<div style="height:${(100 * c / top).toFixed(1)}%" title="${(lo + i * w).toFixed(1)}–${(lo + (i + 1) * w).toFixed(1)}${unit}: ${c}"></div>`).join("")}</div>` +
    `<div class="hist-axis"><span>${lo}${unit}</span><span>${(lo + hi) / 2}${unit}</span><span>${hi}${unit}</span></div>`;
}
