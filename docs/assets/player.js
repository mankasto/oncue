// The project page's scene animations, drawn from the case browser's data (cases/data/case_N.js)
// with the browser's drawing helpers (common.js): one loop per occlusion family, and a player that
// runs one test twice, with masked and with unoccluded observation, side by side on one clock.
// window.DEMO (written by build_page.py) names the cases: {families: [...], compare: [...]}.
"use strict";
const PLANNER_NAMES = { reactive: "IDM-base", conservative: "IDM-conservative", occlusion_aware: "IDM-occlusion-aware",
  pdm_closed: "PDM-Closed", latentdriver: "LatentDriver" };
// the paper's construction families by site kind: the hiding vehicle parked at the kerb, beside a
// median or off the road all make a parked-vehicle crossing
const FAMILY_NAMES = { kerb: "Parked-vehicle crossing", median: "Parked-vehicle crossing", lot: "Parked-vehicle crossing",
  queue: "Queue-gap crossing", lane: "Intersecting lanes" };
const TEST_NAMES = { reaction: "a Reactive test", anticipation: "an Anticipatory test" };
const MODES = [  // observation mode, panel title, ego color (as IDM-base's runs in the case browser)
  ["masked", "Masked observation", "#dc2626"],
  ["see", "Unoccluded observation", "#16a34a"],
];
const BEFORE_REVEAL_S = 2.0;  // a loop starts this long before the reveal (or at the test start)
const STRIP_LEFT_PX = 48, STRIP_RIGHT_PX = 14;  // the speed strip's plot margins
const AFTER_CONTACT_S = 2.0;  // and ends this long after the reference contact (or a later collision)
const HOLD_S = 1.2;  // pause on the last frame before a loop restarts
const RATES = [1, 0.5, 0.25];  // playback speeds of the comparison player, cycled by its speed button
const MIN_WIDTH_M = 30, MIN_HEIGHT_M = 14;  // the smallest view, so short encounters are not over-zoomed
const PAD_PX = 16;  // space around the encounter, in CSS pixels
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const players = [];

function loadCase(i) {
  window.CASES = window.CASES || {};
  if (window.CASES[i]) return Promise.resolve(window.CASES[i]);
  return new Promise((ok, fail) => {
    const s = document.createElement("script");
    s.src = `cases/data/case_${i}.js`; s.onload = () => ok(window.CASES[i]); s.onerror = fail;
    document.head.appendChild(s);
  });
}

// ---------- one scene at one frame ----------
const seconds = (c, f) => (f - c.current_idx) * c.dt;
// a closed-loop run stops at a collision: its scene holds the last frame it reached
const runFrame = (c, run, f) => run ? Math.min(f, c.egos[run].until - 1) : f;
const egoOf = (c, run) => run ? c.egos[run] : c.egos.reference;
const targetPath = (c, run) => (run && c.egos[run].target) || c.target.xyh;
// a run: its planner knows the target; the reference: the target is revealed (the confirmed reveal t_r
// of the paper; visibility may lapse again after it, as at the far side of an intersection)
function knows(c, run, f) {
  if (!run) return f >= c.current_idx + Math.round(c.label.t_r / c.dt);
  const known = c.outcome[run].known_s;
  return known !== null && seconds(c, f) >= known - 1e-6;
}
function contactFrame(c, run) {
  const o = c.outcome[run];
  return o.hit ? c.current_idx + Math.round(o.contact_s / c.dt) : null;
}
const firstFrame = c => Math.max(c.current_idx, c.current_idx + Math.round((c.label.t_r - BEFORE_REVEAL_S) / c.dt));
function lastFrame(c, runs) {  // the reference contact (or the latest collision) plus a short tail
  const t = Math.max(c.label.t_c, ...runs.map(r => c.outcome[r].hit ? c.outcome[r].contact_s : 0));
  return Math.min(c.times.length - 1, c.current_idx + Math.round((t + AFTER_CONTACT_S) / c.dt));
}
function sceneBox(c, runs, start, end) {  // what every panel of a player shows: both egos and the target
  const pts = [];
  for (const run of runs.length ? runs : [null]) {
    for (let f = start; f <= end; f++) {
      const g = runFrame(c, run, f), e = egoOf(c, run).xyh[g], tg = targetPath(c, run)[g];
      if (e) pts.push(e); if (tg) pts.push(tg);
    }
  }
  let [x0, y0, x1, y1] = boundsOf(pts);
  const grow = (lo, hi, min) => { const d = Math.max(0, min - (hi - lo)) / 2; return [lo - d, hi + d]; };
  [x0, x1] = grow(x0, x1, MIN_WIDTH_M); [y0, y1] = grow(y0, y1, MIN_HEIGHT_M);
  return [x0, y0, x1, y1];
}
function fit(canvas, box, dpr) {
  const [x0, y0, x1, y1] = box, pad = PAD_PX * dpr;
  const scale = Math.min((canvas.width - 2 * pad) / (x1 - x0), (canvas.height - 2 * pad) / (y1 - y0));
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, ox = canvas.width / 2, oy = canvas.height / 2;
  return [p => [ox + (p[0] - cx) * scale, oy - (p[1] - cy) * scale], scale];
}
function sized(canvas) {
  const dpr = window.devicePixelRatio || 1, w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  return [ctx, dpr];
}
function pill(ctx, x, y, text, color, dpr) {  // a rounded label above a point, kept inside the canvas
  ctx.font = `600 ${11 * dpr}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 12 * dpr, h = 18 * dpr;
  const bx = Math.min(Math.max(4 * dpr, x + 10 * dpr), ctx.canvas.width - w - 4 * dpr);
  let by = y - h - 8 * dpr; if (by < 4 * dpr) by = y + 8 * dpr;
  ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(bx, by, w, h, 9 * dpr); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(text, bx + 6 * dpr, by + h / 2 + 0.5 * dpr);
  ctx.textBaseline = "alphabetic";
}
function scaleBar(ctx, scale, dpr) {
  const meters = [5, 10, 20, 50].find(m => m * scale > 50 * dpr) || 50, x1 = ctx.canvas.width - 14 * dpr, y = ctx.canvas.height - 14 * dpr;
  ctx.fillStyle = css("--muted"); ctx.fillRect(x1 - meters * scale, y, meters * scale, 2 * dpr);
  ctx.font = `${11 * dpr}px system-ui, sans-serif`; ctx.textAlign = "right"; ctx.fillText(`${meters} m`, x1, y - 6 * dpr);
}

// run: a closed-loop run key ("reactive_masked", ...), or null for the reference ego of the log
// color: a CSS color or a page token ("--ego"), read at each draw so a theme change applies
// start: the loop's first frame, where the ego's trail begins
function drawScene(canvas, c, run, f, start, view, color, labels = true) {
  if (color.startsWith("--")) color = css(color);
  const [ctx, dpr] = sized(canvas), [T, scale] = fit(canvas, view, dpr), g = runFrame(c, run, f);
  ctx.fillStyle = css("--map"); ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawStatic(ctx, T, c.static);
  drawMap(ctx, T, c.map, dpr);
  drawRoadUsers(ctx, T, scale, dpr, c, g);
  const ego = egoOf(c, run), e = ego.xyh[g], target = targetPath(c, run)[g], seen = knows(c, run, g);
  ctx.globalAlpha = .3; polyline(ctx, T, ego.xyh.slice(start, g + 1)); ctx.strokeStyle = color; ctx.lineWidth = 1.4 * dpr; ctx.stroke();
  ctx.globalAlpha = 1;
  if (e && target) {  // the ego's sight line: green once the target is known (visible, for the reference)
    polyline(ctx, T, [e, target]); ctx.strokeStyle = seen ? css("--good") : css("--muted"); ctx.lineWidth = 1.2 * dpr;
    ctx.setLineDash([2 * dpr, 4 * dpr]); ctx.stroke(); ctx.setLineDash([]);
  }
  if (target) {  // a soft halo, so a pedestrian of half a meter still shows at the page's zoom
    const [x, y] = T(target); ctx.globalAlpha = .18; ctx.fillStyle = css("--target");
    ctx.beginPath(); ctx.arc(x, y, Math.max(10 * dpr, 1.6 * scale), 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    const shape = c.kind === "pedestrian" ? "pedestrian" : "vehicle";
    withTrail(ctx, targetPath(c, run), g, c.dt, p => softShape(ctx, T, scale, dpr, p, c.target.size, shape, css("--target")), false);
    box(ctx, T, target, c.target.size, seen ? css("--target") : null, css("--target"), !seen, 2 * dpr);
  }
  drawEgo(ctx, T, scale, dpr, ego.xyh, g, c.dt, c.ego_size, color);
  const hitAt = run && contactFrame(c, run);
  if (hitAt !== null && hitAt !== undefined && f >= hitAt && target) {
    const [x, y] = T(target); ctx.strokeStyle = css("--danger"); ctx.lineWidth = 2.5 * dpr;
    ctx.beginPath(); ctx.arc(x, y, Math.max(14 * dpr, 2.2 * scale), 0, 7); ctx.stroke();
  }
  if (!labels) return;
  if (target) {
    const [x, y] = T(target);
    const text = hitAt !== null && hitAt !== undefined && f >= hitAt ? "collision"
      : run && run.endsWith("_see") ? "known through the occluder" : seen ? (run ? "seen" : "exposed") : "hidden";
    pill(ctx, x, y, text, text === "collision" ? css("--danger") : css("--target"), dpr);
  }
  const clock = `t = ${seconds(c, f) >= 0 ? "+" : ""}${seconds(c, f).toFixed(1)} s`;  // on a backing, over any map line
  ctx.font = `600 ${12 * dpr}px system-ui, sans-serif`; ctx.textAlign = "left";
  ctx.fillStyle = css("--map"); ctx.globalAlpha = .85;
  ctx.beginPath(); ctx.roundRect(6 * dpr, 6 * dpr, ctx.measureText(clock).width + 12 * dpr, 20 * dpr, 6 * dpr); ctx.fill();
  ctx.globalAlpha = 1; ctx.fillStyle = css("--muted"); ctx.fillText(clock, 12 * dpr, 20 * dpr);
  scaleBar(ctx, scale, dpr);
}

// the two runs' speeds on one time axis, with the reveal, the reference contact and the play head
function drawStrip(canvas, p) {
  const [ctx, dpr] = sized(canvas), c = p.c, px = v => v * dpr;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const left = px(STRIP_LEFT_PX), right = px(STRIP_RIGHT_PX), top = px(20), bottom = canvas.height - px(20);
  const t0 = seconds(c, p.start), t1 = seconds(c, p.end);
  const X = t => left + (canvas.width - left - right) * (t - t0) / (t1 - t0);
  const speeds = p.runs.map(r => c.egos[r].v.slice(p.start, c.egos[r].until).slice(0, p.end - p.start + 1));
  const vmax = Math.max(5, ...speeds.flat()), Y = v => bottom - (bottom - top) * v / vmax;
  ctx.fillStyle = css("--chip"); ctx.fillRect(X(t0), top, X(c.label.t_r) - X(t0), bottom - top);  // the target hidden
  ctx.font = `${11 * dpr}px system-ui, sans-serif`; ctx.textBaseline = "middle";
  for (const v of [0, vmax]) {
    ctx.strokeStyle = css("--line"); ctx.lineWidth = px(1); ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(canvas.width - right, Y(v)); ctx.stroke();
    ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.fillText(`${v.toFixed(0)} m/s`, left - px(6), Y(v));
  }
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "center";
  for (let t = Math.ceil(t0 + 1e-9); t <= t1 + 1e-9; t += 1) { ctx.fillStyle = css("--muted"); ctx.fillText(`+${t} s`, X(t), bottom + px(15)); }
  ctx.textAlign = "left";
  let lastRight = -Infinity;  // event labels side by side when the two times are close
  for (const [t, label, col] of [[c.label.t_r, "exposure", css("--target")], [c.label.t_c, "reference contact", css("--danger")]]) {
    ctx.strokeStyle = col; ctx.lineWidth = px(1.2); ctx.beginPath(); ctx.moveTo(X(t), top - px(4)); ctx.lineTo(X(t), bottom); ctx.stroke();
    const lx = Math.max(X(t) + px(4), lastRight + px(6));
    ctx.fillStyle = col; ctx.fillText(label, lx, top - px(7)); lastRight = lx + ctx.measureText(label).width;
  }
  p.runs.forEach((r, k) => {
    ctx.beginPath(); speeds[k].forEach((v, i) => { const x = X(t0 + i * c.dt), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = p.colors[k]; ctx.lineWidth = px(2.2); ctx.stroke();
    const o = c.outcome[r];
    if (o.hit) { ctx.fillStyle = css("--danger"); ctx.beginPath(); ctx.arc(X(o.contact_s), Y(speeds[k][speeds[k].length - 1]), px(4.5), 0, 7); ctx.fill(); }
  });
  const x = X(seconds(c, p.frame)); ctx.strokeStyle = css("--accent"); ctx.lineWidth = px(2); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
}

// ---------- players ----------
function render(p) {
  if (!p.c) return;
  for (const panel of p.panels) drawScene(panel.canvas, p.c, panel.run, p.frame, p.start, p.box, panel.color);
  if (p.strip) drawStrip(p.strip, p);
  if (p.button) { p.button.textContent = p.playing ? "❚❚" : "▶"; p.button.setAttribute("aria-label", p.playing ? "Pause" : "Play"); }
}
function show(p, c, runs) {  // start a player on a case
  p.c = c; p.runs = runs; p.start = firstFrame(c); p.end = lastFrame(c, runs);
  p.box = sceneBox(c, runs, p.start, p.end);
  p.frame = reducedMotion ? Math.min(p.end, c.current_idx + Math.round(c.label.t_c / c.dt)) : p.start;
  p.holdFrom = 0; render(p);
}
function toggle(p) { p.playing = !p.playing; render(p); }
function observe(p, el) {
  p.el = el; p.visible = false; p.playing = !reducedMotion; p.last = 0; players.push(p);
  new IntersectionObserver(es => { p.visible = es[0].isIntersecting; }).observe(el);
}
function animate(now) {
  for (const p of players) {
    if (!p.c || !p.visible || !p.playing || !p.last) { p.last = now; continue; }
    const frameMs = p.c.dt * 1000 / (p.rate || 1);  // real time at the chosen speed, even when frames come late
    const steps = Math.floor((now - p.last) / frameMs);
    if (steps < 1) continue;
    p.last += steps * frameMs;
    if (p.frame < p.end) p.frame = Math.min(p.end, p.frame + steps);
    else if (!p.holdFrom) { p.holdFrom = now; continue; }
    else if (now - p.holdFrom < HOLD_S * 1000) continue;
    else { p.holdFrom = 0; p.frame = p.start; }
    render(p);
  }
  requestAnimationFrame(animate);
}

function familyLoops(root, picks, cases) {
  root.innerHTML = picks.map(() => `<figure><canvas class="scene" role="img"></canvas><figcaption></figcaption></figure>`).join("");
  picks.forEach((pick, k) => {
    const fig = root.children[k], c = cases[pick.case], canvas = fig.querySelector("canvas");
    const p = { panels: [{ canvas, run: null, color: "--ego" }] };
    canvas.setAttribute("aria-label", `${FAMILY_NAMES[pick.kind]}: animation of a ${c.kind} exposed to the reference ego`);
    fig.querySelector("figcaption").innerHTML = `<b>${FAMILY_NAMES[pick.kind]}</b>, ${c.kind}, W = ${c.label.W.toFixed(1)} s`;
    canvas.onclick = () => toggle(p);
    observe(p, fig); show(p, c, []);
  });
}

function comparePlayer(root, picks, cases) {
  const target = c => c.kind, family = c => FAMILY_NAMES[c.label.site_kind];
  root.innerHTML = `<div class="cmp-panels">${MODES.map(([, title]) =>
    `<figure class="panel"><div class="panel-head"><b>${title}</b><span class="badge"></span></div><canvas class="scene" role="img"></canvas></figure>`).join("")}</div>
    <div class="cmp-bar"><button class="play" aria-label="Pause">❚❚</button><button class="speed" aria-label="Playback speed">1×</button><canvas class="strip" aria-label="Ego speeds over time"></canvas></div>
    <p class="cmp-caption"></p>
    <div class="thumbs" role="tablist">${picks.map((pick, k) => {
      const c = cases[pick.case];
      return `<button class="thumb" role="tab" data-k="${k}"><canvas></canvas><span>${target(c)[0].toUpperCase() + target(c).slice(1)}, ${family(c).toLowerCase()}</span></button>`;
    }).join("")}</div>`;
  const canvases = root.querySelectorAll(".panel canvas"), badges = root.querySelectorAll(".badge");
  const p = { panels: MODES.map(([, , color], k) => ({ canvas: canvases[k], run: null, color })), colors: MODES.map(m => m[2]),
    strip: root.querySelector(".strip"), button: root.querySelector(".play") };
  p.button.onclick = () => toggle(p);
  const speed = root.querySelector(".speed");
  speed.onclick = () => {
    p.rate = RATES[(RATES.indexOf(p.rate || 1) + 1) % RATES.length];
    speed.textContent = `${p.rate}×`; speed.setAttribute("aria-label", `Playback speed ${p.rate}×`);
  };
  for (const panel of p.panels) panel.canvas.onclick = () => toggle(p);
  let drag = false;
  const scrub = e => {
    const r = p.strip.getBoundingClientRect(), share = (e.clientX - r.left - STRIP_LEFT_PX) / (r.width - STRIP_LEFT_PX - STRIP_RIGHT_PX);
    p.frame = Math.max(p.start, Math.min(p.end, p.start + Math.round(share * (p.end - p.start)))); p.playing = false; render(p);
  };
  p.strip.addEventListener("pointerdown", e => { drag = true; p.strip.setPointerCapture(e.pointerId); scrub(e); });
  p.strip.addEventListener("pointermove", e => { if (drag) scrub(e); });
  p.strip.addEventListener("pointerup", () => { drag = false; });
  const select = k => {
    const pick = picks[k], c = cases[pick.case], runs = MODES.map(([m]) => `${pick.planner}_${m}`);
    runs.forEach((r, j) => {
      p.panels[j].run = r; const o = c.outcome[r];
      badges[j].className = `badge ${o.hit ? "hit" : "safe"}`; badges[j].textContent = o.hit ? `hits the ${c.kind}` : `avoids the ${c.kind}`;
      p.panels[j].canvas.setAttribute("aria-label", `${MODES[j][1]}: ${PLANNER_NAMES[pick.planner]} ${o.hit ? "hits" : "avoids"} the ${c.kind}`);
    });
    root.querySelector(".cmp-caption").innerHTML = compareCaption(c, pick.planner);
    for (const b of root.querySelectorAll(".thumb")) b.setAttribute("aria-selected", b.dataset.k === String(k));
    show(p, c, runs);
  };
  for (const b of root.querySelectorAll(".thumb")) b.onclick = () => { p.playing = !reducedMotion; select(+b.dataset.k); };
  observe(p, root); select(0);
  root.querySelectorAll(".thumb canvas").forEach((canvas, k) => {  // each scene at its collision, without labels
    const pick = picks[k], c = cases[pick.case], run = `${pick.planner}_masked`, start = firstFrame(c);
    const thumb = { canvas, c, run, start, box: sceneBox(c, [run], start, lastFrame(c, [run])), frame: contactFrame(c, run) ?? start };
    thumbs.push(thumb); drawThumb(thumb);
  });
}
const thumbs = [];
function drawThumb(th) { drawScene(th.canvas, th.c, th.run, th.frame, th.start, th.box, MODES[0][2], false); }

function compareCaption(c, planner) {
  const lab = c.label, m = c.outcome[`${planner}_masked`], s = c.outcome[`${planner}_see`], f = v => (+v).toFixed(1);
  const masked = m.hit
    ? `sees the ${c.kind} at +${f(m.known_s)} s${m.brake_s !== null ? `, brakes at +${f(m.brake_s)} s` : ""} and hits it at ${f(m.contact_speed)} m/s`
    : `avoids it`;
  const see = s.hit ? `hits it at ${f(s.contact_speed)} m/s` : `${s.brake_s !== null ? `brakes from +${f(s.brake_s)} s and ` : ""}avoids it`;
  return `<b>${PLANNER_NAMES[planner]} on ${TEST_NAMES[lab.test]}.</b> ${FAMILY_NAMES[lab.site_kind]} with a ${c.kind}, requested window ` +
    `W = ${f(lab.W)} s: the ${c.kind} is hidden from the reference ego until +${f(lab.t_r)} s and meets it at +${f(lab.t_c)} s. ` +
    `With masked observation the planner ${masked}; with unoccluded observation it ${see}. Times are seconds after the test starts.`;
}

async function startDemos() {
  const demo = window.DEMO, wanted = [...new Set([...demo.families, ...demo.compare].map(d => d.case))];
  const cases = Object.fromEntries(await Promise.all(wanted.map(async i => [i, await loadCase(i)])));
  familyLoops(document.getElementById("families"), demo.families, cases);
  comparePlayer(document.getElementById("compare"), demo.compare, cases);
  const redraw = () => { for (const p of players) render(p); thumbs.forEach(drawThumb); };
  window.addEventListener("resize", redraw);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  requestAnimationFrame(animate);
}
startDemos();
