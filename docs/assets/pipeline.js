// The Method section's step-by-step build of one test, drawn from the case's own data with the
// players' helpers (common.js, player.js): the recorded scene, the request, occlusion, the two timed
// waypoints, the hidden start, the generated motion, its label and the closed loop. The steps'
// titles and captions are in the page (#pipeline-steps); window.PIPELINE names the case.
"use strict";
(() => {
  const root = document.getElementById("pipeline");
  if (!root || !window.PIPELINE) return;
  const STEPS = JSON.parse(document.getElementById("pipeline-steps").textContent);
  const RUNS = [["reactive_masked", "#dc2626", "masked"], ["reactive_see", "#16a34a", "unoccluded"]];
  const SHADOW_M = 60;  // how far the occluder's shadow is drawn from the reference ego
  const HOLD_MS = 1600;  // pause at a step's last frame before autoplay moves on
  const TAIL_S = 1.5;  // frames shown after the reference contact
  const canvas = root.querySelector("canvas"), caption = root.querySelector(".pipe-caption");
  const chips = root.querySelector(".pipe-steps"), play = root.querySelector(".pipe-play");
  let c = null, view = null, step = 0, frame = 0, last = 0, holdFrom = 0, playing = !reducedMotion;

  chips.innerHTML = STEPS.map((s, k) => `<button type="button" data-k="${k}"><b>${k + 1}</b> ${s.title}</button>`).join("");
  chips.onclick = e => { const b = e.target.closest("button"); if (b) go(+b.dataset.k); };
  root.querySelector(".pipe-prev").onclick = () => go((step + STEPS.length - 1) % STEPS.length);
  root.querySelector(".pipe-next").onclick = () => go((step + 1) % STEPS.length);
  play.onclick = () => { playing = !playing; showPlay(); };
  const showPlay = () => { play.textContent = playing ? "❚❚" : "▶"; play.setAttribute("aria-label", playing ? "Pause" : "Play"); };

  // frame ranges: the test's start, the shown start (as the players: 2 s before exposure), the
  // requested exposure and contact, and a short tail
  const f0 = () => c.current_idx, fs = () => firstFrame(c), fe = () => c.current_idx + Math.round(c.label.t_r / c.dt);
  const fc = () => c.current_idx + Math.round(c.label.t_c / c.dt);
  const fEnd = () => Math.min(c.times.length - 1, fc() + Math.round(TAIL_S / c.dt));
  const RANGES = [() => [fs(), fEnd()], () => [fs(), fc()], () => [fs(), fe()], () => [fe(), fe()],
    () => [fs(), fs()], () => [fs(), fEnd()], () => [fc(), fc()], () => [fs(), fEnd()]];
  const range = () => RANGES[step]();

  function go(k) {
    step = k; [frame] = range(); holdFrom = 0; last = 0;
    caption.innerHTML = STEPS[k].caption;
    for (const b of chips.children) b.setAttribute("aria-pressed", b.dataset.k === String(k));
    draw();
  }

  function shadow(ego, occluder, size) {  // the occluder's shadow from the ego: its two outer corners pushed away
    const [l, w] = size, cos = Math.cos(occluder[2]), sin = Math.sin(occluder[2]);
    const corners = [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]
      .map(([a, b]) => [occluder[0] + a * cos - b * sin, occluder[1] + a * sin + b * cos]);
    const base = Math.atan2(occluder[1] - ego[1], occluder[0] - ego[0]);
    const turn = p => { let d = Math.atan2(p[1] - ego[1], p[0] - ego[0]) - base; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; };
    const sorted = corners.sort((p, q) => turn(p) - turn(q)), lo = sorted[0], hi = sorted[3];
    const far = p => { const d = Math.hypot(p[0] - ego[0], p[1] - ego[1]); return [ego[0] + (p[0] - ego[0]) / d * SHADOW_M, ego[1] + (p[1] - ego[1]) / d * SHADOW_M]; };
    return [lo, hi, far(hi), far(lo)];
  }

  function ring(ctx, T, p, color, radius, dpr, filled = false) {
    const [x, y] = T(p); ctx.beginPath(); ctx.arc(x, y, radius * dpr, 0, 7);
    if (filled) { ctx.fillStyle = color; ctx.fill(); } else { ctx.strokeStyle = color; ctx.lineWidth = 2.5 * dpr; ctx.stroke(); }
  }
  function dashedPath(ctx, T, points, color, dpr) {
    polyline(ctx, T, points); ctx.strokeStyle = color; ctx.lineWidth = 1.6 * dpr; ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.stroke(); ctx.setLineDash([]);
  }
  function bars(ctx, dpr, rows) {  // the oracle's two shares of safe maneuvers, top right
    const w = 230 * dpr, h = (18 + rows.length * 30) * dpr, x = ctx.canvas.width - w - 12 * dpr, y = 12 * dpr;
    ctx.fillStyle = css("--map"); ctx.globalAlpha = .92; ctx.beginPath(); ctx.roundRect(x, y, w, h, 8 * dpr); ctx.fill(); ctx.globalAlpha = 1;
    ctx.font = `${11.5 * dpr}px system-ui, sans-serif`; ctx.textAlign = "left";
    rows.forEach(([label, share], k) => {
      const ty = y + (22 + k * 30) * dpr;
      ctx.fillStyle = css("--muted"); ctx.fillText(`${label}: ${Math.round(share * 100)}%`, x + 10 * dpr, ty);
      ctx.fillStyle = css("--line"); ctx.fillRect(x + 10 * dpr, ty + 5 * dpr, w - 20 * dpr, 5 * dpr);
      ctx.fillStyle = css("--good"); ctx.fillRect(x + 10 * dpr, ty + 5 * dpr, (w - 20 * dpr) * share, 5 * dpr);
    });
  }

  function draw() {
    if (!c) return;
    const [ctx, dpr] = sized(canvas), [T, scale] = fit(canvas, view, dpr);
    const ref = c.egos.reference.xyh, target = c.target.xyh, occluder = c.tracks[c.occluder];
    const shown = frame;
    ctx.fillStyle = css("--map"); ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawStatic(ctx, T, c.static); drawMap(ctx, T, c.map, dpr);
    if (step === 2 || step === 4) {  // the parked vehicle's shadow from the reference ego
      ctx.fillStyle = css("--muted"); ctx.globalAlpha = .16; ctx.beginPath();
      shadow(ref[shown], occluder.xyh[shown], occluder.size).forEach((p, i) => { const [x, y] = T(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    }
    drawRoadUsers(ctx, T, scale, dpr, c, shown);
    if (step >= 2 && step <= 4) {  // the occluder outlined, the target's route dashed
      box(ctx, T, occluder.xyh[shown], occluder.size, null, css("--accent"), false, 2.5 * dpr);
      dashedPath(ctx, T, target.slice(f0(), fEnd() + 1).filter(Boolean), css("--target"), dpr);  // from the hidden start
    }
    if (step === 7) {  // the closed loop: one planner, masked and unoccluded
      for (const [run, color] of RUNS) {
        const g = runFrame(c, run, frame), tg = targetPath(c, run)[g];
        if (tg) { const shape = c.kind === "pedestrian" ? "pedestrian" : "vehicle"; softShape(ctx, T, scale, dpr, tg, c.target.size, shape, css("--target")); }
        drawEgo(ctx, T, scale, dpr, c.egos[run].xyh, g, c.dt, c.ego_size, color);
        const hit = contactFrame(c, run);
        if (hit !== null && frame >= hit && tg) ring(ctx, T, tg, css("--danger"), 16, dpr);
      }
      RUNS.forEach(([run, color, mode], k) => {
        ctx.font = `600 ${12 * dpr}px system-ui, sans-serif`; ctx.textAlign = "left"; ctx.fillStyle = color;
        ctx.fillText(`IDM-base, ${mode}: ${c.outcome[run].hit ? "hits" : "avoids"}`, 12 * dpr, (44 + k * 18) * dpr);
      });
    } else {
      const g = Math.min(shown, ref.length - 1);
      drawEgo(ctx, T, scale, dpr, ref, g, c.dt, c.ego_size, css("--ego"));
      if (step >= 5) {  // the generated target, hidden until its exposure
        const seen = g >= fe(), tg = target[g];
        if (tg) {
          polyline(ctx, T, [ref[g], tg]); ctx.strokeStyle = seen ? css("--good") : css("--muted"); ctx.lineWidth = 1.2 * dpr;
          ctx.setLineDash([2 * dpr, 4 * dpr]); ctx.stroke(); ctx.setLineDash([]);
          const shape = c.kind === "pedestrian" ? "pedestrian" : "vehicle";
          withTrail(ctx, target, g, c.dt, p => softShape(ctx, T, scale, dpr, p, c.target.size, shape, css("--target")), false);
          box(ctx, T, tg, c.target.size, seen ? css("--target") : null, css("--target"), !seen, 2 * dpr);
          if (step === 5) pill(ctx, ...T(tg), seen ? "exposed" : "hidden", css("--target"), dpr);
        }
      }
    }
    // marks that the steps add one by one
    const pc = target[fc()], pe = target[fe()], start = target[f0()];
    if (step === 1 && frame === fc()) { ring(ctx, T, pc, css("--danger"), 12, dpr); pill(ctx, ...T(pc), `contact point C, ${c.label.t_c.toFixed(1)} s`, css("--danger"), dpr); }
    if (step === 2 && frame === fe()) {
      polyline(ctx, T, [ref[fe()], pe]); ctx.strokeStyle = css("--good"); ctx.lineWidth = 1.6 * dpr; ctx.stroke();
      ring(ctx, T, pe, css("--good"), 9, dpr); pill(ctx, ...T(pe), `first visible point, ${c.label.t_r.toFixed(1)} s`, css("--good"), dpr);
    }
    if (step === 3 || step === 4 || step === 5) {
      ring(ctx, T, pe, css("--good"), 9, dpr); ring(ctx, T, pc, css("--danger"), 9, dpr);
      if (step === 3) {
        pill(ctx, ...T(pe), `exposure waypoint, ${c.label.t_r.toFixed(1)} s`, css("--good"), dpr);
        pill(ctx, ...T(pc), `contact waypoint, ${c.label.t_c.toFixed(1)} s`, css("--danger"), dpr);
      }
    }
    if (step === 4) {
      box(ctx, T, start, c.target.size, null, css("--target"), true, 2.2 * dpr);
      pill(ctx, ...T(start), "hidden start", css("--target"), dpr);
    }
    if (step === 6) {
      ring(ctx, T, pc, css("--danger"), 9, dpr);
      bars(ctx, dpr, [["Safe responses after exposure", c.label.ma], ["Safe responses from early visibility", c.label.ma0]]);
      pill(ctx, ...T(target[fc()]), c.label.test === "anticipation" ? "Anticipatory test" : "Reactive test", css("--accent"), dpr);
    }
    const clock = `t = +${Math.max(0, seconds(c, frame)).toFixed(1)} s`;
    ctx.font = `600 ${12 * dpr}px system-ui, sans-serif`; ctx.textAlign = "left"; ctx.fillStyle = css("--muted"); ctx.fillText(clock, 12 * dpr, 22 * dpr);
    scaleBar(ctx, scale, dpr);
  }

  function animate(now) {
    if (c && root.dataset.visible === "1") {
      const [a, b] = range();
      if (!last) last = now;
      const steps = Math.floor((now - last) / (c.dt * 1000));
      if (steps >= 1) {
        last += steps * c.dt * 1000;
        if (frame < b) { frame = Math.min(b, frame + steps); draw(); }
        else if (!holdFrom) holdFrom = now;
        else if (now - holdFrom > HOLD_MS) {
          holdFrom = 0;
          if (playing) go((step + 1) % STEPS.length); else { frame = a; draw(); }
        }
      }
    } else last = now;
    requestAnimationFrame(animate);
  }

  new IntersectionObserver(es => { root.dataset.visible = es[0].isIntersecting ? "1" : "0"; }).observe(root);
  loadCase(window.PIPELINE.case).then(found => {
    c = found;
    const pts = [];
    const ends = [fs(), fEnd()];
    pts.push(c.target.xyh[f0()]);  // the hidden start
    for (let f = ends[0]; f <= ends[1]; f++) {
      for (const xyh of [c.egos.reference.xyh, c.target.xyh, c.egos.reactive_masked.xyh, c.egos.reactive_see.xyh]) if (xyh[f]) pts.push(xyh[f]);
    }
    let [x0, y0, x1, y1] = boundsOf(pts);
    const grow = (lo, hi, min) => { const d = Math.max(0, min - (hi - lo)) / 2; return [lo - d, hi + d]; };
    [x0, x1] = grow(x0, x1, MIN_WIDTH_M); [y0, y1] = grow(y0, y1, MIN_HEIGHT_M);
    view = [x0, y0, x1, y1];
    showPlay(); go(0);
    window.addEventListener("resize", draw);
    requestAnimationFrame(animate);
  });
})();
