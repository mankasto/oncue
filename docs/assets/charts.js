// The page's own drawings of the paper's Figures 3 and 6, from the data the paper's figure code
// plots (figure_data.py writes it, build_page.py puts it in window.CHARTS). Hovering a mark shows
// its numbers; clicking a planner in Figure 6's legend hides or shows it.
"use strict";
(() => {
  if (!window.CHARTS) return;
  const NS = "http://www.w3.org/2000/svg";
  const KINDS = ["pedestrian", "cyclist", "vehicle"];
  const title = kind => kind[0].toUpperCase() + kind.slice(1);
  const signed = (v, digits = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);

  function el(parent, tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    parent.appendChild(node);
    return node;
  }
  const scale = (d0, d1, r0, r1) => v => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  const path = points => points.map(([x, y], k) => `${k ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const band = (xs, lows, highs) => path([...xs.map((x, k) => [x, highs[k]]), ...xs.map((x, k) => [x, lows[k]]).reverse()]) + "Z";

  function chart(id, width, height) {
    const figure = document.getElementById(id);
    const body = figure.querySelector(".chart-body");
    const svg = el(body, "svg", { viewBox: `0 0 ${width} ${height}` });
    const tip = document.createElement("div");
    tip.className = "chart-tip";
    body.appendChild(tip);
    const hover = (mark, text) => {
      mark.classList.add("mark");
      mark.addEventListener("pointerenter", () => { tip.textContent = text; tip.style.opacity = 1; });
      mark.addEventListener("pointermove", e => {
        const box = body.getBoundingClientRect();
        tip.style.left = `${Math.min(e.clientX - box.left + 12, box.width - tip.offsetWidth - 4)}px`;
        tip.style.top = `${e.clientY - box.top - tip.offsetHeight - 10}px`;
      });
      mark.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
    };
    return { svg, hover };
  }

  function axes(svg, frame, xs, ys, xTicks, yTicks, xLabels) {
    const [left, top, w, h] = frame;
    for (const t of yTicks) {
      el(svg, "line", { x1: left, x2: left + w, y1: ys(t), y2: ys(t), class: "grid" });
      el(svg, "text", { x: left - 6, y: ys(t) + 4, class: "tick", "text-anchor": "end" }, t === 0 ? "0" : String(t).replace("-", "−"));
    }
    xTicks.forEach((t, k) => el(svg, "text", { x: xs(t), y: top + h + 15, class: "tick", "text-anchor": "middle" }, xLabels ? xLabels[k] : String(t)));
    el(svg, "line", { x1: left, x2: left, y1: top, y2: top + h, class: "axis" });
    el(svg, "line", { x1: left, x2: left + w, y1: top + h, y2: top + h, class: "axis" });
  }

  // ---- Figure 3: temporal control ------------------------------------------------------------
  const BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"];
  function blues(share) {  // matplotlib's Blues at a share in [0, 1]
    const at = Math.max(0, Math.min(1, share)) * (BLUES.length - 1), k = Math.min(BLUES.length - 2, Math.floor(at));
    const mix = (a, b, f) => Math.round(a + (b - a) * f);
    const [p, q] = [BLUES[k], BLUES[k + 1]].map(c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)));
    return `rgb(${p.map((v, i) => mix(v, q[i], at - k)).join(",")})`;
  }
  const COUNT_MAX = 1000;  // the colour scale: 1 to 1000 outputs per marker, log
  const RAW = "#42949E", SELECTED = "#155797", ON_TIME = "#DA853D";

  function timeControl(data) {
    const { svg, hover } = chart("chart-time-control", 1000, 262);
    // (a) and (b) side by side, each a row of three panels (pedestrian, cyclist, vehicle)
    const top = 64, size = 130, left0 = 62, step = 148, limit = 0.45;
    el(svg, "text", { x: 8, y: 20, class: "panel-title" }, "(a) Exposure and contact timing");
    KINDS.forEach((kind, i) => {
      const left = left0 + i * step, cell = data.timing[kind];
      const xs = scale(-limit, limit, left, left + size), ys = scale(-limit, limit, top + size, top);
      const clip = el(el(svg, "defs"), "clipPath", { id: `clip-${kind}` });
      el(clip, "rect", { x: left, y: top, width: size, height: size });
      const on = data.on_time_s;
      el(svg, "path", { d: path([[xs(-limit), ys(-limit - on)], [xs(limit), ys(limit - on)], [xs(limit), ys(limit + on)], [xs(-limit), ys(-limit + on)]]) + "Z",
        class: "tolerance", "clip-path": `url(#clip-${kind})` });
      el(svg, "line", { x1: xs(-limit), y1: ys(-limit), x2: xs(limit), y2: ys(limit), class: "diagonal" });
      el(svg, "line", { x1: xs(0), x2: xs(0), y1: top, y2: top + size, class: "zero" });
      el(svg, "line", { x1: left, x2: left + size, y1: ys(0), y2: ys(0), class: "zero" });
      axes(svg, [left, top, size, size], xs, ys, [-0.4, 0, 0.4], i === 0 ? [-0.4, 0, 0.4] : []);
      for (const [x, y, n] of cell.points) {
        const dot = el(svg, "circle", { cx: xs(x), cy: ys(y), r: 5.6, fill: blues(Math.log10(n) / Math.log10(COUNT_MAX)), class: "count-dot",
          "clip-path": `url(#clip-${kind})` });  // as in the paper: errors beyond the axes are cut off
        hover(dot, `${title(kind)}: exposure error ${signed(x, 1)} s, contact error ${signed(y, 1)} s, ${n.toLocaleString("en")} outputs`);
      }
      el(svg, "rect", { x: xs(-on), y: ys(on), width: xs(on) - xs(-on), height: ys(-on) - ys(on), class: "on-time", stroke: ON_TIME });
      el(svg, "text", { x: left + size / 2, y: top - 22, class: "kind", "text-anchor": "middle" }, title(kind));
      el(svg, "text", { x: left + size / 2, y: top - 7, class: "share", "text-anchor": "middle" }, `${cell.on_time_pct.toFixed(1)}% on time`);
    });
    el(svg, "text", { x: left0 + step + size / 2, y: top + size + 34, class: "label", "text-anchor": "middle" }, "Exposure-time error (s)");
    el(svg, "text", { transform: `translate(18 ${top + size / 2}) rotate(-90)`, class: "label", "text-anchor": "middle" }, "Contact-time error (s)");
    // the count colour bar, upright between (a) and (b)
    const grad = el(el(svg, "defs"), "linearGradient", { id: "count-gradient", x1: 0, y1: 1, x2: 0, y2: 0 });
    BLUES.forEach((c, k) => el(grad, "stop", { offset: k / (BLUES.length - 1), "stop-color": c }));
    const barLeft = left0 + 2 * step + size + 14, barTop = top + 14, barHeight = size - 14;
    el(svg, "rect", { x: barLeft, y: barTop, width: 9, height: barHeight, fill: "url(#count-gradient)", class: "bar" });
    el(svg, "text", { x: barLeft + 4, y: barTop - 7, class: "tick", "text-anchor": "middle" }, "Count");
    [1, 10, 100, 1000].forEach((n, k) => el(svg, "text", { x: barLeft + 13, y: barTop + barHeight - (k / 3) * barHeight + 4, class: "tick" }, n.toLocaleString("en")));

    // (b) window error by requested W
    const left0b = 614, widthB = 118, stepB = 132;
    el(svg, "text", { x: left0b - 50, y: 20, class: "panel-title" }, "(b) Reaction-window control");
    KINDS.forEach((kind, i) => {
      const left = left0b + i * stepB, rows = data.windows[kind];
      const xs = scale(0.36, 3.14, left, left + widthB), ys = scale(-0.43, 0.88, top + size, top);
      const clip = el(el(svg, "defs"), "clipPath", { id: `clip-window-${kind}` });
      el(clip, "rect", { x: left, y: top, width: widthB, height: size });
      axes(svg, [left, top, widthB, size], xs, ys, [1, 2, 3], i === 0 ? [-0.4, 0, 0.4, 0.8] : []);
      for (const w of data.requested_w) el(svg, "line", { x1: xs(w), x2: xs(w), y1: top + size, y2: top + size + 4, class: "axis" });
      const plot = el(svg, "g", { "clip-path": `url(#clip-window-${kind})` });  // bands beyond the axes are cut off
      el(plot, "line", { x1: left, x2: left + widthB, y1: ys(0), y2: ys(0), class: "zero-dotted" });
      el(svg, "text", { x: left + widthB / 2, y: top - 7, class: "kind", "text-anchor": "middle" }, title(kind));
      for (const [name, colour, dashed] of [["raw", RAW, true], ["selected", SELECTED, false]]) {
        const x = rows.map(r => xs(r.W)), med = rows.map(r => ys(r[name][0]));
        el(plot, "path", { d: band(x, rows.map(r => ys(r[name][1])), rows.map(r => ys(r[name][2]))), fill: colour, "fill-opacity": 0.14 });
        el(plot, "path", { d: path(x.map((v, k) => [v, med[k]])), stroke: colour, "stroke-width": 1.8, fill: "none", "stroke-dasharray": dashed ? "5 3" : "" });
        rows.forEach((r, k) => {
          const label = name === "raw" ? "Raw candidates" : "Selected output";
          const text = `${title(kind)}, W = ${r.W} s, ${label}: median ${signed(r[name][0])} s (10th ${signed(r[name][1])}, 90th ${signed(r[name][2])})`;
          const mark = dashed
            ? el(svg, "rect", { x: x[k] - 3.5, y: med[k] - 3.5, width: 7, height: 7, fill: colour })
            : el(svg, "circle", { cx: x[k], cy: med[k], r: 3.8, fill: colour });
          hover(mark, text);
        });
      }
    });
    el(svg, "text", { transform: `translate(${left0b - 42} ${top + size / 2}) rotate(-90)`, class: "label", "text-anchor": "middle" }, "Window error (s)");
    el(svg, "text", { x: left0b + stepB + widthB / 2, y: top + size + 34, class: "label", "text-anchor": "middle" }, "Requested reaction window W (s)");
    const legendY = top + size + 56, legendX = left0b + 20;
    el(svg, "line", { x1: legendX, x2: legendX + 24, y1: legendY, y2: legendY, stroke: RAW, "stroke-width": 1.8, "stroke-dasharray": "5 3" });
    el(svg, "rect", { x: legendX + 8.5, y: legendY - 3.5, width: 7, height: 7, fill: RAW });
    el(svg, "text", { x: legendX + 30, y: legendY + 4, class: "label" }, "Raw candidates");
    el(svg, "line", { x1: legendX + 180, x2: legendX + 204, y1: legendY, y2: legendY, stroke: SELECTED, "stroke-width": 1.8 });
    el(svg, "circle", { cx: legendX + 192, cy: legendY, r: 3.8, fill: SELECTED });
    el(svg, "text", { x: legendX + 210, y: legendY + 4, class: "label" }, "Selected output");
  }

  // ---- Figure 6: response time separates planner behavior ------------------------------------
  const PLANNERS = {  // Figure 6's colours, markers and dashes
    "IDM-base": { color: "#8597A9", marker: "circle", dash: "6 4" },
    "PDM-Closed": { color: "#185B9B", marker: "diamond", dash: "" },
    "LatentDriver": { color: "#956F9E", marker: "triangle", dash: "" },
  };
  const LABEL_GAP = 16;

  function marker(svg, shape, x, y, color) {
    if (shape === "circle") return el(svg, "circle", { cx: x, cy: y, r: 5.5, fill: color });
    if (shape === "diamond") return el(svg, "path", { d: `M${x},${y - 7}L${x + 7},${y}L${x},${y + 7}L${x - 7},${y}Z`, fill: color });
    return el(svg, "path", { d: `M${x},${y - 7}L${x + 6.5},${y + 5}L${x - 6.5},${y + 5}Z`, fill: color });
  }

  function matchedWindows(curves) {
    const { svg, hover } = chart("chart-matched-windows", 1000, 330);
    const hidden = new Set(), layers = {};
    const top = 62, height = 214, width = 240, left0 = 78, step = 300;
    KINDS.forEach((kind, i) => {
      const left = left0 + i * step;
      const xs = scale(0.85, 3.15, left, left + width), ys = scale(0, 100, top + height, top);
      axes(svg, [left, top, width, height], xs, ys, [1, 2, 3], [0, 50, 100]);
      el(svg, "text", { x: left + width / 2, y: top - 10, class: "kind", "text-anchor": "middle" }, title(kind));
      el(svg, "text", { x: left + width / 2, y: top + height + 36, class: "label", "text-anchor": "middle" }, "Window W (s)");
      const ends = [];
      for (const curve of curves.filter(c => c.kind === kind)) {
        const style = PLANNERS[curve.planner], g = el(svg, "g");
        (layers[curve.planner] ||= []).push(g);
        const x = [1, 2, 3].map(xs), e = curve.estimates;
        el(g, "path", { d: band(x, e.map(v => ys(v.lo)), e.map(v => ys(v.hi))), fill: style.color, "fill-opacity": 0.13 });
        el(g, "path", { d: path(x.map((v, k) => [v, ys(e[k].value)])), stroke: style.color, "stroke-width": 2.4, fill: "none", "stroke-dasharray": style.dash });
        e.forEach((v, k) => hover(marker(g, style.marker, x[k], ys(v.value), style.color),
          `${curve.planner}, ${kind}, W = ${k + 1} s: ${v.value.toFixed(1)}% [${v.lo.toFixed(1)}, ${v.hi.toFixed(1)}], ${curve.n} situations`));
        ends.push({ g, y: ys(e[2].value), at: ys(e[2].value), x: x[2], curve, style });
      }
      ends.sort((a, b) => a.y - b.y);
      for (let k = 1; k < ends.length; k++) ends[k].at = Math.max(ends[k].at, ends[k - 1].at + LABEL_GAP);
      for (const end of ends) {
        el(end.g, "line", { x1: end.x + 3, y1: end.y, x2: end.x + 13, y2: end.at, stroke: end.style.color, "stroke-width": 0.8 });
        el(end.g, "text", { x: end.x + 16, y: end.at + 4, class: "count", fill: end.style.color }, `n=${end.curve.n}`);
      }
    });
    el(svg, "text", { transform: `translate(26 ${top + height / 2}) rotate(-90)`, class: "label", "text-anchor": "middle" }, "Target collision (%)");
    // legend: click a planner to hide or show it
    let x = 300;
    for (const [name, style] of Object.entries(PLANNERS)) {
      const item = el(svg, "g", { class: "legend-item", tabindex: 0, role: "button", "aria-pressed": "true" });
      el(item, "line", { x1: x, x2: x + 30, y1: 16, y2: 16, stroke: style.color, "stroke-width": 2.4, "stroke-dasharray": style.dash });
      marker(item, style.marker, x + 15, 16, style.color);
      el(item, "text", { x: x + 38, y: 21, class: "label" }, name);
      const toggle = () => {
        hidden.has(name) ? hidden.delete(name) : hidden.add(name);
        for (const g of layers[name]) g.style.display = hidden.has(name) ? "none" : "";
        item.setAttribute("aria-pressed", String(!hidden.has(name)));
        item.style.opacity = hidden.has(name) ? 0.35 : 1;
      };
      item.addEventListener("click", toggle);
      item.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      x += 150 + name.length * 4;
    }
  }

  timeControl(window.CHARTS.time_control);
  matchedWindows(window.CHARTS.matched_windows);
})();
