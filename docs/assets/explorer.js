// Q5's explorer: the supplement's same-situation table (a), per target type and observation mode,
// drawn as one line per planner from its collision at W* = 1 s to 1 s minus the paired change.
// The numbers printed are the table's own; window.Q5 (written by build_page.py) holds them.
"use strict";
(() => {
  const root = document.getElementById("q5-explorer");
  if (!root || !window.Q5) return;
  const KINDS = [["pedestrian", "Pedestrian"], ["cyclist", "Cyclist"], ["vehicle", "Vehicle"]];
  const MODES = [["masked", "Masked"], ["see", "Unoccluded"]];
  // Figure 6's colours for its three planners; the two other IDM variants in nearby greys and green
  const STYLE = {
    "IDM-base": { color: "#8597A9", dash: "5 4" },
    "IDM-conservative": { color: "#B3BCC6", dash: "1.5 3" },
    "IDM-occlusion-aware": { color: "#4C9A8A", dash: "" },
    "PDM-Closed": { color: "#185B9B", dash: "" },
    "LatentDriver": { color: "#956F9E", dash: "" },
  };
  const W = 560, H = 270, LEFT = 46, RIGHT = 400, TOP = 18, BOTTOM = 236;  // chart frame (viewBox units)
  const LABEL_GAP = 15;  // the least vertical distance between two planners' end labels
  const y = rate => BOTTOM - (rate / 100) * (BOTTOM - TOP);
  const state = { kind: "cyclist", mode: "masked" };
  const fmt = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1);

  function segmented(key, options) {
    return `<div class="seg" role="group" data-key="${key}">${options.map(([value, label]) =>
      `<button type="button" data-value="${value}">${label}</button>`).join("")}</div>`;
  }
  root.querySelector(".explorer-controls").innerHTML = segmented("kind", KINDS) + segmented("mode", MODES);
  for (const group of root.querySelectorAll(".seg")) {
    group.onclick = e => {
      const button = e.target.closest("button");
      if (!button) return;
      state[group.dataset.key] = button.dataset.value;
      draw();
    };
  }

  function endLabels(rows) {  // end labels at their line's end, pushed apart where they would touch
    const placed = rows.map(r => ({ r, at: y(r.at1 - r.delta) })).sort((a, b) => a.at - b.at);
    for (let k = 1; k < placed.length; k++) placed[k].at = Math.max(placed[k].at, placed[k - 1].at + LABEL_GAP);
    const overflow = placed.length ? placed[placed.length - 1].at - (BOTTOM + 4) : 0;
    if (overflow > 0) for (const p of placed) p.at -= overflow;
    return placed;
  }

  function draw() {
    for (const group of root.querySelectorAll(".seg")) {
      for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.value === state[group.dataset.key]);
    }
    const rows = window.Q5.rows[state.kind][state.mode];
    const drawn = rows.filter(r => r.at1 !== null);
    const grid = [0, 25, 50, 75, 100].map(v =>
      `<line x1="${LEFT}" x2="${RIGHT}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${LEFT - 8}" y="${y(v) + 4}" class="tick" text-anchor="end">${v}</text>`).join("");
    const lines = drawn.map(r => {
      const s = STYLE[r.planner], y1 = y(r.at1), y3 = y(r.at1 - r.delta);
      return `<g><title>${r.planner}: ${r.at1.toFixed(1)}% at 1 s, change ${fmt(r.delta)} points</title>
        <line x1="${LEFT + 20}" x2="${RIGHT - 20}" y1="${y1}" y2="${y3}" stroke="${s.color}" stroke-width="2.4" stroke-dasharray="${s.dash}"/>
        <circle cx="${LEFT + 20}" cy="${y1}" r="4.2" fill="${s.color}"/><circle cx="${RIGHT - 20}" cy="${y3}" r="4.2" fill="${s.color}"/></g>`;
    }).join("");
    const labels = endLabels(drawn).map(({ r, at }) =>
      `<text x="${RIGHT - 8}" y="${at + 4}" class="name" fill="${STYLE[r.planner].color}">${r.planner}</text>`).join("");
    root.querySelector("svg").innerHTML = `${grid}
      <text x="${LEFT + 20}" y="${H - 12}" class="tick" text-anchor="middle">W* = 1 s</text>
      <text x="${RIGHT - 20}" y="${H - 12}" class="tick" text-anchor="middle">W* = 3 s</text>
      <text transform="translate(12 ${(TOP + BOTTOM) / 2}) rotate(-90)" class="tick" text-anchor="middle">Collision (%)</text>
      ${lines}${labels}`;
    root.querySelector("tbody").innerHTML = rows.map(r => `<tr>
      <td><i style="background:${STYLE[r.planner].color}"></i>${r.planner}</td><td>${r.situations}</td>
      <td>${r.at1 === null ? "–" : r.at1.toFixed(1)}</td><td>${r.delta === null ? "–" : fmt(r.delta)}</td></tr>`).join("");
    if (window.resortTable) window.resortTable(root.querySelector("table"));  // keep a chosen column order
  }
  root.querySelector("svg").setAttribute("viewBox", `0 0 ${W} ${H}`);
  draw();
})();
