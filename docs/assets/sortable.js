// Tables marked data-sortable sort by a column when its header is clicked: first best first (by the
// header's arrow: ↑ higher is better, ↓ lower is better; no arrow: largest or A to Z first), then the
// other way, then back to the paper's order. window.resortTable(table) reapplies the current order
// after a script has rewritten the rows (the Q5 explorer does).
"use strict";
(() => {
  const states = new WeakMap();  // table -> {column, direction (1 best first, -1 reversed, 0 paper order)}

  const value = cell => {
    const text = cell.textContent.trim().replace("−", "-");
    const number = parseFloat(text.replace(/^\+/, ""));
    return Number.isNaN(number) ? text.toLowerCase() : number;
  };
  const higherIsBetter = header => !header.textContent.includes("↓");

  function apply(table) {
    const state = states.get(table), body = table.tBodies[0];
    const rows = [...body.rows];
    rows.forEach((row, k) => { if (row.dataset.order === undefined) row.dataset.order = k; });
    const headers = [...table.tHead.rows[table.tHead.rows.length - 1].cells];
    headers.forEach((th, k) => {
      const on = state && state.direction && state.column === k;
      th.setAttribute("aria-sort", on ? (state.ascending ? "ascending" : "descending") : "none");
      th.querySelector(".sort-mark").textContent = on ? (state.ascending ? " ▲" : " ▼") : "";
    });
    if (!state || !state.direction) {
      rows.sort((a, b) => a.dataset.order - b.dataset.order);
    } else {
      rows.sort((a, b) => {
        const x = value(a.cells[state.column]), y = value(b.cells[state.column]);
        const missingX = typeof x === "string" && /^[–-]?$/.test(x), missingY = typeof y === "string" && /^[–-]?$/.test(y);
        if (missingX !== missingY) return missingX ? 1 : -1;  // empty cells last either way
        const order = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
        return state.ascending ? order : -order;
      });
    }
    for (const row of rows) body.appendChild(row);
  }

  for (const table of document.querySelectorAll("table[data-sortable]")) {
    const headers = [...table.tHead.rows[table.tHead.rows.length - 1].cells];
    headers.forEach((th, k) => {
      th.classList.add("sortable");
      th.tabIndex = 0;
      th.insertAdjacentHTML("beforeend", '<span class="sort-mark" aria-hidden="true"></span>');
      const click = () => {
        const old = states.get(table);
        const direction = old && old.column === k ? [1, -1, 0][([1, -1, 0].indexOf(old.direction) + 1) % 3] : 1;
        const textual = k === 0;  // the first column names the row
        const bestAscending = textual || !higherIsBetter(th);
        states.set(table, { column: k, direction, ascending: direction === 1 ? bestAscending : !bestAscending });
        apply(table);
      };
      th.onclick = click;
      th.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); click(); } };
    });
  }
  window.resortTable = table => { if (states.get(table)) apply(table); };
})();
