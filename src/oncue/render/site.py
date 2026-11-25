"""Assemble the offline site: home page, method explainer and case viewer (``oncue site``).

Home-page claims come from a run folder's analyze tables; a claim whose table is missing is
left out.
"""

from __future__ import annotations

import html
import json
import shutil
from importlib.resources import files
from pathlib import Path

PAGES = (
    ("home", "index.html", "Home", "首页"),
    ("explained", "explained/index.html", "Method", "方法"),
    ("cases", "cases/index.html", "OnCue tests", "OnCue 测试"),
)
# Analyze tables the home-page claims read, resolved under a run folder.
RESULTS = {
    "logs": "analyze-logs/all.json",
    "generation": "analyze-generation/all.json",
    "closedloop": "analyze-closedloop/all.json",
}
LOG_EGO = "Reference ego"  # the ego OnCue's labels are judged against
# The start the claims read per target type: the one most of its tests use.
CLAIM_START = {"pedestrian": "triggered", "vehicle": "fixed"}


def build_site(
    out: str | Path,
    cases: str | Path | None = None,
    explained: str | Path | None = None,
    results: str | Path | None = None,
) -> Path:
    """Copy the built pages under ``out`` and link them. Return the path to the home page.

    ``results`` is the run folder whose analyze tables the home page's claims read.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts = (("cases", cases, '<nav id="toolbar" class="card" aria-label="views">'),)
    present = _copy_pages(out, parts, explained)
    _link_pages(out, parts, explained, present)
    (out / "index.html").write_text(_home_page(present, results))
    return out / "index.html"


def _copy_pages(out: Path, parts: tuple, explained: str | Path | None) -> set[str]:
    """Copy the given pages under ``out``; return the keys of the site's pages, home included."""
    present = {"home"}
    for key, source, _ in parts:
        if source is not None:
            target = out / key
            if Path(source).resolve() != target.resolve():
                shutil.copytree(source, target, dirs_exist_ok=True)
            present.add(key)
    if explained is not None:
        (out / "explained").mkdir(exist_ok=True)
        shutil.copyfile(explained, out / "explained" / "index.html")
        present.add("explained")
    return present


def _link_pages(out: Path, parts: tuple, explained: str | Path | None, present: set[str]) -> None:
    """Inject the shared site navigation into every copied page."""
    for key, source, anchor in parts:
        if source is not None:
            _inject(out / key / "index.html", anchor, key, present)
    if explained is not None:
        _inject(out / "explained" / "index.html", "<body>", "explained", present)


def _inject(page: Path, anchor: str, current: str, present: set[str]) -> None:
    html = page.read_text()
    if 'class="site-nav"' in html:
        return
    if anchor not in html:
        raise ValueError(f"{page}: navigation anchor {anchor!r} not found")
    html = html.replace(anchor, anchor + _nav(current, present), 1)
    style, script = _template("site_nav.html").split("<!--script-->")
    html = html.replace("</head>", style + "</head>", 1).replace("</body>", script + "</body>", 1)
    page.write_text(html)


def _nav(current: str, present: set[str]) -> str:
    links = "".join(
        f'<a href="../{href}" class="{"on" if key == current else ""}" '
        f'data-en="{en}" data-zh="{zh}">{en}</a>'
        for key, href, en, zh in PAGES
        if key in present
    )
    return f'<span class="site-nav">{links}<span class="sep"></span></span>'


def _template(name: str) -> str:
    return files("oncue.render").joinpath(f"templates/{name}").read_text()


def _home_page(present: set[str], results: str | Path | None) -> str:
    """Return the home page HTML with the claims from ``results`` filled in."""
    home = _template("home.html")
    for key in ("explained", "cases"):
        if key not in present:  # drop the card for a page this site does not carry
            start = home.find(f'<a class="card" href="{key}/index.html">')
            if start >= 0:
                home = home[:start] + home[home.index("</a>", start) + len("</a>") :]
    claims, source = home_claims(results) if results is not None else ([], None)
    home = home.replace("<!--claims-->", _claims_html(claims))
    return home.replace("<!--source-->", _source_html(results, source))


def home_claims(results: str | Path) -> tuple[list[tuple[str, str, str]], str | None]:
    """Return ``(claims, commit)`` from a run folder's analyze tables and its ``commit`` file.

    Each claim is ``(number, english, chinese)``; claims whose table is missing are left out.
    """
    tables = {name: _read_table(Path(results) / path) for name, path in RESULTS.items()}
    claims = [
        _logged_against_inserted(tables["logs"], tables["generation"]),
        _masked_against_see(tables["closedloop"]),
        _progress_kept(tables["closedloop"]),
    ]
    stamp = Path(results) / "commit"  # a run's commit stamp, if its runner wrote one
    commit = stamp.read_text().strip() if stamp.exists() else None
    return [claim for claim in claims if claim is not None], commit


def _read_table(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _logged_against_inserted(logs: dict | None, generation: dict | None):
    """Return the "logged road users vs. inserted targets" anticipation-test claim."""
    if logs is None or generation is None:
        return None
    logged = next((c for c in logs["result"] if c["name"] == LOG_EGO), None)
    inserted = [
        c["A"]
        for c in generation["result"]["methods"]
        if c["run"] == "oncue" and c["generator"] == "model" and "A" in c
    ]
    if logged is None or not inserted:
        return None
    return (
        f"{logged['A_tests']:,} / {sum(inserted):,}",
        f"anticipation tests in {logged['logs']:,} logged scenes: from logged road users vs. "
        "from inserted ones",
        f"{logged['logs']:,} 个日志场景里的预判测试：日志自带的与插入目标得到的",
    )


def _masked_against_see(closedloop: dict | None):
    """Return the "masked vs. see-through IDM-base hit rate" claim for pedestrian anticipation."""
    cell = _closed_loop_cell(closedloop, "pedestrian", "reactive")
    if cell is None:
        return None
    masked, see = _value(cell["masked"]["hit_pct"]), _value(cell["see"]["hit_pct"])
    if masked is None or see is None:
        return None
    return (
        f"{masked:.0f}% / {see:.0f}%",
        "masked vs. see-through IDM-base hitting the pedestrian on anticipation tests",
        "预判测试上，屏蔽版与透视版基础式 IDM 撞到行人的比例",
    )


def _progress_kept(closedloop: dict | None):
    """Return the progress claim: occlusion-aware over IDM-base 4 s distance, per target type."""
    kept = []
    for kind in CLAIM_START:
        aware = _closed_loop_cell(closedloop, kind, "occlusion_aware")
        reactive = _closed_loop_cell(closedloop, kind, "reactive")
        if aware is None or reactive is None:
            return None
        aware_m, reactive_m = _value(aware["distance_4s_m"]), _value(reactive["distance_4s_m"])
        if aware_m is None or not reactive_m:
            return None
        kept.append(aware_m / reactive_m)
    return (
        " / ".join(f"{share:.0%}" for share in kept),
        "normal progress kept by occlusion-aware planning: pedestrian vs. vehicle tests",
        "遮挡感知规划平时保留的行驶距离：行人与车辆测试",
    )


def _closed_loop_cell(closedloop: dict | None, kind: str, planner: str) -> dict | None:
    """Return the closed-loop cell for ``planner`` on anticipation tests of ``kind``.

    None unless the table is grouped by exactly the claim's four fields (a finer grouping, such
    as by ``W_req``, splits the number).
    """
    if closedloop is None:
        return None
    group = {
        "kind": kind,
        "generator": "model",
        "test": "anticipation",
        "start": CLAIM_START[kind],
    }
    return next(
        (c for c in closedloop["result"] if c["planner"] == planner and c["group"] == group),
        None,
    )


def _value(estimate: object) -> float | None:
    """Return an estimate's value, handling ``report.as_json`` dicts and ``"na"`` placeholders."""
    if isinstance(estimate, dict):
        return estimate["value"]
    return None if estimate == "na" else float(estimate)


def _claims_html(claims: list[tuple[str, str, str]]) -> str:
    if not claims:
        return ""
    items = "".join(
        f'<div class="claim"><b>{html.escape(number)}</b><span data-en="{html.escape(en)}" '
        f'data-zh="{html.escape(zh)}"></span></div>'
        for number, en, zh in claims
    )
    return f'<div class="claims">{items}</div>'


def _source_html(results: str | Path | None, commit: str | None) -> str:
    """Return the footer linking the claims to the run folder and the code's commit."""
    if results is None:
        return ""
    name = html.escape(Path(results).resolve().name)
    commit = commit or "unknown"
    en = f" Numbers from run {name}, code {commit[:7]}."
    zh = f" 数字来自运行 {name}，代码 {commit[:7]}。"
    return f'<span data-en="{html.escape(en)}" data-zh="{html.escape(zh)}"></span>'
