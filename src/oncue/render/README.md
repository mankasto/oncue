# render

Offline HTML pages of OnCue's outputs: the case browser (`oncue cases`), which replays chosen
tests with the reference ego and the three IDM planners, and a small site around it (`oncue site`).
Every file of a built page is local, so it opens straight from disk. Start from `render_cases`
in `cases.py`.

## Commands

```
oncue cases LABELS... --scenes SCENES... --spec SPEC --closed CLOSEDLOOP... --out DIR
oncue site DIR --cases CASES_DIR [--results RUN_DIR] [--explained PAGE.html]
```

- `cases` picks up to `--per-kind` tests (default `CASES_PER_KIND`, 4) per target type, site
  kind and label, the most telling first; `--closed` supplies closed-loop records, which only
  rank the tests. It then replays the picked tests from `--scenes` with the planners in
  `CASE_PLANNERS` (IDM-base, IDM-conservative, IDM-occlusion-aware), masked and unoccluded,
  since the outputs keep outcomes, not trajectories; PDM-Closed, the learned planners and outside
  planners are not replayed. `--workers` sets the processes that replay them (default `WORKERS`, 4).
- `site` copies a built case browser into a site with a home page; `--results` takes the
  home page's numbers from a run folder's analyze tables (left out without it), and
  `--explained` adds a method page.
- Open `DIR/index.html` in a browser.

## Files

- `cases.py`: picking the cases, replaying them and writing the browser's data and page.
- `payload.py`: scene data the pages share: rounding, map features and the tracks near a region.
- `site.py`: the site around the browser.
- `templates/`: the pages (`cases.html`, `home.html`, `site_nav.html`) and their shared
  `common.js` and `common.css`.

## Dependencies

Imports `utils/` and `core/`; replaying a case imports `generation/` and `closedloop/`, so a case
runs as it did in the closed loop.
