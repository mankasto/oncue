# evaluate

What the experiments' analysis code shares: 95 % intervals with the drive as the unit, how the
rows of the outputs are read, and how a table is shown and written. Start from `outputs.py`, then
`rates.py`.

## Files

| File | What it holds |
| --- | --- |
| `bootstrap.py` | 95 % intervals resampled by drive or log (`Estimate`) |
| `outputs.py` | reading generate, label and closed-loop rows and their settings records |
| `rates.py` | what a closed-loop run counts as (`run_flags`), counts and rates with their intervals |
| `groups.py` | grouping rows by flaw group, severity and what hides the target |
| `report.py` | showing and writing the tables (`text`, `write`) |

## Inputs and outputs

- In: generate, label and closed-loop outputs (`outputs.output_rows`) with their `.settings.json`
  (`outputs.shared_settings`).
- Out: estimates with the counts behind them, and the tables' JSON (`report.write`).

## Dependencies

Imports `utils/`, `core/`, `data/`, `generation/` and `closedloop/`.
