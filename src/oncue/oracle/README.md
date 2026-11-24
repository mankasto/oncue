# oracle

The budgeted oracle: could the ego have avoided the target? It tries the first B candidates of a
fixed list of ego responses, executes each with `core/vehicle.py`, checks it, and says whether
any is safe. `labels.py` combines two such judgements into the test label. Start from
`labels.label_sample`.

## How it is called

The oracle has no command of its own.

- `oncue label` calls `labels.label_sample` on every sample; `--budget N` sets B (default
  `OracleConfig.budget`, 288).
- `oncue baseline evaluate` labels each adversary row the same way (`baselines/exchange.py`).
- `python -m experiments logs` labels the logged road users that can be evaluated.
- The closed loop reuses `checks.clearances` and `responses.EgoPath`.

## Files

| File | What it holds |
| --- | --- |
| `evaluate.py` | `Oracle`: runs the candidates for one information condition |
| `responses.py` | the fixed candidate list and each candidate's desired motion |
| `checks.py` | checks on executed candidates: dynamics, road, clearance, fault |
| `labels.py` | a sample's label from its two conditions (`label_sample`) |

## Inputs and outputs

- In: the scene with the target inserted (`core.types.Scene`), the target's reveal event and
  contact frame, and the configs (`OracleConfig`, `CandidateConfig`, `VisibilityConfig` in
  `core/config.py`).
- Out: `label_sample` returns the fields the label step writes into the sample row: `test`
  (reaction, anticipation, unresolved or not_judged), `oracle_revealed`, `oracle_early`, `budget`
  and `candidate_list` (the list's digest).

## Dependencies

Imports only `utils/`, `core/` and `data/`.
