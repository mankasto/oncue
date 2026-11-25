# Examples

Three short scripts built on OnCue's public modules (`oncue.core`, `oncue.generation`,
`oncue.closedloop`, `oncue.evaluate`). Run them from the repository root in the `oncue`
environment; `--help` prints each script's options.

| Script | What it shows | What it reads |
| --- | --- | --- |
| [browse_tests.py](browse_tests.py) | How many tests a label output holds by target type, label and requested window W, and one test's timing (exposure, contact, W). | `oncue label` outputs |
| [replay_test.py](replay_test.py) | One test replayed with one planner in the recorded log, with masked and with unoccluded observation, and what each run counts as (collision, clean pass, stall, other contact, road departure). | One `oncue label` output and its scene shards |
| [collision_rates.py](collision_rates.py) | Target collision rates by target type, label and planner, masked and unoccluded, with 95 % intervals. | `oncue closedloop` outputs |
| [my_planner.py](my_planner.py) | A template planner (route following, hard braking for a known road user in the lane) that `--planner-module my_planner` adds to the closed loop as `cautious`; see the main README's "Testing your own planner". | Imported by `oncue closedloop` |

With the outputs of the pipeline in the main README:

```bash
python examples/browse_tests.py labels.jsonl
python examples/replay_test.py labels.jsonl scenes --planner pdm_closed
python examples/collision_rates.py closedloop.jsonl
```
