---
name: ab-test-experiments
description: >-
  Run parallel multi-window A/B backtest experiments for the LLM trading bot.
  Use when testing config, prompt, or strategy changes; comparing baseline vs
  variant performance; running ab sweeps; evaluating whether a change improves
  trading results; or the user mentions A/B tests, experiments, or backtest sweeps.
---

# A/B Test Experiments

## Core rules

1. **Run all variants in parallel** — never run backtests sequentially when comparing options.
2. **Use 3–5 date ranges** — every variant must be tested on the same windows; aggregate metrics across windows before deciding.
3. **Default model: `gpt-5.4-nano`** — use for all experiments unless the change under test is an LLM model comparison.
4. **Hold everything else constant** — only change the variable being tested (prompt, timeframe, candle history, feature flag, etc.).

## When to use which model

| Experiment type | Model |
|-----------------|-------|
| Prompt, timeframe, candle history, HTF, chart/drawdown flags, risk rules | `gpt-5.4-nano` |
| Comparing LLM models | Use the models being compared (still run parallel + multi-window) |

Override via `settings.model_copy(update={"openai_model": "..."})` or `OPENAI_MODEL` in `.env` for the duration of the sweep only. Restore the user's default afterward.

## Workflow

```
Experiment checklist:
- [ ] Define hypothesis and single variable to change
- [ ] Pick 3–5 non-overlapping date ranges (same length per window)
- [ ] Define control + variant(s) as Settings overrides
- [ ] Run all (variant × window) jobs in parallel
- [ ] Aggregate avg return, win rate, Sharpe ratio, drawdown per variant
- [ ] Compare variant vs control on averaged metrics
- [ ] Save JSON results to results/ and summarize findings
```

## Running experiments

### Built-in sweep (timeframe × candle history)

The repo includes a multi-window parallel sweep:

```bash
source .venv/bin/activate
python scripts/run_ab_sweep.py --workers 4
```

This script (`scripts/run_ab_sweep.py`):
- Uses `gpt-5.4-nano` by default
- Runs 4 weekly windows × 6 variants (1h/4h × 30/50/100 candle history)
- Executes backtests in parallel via `ThreadPoolExecutor`
- Aggregates `avg_return_pct`, win rate, avg Sharpe ratio, and drawdown across windows
- Saves to `results/ab_sweep_multi_<timestamp>.json`

Adjust `DATE_RANGES`, `TIMEFRAMES`, `CANDLE_HISTORIES`, or `MODEL` at the top of the script when the experiment requires it.

### Custom A/B (baseline vs one change)

For prompt edits, feature toggles, or other single-variable tests:

1. Read `scripts/run_ab_sweep.py` as the template — reuse `_run_variant`, `_aggregate`, and parallel execution.
2. Define two (or more) named variants, e.g. `control` and `variant`.
3. Apply overrides with `base.model_copy(update={...})`.
4. Run every `(variant, date_range)` pair in parallel using `run_backtest_for_range` from `llm_trading_bot.runners.backtest`.
5. Bucket aggregation by variant name (not just timeframe/candle_history).

Minimal variant runner pattern:

```python
from llm_trading_bot.config import get_settings
from llm_trading_bot.runners.backtest import run_backtest_for_range

MODEL = "gpt-5.4-nano"  # unless testing models

def run_variant(base, *, label, overrides, from_date, to_date):
    settings = base.model_copy(update={"openai_model": MODEL, **overrides})
    result = run_backtest_for_range(settings, from_date, to_date, initial_cash=settings.starting_balance)
    return {"label": label, "from": from_date, "to": to_date, **result}
```

Launch all jobs with `concurrent.futures.ThreadPoolExecutor`. Default `--workers 4`; increase if API rate limits allow.

## Choosing date ranges

- Use **3–5 windows** of equal length (e.g. 7-day weeks).
- Prefer **recent, contiguous** ranges with available exchange data.
- Avoid overlapping windows — each range is an independent sample.
- Update `DATE_RANGES` in the sweep script or pass them as CLI args if you add that support.

Example (4 × 1-week windows):

```python
DATE_RANGES = [
    ("2025-05-12", "2025-05-19"),
    ("2025-05-19", "2025-05-26"),
    ("2025-05-26", "2025-06-02"),
    ("2025-06-02", "2025-06-09"),
]
```

Shift ranges forward as new data becomes available; keep window count at 3–5.

## Metrics and decision criteria

Per variant, compute across all windows:

| Metric | How |
|--------|-----|
| **Avg return %** | Mean of `return_pct` per window |
| **Win rate** | Windows with positive return / total windows |
| **Total PnL** | Sum of `pnl` across windows |
| **Avg Sharpe ratio** | Mean of `sharpe_ratio` per window (from backtest result; annualized from per-bar returns) |
| **Avg / max drawdown** | Mean and max of `max_drawdown_pct` |

Each backtest returns `sharpe_ratio` via `run_backtest_for_range` — capture it in per-run results and include `avg_sharpe_ratio` in the aggregated summary.

**Ship the change** when the variant beats control on **avg return %** and **avg Sharpe ratio** across windows **and** does not materially worsen max drawdown. Treat a single lucky window as insufficient — the average across 3–5 ranges is the decision metric.

When variants trade off (higher return but worse Sharpe or drawdown), report all three and let the user decide.

## Results output

Always persist structured JSON under `results/`:

```json
{
  "model": "gpt-5.4-nano",
  "hypothesis": "Shorter candle history improves responsiveness",
  "date_ranges": [{"from": "...", "to": "..."}],
  "variants": ["control", "variant_a"],
  "results": [],
  "summary": []
}
```

Print a comparison table sorted by `avg_return_pct` descending (also show `avg_sharpe_ratio`). Include per-window breakdowns so the user can spot regime sensitivity.

## Report template

After a sweep, summarize like this:

```markdown
## A/B experiment: [hypothesis]

**Model:** gpt-5.4-nano · **Windows:** N · **Parallel workers:** W

| Variant | Win rate | Avg return % | Avg Sharpe | Total PnL | Avg DD % | Max DD % |
|---------|----------|--------------|------------|-----------|----------|----------|
| ...     | ...      | ...          | ...        | ...       | ...      | ...      |

**Verdict:** [variant/control] wins on avg return and avg Sharpe with [acceptable/worse] drawdown.
**Recommendation:** [ship / reject / iterate]
**Saved:** results/ab_sweep_multi_<timestamp>.json
```

## Prerequisites

- `OPENAI_API_KEY` set in `.env`
- Virtualenv active: `source .venv/bin/activate`
- Package installed: `pip install -e .`

## Common pitfalls

- **Sequential runs** — wastes time and invites inconsistent conditions; always parallelize.
- **Single window** — one week can be noise; require 3–5 ranges.
- **Expensive model for non-model tests** — stick to `gpt-5.4-nano` unless comparing models.
- **Changing multiple variables** — invalidates A/B; isolate one change per experiment.
- **Forgetting warmup** — shorter windows need enough bars for `CANDLE_HISTORY` warmup; use `validate_backtest_window` errors as a signal to widen the range.
