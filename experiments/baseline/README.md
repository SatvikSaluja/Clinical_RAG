Baseline experiment definitions live in `backend/evaluation/baselines.py`
(configuration toggles in `backend/evaluation/runner.py`). Run with:

```bash
python -m backend.evaluation.baselines
```

Measured output is written to `../results/` (one JSON per configuration,
plus `baselines_summary.json`), not to this directory - see
[`RESULTS.md`](../../RESULTS.md) for the reported numbers.
