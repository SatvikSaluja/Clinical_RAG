Ablation experiment definitions live in `backend/evaluation/ablation.py`
(configuration toggles in `backend/evaluation/runner.py`). Run with:

```bash
python -m backend.evaluation.ablation
```

Measured output is written to `../results/` (one JSON per configuration,
plus `ablation_summary.json`), not to this directory - see
[`RESULTS.md`](../../RESULTS.md) for the reported numbers.
