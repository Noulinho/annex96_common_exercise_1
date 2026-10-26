# Final thesis results

Read in order:

1. [No-occupant controller comparison](01_no_occupant_results.ipynb): thesis Tables VII–VIII.
2. [Occupant feedback](02_occupant_results.ipynb): thesis Table IX and paired override counts.
3. [Load shifting](03_load_shifting_results.ipynb): daily profiles and deployment diagnostics.
4. [B-spline occupancy validation](09_bspline_occupancy_validation.ipynb): supporting forecast validation, including a January-only holdout.

The first three notebooks include saved outputs and require only
`requirements-analysis.txt` to rerun. The validation notebook uses the full
simulation environment. Original mixed analysis notebooks live in
[`experiments/analysis/`](../experiments/analysis/).

`data/` contains selected CSV exports, copied without recomputing or rounding
values. `figures/` contains selected original PNG exports. The
[artifact manifest](artifact_manifest.json) records original paths, publication
paths, SHA-256 hashes, the final run identifier, and thesis table mapping.
These exports came from existing saved results; the repository cleanup does
not rerun controllers or change scientific results.

Rebuild the reading notebooks with:

```bash
python scripts/build_results_notebooks.py --execute
```

See [reproduction and provenance](../docs/reproduction.md) for simulation stages.
