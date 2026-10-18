Clean data only: baseline 1808 rows, Uniform(0,1), 20 bins.
2000 independent (baseline, window) pairs per window size.

| window rows | median PSI | p95 PSI | P(PSI > 0.1) | P(PSI > 0.2) | P(PSI > 0.25) | median simulated limit (alpha=0.01) | its false-positive rate (50 baselines x 100 windows) |
|---|---|---|---|---|---|---|---|
| 116 | 0.153 | 0.281 | 87.2% | 23.2% | 8.6% | 0.388 | 1.0% |
| 125 | 0.144 | 0.254 | 84.0% | 17.9% | 5.5% | 0.353 | 1.1% |
| 1000 | 0.029 | 0.047 | 0.0% | 0.0% | 0.0% | 0.057 | 1.1% |
| 5000 | 0.015 | 0.025 | 0.0% | 0.0% | 0.0% | 0.030 | 0.4% |
