# Conformal Beam Prediction

**Status: Complete** ✓

When does conformal prediction help beam management? A cross-scenario analysis.

[Download report (PDF)](report.pdf)

## Abstract

ML-based mmWave beam prediction lacks reliability guarantees. Conformal prediction (CP) addresses this by outputting a prediction set—a set of candidate beams guaranteed to contain the optimal beam with high probability. This project evaluates CP on two DeepMIMO scenarios at 28 GHz. Key findings: (1) CP's benefit is scenario-dependent—on boston5g_28 (83.5% base accuracy), CP outputs 1.19 beams on average; on O1_28 (92.2% accuracy), CP outputs exactly 1 beam; (2) Standard CP exhibits conditional coverage gaps: short-distance users receive only 79% coverage despite a 90% target; (3) Group-conditional CP closes this gap (1.29 beams average).

## How to run

```bash
uv run python run_simulation.py
```

## Project structure

```
src/beampred/    # Python source
run_simulation.py
report.tex       # LaTeX report (IEEE format)
figures/         # Generated plots
data/            # Generated datasets (not tracked)
```
