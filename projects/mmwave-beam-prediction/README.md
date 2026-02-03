# mmWave Beam Prediction

**Status: Complete** ✓

Cross-scenario analysis of conformal prediction for mmWave beam selection.

[Download report (PDF)](report.pdf)

## Abstract

Conformal prediction (CP) can wrap any beam predictor with coverage guarantees, but its practical value depends on the scenario. This project presents the first cross-scenario CP analysis for mmWave beam prediction using two ray-traced DeepMIMO channels at 28 GHz (boston5g_28 and O1_28). Key findings: (1) CP's value is scenario-dependent—on boston5g_28 (~40% top-1) CP adds meaningful coverage, whereas on O1_28 (>90% top-1) prediction sets collapse to singletons; (2) standard CP exhibits conditional coverage gaps (under-covers distant users); (3) group-conditional CP closes this gap. An adaptive fallback protocol reduces overhead by 40-60%.

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
