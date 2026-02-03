# mmWave Beam Prediction

Confidence-aware adaptive beam management with conformal guarantees for 5G mmWave systems.

[Download report (PDF)](report.pdf)

## Abstract

ML-based mmWave beam prediction reduces measurement overhead but provides no reliability guarantees. This project proposes a confidence-aware adaptive beam management framework: (1) cost-aware error analysis showing non-uniform error severity, (2) conformal prediction sets with a novel beam-aware variant, and (3) an adaptive fallback protocol using ML when confident and exhaustive search when uncertain. Evaluated on a 28 GHz channel with 64 beams, the system achieves near-100% effective accuracy while reducing measurement overhead by 40-60%.

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
