# Project: Diffusion Beam Prediction

- always use `uv` for package management. never pip.
- use underscore_case for python files
- python source lives in src/diffbeam/
- report is in report.tex (LaTeX, IEEE format)
- **README.md is the status log and todo tracker — keep it up to date**

## Key Concepts

- DeepSense 6G dataset scenarios 31-34
- Multimodal: images + GPS → beam index prediction
- Diffusion for data augmentation
- **Novel angle: Cross-domain generalization (day→night)**

## Experimental Strategy

### Phase 1: Baseline (current)
- Train on all scenarios (31-34), 80/20 split
- Establishes upper bound for same-domain performance

### Phase 2: Cross-Domain Test
- Train on DAY scenarios (31+32, ~10K samples)
- Test on NIGHT scenarios (33+34, ~8K samples)
- If big accuracy drop → domain shift exists → diffusion story is interesting

### Phase 3: Diffusion Augmentation
- Train diffusion on day images conditioned on beam
- Generate synthetic samples
- Compare: baseline vs +trad_aug vs +diffusion_aug

### Phase 4: Ablations
- Low-data regime (500, 1000, 2000 samples)
- Augmentation ratios (0.5x, 1x, 2x)

## Data

- data/ directory (git-ignored)
- Scenarios 31-32: Day (training for cross-domain)
- Scenarios 33-34: Night (test for cross-domain)

## Commands

```bash
# Baseline (all data)
uv run python run_baseline.py --data-dir data --epochs 30

# Cross-domain (day→night)
uv run python run_baseline.py --data-dir data --epochs 30 --train-scenarios 31 32 --test-scenarios 33 34
```
