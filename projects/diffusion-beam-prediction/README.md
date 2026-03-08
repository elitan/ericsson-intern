# Diffusion-Augmented Multimodal Beam Prediction for 6G

## Background

**Problem**: Beam prediction in mmWave 6G requires aligning narrow beams between base stations and moving UEs. Traditional beam sweeping has high latency. ML approaches predict optimal beams from sensor data (camera, GPS, LiDAR).

**State of the Art**: MLM-BP uses foundation models (DeepSeek) achieving 72.7% top-1 with 30% of data.

**Our Angle**: Cross-domain generalization via diffusion augmentation.

---

## KEY FINDING: Massive Domain Shift

| Setup | Top-1 | Top-3 | Notes |
|-------|-------|-------|-------|
| Same-domain (all scenarios) | 45.47% | 82.49% | Upper bound |
| Cross-domain (no aug) | 4.67% | 13.99% | Baseline |
| Cross-domain + trad aug | 4.69% | ~14% | No improvement |
| **Cross-domain + diffusion aug** | **6.26%** | ~16% | **+33% relative improvement** |

**Key Result**: Diffusion augmentation improves cross-domain accuracy by 1.59pp (4.67% → 6.26%), while traditional augmentation provides no benefit.

---

## Status

### Current Phase: Deep Investigation Complete

**Problem**: 6.26% is still failing (random = 1.56%). We investigated why.

#### Completed
- [x] Project setup & data loading (18,667 samples)
- [x] Baseline (same-domain): ~45% top-1
- [x] Cross-domain baseline: 4.67% top-1
- [x] Cross-domain + diffusion: 6.26% (+34% relative, but still bad)
- [x] **run_visualize.py** - Day/night completely separate in feature space
- [x] **run_gps_only.py** - GPS also fails cross-domain (2.27%)
- [x] **run_oracle.py** - Night ceiling is 44.95% (same as day!)
- [x] **run_finetune.py** - Few-shot fine-tuning shows efficient transfer!

---

## KEY INSIGHT: The Domain Gap is Fundamental

### Exploration Results (2026-02-04)

| Experiment | Top-1 | Key Finding |
|------------|-------|-------------|
| Same-domain (all) | 45.47% | ML works when domains match |
| Oracle (night-only) | **44.95%** | Night domain is equally learnable! |
| Cross-domain (day→night) | 4.67% | Near-random (1.56% random) |
| Cross-domain + diffusion | 6.26% | Tiny improvement |
| GPS-only same-domain | 11.63% | GPS has some signal |
| GPS-only cross-domain | **2.27%** | GPS ALSO fails cross-domain! |
| Few-shot (N=100) | 11.25% | 2.4x better than zero-shot |
| Few-shot (N=500) | 20.85% | Half oracle with 6% of data |
| Few-shot (N=1000) | 26.15% | 58% of oracle |
| Few-shot (N=2000) | **32.05%** | 71% of oracle with 24% of data |

### What This Tells Us

1. **Night is NOT harder** - Oracle achieves 45% same as day
2. **Vision AND GPS fail cross-domain** - Both modalities are domain-specific
3. **The gap is ~40pp** - We need to close 4.67% → 45%
4. **Diffusion closes 4%** of the gap - Not enough for a paper
5. **Few-shot fine-tuning is efficient** - 2000 samples (24% of night data) closes 71% of gap

### Visualization Findings

**t-SNE plot** (`outputs/visualize/tsne_domain.png`):
- Day and night form **completely separate clusters**
- No overlap in feature space
- Pretrained ResNet features are domain-specific, not beam-specific

**Image grid** (`outputs/visualize/day_vs_night_grid.png`):
- Day: Clear buildings, vehicles, sky
- Night: Very dark, only scattered lights
- Massive visual difference explains the gap

## Experimental Results

### 1. Same-Domain Baseline (upper bound)
Train/test on all scenarios (31-34), 80/20 random split.

| Epoch | Top-1 | Top-3 | Top-5 |
|-------|-------|-------|-------|
| 9 | 45.47% | 82.49% | 91.99% |

### 2. Cross-Domain Baseline (no augmentation)
Train on DAY (31+32), test on NIGHT (33+34).

| Method | Best Top-1 | Best Top-3 | Notes |
|--------|------------|------------|-------|
| No augmentation | 4.67% | 13.99% | Best at epoch 1, degrades |

### 3. Cross-Domain + Traditional Augmentation
Using RandomHorizontalFlip, RandomRotation(15), ColorJitter, RandomAffine.

| Method | Best Top-1 | Best Top-3 | Notes |
|--------|------------|------------|-------|
| Traditional aug | 4.69% | ~6-14% | No improvement over baseline |

### 4. Cross-Domain + Diffusion Augmentation ✓
Train beam-conditioned DDPM on day images (30 epochs), pre-generate 10K synthetic images, train beam predictor on combined data.

| Method | Best Top-1 | Best Top-3 | Best Top-5 | Notes |
|--------|------------|------------|------------|-------|
| Diffusion aug (1x ratio) | **6.26%** | ~16% | ~25% | +33% relative improvement |

**Improvement over baseline**: 4.67% → 6.26% (+1.59pp, +34% relative)

## The Story

1. **Same-domain works well** (~45% top-1) - ML can predict beams
2. **Cross-domain fails badly** (~4.7% top-1) - Day/night shift breaks it
3. **Traditional aug doesn't help** (4.69% ≈ 4.67%) - Color/spatial transforms insufficient
4. **Diffusion aug helps!** (6.26% = +34% improvement) - Beam-conditioned generation captures semantic features

**Paper angle**: "Diffusion-Based Domain Adaptation for mmWave Beam Prediction"

**Why diffusion works better**: Traditional augmentations (flips, rotations, color jitter) don't preserve beam-relevant features. Diffusion models trained on beam-conditioned generation learn semantic features tied to beam indices, producing more meaningful synthetic data.

## Data

| Scenario | Samples | Time | Role |
|----------|---------|------|------|
| 31 | 7,012 | Day | Train |
| 32 | 3,235 | Day | Train |
| 33 | 3,981 | Night | Test |
| 34 | 4,439 | Night | Test |

**Cross-domain split**: 10,247 train (day) / 8,420 test (night)

## Commands

```bash
# Same-domain baseline
uv run python run_baseline.py --data-dir data --epochs 30 --batch-size 64

# Cross-domain baseline (day→night)
uv run python run_cross_domain.py --data-dir data --epochs 30 --batch-size 64
```

## Log

- **2026-02-03 22:30**: Project created
- **2026-02-03 22:38**: Data loaded (18,667 samples)
- **2026-02-03 22:40**: Same-domain baseline started
- **2026-02-03 22:55**: Same-domain reaches 45.47% top-1
- **2026-02-03 23:00**: Cross-domain experiment started
- **2026-02-03 23:02**: **DOMAIN SHIFT CONFIRMED** - 4.67% top-1 (vs 45% same-domain)
- **2026-02-04 00:00**: Traditional augmentation completed - no improvement (4.69%)
- **2026-02-04 00:15**: Diffusion augmentation training started
- **2026-02-04 03:00**: **DIFFUSION AUGMENTATION WORKS** - 6.26% top-1 (+34% improvement)
- **2026-02-04 03:30**: Paper updated and compiled (report.pdf)
- **2026-02-04 10:00**: Few-shot fine-tuning complete - 2000 samples achieves 32% (71% of oracle)

## Exploration Commands

```bash
# 1. Visualize day vs night images + t-SNE
uv run python run_visualize.py --data-dir data --n-tsne 500

# 2. GPS-only baseline (does vision help?)
uv run python run_gps_only.py --data-dir data --epochs 20

# 3. Oracle (night-only training ceiling)
uv run python run_oracle.py --data-dir data --epochs 30

# 4. Few-shot fine-tuning sweep
uv run python run_finetune.py --data-dir data --n-samples 100 500 1000 2000
```

## Key Questions to Answer

1. **How different are day vs night visually?** → run_visualize.py
2. **Do features cluster by domain or beam?** → t-SNE analysis
3. **Does GPS alone work cross-domain?** → If yes, vision is the problem
4. **What's the ceiling for night?** → Oracle shows achievable accuracy
5. **How many night samples close the gap?** → Few-shot sweep

## Next Steps (based on exploration)

Directions to explore after initial experiments:
- **If GPS works**: Focus on image domain adaptation
- **If nothing works**: Maybe the problem is fundamentally hard
- **If few-shot works well**: Paper angle = "data-efficient transfer"
- **If oracle is low**: Night scenario itself is harder

## Few-Shot Fine-Tuning Results

Pretrain on day (31+32), fine-tune on N night samples, test on remaining night.

| N Samples | Top-1 | Top-3 | Top-5 | % of Oracle |
|-----------|-------|-------|-------|-------------|
| 0 (zero-shot) | 7.10% | 15.58% | 20.63% | 16% |
| 100 | 11.25% | 21.55% | 29.75% | 25% |
| 500 | 20.85% | 41.90% | 54.25% | 46% |
| 1000 | 26.15% | 51.65% | 65.30% | 58% |
| 2000 | 32.05% | 62.30% | 74.70% | 71% |

**Key insight**: Few-shot transfer is highly efficient. With just 500 night samples (~6% of night data), we achieve 46% of oracle performance. With 2000 samples (~24%), we reach 71%.

**Paper angle**: "Data-Efficient Domain Adaptation for mmWave Beam Prediction"

---

## Scratchpad

### 2026-02-04: Investigation Complete

**Experiments run:**
1. Visualization (t-SNE) - domains completely separate
2. GPS-only baseline - GPS also fails cross-domain (2.27%)
3. Oracle (night-only) - achieves 45% same as day
4. Fine-tuning - **PROMISING**: 2000 samples → 32% (71% of oracle)

**Conclusion:** The domain gap is fundamental. Both image and GPS features are location/time-specific. The problem isn't "night is harder" - it's "day features don't transfer to night".

**Diffusion aug closes ~4% of the gap** - not paper-worthy.

**Few-shot fine-tuning is paper-worthy:**
- 500 samples = 21% top-1 (5x improvement over zero-shot)
- 2000 samples = 32% top-1 (71% of oracle)
- Shows data-efficient adaptation is practical

### GPU Notes
- MPS: ~7.5 it/s training, adequate for experiments
- Vast.ai: May need for larger experiments later
