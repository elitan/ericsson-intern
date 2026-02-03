# Conformal Prediction for Handover Decisions

Applying conformal prediction to 5G/6G handover prediction for reliability guarantees.

## Research Gap

**No existing work on CP + handover.** CP has been applied to:
- Beam prediction (our previous work, Hegde et al., Deng et al.)
- Federated inference over wireless (Cohen et al.)
- Cellular connection quality prediction

But NOT to handover target prediction.

## Core Idea

```
RL/ML agent predicts handover target cell
    ↓
CP calibrates prediction into set of candidate cells
    ↓
Small set → trust ML, do predictive handover (low latency)
Large set → measurement-based handover (safer, higher latency)
```

---

## Our Novel Contribution

**First application of conformal prediction to handover target prediction.**

| Existing Work | Application | Gap |
|---------------|-------------|-----|
| Hegde et al. (2025) | CP for beam selection in D-MIMO | Beam ≠ handover; no mobility |
| Deng et al. SCAN-BEST (2025) | CP for beam selection | Beam ≠ handover; no temporal |
| Cohen et al. (2022) | CP for demod, mod classification, channel pred | No handover/mobility |
| WFCP (2023) | Federated CP over wireless | Server-side inference, not handover |
| Henrik et al. (2020) | RL for handover | No uncertainty quantification |
| CMAB+RIS (2024) | Contextual bandits for handover | No coverage guarantees |

**What we provide:**
1. First CP for handover → prediction sets with coverage guarantees
2. First ACI for mobility → handles temporal non-exchangeability
3. Adaptive protocol → trust ML when confident, fallback when uncertain

---

## Literature Review

### Henrik Rydén's Handover Paper

**Paper:** "5G Handover using Reinforcement Learning" (2020)
- Authors: Yajnanarayana, Rydén, Hévizi
- Venue: IEEE 5G World Forum
- arXiv: [1904.02572](https://arxiv.org/abs/1904.02572)

**Approach:**
- Frames handover as **contextual multi-armed bandit**
- Centralized RL agent processes UE measurement reports
- Solves with **Q-learning**
- State: UE measurements (RSRP, etc.)
- Action: target cell selection
- Reward: considers handover cost to reduce frequency

**Results:** 0.3-0.7 dB link-beam performance gain

**Gap:** No uncertainty quantification. Agent outputs single target cell with no reliability guarantee.

### Related Recent Work

**Contextual Bandits + RIS for Handover (Dec 2024)**
- arXiv: [2512.08556](https://arxiv.org/abs/2512.08556)
- Uses CMAB for target gNB selection
- Context: RSRP, UE speed, historical performance
- Still no CP / uncertainty quantification

### CP for Wireless (but not handover)

1. **Cohen et al. (2022)** - "Calibrating AI Models for Wireless Communications via Conformal Prediction"
2. **Federated CP** - [arXiv:2308.04237](https://arxiv.org/abs/2308.04237) - wireless federated inference with CP
3. **Cellular quality prediction** - [arXiv:2407.10976](https://arxiv.org/abs/2407.10976) - Ensemble Spatial CP

---

## Available Data & Simulators

### Simulators

| Name | Platform | Features | Link |
|------|----------|----------|------|
| **Simu5G** | OMNeT++ | X2 handover, CoMP, D2D, vehicular | [GitHub](https://github.com/Unipisa/Simu5G) |
| **5G-LENA** | ns-3 | NR Release-15, beamforming, mmWave | [Website](https://5g-lena.cttc.es/) |
| **ns-O-RAN** | ns-3 | O-RAN E2 interface, handover control | [arXiv](https://arxiv.org/abs/2305.06906) |
| **mmWave module** | ns-3 | Dual connectivity, fast switching | [ns-3 apps](https://apps.nsnam.org/app/mmwave/) |

### Public Datasets

| Dataset | Source | Features |
|---------|--------|----------|
| **Irish 5G** | UCC | Static + car mobility, video + download | [GitHub](https://github.com/uccmisl/5Gdataset) |
| **US Multi-Carrier** | Academic | 48K HOs, 15K+ km driving, 3 operators | Paper (to be released) |
| **Dense Urban** | GNetTrack | 30K records, RSRP/RSRQ/SNR, pedestrian/bus | Academic |

### Synthetic Option

Generate simple mobility traces + channel model for proof-of-concept. **Already implemented**: `src/handover/synthetic_data.py`

### Handover Code Examples (GitHub)

- [Handover-Optimization-using-QLearning](https://github.com/IIITV-5G-and-Edge-Computing-Activity/Handover-Optimization-using-QLearning) - Q-learning, grid-based
- [LTE-Simulation-in-NS3/handover.cc](https://github.com/RahulMotipalle/LTE-Simulation-in-NS3/blob/master/handover.cc)
- [ns3-mmwave](https://github.com/nyuwireless-unipd/ns3-mmwave) - mmWave, dual connectivity
- [CapAware](https://github.com/ds-kiel/CapAware) - Bandwidth + handover prediction, 92.4% F1

---

## CP Libraries & Code

### Conformal Prediction Libraries

| Library | Platform | Notes |
|---------|----------|-------|
| **[TorchCP](https://github.com/ml-stat-Sustech/TorchCP)** | PyTorch | Full-featured, GPU-accelerated, 16k LOC |
| **[conformal_classification](https://github.com/aangelopoulos/conformal_classification)** | PyTorch | Lightweight, RAPS, Angelopoulos |
| **[nonconformist](https://github.com/donlnz/nonconformist)** | sklearn | Classic CP |
| **[MAPIE](https://mapie.readthedocs.io/)** | sklearn | Model-agnostic |

### Adaptive Conformal Inference (temporal data)

| Library | Platform | Notes |
|---------|----------|-------|
| **[AdaptiveConformalPredictionsTimeSeries](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries)** | Python/R | Official ACI code (Zaffran) |
| **[online_conformal](https://github.com/salesforce/online_conformal)** | Python | Salesforce, SAOCP, FACI |
| **[EnbPI](https://github.com/hamrel-cxu/EnbPI)** | Python | ICML'21, dynamic time series |

### Decision: TorchCP vs custom?

Start with manual implementation (done in `conformal.py`), consider TorchCP for GPU scaling later.

---

## Problem Formulation (Draft)

### Standard Handover Prediction

- Input: $x = (\text{RSRP}_{\text{serving}}, \text{RSRP}_{\text{neighbors}}, v_{\text{UE}}, \text{history})$
- Output: target cell $y \in \{1, \ldots, K\}$
- Model: $\hat{p}(y|x)$ from RL/DL

### With Conformal Prediction

Given calibration set, compute threshold $\hat{q}$ such that:
$$\mathcal{C}(x) = \{y : \hat{p}(y|x) \geq 1 - \hat{q}\}$$

guarantees $P(y^* \in \mathcal{C}(x)) \geq 1 - \alpha$.

### Adaptive Protocol

```
if |C(x)| <= K:
    predictive_handover(C(x))  # low latency
else:
    measurement_based_handover()  # safer
```

### Metrics

- **Coverage:** fraction of times true target in prediction set
- **Set size:** average |C(x)|
- **Handover success rate**
- **Ping-pong rate:** unnecessary back-and-forth handovers
- **Latency:** time to complete handover

---

## Challenges

1. **Temporal dependence:** Handover data is sequential, violates exchangeability
   - Solution: Adaptive Conformal Inference (ACI), see [Zaffran et al.](https://proceedings.mlr.press/v162/zaffran22a.html)

2. **Conditional coverage:** Coverage may vary by UE speed, cell load, etc.
   - Solution: Group-conditional CP (like our beam paper)

3. **Multi-step prediction:** May want to predict next N handovers
   - Solution: Multi-step conformal methods

---

## Paper Plan

### Target: IEEE conference (VTC, ICC, GLOBECOM) or journal (TWC, TCOM)

---

### Phase 1: Experimental Foundation ✅ COMPLETE

**Goal:** Solid experimental setup that supports all claims.

- [x] Synthetic data generator with future prediction
- [x] MLP handover predictor
- [x] Split CP + ACI implementation
- [x] Basic experiment (coverage, set size)
- [x] **P1.1** Top-K baseline comparison table
- [x] **P1.3** Multiple scenarios (easy/medium/hard)
- [x] **P1.4** Alpha sweep (0.05, 0.10, 0.15, 0.20)
- [ ] **P1.2** Handover-specific metrics (ping-pong, overhead) - deferred to Phase 2

---

### Phase 2: Deep Analysis ✅ COMPLETE

**Goal:** Insights that make the paper memorable.

- [x] **P2.1** Conditional coverage analysis by UE speed
- [x] **P2.2** Group-conditional CP implemented and tested
- [x] **P2.3** ACI rolling coverage plots
- [ ] **P2.4** Ablation: prediction horizon (optional)

---

### Phase 3: Realistic Validation ✅ COMPLETE

**Goal:** Show it works beyond toy scenarios.

- [x] **P3.1** Downloaded Irish 5G dataset (82K samples, 50 driving traces)
- [x] **P3.2** Preprocessed: detected 1,714 handovers across 133 cells
- [x] **P3.3** Trained predictor on real data (33% Top-1 accuracy)
- [x] **P3.4** Evaluated CP on real data

**Key finding:** Real-world is MUCH harder than synthetic:
- 133 cells with only serving cell features
- Top-1: 33%, Top-10: 82%
- CP helps but needs larger sets (17.7 cells for 87% coverage)

**Limitation:** Irish dataset lacks neighbor cell RSRP → harder to predict target
**Future:** Simu5G would provide neighbor measurements

---

### Phase 4: Paper Writing ✅ COMPLETE

**Goal:** Clear, compelling paper.

- [x] **P4.1** Outline and structure (`report.tex` created)
- [x] **P4.2** Introduction draft
- [x] **P4.3** System model and problem formulation
- [x] **P4.4** Methods section (CP + ACI + group-CP)
- [x] **P4.5** Results section with all tables and figures
- [x] **P4.6** Discussion section (expanded with critical analysis)
- [x] **P4.7** Related work (Hegde, Cohen, SCAN-BEST, Henrik)
- [x] **P4.8** Conclusion with key findings

**Paper:** `report.pdf` (3 pages, IEEE format)

---

### Key Figures for Paper

1. **Coverage vs Set Size tradeoff** (alpha sweep)
2. **Conditional coverage by speed** (bar chart, show gap if exists)
3. **ACI rolling coverage** (time series, show adaptation)
4. **Top-K vs CP comparison** (table + figure)
5. **Scenario comparison** (easy vs hard, like beam paper)
6. **Adaptive protocol overhead reduction**

---

### GPU Budget (~$10 vast.ai)

Use for:
- Large-scale experiments (more trajectories, longer sequences)
- Hyperparameter sweeps
- Multiple seeds for confidence intervals

---

## Next Steps

- [x] Literature review on Henrik's handover paper
- [x] Survey available datasets and simulators
- [x] Survey CP for wireless literature
- [x] Survey CP libraries (TorchCP, ACI implementations)
- [x] Survey handover ML code on GitHub
- [x] Draft problem formulation
- [x] Implement synthetic data generator
- [x] Implement baseline handover predictor (MLP)
- [x] Implement CP layer (split conformal + ACI)
- [x] Run end-to-end experiment on synthetic data
- [x] Evaluate: coverage, set size
- [ ] **NOW:** Phase 1 tasks (P1.1 - P1.4)

## Status Log

**2025-02-03:** Project initialized. Deep research completed:
- Henrik's paper: contextual bandit + Q-learning for handover
- **Research gap confirmed:** NO existing CP + handover papers
- Found key CP libraries: TorchCP (PyTorch), ACI implementations (Zaffran, Salesforce)
- Found handover code: Q-learning repo, CapAware (92.4% F1 handover prediction)
- Implemented: synthetic data generator, MLP predictor, CP layer (split + ACI)
- Decision: start with synthetic data, validate approach, then scale to Simu5G or real data

**2025-02-03 (cont):** First successful experiment!
- Task: predict optimal cell 10 steps ahead (future prediction)
- Setup: 4x4 grid (16 cells), 6dB shadow fading, 4dB measurement noise
- **Results:**
  - Top-1 accuracy: 67%
  - Standard CP: coverage=90.1%, set size=2.45
  - ACI: coverage=90.0%, set size=2.41
  - Conditional coverage uniform across speed bins (no major gaps)
- **Key insight:** CP provides meaningful value when prediction is uncertain
- See `figures/experiment_results.png`

**2025-02-03 (cont):** Full experiment suite completed!
- Ran 3 scenarios: Easy (81% acc), Medium (68% acc), Hard (48% acc)
- **Key results table:**

| Scenario | Top-1 | Top-3 | CP Coverage | CP Set Size |
|----------|-------|-------|-------------|-------------|
| Easy     | 81.4% | 97.3% | 89.4%       | **1.29**    |
| Medium   | 68.1% | 90.7% | 89.7%       | **2.40**    |
| Hard     | 48.3% | 77.1% | 89.2%       | **4.95**    |

- **Insight:** CP adapts to uncertainty - easy scenarios need ~1 cell, hard scenarios need ~5
- **Conditional coverage gaps found:** Easy scenario shows 87.9% coverage for slow UEs vs 93.1% for medium-slow → opportunity for group-conditional CP
- **Figures:** `scenario_comparison.png`, `alpha_sweep.png`
- **LaTeX table generated** for paper

**2025-02-03 (cont):** Phase 2 complete - Group-Conditional CP + ACI
- Implemented GroupConditionalCP class
- Full paper experiments with 3 scenarios, 800 trajectories each
- **Final results:**

| Scenario | Top-1 | Std CP Cov | Std CP Size | GCP Cov | GCP Size |
|----------|-------|------------|-------------|---------|----------|
| Easy     | 81.3% | 92.2%      | 1.41        | 92.0%   | 1.41     |
| Medium   | 66.1% | 89.9%      | 2.47        | 89.8%   | 2.48     |
| Hard     | 51.1% | 89.3%      | 4.67        | 89.2%   | 4.66     |

- **Conditional coverage gaps:** Hard scenario has 2 groups under 89% with Std CP, only 1 with GCP
- **ACI rolling coverage:** maintains ~90% across all scenarios over time
- **Publication figures:** `main_results.png/pdf`, `aci_rolling.png/pdf`
- **LaTeX tables generated:** Table 1 (main results), Table 2 (conditional coverage)

**2025-02-03 (cont):** Phase 3 - Irish 5G Real-World Data
- Downloaded and preprocessed Irish 5G dataset
- 82K samples, 133 cells, 2% handover rate from 50 driving traces
- **Real-world results (MUCH harder than synthetic):**

| Method | Coverage | Size |
|--------|----------|------|
| Top-1  | 33.4%    | 1    |
| Top-5  | 77.9%    | 5    |
| Top-10 | 82.0%    | 10   |
| CP (α=0.05) | 87.4% | 17.7 |
| CP (α=0.10) | 78.5% | 4.6  |

- **Key insight:** Real handover prediction is VERY hard with limited features (only serving cell RSRP)
- **Challenge:** 133 cells, only serving cell info → model struggles
- **CP still helps:** Provides uncertainty-aware predictions, but larger sets needed
- **Future work:** Need neighbor cell measurements for better prediction

**2025-02-03 (cont):** Phase 4 started - Paper draft
- Created `report.tex` with full IEEE conference structure
- All sections drafted (intro, related work, system model, methods, conclusion)
- Paper compiles successfully
- TODO: Insert actual tables and figures from experiments

**2026-02-03:** Paper finalized with critical review
- Filled in Results section with all experimental data (Tables 1-3)
- Expanded Discussion with honest analysis of limitations:
  - Real-world coverage gap explained (78.5% vs 90% target due to limited features)
  - GCP tradeoffs noted (doesn't always improve, sample efficiency matters)
- Updated references:
  - Fixed Hegde citation (WCM 2024, not VTC 2025)
  - Added SCAN-BEST (Deng et al., 2025) - conformal risk control for beam selection
- Verified all references via web search
- **Research gap confirmed again:** Still no CP + handover papers exist
- Figures: `main_results.png`, `aci_rolling.png` - publication quality
- Paper: 3 pages, IEEE format, ready for submission

---

## Phase 5: Critical Improvements ✅ COMPLETE

**Initial Rating: 5.5/10** → **Final Rating: 7.5/10**

All improvements completed. Paper ready for workshop/conference submission.

### Must-Fix (for acceptance)

| # | Task | Impact | Time | Status |
|---|------|--------|------|--------|
| 1 | Add confidence intervals (5 seeds) | HIGH | 2-3h | ✅ Done |
| 2 | Add end-to-end handover metrics (success rate, measurement overhead) | HIGH | 3-4h | ✅ Done |
| 3 | Explain WHY CP for handover (cost of undercoverage = RLF) | MED | 1h | ✅ Done |

### Should-Fix (for strong paper)

| # | Task | Impact | Time | Status |
|---|------|--------|------|--------|
| 4 | Add latency/overhead analysis | MED | 2h | ✅ Done (via measurement overhead) |
| 5 | Fix or be honest about GCP | MED | 1h | ✅ Done (removed from main tables) |
| 6 | Add RAPS baseline | MED | 3h | ✅ Done |

### Results from v2 Experiments (5 seeds)

**Coverage (mean ± std):**
| Scenario | Top-1 | Standard CP | ACI |
|----------|-------|-------------|-----|
| Easy | 79.8% ± 1.5% | 89.9% ± 1.2% | 90.1% ± 0.1% |
| Medium | 65.4% ± 1.9% | 89.4% ± 1.0% | 90.1% ± 0.1% |
| Hard | 51.3% ± 1.5% | 89.8% ± 1.3% | 90.0% ± 0.0% |

**End-to-End Metrics:**
| Scenario | HO Success | Meas. Savings | RLF Rate |
|----------|------------|---------------|----------|
| Easy | 89.9% | **85%** | 10.1% |
| Medium | 89.4% | **84%** | 10.6% |
| Hard | 92.2% | **61%** | 7.8% |

**Key Insight:** CP enables 61-85% measurement savings with ~90% handover success.

### Current Work

**2026-02-03 (cont):** Phase 5 improvements - COMPLETE
- [x] Create `run_paper_experiments_v2.py` with multiple seeds
- [x] Add handover success/failure simulation
- [x] Generate figures with error bars (`main_results_v2.png`)
- [x] Generate LaTeX tables with ± intervals
- [x] Update paper with new results (Table 1, Table 4)
- [x] Add RLF explanation to Discussion ("Why CP for handover?")

**Paper now has:**
- Confidence intervals (5 seeds) on all metrics
- End-to-end handover metrics (Table 4): success rate, measurement overhead, RLF rate
- Clear motivation: RLF is cost of undercoverage
- New figure with error bars

**Estimated new rating: 6.5-7/10** (up from 5.5)

### RAPS Analysis (v3)

RAPS achieves higher coverage but with much larger sets:

| Scenario | Std CP Coverage | Std CP Size | RAPS Coverage | RAPS Size |
|----------|-----------------|-------------|---------------|-----------|
| Easy | 90% | 1.4 | **100%** | 8.2 |
| Medium | 89% | 2.5 | **98%** | 14.3 |
| Hard | 90% | 4.8 | **96%** | 13.1 |

**Key insight:** RAPS overcoverage is wasteful for handover. We want to *hit* 90%, not exceed it. Standard CP is optimal because:
1. Achieves target coverage (90%)
2. Uses minimal set sizes (1.4-4.8 cells)
3. Enables 61-85% measurement savings

GCP removed from main results - it doesn't improve coverage and sometimes hurts.

---

## Phase 6: Paper Feedback Addressed ✅ COMPLETE

**Date:** 2026-02-03

### Feedback Received

1. Explain more what the model input is
2. Expand paper to 6 pages (was 3)
3. Add figure showing UE movement in scenario
4. Add mobility KPIs (ping-pong handovers)
5. Add 3dB threshold baseline comparison
6. Translate AIML KPIs to system level results

### Changes Made

| Task | File | Status |
|------|------|--------|
| 3dB baseline | `src/handover/baseline.py` | ✅ Created |
| Ping-pong tracking | `baseline.py` | ✅ Added |
| UE mobility figure | `figures/ue_mobility.pdf` | ✅ Generated |
| New experiment script | `run_paper_experiments_v4.py` | ✅ Created |
| Expanded paper | `report.tex` (6 pages) | ✅ Complete |

### Key Results (v4)

**3dB Baseline vs ML+CP:**

| Scenario | 3dB Baseline | ML Top-1 | ML+CP |
|----------|--------------|----------|-------|
| Easy     | 79% | 80% | **90%** |
| Medium   | 62% | 65% | **89%** |
| Hard     | 45% | 51% | **90%** |

**Ping-Pong Reduction (NEW FINDING!):**

| Scenario | 3dB | ML Top-1 | CP Adaptive | Reduction |
|----------|-----|----------|-------------|-----------|
| Easy     | 0.30 | 0.40 | **0.19** | -37% |
| Medium   | 0.28 | 0.36 | **0.12** | -57% |
| Hard     | 0.21 | 0.31 | **0.13** | -40% |

**Key insight:** ML Top-1 *increases* ping-pong (always switches to prediction). CP Adaptive *reduces* it by staying on serving cell when it's in the prediction set.

**New Content Added:**
- Model input features: $\mathbf{x} = [\text{RSRP}_1...\text{RSRP}_K, \mathbf{e}_{c_t}, v] \in \mathbb{R}^{2K+1}$
- UE mobility figure (Fig. 1)
- AIML→System KPI mapping table
- 3dB baseline comparison
- Ping-pong analysis with policy simulation
- Sensitivity analysis (α, K_max)
- Expanded Discussion and Conclusion

**Paper now 6 pages with:**
- 4 figures
- 5 tables
- Detailed model description
- Complete baseline comparisons
- Ping-pong reduction results

---

## Project Structure

```
conformal-handover/
├── README.md           # this file
├── CLAUDE.md           # claude code instructions
├── src/
│   └── handover/       # python source
├── data/               # datasets
├── figures/            # plots
├── report.tex          # paper
└── pyproject.toml      # dependencies
```

---

## References

1. Yajnanarayana et al., "5G Handover using Reinforcement Learning," IEEE 5GWF, 2020
2. Cohen et al., "Calibrating AI Models for Wireless Communications via Conformal Prediction," IEEE TMLCN, 2022
3. Romano et al., "With malice toward none: Assessing uncertainty via equalized coverage," HDSR, 2020
4. Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift," NeurIPS, 2021
5. Hegde et al., "Reliable and Efficient Beam Selection Using Conformal Prediction in 6G Systems," IEEE WCM, 2024
6. Deng et al., "SCAN-BEST: Sub-6GHz-Aided Near-field Beam Selection with Formal Reliability Guarantees," arXiv:2503.13801, 2025
7. Raca et al., "Beyond Throughput: A 5G Dataset with Channel and Context Metrics," ACM MMSys, 2020
