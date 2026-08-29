# From Forecasts to Inventory Value: Time Series Foundation Models under Censored Sales

This repository contains the code and reproducibility artifacts for the paper:

> **From Forecasts to Inventory Value: Time Series Foundation Models under Censored Sales**
> Min Xu (2026). Submitted to *Manufacturing & Service Operations Management (MSOM)*.

## Overview

We evaluate whether the forecasting accuracy of time-series foundation models (TSFMs) translates into better inventory decisions under realistic cost structures. Using a 1,836-series retail dataset from Zhao, Li, and Shen (2020), we find that:

- **Chronos-2** reduces static high-service newsvendor cost by 12.21% relative to the origin-specific retuned empirical policy.
- **Censored-demand evaluation** accounts for most of the measured forecast-to-policy attenuation in the calibrated semi-synthetic design.
- A **5×5 service-level–recensoring grid** with familywise simultaneous inference identifies a discrete **conversion boundary**: four of 25 prespecified combinations satisfy the simultaneous conversion criterion.

## Dataset

This project uses the retail inventory dataset from:

> Zhao, L., Li, L., & Shen, Z.-J. M. (2020). Transactional and in-store display data of a large supermarket for data-driven decision-making.
> *Naval Research Logistics*, 67(8), 617–626.
> DOI: [10.1002/nav.21957](https://doi.org/10.1002/nav.21957)

### Download Instructions

1. Access the paper at [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1002/nav.21957).
2. Download the five supplementary Excel files from the "Supporting Information" section:
   - `nav21957-sup-0001-supinfo01.xlsx` (inventory)
   - `nav21957-sup-0002-supinfo02.xlsx` (orders)
   - `nav21957-sup-0003-supinfo03.xlsx` (sales)
   - `nav21957-sup-0004-supinfo04.xlsx` (attributes)
   - `nav21957-sup-0005-supinfo05.xlsx` (shelf life)
3. Place all five files in `data/external/Zhao/`.

### Verification

After downloading, verify file integrity against the SHA-256 hashes in [`artifacts/zhao_grid/FROZEN_MANIFEST.md`](artifacts/zhao_grid/FROZEN_MANIFEST.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/xumin1007/TSFM_Invertory.git
cd TSFM_Invertory

# Create environment (Python 3.12+)
conda create -n tsfm python=3.12
conda activate tsfm

# Install dependencies
pip install torch numpy pandas chronos-forecasting transformers scikit-learn lightgbm
```

## Project Structure

```
src/f2d/                    # Core library
  conventions.py            # Global constants (SEED_BASE=42)
  config.py                 # Paths
  simulation.py             # Carry-state replay engine
  decision.py               # Newsvendor cost, order-up-to policy
  aggregation.py            # PMF convolution
  datasets/zhao.py          # Raw data loader
  models/chronos.py         # Chronos-2 wrapper (revision-pinned)
  models/gbdt_grid.py       # GBDT quantile grid

  run_rolling_origins.py    # Main rolling-origin experiment
  run_halfsynthetic.py      # Semi-synthetic censoring experiment
  run_grid_censoring_alpha.py  # 5×5 grid experiment (frozen)
  run_continuous_replay.py  # Continuous carry-state replay
  run_review_robustness.py  # DGP, dependence, PMF, and tail-calibration checks
  run_ws4_kappa_margin.py   # Margin costs and SKU-specific critical ratios
  audit_*.py                # Audit scripts

artifacts/                  # Experiment outputs
  zhao_grid/                # Grid experiment (frozen)
    grid_results.csv        # B=50,000 results
    FROZEN_MANIFEST.md      # Reproducibility manifest
  zhao_rolling/             # Rolling-origin bootstrap results
  zhao_halfsynthetic/       # Semi-synthetic experiment
  zhao_continuous/          # Continuous replay
  zhao_kappa_margin/        # Margin-cost sensitivity and implied alpha_i
  zhao_review_robustness/   # Distribution, tail, and support-cap audits

paper/
  main.tex                  # MSOM paper source
  references.bib            # Bibliography

tests/
  test_simulation.py        # Unit tests
```

## Reproducing the Grid Experiment

The frozen grid experiment (tag `v1.0-grid-freeze`) can be reproduced with:

```bash
PYTHONPATH=src python -m f2d.run_grid_censoring_alpha \
  --device mps \
  --batch-size 256 \
  --b 50000
```

Replace `--device mps` with `--device cuda` on NVIDIA GPUs or `--device cpu` for CPU-only.

See [`FROZEN_MANIFEST.md`](artifacts/zhao_grid/FROZEN_MANIFEST.md) for full reproduction criteria, environment specifications, and acceptance tolerances.

## Reproducing the Review Robustness Analyses

After placing the five Zhao source files in `data/external/Zhao/`, run:

```bash
PYTHONPATH=src python -m f2d.run_review_robustness \
  --device mps \
  --batch-size 256 \
  --n-draws 50 \
  --copula-draws 16384 \
  --b 10000
```

This command leaves the frozen main and grid artifacts unchanged. It writes
the latent-demand-generator sensitivity, protection-interval dependence and
PMF-reconstruction sensitivity, operational tail-calibration diagnostics, and
run metadata to `artifacts/zhao_review_robustness/`.

See [`FROZEN_MANIFEST.md`](artifacts/zhao_review_robustness/FROZEN_MANIFEST.md)
for the exact model revision, reference environment, input and output hashes,
and acceptance criteria for this robustness run.

The margin-cost analysis changes the economic tradeoff rather than merely
rescaling a fixed critical ratio. Reproduce it with:

```bash
PYTHONPATH=src python -m f2d.run_ws4_kappa_margin \
  --device cpu \
  --batch-size 256 \
  --n-series 2000
```

It writes the complete arm-level costs, implied SKU-specific critical-ratio
distribution, and the focal Chronos-2 comparison to
`artifacts/zhao_kappa_margin/`. The registered production-cap binding audit
and midpoint support-cap scan are stored in
`artifacts/zhao_review_robustness/vmax_binding_audit.csv` and
`vmax_midpoint_sensitivity.csv`.
The exact output and source hashes for these six peer-review revisions are
recorded in
[`PEER_REVIEW_EXTENSION_MANIFEST.md`](artifacts/zhao_review_robustness/PEER_REVIEW_EXTENSION_MANIFEST.md).

## Key Results

| Metric | Value |
|---|---|
| Static high-service cost reduction (Chronos-2 vs. Emp-retuned) | 12.21% |
| Dynamic replay point estimate under logged sales | 3.46% cost reduction |
| Demand-measure DiD (empirical-conditional generator) | −8.28 pp (95% CI: −17.22 to −4.23) |
| Demand-measure DiD (alternative generators) | Poisson: −2.54 pp; negative binomial: −6.03 pp (both 95% CIs exclude zero) |
| PMF-reconstruction sensitivity | Alternative mappings change pooled attenuation by less than 0.1 pp |
| Support-cap binding and sensitivity | 0 of 6,738 focal targets bind; all reported directions retained for `vmax` 40–200; minimum per-series correlation 0.938 |
| Margin-cost sensitivity | Chronos-2 ZS cost is 7.11–8.94% below emp-daily across `kappa_h` 0.10–0.30 |
| Protection-interval dependence | Pooled attenuation is 9.56 pp at ρ=0.25 and 11.70 pp at ρ=0.50 |
| Operational-tail calibration | Chronos-2 coverage exceeds Emp-retuned by 3.63–4.25 pp |
| Conversion region | 4 of 25 grid points |
| Converting points | (α,λ) ∈ {(0.95,0), (0.98,0), (0.98,0.25), (0.98,0.50)} |

## Citation

If you use this code, please cite:

```bibtex
@article{xu2026tsfm,
  title={From Forecasts to Inventory Value: Time Series Foundation Models under Censored Sales},
  author={Xu, Min},
  journal={Working Paper, submitted to Manufacturing \& Service Operations Management},
  year={2026}
}
```

And the dataset:

```bibtex
@article{zhao2020supermarket,
  author={Zhao, Lin and Li, Lefei and Shen, Zuo-Jun Max},
  title={Transactional and In-Store Display Data of a Large Supermarket for Data-Driven Decision-Making},
  journal={Naval Research Logistics},
  volume={67},
  number={8},
  pages={617--626},
  year={2020},
  publisher={Wiley},
  doi={10.1002/nav.21957}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The Zhao et al. (2020) dataset is subject to the publisher's terms and is not redistributed in this repository.
