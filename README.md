# When Better Predictions Fail: Evaluating Foundation Models for Inventory Decisions

This repository contains the code and reproducibility artifacts for the paper:

> **When Better Predictions Fail: Evaluating Foundation Models for Inventory Decisions**
> Min Xu (2026). Submitted to *Manufacturing & Service Operations Management (MSOM)*.

## Overview

We evaluate whether the forecasting accuracy of time-series foundation models (TSFMs) translates into better inventory decisions under realistic cost structures. Using a 1,836-series retail dataset from Zhao et al. (2023), we find that:

- **Chronos-2** reduces static newsvendor cost by 6–14%, but this advantage does not always translate into dynamic inventory savings.
- **Censored-demand evaluation** is the primary source of attenuation — not inventory dynamics.
- A **5×5 service-level–recensoring grid** with familywise simultaneous inference identifies a discrete **conversion boundary**: four of 25 prespecified combinations satisfy the simultaneous conversion criterion.

## Dataset

This project uses the retail inventory dataset from:

> Zhao, X., Zhao, Y., & Song, Z. (2023). An integrated framework for inventory management with supply chain coordination.
> *Naval Research Logistics*, 70(8), 789–808.
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
  audit_*.py                # Audit scripts

artifacts/                  # Experiment outputs
  zhao_grid/                # Grid experiment (frozen)
    grid_results.csv        # B=50,000 results
    FROZEN_MANIFEST.md      # Reproducibility manifest
  zhao_rolling/             # Rolling-origin bootstrap results
  zhao_halfsynthetic/       # Semi-synthetic experiment
  zhao_continuous/          # Continuous replay

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

## Key Results

| Metric | Value |
|---|---|
| Static cost reduction (Chronos-2) | 6–14% |
| Dynamic cost reduction (replay) | Not statistically detectable under censored sales |
| Pooled DiD (semi-synthetic) | −8.28 pp (p < 0.001) |
| Conversion region | 4 of 25 grid points |
| Converting points | (α,λ) ∈ {(0.95,0), (0.98,0), (0.98,0.25), (0.98,0.50)} |

## Citation

If you use this code, please cite:

```bibtex
@article{xu2026tsfm,
  title={When Better Predictions Fail: Evaluating Foundation Models for Inventory Decisions},
  author={Xu, Min},
  journal={Working Paper, submitted to Manufacturing \& Service Operations Management},
  year={2026}
}
```

And the dataset:

```bibtex
@article{zhao2023integrated,
  title={An integrated framework for inventory management with supply chain coordination},
  author={Zhao, Xuan and Zhao, Yao and Song, Zuo-Jun},
  journal={Naval Research Logistics},
  volume={70},
  number={8},
  pages={789--808},
  year={2023},
  publisher={Wiley},
  doi={10.1002/nav.21957}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The Zhao et al. (2023) dataset is subject to the publisher's terms and is not redistributed in this repository.
