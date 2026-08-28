# Frozen Reproducibility Manifest — Grid Censoring Experiment

**Frozen**: 2026-08-28
**Bootstrap resamples**: B = 50,000
**Run time**: ~1054 s (post-revision-pin full rerun, all audits pass, exit 0)
**Output verified**: byte-identical to pre-pin run (`be06c913...f50b7`)

## 1. Git State

```
Frozen source commit:  4265ac5  (all code + paper + grid_results.csv)
Release tag:           v1.0-grid-freeze  (this manifest + tag)
```

## 2. Design Constants

| Parameter | Value | Source |
|---|---|---|
| SEED_BASE | 42 | `src/f2d/conventions.py` |
| R (review period) | 30 days | `src/f2d/run_halfsynthetic.py` |
| LEAD_DAYS | 1 | `src/f2d/run_halfsynthetic.py` |
| VMAX | 60 | `src/f2d/run_halfsynthetic.py` |
| KAPPA_H | 0.20 | `src/f2d/run_halfsynthetic.py` |
| BURN_IN_DAYS | 62 (= 2 × (R + LEAD_DAYS)) | `src/f2d/run_halfsynthetic.py` |
| SCORED_ORIGINS | range(3, 8) → origins 3–7 | `src/f2d/run_halfsynthetic.py` |
| B (default) | 10,000 (overridden to 50,000 via `--b`) | `src/f2d/run_halfsynthetic.py` |
| N_DRAWS | 50 | `src/f2d/run_grid_censoring_alpha.py` |
| GRID_ALPHAS | (0.80, 0.85, 0.90, 0.95, 0.98) | `src/f2d/run_grid_censoring_alpha.py` |
| LAMBDAS | (0.0, 0.25, 0.50, 0.75, 1.0) | `src/f2d/run_grid_censoring_alpha.py` |

## 3. Model & Environment

| Component | Version / ID |
|---|---|
| Model | `amazon/chronos-2` (119.5M params) |
| HuggingFace revision | `29ec3766d36d6f73f0696f85560a422f50e8498c` (2026-06-05) |
| Revision pinned in code | `src/f2d/models/chronos.py::BASE_REVISION` |
| Python | 3.12.2 (conda-forge, Clang 16.0.6) |
| NumPy | 1.26.4 |
| Pandas | 2.1.4 |
| PyTorch | 2.13.0 |
| Transformers | 5.14.1 |
| Chronos | 2.3.1 |
| Platform | macOS 26.5.1, Apple Silicon (arm64) |
| Device | MPS |

The revision is enforced in code via `BaseChronosPipeline.from_pretrained(
BASE_CHECKPOINT, revision=BASE_REVISION, ...)` so that future runs cannot
silently load a different model version.

## 4. Key Results (frozen)

- Conversion region: (0.95,0), (0.98,0), (0.98,0.25), (0.98,0.50)
- R₃ < 0: 11 of 25 grid points
- DiD significant (simultaneous): 16 of 20 non-mechanical-zero entries
- DiD non-significant: all 4 at α = 0.98
- Natural censoring rate: 35.3% (full panel); 36.5% (scored origins, λ=1)
- Critical values: c_{R₀,R₃} = 2.763 (50-dim), c_{A,DiD} = 2.895 (45-dim)

## 5. SHA-256 Hashes

### Raw Zhao data (5 files, `data/external/Zhao/`)

Source: Zhao et al. (2023), Naval Research Logistics, supplementary materials.

```
02b018ddfca26cb6398fa0f9c4031635a763c02a9e62e9c9fa420b3f47833d50  data/external/Zhao/nav21957-sup-0001-supinfo01.xlsx  (inventory)
5d9328aa2f31cb5e38e98bc9399f20cd8d9a81c41d8c09e588ea6c35a337de16  data/external/Zhao/nav21957-sup-0002-supinfo02.xlsx  (orders)
50cb9a0f80cca990c6895f1f2714cc75272fc99ae0aa66cbb8a34b943200b223  data/external/Zhao/nav21957-sup-0003-supinfo03.xlsx  (sales)
796215894ec38484fa5ac27a346d327472ac5ce19ee18e952c753abef1cfa492  data/external/Zhao/nav21957-sup-0004-supinfo04.xlsx  (attrs)
345da5ab16f2837f8adfdfa600ecb9f45944efa1494158445955b2e121817dce  data/external/Zhao/nav21957-sup-0005-supinfo05.xlsx  (shelf)
```

These files are not redistributable. To reproduce, download the five
supplementary Excel files from the published paper and place them in
`data/external/Zhao/`. Verify SHA-256 hashes before running.

### Derived input data (3 files)
```
8acce8dedbc297b99737be016a15430779baf476383fcaffe58ae9805637b13a  artifacts/zhao_halfsynthetic/halfsynthetic_long.csv
fe3a7d34faa7faaa77851b9470a8324afe12909cdf5daec3316fa5eaea49f129  artifacts/zhao_rolling_origins/baseline_selection.csv
a4c408e457443312cf38722a92b748c3d11ee3505e931b013df4d31470c2865f  artifacts/zhao_continuous/common_series.csv
```

### Grid driver and direct dependencies (11 files)
```
d536ab296b8e9431eb4e2077932c1cc80a72b5592714c6cc6be65d25fafebff6  src/f2d/run_grid_censoring_alpha.py
307dc05ec6fd6dabbc089a5442cc39d8cf7c3f51d1db23b7d0d0c133bdbc6e85  src/f2d/run_halfsynthetic.py
6d3ab3622ebbe60f3dcaeb548e824c2785123a3dca8511ef40d3000755916e84  src/f2d/simulation.py
b233181329219e537090b0ca12a6854e62eb6d0614f93961a845f0ec09fd7fab  src/f2d/conventions.py
3e16f37d107dfd31613d4e1e518a73dd55c74fef0060725dee77f366bb47aa17  src/f2d/config.py
8e149fb1862d7332db7f86854c4689b49d01c2af2f017553603e9a240593026d  src/f2d/aggregation.py
c1e5636c6d334af18973eb6f1b758e53b55a0ab6e2ebbd8ee05e5dfce20a532b  src/f2d/decision.py
1310982d3806e392a017a24e89b57028a9ec09d9268458ffa556da5cd3e99150  src/f2d/datasets/zhao.py
1ecb34ca437bfed46aa6b1fafe100d732e6c7df1d5bd6d83c12fe7f3284e2532  src/f2d/models/chronos.py
63f236195a9a5281aac45bb98317b7a2d6e0dadbb65f88d016f9a0c3c5683840  src/f2d/models/gbdt_grid.py
e4f8d86cd82360347a9d9a9b376dfbbc8510ac340a3423552ec9a83f8af7abb4  src/f2d/run_rolling_origins.py
```

### Output (1 file)
```
be06c91383f0ed2217e8f1dfdcaeb0caf540687f4b72dd6649171e2e10ff50b7  artifacts/zhao_grid/grid_results.csv
```

### Paper
```
275a9118b4ff8950682972ddf23d963e54ef40514badfb1d9827599a36ac9980  paper/main.tex
```

## 6. Reproduction Command

Run from repository root:

```bash
PYTHONPATH=src python -m f2d.run_grid_censoring_alpha \
  --device mps \
  --batch-size 256 \
  --b 50000
```

## 7. Reproduction Criteria

**Exact verification** (frozen environment, same hardware):
Byte-identical output is expected on the frozen environment. If it is
not obtained because of MPS nondeterminism, the audit, conversion-set,
and numerical-tolerance criteria below are binding.

**Cross-platform verification** (different hardware, GPU, or package versions):
- All 6 audits must pass (see §8).
- Conversion set must equal {(0.95,0), (0.98,0), (0.98,0.25), (0.98,0.50)}.
- All 95 non-mechanical estimands — 25 R₀, 25 R₃, 25 A, and 20 DiD
  values — must agree within ±0.05 pp.
- All 95 simultaneous interval endpoints (`sim_lo` and `sim_hi`) must
  agree within ±0.5 pp.
- Floating-point non-determinism (GPU reductions, MPS vs CUDA) may cause
  bitwise differences in the CSV; the audit suite and tolerance checks
  above are the binding acceptance criteria, not byte-identical output.

## 8. Audits (all pass at B=50,000)

1. **Baseline re-aggregation**: unrounded 7 estimands from halfsynthetic_long.csv, `np.testing.assert_allclose(atol=1e-8)`
2. **Cell cross-check**: Chronos Y(λ=1)+D(λ=0), Emp Y(λ=1)+D(λ=0); `n_checked == 90,800`
3. **Draw uniqueness**: 50 distinct SHA-256 orderings (`len(order_hashes) == 50`)
4. **SE defense**: all 95 family SEs finite and > 0
5. **Mechanical-zero DiD**: exactly 5 entries at DiD(α,0), explicit `np.divide(where=~zero_se)`
6. **Recensoring rates**: 5 columns (natural, fraction, total, scored_mean, scored_sd)

## 9. Directory Structure (for raw data)

```
data/
  external/
    Zhao/
      nav21957-sup-0001-supinfo01.xlsx   # inventory
      nav21957-sup-0002-supinfo02.xlsx   # orders
      nav21957-sup-0003-supinfo03.xlsx   # sales
      nav21957-sup-0004-supinfo04.xlsx   # attrs
      nav21957-sup-0005-supinfo05.xlsx   # shelf
```
