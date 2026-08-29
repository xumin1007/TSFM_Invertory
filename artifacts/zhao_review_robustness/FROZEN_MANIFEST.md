# Reviewer-Robustness Reproducibility Manifest

This manifest freezes the distributional-robustness analyses added in response
to peer review and is published with tag `v1.1-review-robustness`.

## Scope

The run produces four audit families without modifying the frozen principal
grid or half-synthetic outputs:

1. empirical-conditional, Poisson, and negative-binomial latent-demand DGPs;
2. Gaussian-copula AR(1) protection-interval dependence at
   `rho in {0.25, 0.50}`;
3. 21-level linear-inverse-quantile and 12-level midpoint PMF reconstructions;
4. operational coverage, exceedance, and mean-excess diagnostics.

## Frozen Design

- Chronos-2 checkpoint: `amazon/chronos-2`
- Hugging Face revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Cost-eligible series: 1,135
- Scored origins: 5
- Service levels: 0.95 and 0.98
- Latent-demand draws: 50
- Scrambled Sobol paths per copula aggregation: 16,384
- Bootstrap replications: 10,000
- Global seed: 42
- Review interval: 30 days
- Lead time: 1 day
- Holding-cost rate: 0.20 of unit cost per year

## Exact Command

```bash
PYTHONPATH=src .venv-review/bin/python -m f2d.run_review_robustness \
  --device cpu \
  --batch-size 64 \
  --n-draws 50 \
  --copula-draws 16384 \
  --b 10000
```

The recorded wall-clock time was 3,707.32 seconds.

## Reference Environment

- Platform: Linux 6.18.35, x86_64, glibc 2.39
- Python: 3.12.13
- NumPy: 1.26.4
- pandas: 2.1.4
- SciPy: 1.17.0
- PyArrow: 19.0.1
- PyTorch: 2.13.0
- Transformers: 5.14.1
- chronos-forecasting: 2.3.1
- scikit-learn: 1.8.0

## Input Hashes

The Zhao source workbooks are publisher-supplied and are not redistributed.
Place them in `data/external/Zhao/` with the following names and SHA-256 hashes.

| File | SHA-256 |
|---|---|
| `nav21957-sup-0001-supinfo01.xlsx` | `02b018ddfca26cb6398fa0f9c4031635a763c02a9e62e9c9fa420b3f47833d50` |
| `nav21957-sup-0002-supinfo02.xlsx` | `5d9328aa2f31cb5e38e98bc9399f20cd8d9a81c41d8c09e588ea6c35a337de16` |
| `nav21957-sup-0003-supinfo03.xlsx` | `50cb9a0f80cca990c6895f1f2714cc75272fc99ae0aa66cbb8a34b943200b223` |
| `nav21957-sup-0004-supinfo04.xlsx` | `796215894ec38484fa5ac27a346d327472ac5ce19ee18e952c753abef1cfa492` |
| `nav21957-sup-0005-supinfo05.xlsx` | `345da5ab16f2837f8adfdfa600ecb9f45944efa1494158445955b2e121817dce` |

The audit also requires three tracked upstream artifacts.

| File | SHA-256 |
|---|---|
| `artifacts/zhao_continuous/common_series.csv` | `a4c408e457443312cf38722a92b748c3d11ee3505e931b013df4d31470c2865f` |
| `artifacts/zhao_rolling_origins/baseline_selection.csv` | `fe3a7d34faa7faaa77851b9470a8324afe12909cdf5daec3316fa5eaea49f129` |
| `artifacts/zhao_halfsynthetic/factorial_summary.csv` | `7bbdac809fed3e9e3d56a706940360517d75de84badabf3e23bd5e9734b2050a` |

## Source Hashes

| File | SHA-256 |
|---|---|
| `src/f2d/run_review_robustness.py` | `9b6d2e34db3caf1bcad07c764459309c51f484cc9f6deb967d6da5e28f9e2ea3` |
| `src/f2d/aggregation.py` | `afa0e2d93a64b1b768780e09ddb3fd567e489d54e8889ecb8676ae7862ba57cf` |
| `src/f2d/run_halfsynthetic.py` | `307dc05ec6fd6dabbc089a5442cc39d8cf7c3f51d1db23b7d0d0c133bdbc6e85` |
| `src/f2d/run_rolling_origins.py` | `e4f8d86cd82360347a9d9a9b376dfbbc8510ac340a3423552ec9a83f8af7abb4` |
| `src/f2d/simulation.py` | `6d3ab3622ebbe60f3dcaeb548e824c2785123a3dca8511ef40d3000755916e84` |
| `src/f2d/datasets/zhao.py` | `1310982d3806e392a017a24e89b57028a9ec09d9268458ffa556da5cd3e99150` |
| `src/f2d/models/chronos.py` | `1ecb34ca437bfed46aa6b1fafe100d732e6c7df1d5bd6d83c12fe7f3284e2532` |
| `src/f2d/config.py` | `3e16f37d107dfd31613d4e1e518a73dd55c74fef0060725dee77f366bb47aa17` |
| `src/f2d/conventions.py` | `b233181329219e537090b0ca12a6854e62eb6d0614f93961a845f0ec09fd7fab` |

## Frozen Outputs

| File | SHA-256 |
|---|---|
| `aggregation_sensitivity.csv` | `63c756a2b0adb3c92b98f739c8b656e6595d9d3436b00ee1e2452a6e6cc08698` |
| `latent_dgp_diagnostics.csv` | `7217731b8048054999443124d035a209d2f76542fd19f6bbdf78fa8f3caa9f22` |
| `latent_dgp_sensitivity.csv` | `8b700cbafcaedb2cc2f40d8e2c97bad9c03b421c3eb729d42ddb42bcaf3d7a4f` |
| `run_metadata.csv` | `6c1e8e1f9ebb955373d4a461b6602f6ad1409dd06013357ccabfa6e08205b62f` |
| `tail_calibration.csv` | `04eaebd89c78a1e0b42c857088003350c94b819fe4c6cd3fc2990abd1e5cc09e` |

## Frozen Findings

- Demand-measure DiD: empirical conditional `-8.28` pp, Poisson `-2.54` pp,
  and negative binomial `-6.03` pp; all three 95% intervals exclude zero.
- The two alternative PMF reconstructions change pooled attenuation by less
  than 0.1 pp.
- Pooled attenuation is `9.56` pp at `rho=0.25` and `11.70` pp at
  `rho=0.50`.
- Chronos-2's operational-coverage advantage over the retuned empirical
  baseline ranges from 3.63 to 4.25 pp across the four demand/service cells.

## Verification

Run the full unit-test suite before the experiment:

```bash
python -m pytest -q
```

The frozen workspace passes 82 tests. A reproduction is accepted when the
input hashes match, the model revision is unchanged, the script exits cleanly,
all tests pass, and the five generated CSVs match the hashes above under the
reference environment. On a different compute stack, retain the generated
CSVs and compare every reported point estimate and interval before treating a
hash difference as numerical rather than substantive.
