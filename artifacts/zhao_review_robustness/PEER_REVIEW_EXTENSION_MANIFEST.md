# Peer-Review Six-Item Extension Manifest

This manifest freezes the six evidence-backed revisions layered on the
`v1.1-review-robustness` release. The principal grid, semi-synthetic, PMF,
dependence, latent-DGP, and tail-calibration outputs remain governed by
`FROZEN_MANIFEST.md` in this directory.

## Scope

1. a practical daily-support-cap selection rule, a policy-target binding audit,
   and midpoint `vmax` sensitivity;
2. formal EC reporting of observed-margin costs and SKU-specific critical
   ratios;
3. explicit identification boundaries for the fixed-policy semi-synthetic
   contrast, the exact 98% cost bridge, and the single-dataset formal
   inference;
4. the operational mapping from recensoring fraction to scored-origin
   observability;
5. exact max-|t| simultaneous inference, within-replication contrast
   construction, and zero-sales handling; and
6. two verified adjacent references on censored-demand recovery and offline
   sequential decision-making.

## Economic-Sensitivity Design

- Dataset and window: Zhao validation origins, July and August 2019.
- Sample: the registered 2,000-series draw under the project seed.
- Focal comparison: `chronos2-zs` versus `emp-daily`.
- Holding-cost fractions: `kappa_h in {0.10, 0.15, 0.20, 0.25, 0.30}`.
- Monthly holding cost: `h_i = kappa_h * unit_cost_i / 12`.
- Shortage cost: observed positive unit margin `p_i`.
- SKU-specific target: `alpha_i = p_i / (p_i + h_i)`.
- Each policy target is recomputed row by row from its protection-interval PMF.

Run the complete margin-cost analysis with:

```bash
PYTHONPATH=src python -m f2d.run_ws4_kappa_margin \
  --device cpu \
  --batch-size 256 \
  --n-series 2000
```

The run writes the full arm-level costs, the implied-critical-ratio summaries,
and the deterministic focal comparison to `artifacts/zhao_kappa_margin/`.

## Support-Cap Audit

The registered midpoint reconstruction uses `vmax=60` units per day. The
production audit reconstructs the month-specific convolution edge as
`(days_in_month + lead_days) * vmax` and compares it with both focal policy
targets in `artifacts/zhao_mechanism/mechanism_per_sku.csv`. Across 3,369
Chronos-2 targets and 3,369 empirical-policy targets, no target equals the
1,920-unit edge. The 99th-percentile utilization is 32.4% for Chronos-2 and
29.1% for the empirical policy; the corresponding maxima are 99.7% and 98.7%.

The decision-layer scan then changes only the numerical cap and covers every
reported arm contrast.

| `vmax` | Effect-ratio range | Minimum paired-loss correlation | Directions retained |
|---:|---:|---:|:---:|
| 40 | 0.861--0.952 | 0.9526 | Yes |
| 60 | 1.000--1.000 | 1.0000 | Yes |
| 100 | 0.956--1.088 | 0.9585 | Yes |
| 200 | 0.999--1.120 | 0.9382 | Yes |

The machine-readable export is
`artifacts/zhao_review_robustness/vmax_midpoint_sensitivity.csv`; the binding
audit is `artifacts/zhao_review_robustness/vmax_binding_audit.csv`. Tail
closure at each row's repaired highest quantile is protected by
`tests/test_aggregation.py::test_reconstruction_independent_of_vmax` and
`tests/test_aggregation.py::test_no_mass_above_highest_quantile`. The
month-specific binding reduction is protected by
`tests/test_review_robustness.py::test_support_binding_summary_uses_month_specific_convolution_edge`.

## Frozen Output Hashes

| File | SHA-256 |
|---|---|
| `artifacts/zhao_kappa_margin/sensitivity_kappa_margin.csv` | `247b84d50ace9875209d3037d797a504497ea4e1e8ef1b18235cfc0c1180ef5a` |
| `artifacts/zhao_kappa_margin/implied_alpha_by_kappa.csv` | `aab2ef515b0d8769f38d30de66dbbb71a913b3f5f0b4da2543d68cd02f958ff9` |
| `artifacts/zhao_kappa_margin/margin_cost_summary.csv` | `7829d9e0968a394068d759084ac9fd6bf883fb11b87f2c4efb12b7a6a96f29a6` |
| `artifacts/zhao_mechanism/mechanism_per_sku.csv` | `04ef662f9ad734d2a686ba3658c0adf922872a61646d27f568dfd33e3c368353` |
| `artifacts/zhao_review_robustness/vmax_binding_audit.csv` | `3c5e0998c37d27a961e8ee4a858db37f3138776ec830fe6ced8ad40485893d2b` |
| `artifacts/zhao_review_robustness/vmax_midpoint_sensitivity.csv` | `97f471dfd72726dbdcfc3b1db236a4bae1f45ddf09e4f308f6a6c45357611bbf` |
| `artifacts/zhao_decision/checks/layer_b.json` | `06e95541bc61545edf411ec9525a1965619f036f66c9b76c22249faaee36aac9` |
| `artifacts/zhao_decision/layer_b_summary.csv` | `7d584eb4ded38cd82652b6afd075552494e9e783629d4d1bb7c5a9d5243dafa1` |

## Source and Manuscript Hashes

| File | SHA-256 |
|---|---|
| `src/f2d/run_ws4_kappa_margin.py` | `ecee61b499c356e64d7cf7bfad5cf79d9d87d0b03176370e40f778082395b278` |
| `src/f2d/run_mechanism_analysis.py` | `6b945623c1c7bb37c0999c41cbfab1154ff63a488a7aadbd680bc9f01abe88f7` |
| `tests/test_review_robustness.py` | `aa06c849b0b3c7f7e0f4861657808b36ffba8126c710c32a41ffd58449c38074` |
| `tests/test_aggregation.py` | `7c121f010e58531a87f8f08d89abdc495446090c2a4d52296778cad3e60ed014` |
| `docs/07_decision_layer.md` | `cac866bd581e263847dd46518d61631336e4d28376bb0a01c5b9f7c072dc3ab4` |
| `paper/main.tex` | `2eced14570bde182d53955000ed44e354abfb843948c68e4c8c48f39331de694` |
| `paper/online_appendix.tex` | `6c810831c8003057bdce6de1b1db857bc8af88a347383558923554c090cc650b` |
| `paper/references.bib` | `b1910c2f9202bff3d5287a968f778c49b5bcde1cb452fb89676227dd427a1e43` |

## Verification

```bash
python -m pytest -q
```

The frozen extension passes 84 tests. It is accepted when the hashes above
match, the margin summary is an exact deterministic reduction of its two full
output tables, the binding audit is an exact reduction of the registered
policy targets, all midpoint support-cap rows retain their reported directions,
and both manuscript sources compile without unresolved citations or
cross-references.
