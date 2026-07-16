# Ordinal Modeling Study

This package contains the ordinal-axis evaluation used to test whether an image axis preserves ordered visual labels. The current paper-facing study uses fixed CLIP embeddings, normalized ordinal targets in `[0,1]`, and simulated feedback budgets `b in {0,1,3,5,10,20}` with five repetitions.

## Current Dataset Suite

The expanded study has 87 axes across eight prepared UI datasets.

| dataset | target families |
| --- | --- |
| `SCUTFBP5500` | facial attractiveness plus low-level visual features |
| `ISIC2024` | lesion geometry/asymmetry/irregularity plus low-level visual features |
| `Messidor2` | diabetic-retinopathy severity plus low-level visual features |
| `VinDrMammo` | BI-RADS, breast density, finding size, grayscale low-level features |
| `AADB` | aesthetics, color harmony, vivid color, composition, low-level visual features |
| `LaMem` | memorability plus low-level visual features |
| `OASIS` | valence, arousal, beauty, low-level visual features |
| `HousePrices` | price, bedrooms, bathrooms, area, low-level visual features |

Axis categories are `low_level`, `medium_level`, `high_level`, and `abstract`.

## Metrics

| metric | meaning |
| --- | --- |
| `spearman` | rank correlation between learned scores and ground-truth ordinal labels |
| `kendall_tau` | sampled pairwise ordering agreement |
| `mae_isotonic` | MAE after isotonic calibration from score ranks to labels |
| `qwk` | quadratic weighted kappa after discretizing calibrated scores |
| `axis_r2` | linear-fit quality of the bin-level ground-truth trend along the learned axis |
| `monotonicity_violations` | adjacent axis bins where mean ground-truth label decreases |

## Compared Methods

| method | role |
| --- | --- |
| `text_prior` | zero-label CLIP direction from low/high anchors |
| `prompt_ladder` | zero-shot ordered prompt ladder |
| `reaxis_random` | Gaussian AxisBayes scalar-feedback ReAxis with random feedback |
| `reaxis_active` | Gaussian AxisBayes scalar-feedback ReAxis with uncertainty-selected feedback |
| `reaxis_gaussian_tuned` | best compact Gaussian sweep setting, evaluated as an ablation |
| `reaxis_log` | Gaussian ReAxis after log-normalizing skewed raw targets |
| `reaxis_pairwise` | prior-regularized pairwise ReAxis ablation that nearly matches RankSVM |
| `prior_affine` | affine calibration of the text-prior score |
| `label_mean` | constant label-mean sanity baseline |
| `ols_linear`, `bayesian_ridge`, `ordinal_ridge`, `linear_svr`, `pcr_ridge`, `elastic_net` | label-only pointwise linear controls |
| `rank_svm` | label-only pairwise ranking baseline |
| `knn_ordinal` | label-only nearest-neighbor propagation |
| `kernel_ridge` | label-only nonlinear RBF regressor |

Rank-mode ReAxis, centered ReAxis, and calibrated-residual ReAxis are retained as diagnostics in older artifacts, but they are not part of the current paper-facing comparison.

## Headline Results

The paper-facing interpretation is a warm-start transition regime:

| budget | best ReAxis | ReAxis rho | best pure label-only | label-only rho | margin |
| ---: | --- | ---: | --- | ---: | ---: |
| 0 | `reaxis_pairwise` | 0.216 | `label_mean` | 0.000 | +0.216 |
| 1 | `reaxis_active` | 0.220 | `label_mean` | 0.000 | +0.220 |
| 3 | `reaxis_active` | 0.238 | `ols_linear` | 0.135 | +0.103 |
| 5 | `reaxis_active` | 0.253 | `ols_linear` | 0.205 | +0.048 |
| 10 | `reaxis_pairwise` | 0.298 | `ols_linear` | 0.292 | +0.006 |
| 20 | `reaxis_pairwise` | 0.372 | `rank_svm` | 0.375 | -0.003 |

Scalar-Gaussian ReAxis improves over the text prior but does not close the label-only RankSVM/OLS gap at high budget: `reaxis_random` reaches `0.313`, tuned Gaussian reaches `0.316`, and `rank_svm` reaches `0.375` at `b=20`. The pairwise ReAxis ablation reaches `0.372`, showing that the main gap is the scalar-Gaussian objective, not linearity or the text prior.

## Main Entry Points

| command | purpose |
| --- | --- |
| `python backend/run_ordinal_modeling_study.py ...` | shared ordinal runner for task/method subsets |
| `python backend/run_ordinal_gaussian_sweep.py ...` | Gaussian scalar-feedback hyperparameter sweep |
| `python backend/run_ordinal_centered_sweep.py ...` | older centered Gaussian diagnostic sweep |
| `python backend/run_ordinal_prompt_sweep.py ...` | prompt-initialization diagnostic sweep |
| `python backend/plot_ordinal_prompt_baselines.py` | combined prompt/baseline Spearman plot |

## Main Artifacts

| file | contents |
| --- | --- |
| `backend/experiments/ordinal_modeling_diagnostics/ordinal_modeling_study_paper_section.md` | current paper-facing technical modeling-study draft |
| `backend/experiments/ordinal_modeling_diagnostics/ordinal_all_methods_summary_for_paper.csv` | consolidated method/budget metrics |
| `backend/experiments/ordinal_modeling_diagnostics/ordinal_all_methods_spearman_curve_for_paper.csv` | Spearman curve table used in the draft |
| `backend/experiments/ordinal_modeling_diagnostics/ordinal_all_methods_b10_b20_for_paper.csv` | compact b10/b20 metric table |
| `backend/experiments/ordinal_modeling_diagnostics/ordinal_best_reaxis_vs_label_only_for_paper.csv` | transition-regime comparison |
| `backend/experiments/ordinal_modeling_expanded/metrics.csv` | expanded Gaussian ReAxis and original baselines |
| `backend/experiments/ordinal_label_only_more/metrics.csv` | added label-only methods and `reaxis_pairwise` |
| `backend/experiments/ordinal_modeling_expanded_log/metrics.csv` | log-target ablation |
| `backend/experiments/ordinal_gaussian_sweep_top_full/ordinal_gaussian_sweep_results.csv` | full-metric tuned Gaussian run |

## Interpretation Notes

Use `rank_svm` as a strong high-budget label-only rank baseline. It is strong because it directly optimizes pairwise order and converts sparse labels into many constraints.

Use scalar-Gaussian ReAxis as the closest implementation of the paper-method feedback model. It gives a real but modest feedback progression.

Use `reaxis_pairwise` as an ablation showing what happens when ReAxis keeps the text-prior warm start but switches to an ordinal ranking objective. It should be described as diagnostic unless the method section is updated to include pairwise feedback as a first-class model.
