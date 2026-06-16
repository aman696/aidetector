# Generator-Family x Architecture Analysis (2026-06-15T15:22:59)

Test rows: 6414 | holdout rows: 2520 | features: 855

Real-anchored: each all-AI family is scored against the pooled real images so AUC is defined (the per-family n/a in the eval report is fixed here). Buckets: udiff=U-Net diffusion, pdiff=pixel diffusion, flow=rectified-flow/DiT, ar=autoregressive, undisc=undisclosed. Architecture map + sources: paper_notes/architectures.md.

## Part C - architecture buckets (clean condition)
### Real-anchored AUC by bucket

| bucket | exposure | n families | mean | median | families |
|---|---|---|---|---|---|
| ar | all | 5 | 0.940 | 0.916 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| ar | seen | 5 | 0.940 | 0.916 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| flow | all | 6 | 0.894 | 0.895 | chroma, flux_1, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | seen | 5 | 0.910 | 0.907 | chroma, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | holdout | 1 | 0.811 | 0.811 | flux_1 |
| pdiff | all | 1 | 0.985 | 0.985 | glide |
| pdiff | seen | 1 | 0.985 | 0.985 | glide |
| udiff | all | 19 | 0.951 | 0.946 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, ideogram_3.0, imagen_3.0_002, imagen_4.0, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | seen | 17 | 0.953 | 0.961 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, imagen_3.0_002, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | holdout | 2 | 0.936 | 0.936 | ideogram_3.0, imagen_4.0 |
| undisc | all | 3 | 0.922 | 0.935 | halfmoon_4_4_25, midjourney_7, recraft_v3 |
| undisc | seen | 1 | 0.935 | 0.935 | halfmoon_4_4_25 |
| undisc | holdout | 2 | 0.915 | 0.915 | midjourney_7, recraft_v3 |

### Pd@5%FAR by bucket

| bucket | exposure | n families | mean | median | families |
|---|---|---|---|---|---|
| ar | all | 5 | 0.693 | 0.615 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| ar | seen | 5 | 0.693 | 0.615 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| flow | all | 6 | 0.487 | 0.385 | chroma, flux_1, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | seen | 5 | 0.508 | 0.385 | chroma, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | holdout | 1 | 0.381 | 0.381 | flux_1 |
| pdiff | all | 1 | 0.923 | 0.923 | glide |
| pdiff | seen | 1 | 0.923 | 0.923 | glide |
| udiff | all | 19 | 0.753 | 0.769 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, ideogram_3.0, imagen_3.0_002, imagen_4.0, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | seen | 17 | 0.769 | 0.769 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, imagen_3.0_002, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | holdout | 2 | 0.611 | 0.611 | ideogram_3.0, imagen_4.0 |
| undisc | all | 3 | 0.651 | 0.667 | halfmoon_4_4_25, midjourney_7, recraft_v3 |
| undisc | seen | 1 | 0.667 | 0.667 | halfmoon_4_4_25 |
| undisc | holdout | 2 | 0.643 | 0.643 | midjourney_7, recraft_v3 |

### Decisive checks
- (i) seen-only mean AUC by bucket: {"ar": 0.94, "flow": 0.91, "udiff": 0.953, "pdiff": 0.985, "undisc": 0.935}
- (ii) held-out diffusion {"midjourney_7": 0.939, "imagen_4.0": 0.925, "ideogram_3.0": 0.946} vs non-diffusion {"flux_1": 0.811, "recraft_v3": 0.891} (diff - nondiff = 0.086)
- (iii) SD3 (flow) vs SD (udiff): {"stable_diffusion_3": 0.907, "sd_1.5": 0.943, "stable_diffusion_2": 0.995}

## Part B - per-family (clean condition, real-anchored)
### By family (clean)

| family | bucket | seen? | n | AUC | Pd@5%FAR | mean p(AI) | acc@0.5 |
|---|---|---|---|---|---|---|---|
| aurora_20_1_25 | ar | seen | 13 | 0.990 | 1.000 | 0.945 | 1.000 |
| gemini | ar | seen | 42 | 0.985 | 0.929 | 0.935 | 1.000 |
| grok_2_image_1212 | ar | seen | 13 | 0.916 | 0.615 | 0.778 | 0.846 |
| gpt_image_1 | ar | seen | 13 | 0.911 | 0.462 | 0.714 | 0.769 |
| gpt | ar | seen | 13 | 0.899 | 0.462 | 0.693 | 0.692 |
| lumina_17_2_25 | flow | seen | 13 | 0.957 | 0.846 | 0.866 | 0.846 |
| mystic | flow | seen | 13 | 0.926 | 0.615 | 0.758 | 0.769 |
| stable_diffusion_3 | flow | seen | 13 | 0.907 | 0.385 | 0.694 | 0.769 |
| hidream_i1_full | flow | seen | 13 | 0.883 | 0.308 | 0.641 | 0.692 |
| chroma | flow | seen | 13 | 0.878 | 0.385 | 0.684 | 0.769 |
| flux_1 | flow | holdout | 63 | 0.811 | 0.381 | 0.548 | 0.540 |
| glide | pdiff | seen | 13 | 0.985 | 0.923 | 0.929 | 1.000 |
| midjourney_v5 | udiff | seen | 13 | 0.996 | 1.000 | 0.974 | 1.000 |
| stable_diffusion_2 | udiff | seen | 13 | 0.995 | 1.000 | 0.973 | 1.000 |
| dalle_3 | udiff | seen | 13 | 0.977 | 0.846 | 0.913 | 1.000 |
| firefly | udiff | seen | 13 | 0.976 | 0.923 | 0.903 | 1.000 |
| sd_1.5_dreamshaper | udiff | seen | 13 | 0.972 | 0.846 | 0.887 | 0.923 |
| dalle_2 | udiff | seen | 13 | 0.970 | 0.846 | 0.874 | 0.923 |
| frames_23_1_25 | udiff | seen | 13 | 0.964 | 0.769 | 0.852 | 0.923 |
| recraft_v2 | udiff | seen | 13 | 0.962 | 0.769 | 0.849 | 0.846 |
| stable_diffusion_1_3 | udiff | seen | 13 | 0.961 | 0.923 | 0.874 | 0.923 |
| ideogram_3.0 | udiff | holdout | 63 | 0.946 | 0.667 | 0.813 | 0.889 |
| sd_1.5 | udiff | seen | 13 | 0.943 | 0.769 | 0.800 | 0.846 |
| imagen_3.0_002 | udiff | seen | 13 | 0.937 | 0.692 | 0.831 | 0.923 |
| ideogram_2.0 | udiff | seen | 13 | 0.936 | 0.846 | 0.821 | 0.846 |
| midjourney_6 | udiff | seen | 13 | 0.928 | 0.538 | 0.734 | 0.769 |
| stable_diffusion_xl | udiff | seen | 13 | 0.927 | 0.538 | 0.757 | 0.923 |
| imagen_4.0 | udiff | holdout | 63 | 0.925 | 0.556 | 0.754 | 0.794 |
| stable_diffusion_1_4 | udiff | seen | 13 | 0.925 | 0.769 | 0.798 | 0.846 |
| sd_1.5_epicdream | udiff | seen | 13 | 0.921 | 0.462 | 0.734 | 0.846 |
| sd_2.1 | udiff | seen | 13 | 0.909 | 0.538 | 0.719 | 0.692 |
| midjourney_7 | undisc | holdout | 63 | 0.939 | 0.730 | 0.809 | 0.841 |
| halfmoon_4_4_25 | undisc | seen | 9 | 0.935 | 0.667 | 0.765 | 0.889 |
| recraft_v3 | undisc | holdout | 63 | 0.891 | 0.556 | 0.714 | 0.730 |

## Part E.2 - signal-family ablation probe (group-aware OOF AUC)
| subset | n feat | udiff | pdiff | flow | ar | undisc |
|---|---|---|---|---|---|---|
| freq_forensic | 24 | 0.831 | 0.790 | 0.754 | 0.818 | 0.791 |
| all_classical | 85 | 0.858 | 0.820 | 0.809 | 0.905 | 0.857 |
| embedding | 768 | 0.832 | 0.860 | 0.802 | 0.851 | 0.828 |
| emb_drift | 2 | 0.513 | 0.410 | 0.478 | 0.506 | 0.560 |

## Part E.1 - embedding gain per family (hybrid AUC - classical AUC)
| family | bucket | classical | hybrid | embedding gain |
|---|---|---|---|---|
| dalle_2 | udiff | 0.659 | 0.970 | +0.311 |
| grok_2_image_1212 | ar | 0.802 | 0.916 | +0.114 |
| sd_1.5 | udiff | 0.834 | 0.943 | +0.109 |
| stable_diffusion_1_3 | udiff | 0.857 | 0.961 | +0.104 |
| glide | pdiff | 0.905 | 0.985 | +0.080 |
| lumina_17_2_25 | flow | 0.878 | 0.957 | +0.079 |
| recraft_v2 | udiff | 0.888 | 0.962 | +0.074 |
| stable_diffusion_2 | udiff | 0.925 | 0.995 | +0.070 |
| midjourney_v5 | udiff | 0.961 | 0.996 | +0.035 |
| dalle_3 | udiff | 0.943 | 0.977 | +0.034 |
| mystic | flow | 0.894 | 0.926 | +0.032 |
| aurora_20_1_25 | ar | 0.961 | 0.990 | +0.028 |
| frames_23_1_25 | udiff | 0.936 | 0.964 | +0.028 |
| midjourney_6 | udiff | 0.901 | 0.928 | +0.027 |
| halfmoon_4_4_25 | undisc | 0.912 | 0.935 | +0.023 |
| sd_2.1 | udiff | 0.886 | 0.909 | +0.023 |
| gpt_image_1 | ar | 0.891 | 0.911 | +0.020 |
| sd_1.5_dreamshaper | udiff | 0.955 | 0.972 | +0.017 |
| ideogram_2.0 | udiff | 0.922 | 0.936 | +0.014 |
| firefly | udiff | 0.963 | 0.976 | +0.013 |
| stable_diffusion_xl | udiff | 0.916 | 0.927 | +0.012 |
| imagen_3.0_002 | udiff | 0.932 | 0.937 | +0.004 |
| imagen_4.0 | udiff | 0.930 | 0.925 | -0.004 |
| hidream_i1_full | flow | 0.890 | 0.883 | -0.007 |
| gemini | ar | 0.992 | 0.985 | -0.008 |
| stable_diffusion_1_4 | udiff | 0.938 | 0.925 | -0.013 |
| midjourney_7 | undisc | 0.960 | 0.939 | -0.020 |
| stable_diffusion_3 | flow | 0.936 | 0.907 | -0.030 |
| flux_1 | flow | 0.842 | 0.811 | -0.031 |
| chroma | flow | 0.911 | 0.878 | -0.033 |
| ideogram_3.0 | udiff | 0.982 | 0.946 | -0.035 |
| sd_1.5_epicdream | udiff | 0.967 | 0.921 | -0.046 |
| recraft_v3 | undisc | 0.948 | 0.891 | -0.058 |
| gpt | ar | 0.961 | 0.899 | -0.062 |

## Part D - classical features whose diffusion signal collapses on flow
Standardized distance from real mean (z); ranked by |z_udiff| - |z_flow| (the gap that disappears).
| feature | z_udiff | z_flow | z_ar | |z_udiff|-|z_flow| |
|---|---|---|---|---|
| drift_eig_condition_number | 0.573 | 0.043 | 0.406 | +0.530 |
| noise_block_var_std | -0.482 | -0.044 | -0.486 | +0.438 |
| drift_gradient_variance | -0.404 | 0.149 | -0.479 | +0.255 |
| drift_noise_variance | 0.327 | 0.091 | 0.263 | +0.236 |
| lbp_entropy | 0.787 | 0.574 | 0.718 | +0.213 |
| chroma_std_ratio | 0.437 | 0.240 | -0.083 | +0.197 |
| gradient_variance | -0.433 | -0.242 | -0.593 | +0.192 |
| eig_ratio_2_3 | 0.406 | 0.221 | 0.409 | +0.185 |
| eig_condition_number | 0.313 | 0.169 | 0.390 | +0.145 |
| drift_spectral_slope | 0.412 | 0.269 | 0.968 | +0.144 |
| npr_skewness | 0.141 | 0.001 | 0.144 | +0.141 |
| noise_spectral_entropy | 0.271 | 0.147 | 0.060 | +0.124 |
| meta_tag_count | 0.118 | 0.000 | 0.000 | +0.118 |
| drift_noise_autocorrelation | 0.339 | 0.224 | 1.044 | +0.115 |
| eig_ratio_1_2 | 0.164 | 0.055 | 0.043 | +0.109 |
| drift_noise_spectral_entropy | 0.236 | 0.137 | 0.895 | +0.099 |
| patch_dominance_mean | 0.183 | 0.091 | 0.352 | +0.092 |
| noise_skewness | 0.111 | -0.056 | 0.420 | +0.054 |
| drift_dct_zigzag_decay | 0.107 | -0.061 | 0.405 | +0.046 |
| wavelet_hh_energy | 0.341 | 0.297 | -0.030 | +0.044 |

## Part E.3 - resolution-controlled re-check (weak families)
Real-anchored AUC within each resolution bucket (reals from the same bucket). Low across buckets = not just a resolution artifact.
| family | <400 | 400-800 | >800 |
|---|---|---|---|
| chroma | n/a | 0.819 | 0.711 |
| gpt | n/a | n/a | 0.721 |
| hidream_i1_full | n/a | 0.980 | 0.666 |

