# Generator-Family x Architecture Analysis (2026-06-13T18:58:50)

Test rows: 6414 | holdout rows: 2520 | features: 855

Real-anchored: each all-AI family is scored against the pooled real images so AUC is defined (the per-family n/a in the eval report is fixed here). Buckets: udiff=U-Net diffusion, pdiff=pixel diffusion, flow=rectified-flow/DiT, ar=autoregressive, undisc=undisclosed. Architecture map + sources: paper_notes/architectures.md.

## Part C - architecture buckets (clean condition)
### Real-anchored AUC by bucket

| bucket | exposure | n families | mean | median | families |
|---|---|---|---|---|---|
| ar | all | 5 | 0.943 | 0.921 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| ar | seen | 5 | 0.943 | 0.921 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| flow | all | 6 | 0.902 | 0.909 | chroma, flux_1, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | seen | 5 | 0.918 | 0.913 | chroma, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | holdout | 1 | 0.820 | 0.820 | flux_1 |
| pdiff | all | 1 | 0.964 | 0.964 | glide |
| pdiff | seen | 1 | 0.964 | 0.964 | glide |
| udiff | all | 19 | 0.953 | 0.952 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, ideogram_3.0, imagen_3.0_002, imagen_4.0, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | seen | 17 | 0.954 | 0.952 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, imagen_3.0_002, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | holdout | 2 | 0.942 | 0.942 | ideogram_3.0, imagen_4.0 |
| undisc | all | 3 | 0.923 | 0.932 | halfmoon_4_4_25, midjourney_7, recraft_v3 |
| undisc | seen | 1 | 0.944 | 0.944 | halfmoon_4_4_25 |
| undisc | holdout | 2 | 0.913 | 0.913 | midjourney_7, recraft_v3 |

### Pd@5%FAR by bucket

| bucket | exposure | n families | mean | median | families |
|---|---|---|---|---|---|
| ar | all | 5 | 0.724 | 0.615 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| ar | seen | 5 | 0.724 | 0.615 | aurora_20_1_25, gemini, gpt, gpt_image_1, grok_2_image_1212 |
| flow | all | 6 | 0.512 | 0.462 | chroma, flux_1, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | seen | 5 | 0.538 | 0.538 | chroma, hidream_i1_full, lumina_17_2_25, mystic, stable_diffusion_3 |
| flow | holdout | 1 | 0.381 | 0.381 | flux_1 |
| pdiff | all | 1 | 0.769 | 0.769 | glide |
| pdiff | seen | 1 | 0.769 | 0.769 | glide |
| udiff | all | 19 | 0.773 | 0.778 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, ideogram_3.0, imagen_3.0_002, imagen_4.0, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | seen | 17 | 0.783 | 0.846 | dalle_2, dalle_3, firefly, frames_23_1_25, ideogram_2.0, imagen_3.0_002, midjourney_6, midjourney_v5, recraft_v2, sd_1.5, sd_1.5_dreamshaper, sd_1.5_epicdream, sd_2.1, stable_diffusion_1_3, stable_diffusion_1_4, stable_diffusion_2, stable_diffusion_xl |
| udiff | holdout | 2 | 0.690 | 0.690 | ideogram_3.0, imagen_4.0 |
| undisc | all | 3 | 0.661 | 0.667 | halfmoon_4_4_25, midjourney_7, recraft_v3 |
| undisc | seen | 1 | 0.667 | 0.667 | halfmoon_4_4_25 |
| undisc | holdout | 2 | 0.659 | 0.659 | midjourney_7, recraft_v3 |

### Decisive checks
- (i) seen-only mean AUC by bucket: {"ar": 0.943, "flow": 0.918, "udiff": 0.954, "pdiff": 0.964, "undisc": 0.944}
- (ii) held-out diffusion {"midjourney_7": 0.932, "imagen_4.0": 0.932, "ideogram_3.0": 0.952} vs non-diffusion {"flux_1": 0.82, "recraft_v3": 0.894} (diff - nondiff = 0.082)
- (iii) SD3 (flow) vs SD (udiff): {"stable_diffusion_3": 0.905, "sd_1.5": 0.944, "stable_diffusion_2": 0.994}

## Part B - per-family (clean condition, real-anchored)
### By family (clean)

| family | bucket | seen? | n | AUC | Pd@5%FAR | mean p(AI) | acc@0.5 |
|---|---|---|---|---|---|---|---|
| aurora_20_1_25 | ar | seen | 13 | 0.990 | 1.000 | 0.995 | 1.000 |
| gemini | ar | seen | 42 | 0.987 | 0.929 | 0.980 | 1.000 |
| gpt_image_1 | ar | seen | 13 | 0.921 | 0.615 | 0.778 | 0.846 |
| grok_2_image_1212 | ar | seen | 13 | 0.919 | 0.615 | 0.844 | 0.846 |
| gpt | ar | seen | 13 | 0.896 | 0.462 | 0.745 | 0.769 |
| lumina_17_2_25 | flow | seen | 13 | 0.963 | 0.846 | 0.897 | 0.923 |
| mystic | flow | seen | 13 | 0.927 | 0.615 | 0.791 | 0.769 |
| hidream_i1_full | flow | seen | 13 | 0.913 | 0.308 | 0.759 | 0.846 |
| stable_diffusion_3 | flow | seen | 13 | 0.905 | 0.385 | 0.751 | 0.769 |
| chroma | flow | seen | 13 | 0.882 | 0.538 | 0.785 | 0.846 |
| flux_1 | flow | holdout | 63 | 0.820 | 0.381 | 0.569 | 0.571 |
| glide | pdiff | seen | 13 | 0.964 | 0.769 | 0.916 | 0.923 |
| midjourney_v5 | udiff | seen | 13 | 0.997 | 1.000 | 0.999 | 1.000 |
| stable_diffusion_2 | udiff | seen | 13 | 0.994 | 0.923 | 0.994 | 1.000 |
| dalle_3 | udiff | seen | 13 | 0.983 | 0.846 | 0.961 | 1.000 |
| firefly | udiff | seen | 13 | 0.979 | 0.923 | 0.962 | 1.000 |
| sd_1.5_dreamshaper | udiff | seen | 13 | 0.974 | 0.846 | 0.932 | 0.923 |
| dalle_2 | udiff | seen | 13 | 0.973 | 0.846 | 0.942 | 0.923 |
| recraft_v2 | udiff | seen | 13 | 0.961 | 0.846 | 0.902 | 0.923 |
| stable_diffusion_1_3 | udiff | seen | 13 | 0.957 | 0.846 | 0.912 | 0.923 |
| ideogram_3.0 | udiff | holdout | 63 | 0.952 | 0.778 | 0.890 | 0.921 |
| frames_23_1_25 | udiff | seen | 13 | 0.952 | 0.769 | 0.880 | 0.846 |
| imagen_3.0_002 | udiff | seen | 13 | 0.947 | 0.692 | 0.885 | 0.923 |
| sd_1.5 | udiff | seen | 13 | 0.944 | 0.769 | 0.872 | 0.923 |
| ideogram_2.0 | udiff | seen | 13 | 0.943 | 0.846 | 0.861 | 0.846 |
| imagen_4.0 | udiff | holdout | 63 | 0.932 | 0.603 | 0.834 | 0.905 |
| stable_diffusion_xl | udiff | seen | 13 | 0.930 | 0.615 | 0.822 | 0.846 |
| sd_1.5_epicdream | udiff | seen | 13 | 0.927 | 0.615 | 0.830 | 0.923 |
| midjourney_6 | udiff | seen | 13 | 0.926 | 0.538 | 0.797 | 0.769 |
| stable_diffusion_1_4 | udiff | seen | 13 | 0.923 | 0.769 | 0.829 | 0.846 |
| sd_2.1 | udiff | seen | 13 | 0.911 | 0.615 | 0.749 | 0.692 |
| halfmoon_4_4_25 | undisc | seen | 9 | 0.944 | 0.667 | 0.871 | 0.889 |
| midjourney_7 | undisc | holdout | 63 | 0.932 | 0.730 | 0.839 | 0.841 |
| recraft_v3 | undisc | holdout | 63 | 0.894 | 0.587 | 0.765 | 0.762 |

## Part E.2 - signal-family ablation probe (group-aware OOF AUC)
| subset | n feat | udiff | pdiff | flow | ar | undisc |
|---|---|---|---|---|---|---|
| freq_forensic | 24 | 0.882 | 0.136 | 0.850 | 0.904 | 0.899 |
| all_classical | 85 | 0.878 | 0.479 | 0.847 | 0.922 | 0.902 |
| embedding | 768 | 0.832 | 0.860 | 0.802 | 0.851 | 0.828 |
| emb_drift | 2 | 0.513 | 0.410 | 0.478 | 0.506 | 0.560 |

## Part E.1 - embedding gain per family (hybrid AUC - classical AUC)
| family | bucket | classical | hybrid | embedding gain |
|---|---|---|---|---|
| glide | pdiff | 0.657 | 0.964 | +0.308 |
| dalle_2 | udiff | 0.721 | 0.973 | +0.251 |
| stable_diffusion_1_3 | udiff | 0.872 | 0.957 | +0.084 |
| grok_2_image_1212 | ar | 0.854 | 0.919 | +0.065 |
| recraft_v2 | udiff | 0.904 | 0.961 | +0.057 |
| stable_diffusion_2 | udiff | 0.949 | 0.994 | +0.045 |
| lumina_17_2_25 | flow | 0.922 | 0.963 | +0.041 |
| frames_23_1_25 | udiff | 0.912 | 0.952 | +0.040 |
| ideogram_2.0 | udiff | 0.906 | 0.943 | +0.037 |
| midjourney_v5 | udiff | 0.969 | 0.997 | +0.027 |
| stable_diffusion_xl | udiff | 0.905 | 0.930 | +0.025 |
| mystic | flow | 0.903 | 0.927 | +0.023 |
| aurora_20_1_25 | ar | 0.970 | 0.990 | +0.020 |
| sd_1.5 | udiff | 0.928 | 0.944 | +0.016 |
| dalle_3 | udiff | 0.977 | 0.983 | +0.006 |
| stable_diffusion_1_4 | udiff | 0.917 | 0.923 | +0.006 |
| sd_2.1 | udiff | 0.906 | 0.911 | +0.005 |
| imagen_3.0_002 | udiff | 0.944 | 0.947 | +0.003 |
| gpt_image_1 | ar | 0.920 | 0.921 | +0.002 |
| midjourney_6 | udiff | 0.925 | 0.926 | +0.001 |
| gemini | ar | 0.996 | 0.987 | -0.009 |
| firefly | udiff | 0.992 | 0.979 | -0.012 |
| sd_1.5_dreamshaper | udiff | 0.990 | 0.974 | -0.016 |
| midjourney_7 | undisc | 0.949 | 0.932 | -0.016 |
| sd_1.5_epicdream | udiff | 0.946 | 0.927 | -0.018 |
| halfmoon_4_4_25 | undisc | 0.964 | 0.944 | -0.020 |
| imagen_4.0 | udiff | 0.958 | 0.932 | -0.026 |
| recraft_v3 | undisc | 0.924 | 0.894 | -0.030 |
| stable_diffusion_3 | flow | 0.936 | 0.905 | -0.031 |
| hidream_i1_full | flow | 0.944 | 0.913 | -0.032 |
| ideogram_3.0 | udiff | 0.989 | 0.952 | -0.036 |
| chroma | flow | 0.925 | 0.882 | -0.043 |
| gpt | ar | 0.945 | 0.896 | -0.049 |
| flux_1 | flow | 0.892 | 0.820 | -0.072 |

## Part D - classical features whose diffusion signal collapses on flow
Standardized distance from real mean (z); ranked by |z_udiff| - |z_flow| (the gap that disappears).
| feature | z_udiff | z_flow | z_ar | |z_udiff|-|z_flow| |
|---|---|---|---|---|
| drift_eig_condition_number | 1.781 | -0.197 | 2.159 | +1.585 |
| noise_block_var_std | -0.482 | -0.044 | -0.486 | +0.438 |
| drift_gradient_variance | -0.457 | 0.144 | -0.539 | +0.313 |
| lbp_entropy | 0.787 | 0.574 | 0.718 | +0.213 |
| chroma_std_ratio | 0.437 | 0.240 | -0.083 | +0.197 |
| gradient_variance | -0.433 | -0.242 | -0.593 | +0.192 |
| eig_ratio_2_3 | 0.406 | 0.221 | 0.409 | +0.185 |
| slope_r_squared | -0.281 | -0.100 | -0.728 | +0.181 |
| drift_spectral_slope | 0.160 | -0.002 | 0.851 | +0.158 |
| eig_condition_number | 0.313 | 0.169 | 0.390 | +0.145 |
| npr_skewness | 0.141 | 0.001 | 0.144 | +0.141 |
| drift_high_freq_ratio | 0.166 | 0.040 | 0.860 | +0.126 |
| noise_spectral_entropy | 0.271 | 0.147 | 0.060 | +0.124 |
| meta_tag_count | 0.118 | 0.000 | 0.000 | +0.118 |
| spectral_slope | 0.150 | 0.032 | -0.225 | +0.117 |
| eig_ratio_1_2 | 0.164 | 0.055 | 0.043 | +0.109 |
| band_mid_ratio | 0.133 | -0.038 | 0.229 | +0.095 |
| patch_dominance_mean | 0.183 | 0.091 | 0.352 | +0.092 |
| drift_texture_rich_mean | -0.235 | -0.161 | -0.024 | +0.074 |
| gradient_laplacian_variance | -0.425 | -0.369 | -0.638 | +0.055 |

## Part E.3 - resolution-controlled re-check (weak families)
Real-anchored AUC within each resolution bucket (reals from the same bucket). Low across buckets = not just a resolution artifact.
| family | <400 | 400-800 | >800 |
|---|---|---|---|
| chroma | n/a | 0.829 | 0.737 |
| gpt | n/a | n/a | 0.710 |

