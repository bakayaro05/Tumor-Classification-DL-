# FINAL PROJECT QC REPORT

Generated during preparation of the final deliverables (read-only with respect to repository evidence).
Every number below was **re-read directly from the raw repository JSON/CSV/joblib/source files** by `gen_docs.py`, independently of the
fact sheet used to write the report, and then searched for in the generated report (Markdown **and** PDF text) and, where marked, in the PPT
(slide text, tables and speaker notes).

## 1. Summary

| Check family | Result |
|---|---|
| Automated numeric / label checks | **174 PASS, 0 FAIL** (of 174) |
| Frozen-artifact SHA-256 hashes | 75/75 match |
| PPT slides with speaker notes | 43/43 (missing: none) |
| Claim-rule terminology scan | see §4 |
| Visual QC | see §5 |

## 2. Automated checks (value → present in report? → present in PPT?)

"in report" = found in both FINAL_PROJECT_REPORT.md and the PDF text. PPT presence is only required for headline numbers.

| Area | Item | Verified value | In report | In PPT | Status | Source / field |
|---|---|---|---|---|---|---|
| Phase 1 | vgg16 PCA+SVM test acc | 94.97% | yes | yes | **PASS** | `experiments/vgg16/results/final_results.json → test.accuracy` |
| Phase 1 | vgg16 neural-head test acc | 93.67% | yes | yes | **PASS** | `experiments/vgg16/results/neural_head_results.json → test.accuracy` |
| Phase 1 | vgg16 best epoch | 16 | yes | yes | **PASS** | `final_results.json → neural_network.best_epoch` |
| Phase 1 | vgg19 PCA+SVM test acc | 93.52% | yes | yes | **PASS** | `experiments/vgg19/results/final_results.json → test.accuracy` |
| Phase 1 | vgg19 neural-head test acc | 93.14% | yes | yes | **PASS** | `experiments/vgg19/results/neural_head_results.json → test.accuracy` |
| Phase 1 | vgg19 best epoch | 18 | yes | yes | **PASS** | `final_results.json → neural_network.best_epoch` |
| Phase 1 | resnet50 PCA+SVM test acc | 98.09% | yes | yes | **PASS** | `experiments/resnet50/results/final_results.json → test.accuracy` |
| Phase 1 | resnet50 neural-head test acc | 97.33% | yes | yes | **PASS** | `experiments/resnet50/results/neural_head_results.json → test.accuracy` |
| Phase 1 | resnet50 best epoch | 7 | yes | yes | **PASS** | `final_results.json → neural_network.best_epoch` |
| Phase 1 | resnet152 PCA+SVM test acc | 98.40% | yes | yes | **PASS** | `experiments/resnet152/results/final_results.json → test.accuracy` |
| Phase 1 | resnet152 neural-head test acc | 98.17% | yes | yes | **PASS** | `experiments/resnet152/results/neural_head_results.json → test.accuracy` |
| Phase 1 | resnet152 best epoch | 8 | yes | yes | **PASS** | `final_results.json → neural_network.best_epoch` |
| Phase 1 | efficientnet_b3 PCA+SVM test acc | 98.47% | yes | yes | **PASS** | `experiments/efficientnet_b3/results/final_results.json → test.accuracy` |
| Phase 1 | efficientnet_b3 neural-head test acc | 98.02% | yes | yes | **PASS** | `experiments/efficientnet_b3/results/neural_head_results.json → test.accuracy` |
| Phase 1 | efficientnet_b3 best epoch | 6 | yes | yes | **PASS** | `final_results.json → neural_network.best_epoch` |
| Phase 1 | EfficientNet-B3 neural→SVM test gain | +0.46 pp | yes | yes | **PASS** | `derived from the two JSONs above` |
| Leakage audit | cross_split_exact_or_pixel_duplicate_pairs | 464 | yes | yes | **PASS** | `duplicate_leakage_audit.json → cross_split_exact_or_pixel_duplicate_pairs` |
| Leakage audit | high_confidence_near_duplicate_pairs | 98 | yes | yes | **PASS** | `duplicate_leakage_audit.json → high_confidence_near_duplicate_pairs` |
| Leakage audit | material_pairs | 562 | yes | yes | **PASS** | `duplicate_leakage_audit.json → material_pairs` |
| Leakage audit | near_duplicate_review_pairs | 481 | yes | yes | **PASS** | `duplicate_leakage_audit.json → near_duplicate_review_pairs` |
| material_clean (P1) | vgg16 acc | 94.816% | yes | yes | **PASS** | `leakage_corrected_summary.csv → accuracy_pct (policy=material_clean)` |
| material_clean (P1) | vgg16 macro F1 | 94.302% | yes | yes | **PASS** | `leakage_corrected_summary.csv → macro_f1_pct` |
| material_clean (P1) | vgg19 acc | 93.025% | yes | yes | **PASS** | `leakage_corrected_summary.csv → accuracy_pct (policy=material_clean)` |
| material_clean (P1) | vgg19 macro F1 | 92.415% | yes | yes | **PASS** | `leakage_corrected_summary.csv → macro_f1_pct` |
| material_clean (P1) | resnet50 acc | 98.303% | yes | yes | **PASS** | `leakage_corrected_summary.csv → accuracy_pct (policy=material_clean)` |
| material_clean (P1) | resnet50 macro F1 | 98.134% | yes | yes | **PASS** | `leakage_corrected_summary.csv → macro_f1_pct` |
| material_clean (P1) | resnet152 acc | 98.398% | yes | yes | **PASS** | `leakage_corrected_summary.csv → accuracy_pct (policy=material_clean)` |
| material_clean (P1) | resnet152 macro F1 | 98.315% | yes | yes | **PASS** | `leakage_corrected_summary.csv → macro_f1_pct` |
| material_clean (P1) | efficientnet_b3 acc | 98.680% | yes | yes | **PASS** | `leakage_corrected_summary.csv → accuracy_pct (policy=material_clean)` |
| material_clean (P1) | efficientnet_b3 macro F1 | 98.511% | yes | yes | **PASS** | `leakage_corrected_summary.csv → macro_f1_pct` |
| Leakage policies | exact_clean n | 1,095 | yes | yes | **PASS** | `leakage_corrected_summary.csv → test_n` |
| Leakage policies | material_clean n | 1,061 | yes | yes | **PASS** | `leakage_corrected_summary.csv → test_n` |
| Leakage policies | conservative_clean n | 926 | yes | yes | **PASS** | `leakage_corrected_summary.csv → test_n` |
| Phase 2A | vgg16 SVM val (best_val_within_30) | 95.888% | yes | yes | **PASS** | `phase2/experiments_30epoch/vgg16/evaluations/best_val_within_30/results/final_results.json → validation.accuracy` |
| Phase 2A | vgg16 internal test | 96.186% | yes | yes | **PASS** | `same → test.accuracy` |
| Phase 2A | vgg19 SVM val (best_val_within_30) | 95.626% | yes | yes | **PASS** | `phase2/experiments_30epoch/vgg19/evaluations/best_val_within_30/results/final_results.json → validation.accuracy` |
| Phase 2A | vgg19 internal test | 95.271% | yes | yes | **PASS** | `same → test.accuracy` |
| Phase 2A | resnet50 SVM val (best_val_within_30) | 98.688% | yes | yes | **PASS** | `phase2/experiments_30epoch/resnet50/evaluations/best_val_within_30/results/final_results.json → validation.accuracy` |
| Phase 2A | resnet50 internal test | 98.932% | yes | yes | **PASS** | `same → test.accuracy` |
| Phase 2A | resnet152 SVM val (best_val_within_30) | 98.600% | yes | yes | **PASS** | `phase2/experiments_30epoch/resnet152/evaluations/best_val_within_30/results/final_results.json → validation.accuracy` |
| Phase 2A | resnet152 internal test | 98.703% | yes | yes | **PASS** | `same → test.accuracy` |
| Phase 2A | efficientnet_b3 SVM val (best_val_within_30) | 99.300% | yes | yes | **PASS** | `phase2/experiments_30epoch/efficientnet_b3/evaluations/best_val_within_30/results/final_results.json → validation.accuracy` |
| Phase 2A | efficientnet_b3 internal test | 99.314% | yes | yes | **PASS** | `same → test.accuracy` |
| Phase 2A | ResNet152 ep10 material_clean acc | 98.869% | yes | yes | **PASS** | `PHASE2A_PRIMARY_material_clean_comparison.csv → accuracy_pct` |
| Phase 2A | ResNet152 ep10 material_clean F1 | 98.788% | yes | no | **PASS** | `same → macro_f1_pct` |
| State labels | vgg16 Phase 2A frozen state | best_val_within_30 @ epoch 30 | yes | yes | **PASS** | `frozen_15_selection.csv → checkpoint_state, selected_epoch` |
| State labels | vgg19 Phase 2A frozen state | best_val_within_30 @ epoch 30 | yes | yes | **PASS** | `frozen_15_selection.csv → checkpoint_state, selected_epoch` |
| State labels | resnet50 Phase 2A frozen state | best_val_within_30 @ epoch 30 | yes | yes | **PASS** | `frozen_15_selection.csv → checkpoint_state, selected_epoch` |
| Phase 2B | vgg16 lr_5e-4 val | 96.500% | yes | yes | **PASS** | `frozen_15_selection.csv → internal_validation_accuracy` |
| Phase 2B | vgg19 dropout_0.5 val | 95.538% | yes | yes | **PASS** | `frozen_15_selection.csv → internal_validation_accuracy` |
| Phase 2B | resnet50 dropout_0.2 val | 99.125% | yes | yes | **PASS** | `frozen_15_selection.csv → internal_validation_accuracy` |
| Phase 2B | resnet152 lstm_512 val | 99.300% | yes | yes | **PASS** | `frozen_15_selection.csv → internal_validation_accuracy` |
| Phase 2B | efficientnet_b3 lstm_512 val | 99.388% | yes | yes | **PASS** | `frozen_15_selection.csv → internal_validation_accuracy` |
| Freeze | SHA-256 hashes matching | 75/75 | yes | yes | **PASS** | `frozen_15_manifest.json → candidates[].sha256 vs recomputed` |
| BDNeuro | raw MRIs | 5,941 | yes | yes | **PASS** | `bdneuro_overlap_audit/summary.json → bdneuro_images` |
| BDNeuro | exact file pairs | 3,421 | yes | yes | **PASS** | `summary.json → cross_dataset.exact_file_duplicate_pairs` |
| BDNeuro | exact pixel pairs | 3,519 | yes | yes | **PASS** | `summary.json → cross_dataset.exact_pixel_duplicate_pairs` |
| BDNeuro | unique exact-pixel images excluded | 3,297 | yes | yes | **PASS** | `cohort_summary.json → unique_exact_pixel_images_excluded` |
| BDNeuro | exact-clean cohort | 2,644 | yes | yes | **PASS** | `cohort_summary.json → exact_pixel_clean_images_kept` |
| BDNeuro | kept glioma | 1,159 | yes | yes | **PASS** | `cohort_summary.json → kept_by_class` |
| BDNeuro | kept meningioma | 472 | yes | yes | **PASS** | `cohort_summary.json → kept_by_class` |
| BDNeuro | kept notumor | 412 | yes | yes | **PASS** | `cohort_summary.json → kept_by_class` |
| BDNeuro | kept pituitary | 601 | yes | yes | **PASS** | `cohort_summary.json → kept_by_class` |
| BDNeuro | balanced total | 1,648 | yes | yes | **PASS** | `balanced_subset_summary.json → balanced_total` |
| BDNeuro full | Phase 1 vgg16 acc | 77.61% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 1 vgg16 macro F1 | 75.50% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2A vgg16 acc | 80.07% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2A vgg16 macro F1 | 78.71% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2B vgg16 acc | 77.57% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2B vgg16 macro F1 | 76.01% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 1 vgg19 acc | 75.76% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 1 vgg19 macro F1 | 74.80% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2A vgg19 acc | 79.27% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2A vgg19 macro F1 | 78.12% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2B vgg19 acc | 75.00% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2B vgg19 macro F1 | 74.27% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 1 resnet50 acc | 81.09% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 1 resnet50 macro F1 | 79.93% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2A resnet50 acc | 81.73% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2A resnet50 macro F1 | 81.42% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2B resnet50 acc | 86.04% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2B resnet50 macro F1 | 84.83% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 1 resnet152 acc | 81.28% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 1 resnet152 macro F1 | 80.84% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2A resnet152 acc | 88.31% | yes | yes | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2A resnet152 macro F1 | 87.85% | yes | yes | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2B resnet152 acc | 86.54% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2B resnet152 macro F1 | 85.67% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 1 efficientnet_b3 acc | 83.09% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 1 efficientnet_b3 macro F1 | 81.93% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2A efficientnet_b3 acc | 86.65% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2A efficientnet_b3 macro F1 | 85.12% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 2B efficientnet_b3 acc | 86.57% | yes | no | **PASS** | `comparison_15_models_bdneuro_full/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro full | Phase 2B efficientnet_b3 macro F1 | 85.53% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro full | Phase 1 mean acc | 79.77% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| BDNeuro full | Phase 2A mean acc | 83.21% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| BDNeuro full | Phase 2B mean acc | 82.34% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| BDNeuro balanced | Phase 1 vgg16 acc | 77.18% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 1 vgg16 macro F1 | 76.82% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2A vgg16 acc | 80.70% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2A vgg16 macro F1 | 80.62% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2B vgg16 acc | 78.16% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2B vgg16 macro F1 | 77.88% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 1 vgg19 acc | 77.49% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 1 vgg19 macro F1 | 77.64% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2A vgg19 acc | 81.13% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2A vgg19 macro F1 | 81.24% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2B vgg19 acc | 77.00% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2B vgg19 macro F1 | 77.15% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 1 resnet50 acc | 83.74% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 1 resnet50 macro F1 | 83.80% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2A resnet50 acc | 86.77% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2A resnet50 macro F1 | 86.90% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2B resnet50 acc | 87.86% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2B resnet50 macro F1 | 87.87% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 1 resnet152 acc | 85.13% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 1 resnet152 macro F1 | 85.13% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2A resnet152 acc | 90.47% | yes | yes | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2A resnet152 macro F1 | 90.50% | yes | yes | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2B resnet152 acc | 89.81% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2B resnet152 macro F1 | 89.76% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 1 efficientnet_b3 acc | 85.44% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 1 efficientnet_b3 macro F1 | 85.60% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2A efficientnet_b3 acc | 87.86% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2A efficientnet_b3 macro F1 | 87.79% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 2B efficientnet_b3 acc | 88.83% | yes | no | **PASS** | `comparison_15_models_bdneuro_balanced/external_comparison_15.csv → external_accuracy_pct` |
| BDNeuro balanced | Phase 2B efficientnet_b3 macro F1 | 88.96% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| BDNeuro balanced | Phase 1 mean acc | 81.80% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| BDNeuro balanced | Phase 2A mean acc | 85.39% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| BDNeuro balanced | Phase 2B mean acc | 84.33% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| Web-40 | Phase 1 vgg16 acc | 25.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 1 vgg16 macro F1 | 14.58% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2A vgg16 acc | 30.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2A vgg16 macro F1 | 23.23% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2B vgg16 acc | 30.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2B vgg16 macro F1 | 21.59% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 1 vgg19 acc | 35.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 1 vgg19 macro F1 | 28.89% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2A vgg19 acc | 37.5% | yes | yes | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2A vgg19 macro F1 | 27.78% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2B vgg19 acc | 35.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2B vgg19 macro F1 | 25.45% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 1 resnet50 acc | 30.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 1 resnet50 macro F1 | 26.49% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2A resnet50 acc | 42.5% | yes | yes | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2A resnet50 macro F1 | 38.18% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2B resnet50 acc | 40.0% | yes | yes | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2B resnet50 macro F1 | 33.91% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 1 resnet152 acc | 35.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 1 resnet152 macro F1 | 28.08% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2A resnet152 acc | 37.5% | yes | yes | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2A resnet152 macro F1 | 31.66% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2B resnet152 acc | 35.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2B resnet152 macro F1 | 25.74% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 1 efficientnet_b3 acc | 40.0% | yes | yes | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 1 efficientnet_b3 macro F1 | 34.78% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2A efficientnet_b3 acc | 42.5% | yes | yes | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2A efficientnet_b3 macro F1 | 31.97% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 2B efficientnet_b3 acc | 35.0% | yes | no | **PASS** | `comparison_15_models_web_stress_40/external_comparison_15.csv → external_accuracy_pct` |
| Web-40 | Phase 2B efficientnet_b3 macro F1 | 28.07% | yes | no | **PASS** | `same → external_macro_f1_pct` |
| Web-40 | Phase 1 mean acc | 33.00% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| Web-40 | Phase 2A mean acc | 38.00% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| Web-40 | Phase 2B mean acc | 35.00% | yes | yes | **PASS** | `mean of external_accuracy_pct by stage` |
| Web-40 | EfficientNet-B3 Phase 1 = 40.0% (not 42.5%) | 40.0% | yes | yes | **PASS** | `external_comparison_15.csv` |
| Web-40 | lock status | LOCKED_BEFORE_INFERENCE | yes | yes | **PASS** | `lock_record.json → status` |
| XAI | Layer 3 G/P/Poor | 1/9/5 | yes | yes | **PASS** | `gradcam CSV → layer3_rating` |
| XAI | Layer 4 G/P/Poor | 3/2/10 | yes | yes | **PASS** | `gradcam CSV → layer4_rating` |
| XAI | Fused G/P/Poor | 2/6/7 | yes | yes | **PASS** | `gradcam CSV → fused_rating` |
| XAI | occlusion total | 1,859 | yes | yes | **PASS** | `occlusion CSV → occlusion_count sum` |
| XAI | occlusion flips | 58 | yes | yes | **PASS** | `occlusion CSV → prediction_flip_fraction × count` |
| Source | Augmentation (flip p=0.5 ×2; ColorJitter 0.1/0.2/0.2/0.1) | present | yes | yes | **PASS** | `src/data_pipeline_torch.py lines 65–78` |
| Source | GUI artifact root = experiments/<backbone>/ (Phase 1) | experiments | yes | yes | **PASS** | `gui/model_registry.py lines 51–60` |
| Artifacts | SVM probability (Phase 1 ×5 / legacy) | {False} / True | yes | n/a | **PASS** | `experiments/*/models/svm.joblib; models/resnet152_lstm_pca_svm.joblib` |
| Phase 2B | runs completed (30/30) / downstream evals / test keys | 35 / 70 / 0 | yes | yes | **PASS** | `phase2/hyperparameter_dl/**` |

## 3. Manual / semantic checks

| Area | Check | Result |
|---|---|---|
| Epochs | Phase 1 budgets: VGG16 19/30, VGG19 21/30, ResNet50 10/10, ResNet152 10/10, EfficientNet-B3 9/10 (training_results.json → epochs_completed/epochs_requested); Phase 2A and all 35 Phase 2B runs 30/30 | PASS — stated in report §6.1, §8.2, §9.1 |
| Epoch indexing | best_epoch is 1-based in all current JSONs (history.val_accuracy[best_epoch−1] = best_val_accuracy) | PASS — report §6.1, App. H |
| State names | VGG16/VGG19/ResNet50 Phase 2A representative = **best_val_within_30 @ epoch 30** (not epoch_30_model; files differ by hash) | PASS — report §8.3, Table "Phase 2A results"; PPT slides 18–19 |
| Phase labels | Phase 1 / 2A / 2B used consistently; Phase 2D presented before 2A with explanation | PASS |
| Final candidate | ResNet152 Phase 2A, best_val_within_30, epoch 10; RMSprop, LR 1e-4, LSTM 256, dropout 0.3, Dense 128, Scaler → PCA 64 → linear SVM C=1 | PASS — report §12.5, PPT slide 29 |
| Selection framing | 15 candidates frozen on internal validation only; representative = retrospective synthesis of internal + structured BDNeuro evidence; **web-40 not used**; selection optimism disclosed | PASS — report §12.5, §16 item 10; PPT slide 29 + notes |
| External cohorts | 5,941 raw → 3,297 excluded (exact pixel) → 2,644 primary; 1,648 balanced nested (not independent) | PASS |
| Web results | Range 25.0–42.5%; EfficientNet-B3 Phase 1 = 40.0% (corrects doc 15 §8 "Phase1/2A 42.5%"); stage means 33.00/38.00/35.00%; macro-F1 means 26.56/30.56/26.95% | PASS (note: Phase 1 macro-F1 mean is 26.565 → reported as 26.56%) |
| Web bias | mean recall 12.7/30.7/90.7/7.3%; mean predicted counts 2.1/8.3/28.3/1.3 per 40; 600 predictions | PASS |
| XAI counts | Grad-CAM L3 1/9/5, L4 3/2/10, fused 2/6/7; best map 9/6/0; agree 20/20; both correct 19/20; Te-pi_0078; occlusion 2/2/5, 4/3/2, 58/1,859 flips (3.12%), median 0, 5/11 any flip | PASS |
| XAI model | Both XAI studies used the **Phase 1** ResNet152 (experiments/resnet152, epoch 8) via the GUI registry — stated explicitly | PASS — report Ch. 13 intro, PPT slide 30 |
| SVM probability | probability=False for all Phase 1 and Phase 2 SVMs; only legacy root-level prototype True | PASS — report §3.9 |
| Augmentation | verified in source (lines 65–78) | PASS — report §4.6, PPT slide 11 |
| GUI artifact paths | experiments/<backbone>/ → Phase 1 pipelines; GUI does not load the final representative | PASS — report §14.6, PPT slide 32 |
| Phase completion | Phases 1, 2A, 2B, 2C, 2D, 3, 4, 5, 6 all marked COMPLETE; web-40 marked as beyond plan | PASS |
| Historical docs | FINAL_QC_BLOCKER.md "NOT locked" presented as historical state; leakage warning retained as valid | PASS — report §7.6, App. H |
| Phase 2B test usage | no test features/metrics in any Phase 2B result (0 test keys) — presented as intentional | PASS |
| References | [1]–[15] identical to Review-I verified list; [16]–[17] datasets; [18]–[23] standard references from earlier project report | PASS with open items (see §6) |

## 4. Claim-rule terminology scan

Counts of *asserting* uses of forbidden phrasings (negated uses such as "not tumour segmentation" or "do not call 42.5% clinical accuracy" are excluded by the patterns; all
remaining contextual uses were reviewed manually and are negations or caveats).

| Forbidden claim | Report hits | PPT hits | Status |
|---|---|---|---|
| patient-independent | 0 | 0 | PASS |
| clinical accuracy (asserted) | 0 | 0 | PASS |
| ensemble (asserted for this project) | 0 | 0 | PASS |
| SVM probabilities (asserted) | 0 | 0 | PASS |
| Phase 2C pending | 0 | 0 | PASS |
| DenseNet121 final | 0 | 0 | PASS |
| PCA-only ablation (asserted) | 0 | 0 | PASS |
| tumour segmentation (asserted) | 0 | 0 | PASS |

Additional manual review (all occurrences printed in context and read): "patient-level independence" (always "cannot be established"), "segmentation" (always negated or
referring to rejected source images), "ensemble" (only literature or "not an ensemble"), "probability" (only probability=False / "not probabilities"), "significant"
(only "not claimed"/"no significance tests"), "independent external" (objective reworded to "overlap-audited"; "Raw BDNeuro 5,941 is NOT an independent external set").

## 5. Visual / file QC


## 5. Visual and file QC

### Presentation (DL_TUMOR_CLASSIFICATION_FINAL_REVIEW.pptx / .pdf)
- Reopened programmatically (python-pptx): 43 slides = **37 main + 6 backup**; **43/43 slides have speaker notes** (every main slide's notes say what to present, why it matters, the caveat and likely viva questions).
- Automated layout checker (qc_ppt.py): shapes within slide bounds, estimated wrapped-text height of every text box/shape/table cell against its box and the footer bar, image integrity, text-box/picture/table overlaps: **0 issues**. (First pass found the logo overlapping the title box by 0.18 in, a geometry inherited from the Review-I deck; titles were moved below the logo.)
- Rendered to PDF with headless LibreOffice 25.8 and inspected slide by slide as contact sheets plus full-size views of the densest slides (3, 16, 21, 29): no clipping, no overflow, no missing images, no broken charts.
- Font substitution during rendering: Times New Roman → Liberation Serif and Calibri → Carlito, both **metric-compatible**, so line breaks match PowerPoint closely. On a machine with Times New Roman/Calibri, PowerPoint uses the original fonts.
- Design fidelity to Review-I: 10 × 7.5 in; SASTRA logo top-right; Times New Roman titles; grey subtitle (#595959); orange/rust footer bars (#ED800B/#BD582C) with "*" and slide number; rounded boxes in #E2EFDA / #FCE4D6 with black outline; #E36C0A connectors; tables with #F2F2F2 bold header.
- Charts are rendered from repository data (matplotlib PNGs). Chart text inside a few embedded figures (Phase 2B heatmap, BDNeuro funnel) is small at projector distance; the key numbers are repeated in slide tables and text.

### Report (FINAL_PROJECT_REPORT.pdf)
- 68 A4 pages: title, acknowledgement placeholder, abstract, TOC (2 levels), list of 22 figures, list of 38 tables (~9 pages front matter), Chapters 1–18 (~52 pages), references and Appendices A–H (~7 pages). This exceeds the 35–55 guideline; the extra length is content (full 15-pipeline tables, 35-run table, appendices), not padding.
- Page numbers on every page after the title; running header; PDF bookmarks for every chapter and section; TOC/LOF/LOT page numbers resolved by multi-pass build.
- Landscape pages (39–40) used for the two 10-column external-validation tables; all other tables fit portrait width with wrapped cells; repeated header rows on tables that break across pages.
- Contact sheet of all pages plus full-size inspection of the title page, TOC, Ch. 6, 12, 13 and landscape pages: no clipped text, no overlapping figures, no accidental blank pages. Sparse pages occur only where a chapter ends before the next chapter's new page.
- Fonts embedded: Liberation Serif/Mono (full glyph coverage for →, ≈, ≤, ×, −).
- Every numbered figure and table is cross-referenced from text through resolved {fig:}/{tab:} references (an unresolved reference fails the build).

### DOCX (FINAL_PROJECT_REPORT.docx)
- Reopened with python-docx (600 paragraphs, 59 tables, 26 images, 3 sections incl. landscape section) and converted by LibreOffice to PDF (60 pages) without error.
- TOC, list of figures and list of tables are Word fields: **right-click → Update Field** (or Ctrl+A, F9) after opening in Word to populate them.

## 6. Unresolved items / open discrepancies (for the team before submission)
1. **Acknowledgement** page is a placeholder; no certificate/declaration wording exists in project files, so none was invented. Add institution-prescribed pages if required.
2. **Dataset citations incomplete:** the BDNeuro V8 README leaves licence and citation as "[FILL IN]"; Nickparvar dataset version/licence/access date are unconfirmed. Both are flagged in references [16]–[17].
3. **Historical (non machine-readable) figures** reported with that label: VGG16 collapse (~28% validation), legacy ResNet152 prototype (~97.64%), the initial NOR_01 pHash-0 audit (only the visual-review image and final audit are retained), and the VGG16 lr_1e-5 queue-skip note.
4. **GUI serves Phase 1 pipelines**, not the final ResNet152 Phase 2A representative (stated in report and slides; changing the registry is a software task deliberately not done).
5. **XAI was performed on the Phase 1 ResNet152** (epoch 8), not on the final representative.
6. **Web provenance:** repository template `phase2/external_validation/web_stress_40/web_stress_40_manifest.csv` is unfilled (all PENDING); the authoritative source list is the transfer package's `Web_Stress_Test_40_Source_List.xlsx`.
7. **Transfer-package corrections applied (repo evidence wins):** EfficientNet-B3 web Phase 1 = 40.0% (doc 15 §8 said "Phase1/2A 42.5%"); EfficientNet-B3 neural→SVM gain = +0.46 pp (docs said +0.45); VGG16/VGG19/ResNet50 Phase 2A state = best_val_within_30 @ epoch 30 (docs said "epoch_30").
8. **Uncommitted evidence:** most Phase 2/external-validation evidence and these deliverables are not committed to git; commit/archive before submission.
9. Report length (68 pages) exceeds the 35–55 guideline (see §5).
