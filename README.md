# Prot2Prop
Structure-aware fine-tuning of protein language models for joint prediction of multiple developability properties from protein structure inputs.

## Objectives
This project aims to train **one shared adapter + task‑specific heads** for key developability properties relevant to enzymes, binders, synthetic biology, and related protein engineering tasks.

- [ ] Aggregate data from multiple datasets for properties of interest
- [ ] Train a single adapter (or LoRA block) on all properties with a separate head per property.
- [ ] Use a multi‑task loss with masking for missing labels.
- [ ] Repeat training to build ensembles and evaluate ensembling strategies.

### Properties (Primary)
- Thermal unfolding metrics (Tm, ΔG)
- Stability under pH or ionic strength shifts
- Aggregation propensity
- Solubility
- Secretion signal / signal peptide presence
- Protease susceptibility / half-life
- Immunogenicity / epitope likelihood
- Oligomerization state / assembly propensity
- Metal binding / cofactor dependence
- Membrane association / transmembrane topology

## Properties (Secondary)
Properties for future investigation (lower priority and potentially less synergistic with the primary set).
- Binding affinity (protein–protein, protein–ligand)
- Enzymatic activity (kcat, Km, specificity)
- Expression yield / solubility in specific hosts
- Subcellular localization
- Allosteric regulation potential
- Disorder content / intrinsically disordered regions
- PTM site likelihood (phosphorylation, glycosylation, ubiquitination, acetylation)

> I suggest we ignore these for the time being, as they all require some degree of auxiliary information. For example, binding affinity depends on interaction partners, while enzymatic activity and kcat are only meaningful with substrate information. Without this context, I'm of the opinion that these properties would be of limited value. That said, I’ve kept them listed to provide a sense of future direction for the project, once the initial set of primary properties has been implemented.

## Motivation
The current ecosystem provides many task-specific models for properties such as solubility and stability, but few efforts unify these objectives within a single, structure-aware model. A multi-property framework can reduce parameter count and improve throughput, while potentially improving accuracy through shared signal across correlated biochemical traits (e.g., stability-related metrics).

Additionally, widely used tools (e.g., NetSolP-1.0) rely on older base models and datasets. With improved architectures, larger corpora, and better training algorithms now available, a modern, multi-property, structure-aware approach is both timely and high-impact.

## Installation
```sh
# install python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e .

# training-only dependencies
pip install -e ".[dev]"
```
## Download ProteinGym
```sh
# download subs
wget https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip
unzip DMS_ProteinGym_substitutions.zip
rm DMS_ProteinGym_substitutions.zip

# download indels
wget https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_indels.zip
unzip DMS_ProteinGym_substitutions.zip
rm DMS_ProteinGym_substitutions.zip
```

## Taining
1. Once the above data downloads have been complete, run the dta aggregation pipeline.
```sh
python aggregate_data.py
```
2. Tokenize the aggregated data so that it is ready for training
```sh
python tokenize_data.py
```
3. Train the models, currently suggest using L40S GPU(s)
```sh
python train.py
```

## Results (WIP)
### Crude Solubility Results for Reference
```sh
Epoch 1/3: 100%|???????????| 7810/7810 [1:05:11<00:00,  2.00it/s]
Train Loss: 0.5971 | Val Acc: 0.7097 F1: 0.7000
Epoch 2/3: 100%|???????????| 7810/7810 [1:05:04<00:00,  2.00it/s]
Train Loss: 0.5374 | Val Acc: 0.7233 F1: 0.7051
Epoch 3/3: 100%|???????????| 7810/7810 [1:04:57<00:00,  2.00it/s]
Train Loss: 0.5194 | Val Acc: 0.7224 F1: 0.7146
```

### New version initial run
```sh
Epoch 6/10: 100%|██████████████████████| 59052/59052 [7:47:30<00:00,  2.11it/s]
Train Loss: 0.7318 | Val aggregation_propensity:MAE=0.8527 binding_affinity:MAE=7894.5411 enzymatic_activity:MAE=0.4782 expression_yield:MAE=0.7306 folding_stability:MAE=0.7612 material_production:F1=0.8043 membrane_topology:MAE=1.2854 solubility:F1=0.7377 temperature_stability:MAE=0.2187 thermostability:MAE=1.8387
```

### New version with better dataset:
```
Epoch 8/10: 100%|██████████████████████| 19234/19234 [2:03:14<00:00,  2.60it/s]
Train Loss: 0.4440 | Val aggregation_propensity:MAE=1.1890 RMSE=1.5380 expression_yield:MAE=0.7119 RMSE=1.0359 folding_stability:MAE=0.9509 RMSE=1.1577 material_production:ACC=0.7802 F1=0.8325 solubility:ACC=0.7571 F1=0.7348 temperature_stability:ACC=0.9181 F1=0.9200 temperature_stability_abs:MAE=2.3290 RMSE=2.9995
```

### Version 2026-04-26
- Increased task-head capacity beyond the original `LayerNorm + Linear` design by introducing small MLP heads, with the main goal of improving regression calibration and expressiveness.
- Removed `temperature_stability_abs` from the shared multitask setup because it was a small, single-protein dataset that was unlikely to contribute useful transferable signal.
- Added richer validation for classification tasks, including `AUROC` and `AUPRC`, to separate ranking quality from thresholded classification quality.
- Added post-hoc threshold tuning for binary classification tasks to test whether task-specific decision thresholds could improve `acc`, `balanced_acc`, `precision`, `recall`, and `F1`.
- Added post-hoc linear calibration for regression tasks using `y_calibrated = a * y_pred + b` to correct prediction bias and under-dispersion without retraining.
- Result: the architectural head changes were a clear improvement overall, while the post-hoc calibration steps produced modest but informative gains, especially for some regression tasks.

```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7823  0.7648   0.8734     0.8078  0.8393  0.8463  0.9265  0:0.296 1:0.704  0:0.349 1:0.651
solubility             bool   7071   0.7645  0.7692   0.6894     0.7980  0.7397  0.8619  0.8231  0:0.581 1:0.419  0:0.515 1:0.485
temperature_stability  bool   41981  0.9207  0.9211   0.8857     0.9644  0.9234  0.9829  0.9832  0:0.505 1:0.495  0:0.461 1:0.539

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.6190    1.5172    0.8434  1.1004  0.7860
expression_yield        11204  -0.0776     1.1371     0.2937     0.6376    0.6129  1.0296  0.6740
folding_stability       12562  -1.3020     1.2068     -0.6148    1.0979    0.7812  0.9900  0.8133

Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)
task                   cal_n  rep_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    1408   1408   0.0500  0.7834  0.6871   0.8049     0.9169  0.8573  0.8422  0.9250  0:0.290 1:0.710  0:0.192 1:0.808
solubility             3536   3535   0.4200  0.7545  0.7650   0.6652     0.8293  0.7382  0.8604  0.8210  0:0.582 1:0.418  0:0.479 1:0.521
temperature_stability  20991  20990  0.7400  0.9294  0.9295   0.9247     0.9339  0.9293  0.9824  0.9827  0:0.504 1:0.496  0:0.499 1:0.501

Post-hoc Regression Calibration (fit on internal half, report on held-out half)
task                    cal_n  rep_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  860    860    0.9315  -0.3336    -1.8691    1.4176    0.8487  1.1001  0.7727
expression_yield        5602   5602   0.9520  -0.3611    -0.0820    0.6079    0.6087  0.9457  0.6768
folding_stability       6281   6281   0.8913  -0.7491    -1.2964    0.9734    0.5495  0.7004  0.8115
```

### Version 2026-04-28
- Tested a new regression-loss recipe intended to better align strong ranking performance with stronger numeric accuracy.
- Replaced plain `MSE` / `L1` regression loss with Huber loss to make training more robust to noisy or outlier-heavy biological measurements.
- Added a pairwise ranking term to the regression objective:
  `loss = mse_or_huber + lambda * pairwise_ranking_loss`
- The goal was to preserve and strengthen relative ordering, since Spearman correlation was already reasonably strong on several tasks.
- Result: this experiment did not improve the main validation metrics. Overall performance was slightly worse across both classification and regression, so this direction was not retained.

```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7766  0.7590   0.8703     0.8022  0.8349  0.8459  0.9259  0:0.296 1:0.704  0:0.351 1:0.649
solubility             bool   7071   0.7613  0.7669   0.6836     0.8017  0.7380  0.8592  0.8190  0:0.581 1:0.419  0:0.508 1:0.492
temperature_stability  bool   41981  0.9126  0.9132   0.8653     0.9753  0.9170  0.9837  0.9839  0:0.505 1:0.495  0:0.442 1:0.558

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.5628    1.4654    0.8551  1.1350  0.7781
expression_yield        11204  -0.0776     1.1371     0.3753     0.6047    0.6418  1.0886  0.6688
folding_stability       12562  -1.3020     1.2068     -0.5382    1.0880    0.8535  1.0647  0.7964

Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)
task                   cal_n  rep_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    1408   1408   0.0500  0.7947  0.7095   0.8187     0.9129  0.8632  0.8427  0.9241  0:0.290 1:0.710  0:0.209 1:0.791
solubility             3536   3535   0.4700  0.7610  0.7693   0.6764     0.8198  0.7412  0.8592  0.8196  0:0.582 1:0.418  0:0.494 1:0.506
temperature_stability  20991  20990  0.8300  0.9317  0.9317   0.9253     0.9381  0.9316  0.9834  0.9836  0:0.504 1:0.496  0:0.497 1:0.503

Post-hoc Regression Calibration (fit on internal half, report on held-out half)
task                    cal_n  rep_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  860    860    0.9500  -0.3447    -1.8443    1.3959    0.8705  1.1252  0.7645
expression_yield        5602   5602   0.9278  -0.4292    -0.0809    0.5609    0.6317  0.9774  0.6696
folding_stability       6281   6281   0.8789  -0.8238    -1.2960    0.9502    0.5709  0.7265  0.7957
```

### Version 2026-04-29
- Introduced a shared adapter plus small per-task residual adapters so each task can specialize the shared representation without giving up multitask sharing.
- Moved from one fully shared pooled representation to task-specific token adaptation before pooling, allowing each task to derive its own sequence summary from a common adapted backbone.
- This change was intended to improve task-specific calibration and feature specialization while keeping the parameter increase modest.
- Result: this was the strongest run so far. The main gains came from the regression tasks, with substantial improvements in both error (`MAE` / `RMSE`) and ranking quality (`Spearman`), while classification performance remained broadly stable.
- Post-hoc calibration continued to show additional upside for some regression tasks, especially `folding_stability`, suggesting the learned representation is strong and remaining gains may come from better final calibration rather than major architectural changes.
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7773  0.7568   0.8672     0.8073  0.8362  0.8458  0.9262  0:0.296 1:0.704  0:0.345 1:0.655
solubility             bool   7071   0.7657  0.7723   0.6861     0.8132  0.7443  0.8640  0.8296  0:0.581 1:0.419  0:0.503 1:0.497
temperature_stability  bool   41981  0.9207  0.9211   0.8856     0.9644  0.9233  0.9838  0.9842  0:0.505 1:0.495  0:0.461 1:0.539

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.6774    1.5536    0.7268  0.9547  0.8452
expression_yield        11204  -0.0776     1.1371     0.1792     0.6917    0.5464  0.9304  0.7267
folding_stability       12562  -1.3020     1.2068     -0.8125    1.2342    0.6260  0.8305  0.8373

Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)
task                   cal_n  rep_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    1408   1408   0.0500  0.7841  0.6912   0.8078     0.9129  0.8571  0.8473  0.9288  0:0.290 1:0.710  0:0.198 1:0.802
solubility             3536   3535   0.4000  0.7491  0.7632   0.6536     0.8489  0.7386  0.8634  0.8298  0:0.582 1:0.418  0:0.458 1:0.542
temperature_stability  20991  20990  0.7800  0.9330  0.9330   0.9318     0.9334  0.9326  0.9835  0.9839  0:0.504 1:0.496  0:0.503 1:0.497

Post-hoc Regression Calibration (fit on internal half, report on held-out half)
task                    cal_n  rep_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  860    860    0.9488  -0.2566    -1.8822    1.4586    0.7286  0.9532  0.8407
expression_yield        5602   5602   1.0187  -0.2687    -0.0913    0.7097    0.5657  0.8792  0.7279
folding_stability       6281   6281   0.8312  -0.6213    -1.2956    1.0233    0.4853  0.6360  0.8345
```