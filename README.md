# Prot2Prop
Structure-aware fine-tuning of protein language models for joint prediction of multiple developability properties from protein structure inputs.

## Objectives
This project aims to train **one shared adapter + lightweight task-specific residual adapters + task-specific heads** for key developability properties relevant to enzymes, binders, synthetic biology, and related protein engineering tasks.

- [ ] Repeat training to build ensembles and evaluate ensembling strategies.

### Ideas To Consider
- Per-task pooling
- Mixture-of-experts style light specialization
- Train a few different variants with different seeds
- Possibly add ESM-MSA optional embeddings
- Possibly test out these properties added as additional features:
  - Add MW, pI, net charge, charge density.
  - Aromaticity, aliphatic index
  - Low-complexity / entropy and disorder-related residue fractions.
- Generate embeddings for each protein

### Properties (Primary)
- [X] Thermal unfolding metrics (Tm, ΔG)
- [ ] Stability under pH or ionic strength shifts
- [X] Aggregation propensity
- [X] Solubility
- [ ] Secretion signal / signal peptide presence
- [ ] Immunogenicity / epitope likelihood
- [ ] Oligomerization state / assembly propensity
- [ ] Metal binding / cofactor dependence

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
### Core Program
```sh
# clone github repo
git clone https://github.com/NeurosnapInc/Prot2Prop.git
cd Prot2Prop

# install python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e .

# training-only dependencies
pip install -e ".[dev]"
```

### Download ProteinGym
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

### Version 2026-04-30
- Added learned uncertainty weighting so each task contributed through a trainable per-task uncertainty term rather than fixed equal weighting.
- This changed the multitask balance but did not clearly improve the overall result.
- `aggregation_propensity` and `temperature_stability` improved, but `expression_yield` and `folding_stability` both regressed, making the net effect mixed to slightly worse overall.
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7770  0.7565   0.8671     0.8068  0.8359  0.8437  0.9247  0:0.296 1:0.704  0:0.345 1:0.655
solubility             bool   7071   0.7637  0.7704   0.6837     0.8121  0.7424  0.8660  0.8300  0:0.581 1:0.419  0:0.502 1:0.498
temperature_stability  bool   41981  0.9265  0.9269   0.8938     0.9664  0.9287  0.9840  0.9842  0:0.505 1:0.495  0:0.465 1:0.535

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.8579    1.5613    0.6876  0.9205  0.8522
expression_yield        11204  -0.0776     1.1371     0.2081     0.6905    0.5564  0.9491  0.7117
folding_stability       12562  -1.3020     1.2068     -0.7288    1.2170    0.6883  0.8927  0.8322

Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)
task                   cal_n  rep_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    1408   1408   0.0600  0.7919  0.7046   0.8157     0.9129  0.8616  0.8470  0.9279  0:0.290 1:0.710  0:0.206 1:0.794
solubility             3536   3535   0.3800  0.7482  0.7638   0.6504     0.8584  0.7401  0.8667  0.8295  0:0.582 1:0.418  0:0.449 1:0.551
temperature_stability  20991  20990  0.8100  0.9343  0.9343   0.9348     0.9327  0.9337  0.9837  0.9839  0:0.504 1:0.496  0:0.505 1:0.495

Post-hoc Regression Calibration (fit on internal half, report on held-out half)
task                    cal_n  rep_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  860    860    0.9649  -0.0649    -1.9009    1.4968    0.7141  0.9537  0.8401
expression_yield        5602   5602   0.9918  -0.2912    -0.0886    0.6850    0.5730  0.8897  0.7140
folding_stability       6281   6281   0.8328  -0.6897    -1.2955    1.0077    0.5020  0.6540  0.8277
```

### Version 2026-05-22
- Added optional evolutionary-alignment training support for `folding_stability` using precomputed homolog/MSA-derived residue likelihood targets.
- This feature was inspired by the preprint at `https://www.biorxiv.org/content/10.1101/2025.04.25.650688v4`.
- This introduced a target-building script, tokenization/cache support for per-residue alignment supervision, and a residue-likelihood head with an auxiliary correlation-based loss.
- The auxiliary loss is disabled by default through `LAMBDA_EVOLUTIONARY_ALIGNMENT = 0.0`, so this version mainly adds the training pathway and data plumbing rather than changing baseline behavior on its own.
- Compared with the previous best run (`2026-04-29`), this checkpoint looked roughly flat overall: classification moved slightly, ranking metrics were modestly stronger, and uncalibrated `folding_stability` regression was worse (`MAE 0.6435` vs `0.6260`) even though Spearman improved (`0.8440` vs `0.8373`).
- The calibrated `folding_stability` MAE improved from `0.4853` to `0.4725`, an absolute gain of `0.0128` or about `2.6%`, but that effect is small and not cleanly apples-to-apples because older runs used validation-time post-hoc fitting while this checkpoint uses saved checkpoint calibration.
- Practical takeaway: the measured `folding_stability` gain is modest enough that it may be within normal seed-to-seed variance, so this version does not yet make a strong case for keeping the evolutionary-alignment pathway on performance grounds alone.
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7731  0.7485   0.8604     0.8088  0.8338  0.8408  0.9222  0:0.296 1:0.704  0:0.338 1:0.662
solubility             bool   7071   0.7665  0.7733   0.6865     0.8155  0.7455  0.8667  0.8315  0:0.581 1:0.419  0:0.502 1:0.498
temperature_stability  bool   41981  0.9233  0.9237   0.8899     0.9645  0.9257  0.9836  0.9839  0:0.505 1:0.495  0:0.463 1:0.537

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.5866    1.7134    0.7264  0.9615  0.8550
expression_yield        11204  -0.0776     1.1371     0.1771     0.6963    0.5473  0.9305  0.7289
folding_stability       12562  -1.3020     1.2068     -0.7797    1.2460    0.6435  0.8410  0.8440

Checkpoint Classification Calibration Applied
task                   cal_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    2816   0.1200  0.7898  0.7152   0.8203     0.8981  0.8574  0.8408  0.9222  0:0.296 1:0.704  0:0.229 1:0.771
solubility             7071   0.3800  0.7568  0.7708   0.6620     0.8580  0.7474  0.8667  0.8315  0:0.581 1:0.419  0:0.457 1:0.543
temperature_stability  41981  0.6900  0.9311  0.9312   0.9209     0.9418  0.9313  0.9836  0.9839  0:0.505 1:0.495  0:0.494 1:0.506

Checkpoint Regression Calibration Applied
task                    cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   0.8834  -0.4347    -1.8363    1.5136    0.6901  0.9066  0.8550
expression_yield        11204  1.0082  -0.2554    -0.0769    0.7020    0.5684  0.8949  0.7289
folding_stability       12562  0.8285  -0.6566    -1.3026    1.0324    0.4725  0.6238  0.8440
```

### Version 2026-05-22
- This is the final version, no particularly major changes, just training on different seeds and stuff

#### Seed 42
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7699  0.7542   0.8689     0.7926  0.8290  0.8424  0.9250  0:0.296 1:0.704  0:0.358 1:0.642
solubility             bool   7071   0.7618  0.7682   0.6826     0.8074  0.7398  0.8644  0.8271  0:0.581 1:0.419  0:0.504 1:0.496
temperature_stability  bool   41981  0.9219  0.9223   0.8895     0.9618  0.9243  0.9829  0.9833  0:0.505 1:0.495  0:0.464 1:0.536

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.7424    1.5496    0.7569  0.9933  0.8276
expression_yield        11204  -0.0776     1.1371     0.1814     0.6746    0.5560  0.9419  0.6973
folding_stability       12562  -1.3020     1.2068     -0.7425    1.2468    0.6824  0.8844  0.8345

Checkpoint Classification Calibration Applied
task                   cal_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    2816   0.0800  0.7859  0.7024   0.8110     0.9072  0.8564  0.8424  0.9250  0:0.296 1:0.704  0:0.213 1:0.787
solubility             7071   0.4100  0.7534  0.7662   0.6609     0.8459  0.7420  0.8644  0.8271  0:0.581 1:0.419  0:0.463 1:0.537
temperature_stability  41981  0.7500  0.9323  0.9322   0.9320     0.9311  0.9316  0.9829  0.9833  0:0.505 1:0.495  0:0.505 1:0.495

Checkpoint Regression Calibration Applied
task                    cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   0.9451  -0.1877    -1.8344    1.4645    0.7533  0.9850  0.8276
expression_yield        11204  1.0190  -0.2623    -0.0775    0.6874    0.5795  0.9055  0.6973
folding_stability       12562  0.8175  -0.6956    -1.3026    1.0193    0.4931  0.6460  0.8345
```

#### Seed 26
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7773  0.7574   0.8680     0.8063  0.8360  0.8443  0.9262  0:0.296 1:0.704  0:0.346 1:0.654
solubility             bool   7071   0.7692  0.7741   0.6939     0.8044  0.7451  0.8652  0.8262  0:0.581 1:0.419  0:0.514 1:0.486
temperature_stability  bool   41981  0.9215  0.9219   0.8881     0.9628  0.9239  0.9827  0.9830  0:0.505 1:0.495  0:0.463 1:0.537

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.5609    1.6295    0.7310  0.9712  0.8516
expression_yield        11204  -0.0776     1.1371     0.1845     0.6810    0.5551  0.9507  0.6899
folding_stability       12562  -1.3020     1.2068     -0.7235    1.2222    0.6922  0.8978  0.8300

Checkpoint Classification Calibration Applied
task                   cal_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    2816   0.0600  0.7823  0.6898   0.8022     0.9168  0.8557  0.8443  0.9262  0:0.296 1:0.704  0:0.196 1:0.804
solubility             7071   0.5200  0.7722  0.7755   0.7011     0.7960  0.7455  0.8652  0.8262  0:0.581 1:0.419  0:0.524 1:0.476
temperature_stability  41981  0.7500  0.9311  0.9311   0.9298     0.9312  0.9305  0.9827  0.9830  0:0.505 1:0.495  0:0.504 1:0.496

Checkpoint Regression Calibration Applied
task                    cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   0.9235  -0.3933    -1.8348    1.5048    0.6977  0.9227  0.8516
expression_yield        11204  0.9941  -0.2603    -0.0768    0.6770    0.5833  0.9139  0.6899
folding_stability       12562  0.8293  -0.7023    -1.3023    1.0136    0.5013  0.6542  0.8300
```

#### Seed 1
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7784  0.7565   0.8662     0.8103  0.8373  0.8445  0.9268  0:0.296 1:0.704  0:0.342 1:0.658
solubility             bool   7071   0.7737  0.7761   0.7053     0.7909  0.7456  0.8675  0.8306  0:0.581 1:0.419  0:0.530 1:0.470
temperature_stability  bool   41981  0.9280  0.9282   0.9076     0.9515  0.9290  0.9834  0.9837  0:0.505 1:0.495  0:0.481 1:0.519

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.5239    1.6423    0.7238  0.9559  0.8616
expression_yield        11204  -0.0776     1.1371     0.1947     0.7017    0.5545  0.9377  0.7336
folding_stability       12562  -1.3020     1.2068     -0.7347    1.2263    0.6795  0.8800  0.8412

Checkpoint Classification Calibration Applied
task                   cal_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    2816   0.0500  0.7816  0.6823   0.7968     0.9258  0.8565  0.8445  0.9268  0:0.296 1:0.704  0:0.182 1:0.818
solubility             7071   0.4300  0.7662  0.7744   0.6832     0.8250  0.7474  0.8675  0.8306  0:0.581 1:0.419  0:0.494 1:0.506
temperature_stability  41981  0.6700  0.9331  0.9331   0.9342     0.9305  0.9323  0.9834  0.9837  0:0.505 1:0.495  0:0.507 1:0.493

Checkpoint Regression Calibration Applied
task                    cal_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   0.9261  -0.4233    -1.8345    1.5209    0.6809  0.8950  0.8616
expression_yield        11204  0.9956  -0.2710    -0.0772    0.6986    0.5716  0.8973  0.7336
folding_stability       12562  0.8335  -0.6896    -1.3020    1.0222    0.4884  0.6411  0.8412
```

#### Seed M (Older mixed seed version)
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7699  0.7508   0.8649     0.7977  0.8299  0.8399  0.9242  0:0.296 1:0.704  0:0.351 1:0.649
solubility             bool   7071   0.7617  0.7709   0.6764     0.8277  0.7444  0.8660  0.8289  0:0.581 1:0.419  0:0.487 1:0.513
temperature_stability  bool   41981  0.9216  0.9220   0.8867     0.9650  0.9242  0.9834  0.9837  0:0.505 1:0.495  0:0.461 1:0.539

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.5139    1.6291    0.8071  1.0644  0.8263
expression_yield        11204  -0.0776     1.1371     0.1833     0.6743    0.5473  0.9390  0.7222
folding_stability       12562  -1.3020     1.2068     -0.7963    1.2232    0.6400  0.8424  0.8394

Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)
task                   cal_n  rep_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    1408   1408   0.0500  0.7805  0.6800   0.8005     0.9199  0.8561  0.8386  0.9246  0:0.290 1:0.710  0:0.185 1:0.815
solubility             3536   3535   0.5600  0.7706  0.7752   0.6948     0.8035  0.7452  0.8663  0.8287  0:0.582 1:0.418  0:0.517 1:0.483
temperature_stability  20991  20990  0.7600  0.9317  0.9318   0.9282     0.9347  0.9315  0.9831  0.9834  0:0.504 1:0.496  0:0.500 1:0.500

Post-hoc Regression Calibration (fit on internal half, report on held-out half)
task                    cal_n  rep_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  860    860    0.8911  -0.5016    -1.8869    1.4450    0.7882  1.0231  0.8170
expression_yield        5602   5602   1.0229  -0.2723    -0.0884    0.6907    0.5711  0.8862  0.7237
folding_stability       6281   6281   0.8341  -0.6338    -1.2984    1.0151    0.4916  0.6419  0.8348
```

#### 4 Checkpoint Ensemble
- Evaluated a simple four-checkpoint prediction ensemble built from the strongest recent seeds and the older mixed-seed checkpoint.
- Result: the ensemble was competitive, but it did **not** clearly beat the best single-checkpoint runs overall.
- Classification moved only slightly and remained roughly within normal run-to-run variation.
- Regression was mixed: the ensemble was solid, but it generally did not surpass the strongest single-seed checkpoints on the tasks that mattered most, especially when comparing against the best raw `Spearman` / `MAE` results from `aggregation_propensity`, `expression_yield`, and `folding_stability`.
- Practical takeaway: ensembling did not provide a compelling enough gain to justify treating it as the default evaluation or deployment path, so the best single-checkpoint models remain the more meaningful reference point.
```
Classification Tasks
task                   dtype  n      acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    bool   2816   0.7784  0.7620   0.8727     0.8022  0.8360  0.8480  0.9283  0:0.296 1:0.704  0:0.353 1:0.647
solubility             bool   7071   0.7715  0.7778   0.6929     0.8172  0.7499  0.8714  0.8369  0:0.581 1:0.419  0:0.505 1:0.495
temperature_stability  bool   41981  0.9259  0.9262   0.8960     0.9621  0.9279  0.9840  0.9844  0:0.505 1:0.495  0:0.468 1:0.532

Regression Tasks
task                    n      label_mean  label_std  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  ----------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  1720   -1.8365     1.7641     -1.5853    1.5856    0.7190  0.9475  0.8575
expression_yield        11204  -0.0776     1.1371     0.1860     0.6789    0.5498  0.9393  0.7182
folding_stability       12562  -1.3020     1.2068     -0.7492    1.2227    0.6660  0.8661  0.8432

Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)
task                   cal_n  rep_n  thr     acc     bal_acc  precision  recall  f1      auroc   auprc   label_ratio      pred_ratio
---------------------  -----  -----  ------  ------  -------  ---------  ------  ------  ------  ------  ---------------  ---------------
material_production    1408   1408   0.0700  0.7841  0.6840   0.8024     0.9229  0.8585  0.8486  0.9305  0:0.290 1:0.710  0:0.184 1:0.816
solubility             3536   3535   0.4700  0.7661  0.7751   0.6802     0.8299  0.7476  0.8709  0.8358  0:0.582 1:0.418  0:0.491 1:0.509
temperature_stability  20991  20990  0.6800  0.9328  0.9329   0.9250     0.9410  0.9329  0.9837  0.9841  0:0.504 1:0.496  0:0.495 1:0.505

Post-hoc Regression Calibration (fit on internal half, report on held-out half)
task                    cal_n  rep_n  slope   intercept  pred_mean  pred_std  mae     rmse    spearman
----------------------  -----  -----  ------  ---------  ---------  --------  ------  ------  --------
aggregation_propensity  860    860    0.9428  -0.3563    -1.8877    1.4800    0.7062  0.9285  0.8527
expression_yield        5602   5602   1.0186  -0.2744    -0.0890    0.6925    0.5723  0.8868  0.7198
folding_stability       6281   6281   0.8379  -0.6695    -1.2969    1.0199    0.4870  0.6359  0.8398
```