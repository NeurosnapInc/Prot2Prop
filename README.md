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
Epoch 1/10: 100%|██████████████████████| 59052/59052 [7:48:46<00:00,  2.10it/s]
Train Loss: 1.4744 | Val aggregation_propensity:MAE=1.9031 binding_affinity:MAE=52497.9969 enzymatic_activity:MAE=0.6969 expression_yield:MAE=0.7056 folding_stability:MAE=0.8563 material_production:F1=0.8526 membrane_topology:MAE=0.9221 solubility:F1=0.7203 temperature_stability:MAE=0.2316 thermostability:MAE=2.1242
Epoch 2/10: 100%|██████████████████████| 59052/59052 [7:46:51<00:00,  2.11it/s]
Train Loss: 1.2045 | Val aggregation_propensity:MAE=1.1719 binding_affinity:MAE=17223.4967 enzymatic_activity:MAE=0.6721 expression_yield:MAE=0.8159 folding_stability:MAE=0.8227 material_production:F1=0.7900 membrane_topology:MAE=1.1418 solubility:F1=0.7318 temperature_stability:MAE=0.2409 thermostability:MAE=2.6384
Epoch 3/10: 100%|██████████████████████| 59052/59052 [7:47:38<00:00,  2.10it/s]
Train Loss: 0.9002 | Val aggregation_propensity:MAE=1.1414 binding_affinity:MAE=16454.1588 enzymatic_activity:MAE=0.4842 expression_yield:MAE=0.7670 folding_stability:MAE=0.9427 material_production:F1=0.8111 membrane_topology:MAE=0.9959 solubility:F1=0.7294 temperature_stability:MAE=0.2295 thermostability:MAE=2.0764
Epoch 4/10: 100%|██████████████████████| 59052/59052 [7:49:46<00:00,  2.10it/s]
Train Loss: 0.8173 | Val aggregation_propensity:MAE=0.8857 binding_affinity:MAE=7964.2868 enzymatic_activity:MAE=0.4536 expression_yield:MAE=0.6968 folding_stability:MAE=0.7595 material_production:F1=0.7719 membrane_topology:MAE=1.1293 solubility:F1=0.7231 temperature_stability:MAE=0.2167 thermostability:MAE=1.8650
Epoch 5/10: 100%|██████████████████████| 59052/59052 [7:46:53<00:00,  2.11it/s]
Train Loss: 1.2787 | Val aggregation_propensity:MAE=0.8390 binding_affinity:MAE=9205.2723 enzymatic_activity:MAE=0.5042 expression_yield:MAE=0.7088 folding_stability:MAE=0.8430 material_production:F1=0.7905 membrane_topology:MAE=0.9927 solubility:F1=0.7274 temperature_stability:MAE=0.2203 thermostability:MAE=1.9772
Epoch 6/10: 100%|██████████████████████| 59052/59052 [7:47:30<00:00,  2.11it/s]
Train Loss: 0.7318 | Val aggregation_propensity:MAE=0.8527 binding_affinity:MAE=7894.5411 enzymatic_activity:MAE=0.4782 expression_yield:MAE=0.7306 folding_stability:MAE=0.7612 material_production:F1=0.8043 membrane_topology:MAE=1.2854 solubility:F1=0.7377 temperature_stability:MAE=0.2187 thermostability:MAE=1.8387
Epoch 7/10: 100%|██████████████████████| 59052/59052 [7:48:47<00:00,  2.10it/s]
Train Loss: 0.8364 | Val aggregation_propensity:MAE=0.8654 binding_affinity:MAE=7049.0198 enzymatic_activity:MAE=0.4496 expression_yield:MAE=0.7309 folding_stability:MAE=0.8069 material_production:F1=0.8228 membrane_topology:MAE=1.1577 solubility:F1=0.7330 temperature_stability:MAE=0.2120 thermostability:MAE=1.8011
Epoch 8/10: 100%|██████████████████████| 59052/59052 [7:45:15<00:00,  2.12it/s]
Train Loss: 0.9588 | Val aggregation_propensity:MAE=0.8156 binding_affinity:MAE=7664.6571 enzymatic_activity:MAE=0.4539 expression_yield:MAE=0.6896 folding_stability:MAE=0.8291 material_production:F1=0.7948 membrane_topology:MAE=1.1935 solubility:F1=0.7346 temperature_stability:MAE=0.2118 thermostability:MAE=1.8085
Epoch 9/10: 100%|██████████████████████| 59052/59052 [7:46:24<00:00,  2.11it/s]
Train Loss: 1.3263 | Val aggregation_propensity:MAE=0.8555 binding_affinity:MAE=6034.2138 enzymatic_activity:MAE=0.4443 expression_yield:MAE=0.6880 folding_stability:MAE=0.8157 material_production:F1=0.8226 membrane_topology:MAE=1.1865 solubility:F1=0.7365 temperature_stability:MAE=0.2097 thermostability:MAE=1.6818
Epoch 10/10: 100%|██████████████████████| 59052/59052 [7:45:42<00:00,  2.11it/s]
Train Loss: 0.9175 | Val aggregation_propensity:MAE=0.8252 binding_affinity:MAE=6367.3902 enzymatic_activity:MAE=0.4654 expression_yield:MAE=0.6842 folding_stability:MAE=0.8149 material_production:F1=0.8118 membrane_topology:MAE=1.1776 solubility:F1=0.7369 temperature_stability:MAE=0.2082 thermostability:MAE=1.6443
Saved best shared adapter+heads -> ./prostt5_multitask_adapter_best.pt
```

### New version with better dataset:
```
Epoch 1/10: 100%|██████████████████████| 19234/19234 [2:04:15<00:00,  2.58it/s]
Train Loss: 0.6494 | Val aggregation_propensity:MAE=1.7844 RMSE=2.2044 expression_yield:MAE=0.6525 RMSE=0.9819 folding_stability:MAE=0.8009 RMSE=1.0129 material_production:ACC=0.7610 F1=0.8164 solubility:ACC=0.7188 F1=0.7088 temperature_stability:ACC=0.8865 F1=0.8918 temperature_stability_abs:MAE=2.0627 RMSE=2.6679
Epoch 2/10: 100%|██████████████████████| 19234/19234 [2:02:24<00:00,  2.62it/s]
Train Loss: 0.5563 | Val aggregation_propensity:MAE=1.6439 RMSE=2.0353 expression_yield:MAE=0.6809 RMSE=1.0071 folding_stability:MAE=1.0021 RMSE=1.2270 material_production:ACC=0.7645 F1=0.8175 solubility:ACC=0.7296 F1=0.7237 temperature_stability:ACC=0.8987 F1=0.9025 temperature_stability_abs:MAE=2.0389 RMSE=2.6225
Epoch 3/10: 100%|██████████████████████| 19234/19234 [2:02:32<00:00,  2.62it/s]
Train Loss: 0.5324 | Val aggregation_propensity:MAE=1.4617 RMSE=1.8305 expression_yield:MAE=0.7139 RMSE=1.0323 folding_stability:MAE=0.8061 RMSE=1.0178 material_production:ACC=0.7722 F1=0.8250 solubility:ACC=0.7552 F1=0.7264 temperature_stability:ACC=0.9133 F1=0.9104 temperature_stability_abs:MAE=2.0108 RMSE=2.5867
Epoch 4/10: 100%|██████████████████████| 19234/19234 [2:02:42<00:00,  2.61it/s]
Train Loss: 0.5157 | Val aggregation_propensity:MAE=1.2533 RMSE=1.6132 expression_yield:MAE=0.7097 RMSE=1.0429 folding_stability:MAE=0.8241 RMSE=1.0363 material_production:ACC=0.7704 F1=0.8222 solubility:ACC=0.7546 F1=0.7315 temperature_stability:ACC=0.9167 F1=0.9147 temperature_stability_abs:MAE=2.2741 RMSE=2.9710
Epoch 5/10: 100%|██████████████████████| 19234/19234 [2:03:10<00:00,  2.60it/s]
Train Loss: 0.4942 | Val aggregation_propensity:MAE=1.0490 RMSE=1.3967 expression_yield:MAE=0.7557 RMSE=1.0712 folding_stability:MAE=0.9029 RMSE=1.1143 material_production:ACC=0.7572 F1=0.8066 solubility:ACC=0.7283 F1=0.7262 temperature_stability:ACC=0.8952 F1=0.9014 temperature_stability_abs:MAE=2.0777 RMSE=2.7313
Epoch 6/10: 100%|██████████████████████| 19234/19234 [2:03:00<00:00,  2.61it/s]
Train Loss: 0.4750 | Val aggregation_propensity:MAE=1.2680 RMSE=1.6270 expression_yield:MAE=0.7610 RMSE=1.0689 folding_stability:MAE=0.8966 RMSE=1.1043 material_production:ACC=0.7694 F1=0.8206 solubility:ACC=0.7391 F1=0.7293 temperature_stability:ACC=0.8992 F1=0.9047 temperature_stability_abs:MAE=2.3254 RMSE=3.0082
Epoch 7/10: 100%|██████████████████████| 19234/19234 [2:02:45<00:00,  2.61it/s]
Train Loss: 0.4569 | Val aggregation_propensity:MAE=1.2822 RMSE=1.6492 expression_yield:MAE=0.7354 RMSE=1.0632 folding_stability:MAE=0.8455 RMSE=1.0592 material_production:ACC=0.7662 F1=0.8165 solubility:ACC=0.7476 F1=0.7321 temperature_stability:ACC=0.9098 F1=0.9129 temperature_stability_abs:MAE=2.2975 RMSE=2.9705
Epoch 8/10: 100%|██████████████████████| 19234/19234 [2:03:14<00:00,  2.60it/s]
Train Loss: 0.4440 | Val aggregation_propensity:MAE=1.1890 RMSE=1.5380 expression_yield:MAE=0.7119 RMSE=1.0359 folding_stability:MAE=0.9509 RMSE=1.1577 material_production:ACC=0.7802 F1=0.8325 solubility:ACC=0.7571 F1=0.7348 temperature_stability:ACC=0.9181 F1=0.9200 temperature_stability_abs:MAE=2.3290 RMSE=2.9995
```