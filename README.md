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
