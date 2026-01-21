# OpenADMET-ExpansionRx-Challenge

Final submission was generated using the winsorized mean over multiple modeling strategies:

- Three **Chemprop multitask** neural networks
- **TabPFN** trained on a single shared feature set
- **TabPFN** trained with endpoint-specific feature sets

---

## Features

Multiple molecular representations were used:
- **RDKit:** 2D descriptors, ECFP4 (binary & count), 3D descriptors, Avalon, ERG fingerprints
- **ChemAxon:** pKa, logP, logD, H/L/B descriptors (calculated with their Python API)
- **Jazzy:** Interaction energy descriptors
- **CheMeleon:** 2048-bit molecular embeddings
- **Minimol:** 512-bit molecular embeddings
- **AZ-pred:** Predictions from Chemprop MT model published by AZ (https://pubs.acs.org/doi/10.1021/acsomega.5c04861)

---

## Models

### Chemprop multitask
- Chemprop 2.2.1 with CheMeleon foundation model
- Multitask training across all endpoints
- A separate model was trained for each descriptor set provided
via --descriptors-path (ChemAxon, RDKit2D, AZ-pred).

### TabPFN (Single Feature Set)
- `TabPFNRegressor` (v2.5)
- Feature set: RDKit2D + CheMeleon + ChemAxon + AZ-pred
- Feature reduction:
  - Drop constant features
  - Remove correlated features (Pearson > 0.85). Cutoff was chosen to stay <2000 features (TabPFN 2.5 limit)

### TabPFN (Endpoint-Specific)
- `TabPFNRegressor` (v2.5)
- Feature sets selected per endpoint via 5-fold CV
- 2048-bit ECFP4 and CheMeleon excluded due to feature limits in TabPFN 2.5

**Final feature sets:**
- Caco2_Efflux: ChemAxon + ERG + Jazzy + Minimol
- Caco2_Papp_AB: Avalon + ChemAxon + ERG + Jazzy + Minimol + RDKit2D
- HLM_CLint: Avalon + ChemAxon + ERG + Jazzy + Minimol + RDKit2D
- KSOL: Avalon + ChemAxon + ERG + Jazzy + Minimol + RDKit2D
- LogD: Avalon + ChemAxon + ERG + Jazzy + Minimol + RDKit2D
- MBPB: Avalon + ChemAxon + ERG + Jazzy + Minimol + RDKit2D
- MGMB: Avalon + ChemAxon + ERG + Minimol
- MLM_CLint: ChemAxon + Minimol + RDKit2D
- MPPB: ChemAxon + Minimol + RDKit2D

---

## Performance comments
Model performance during training was not used as the primary selection criterion. Instead, models were trained and evaluated directly on the leaderboard, and final model selection was based on leaderboard performance.

## Notes

The following approaches were explored but did not give me any good performance:
- Using predictions from public ADMET models as additional features
- Delta learning with public ADMET data
- Fine-tuning TabPFN
- Finetuning https://github.com/NVIDIA-Digital-Bio/KERMT on challenge dataset.
- AutoGluon ensemble models using the usual suspects. TabPFN 2.5 produced a better model in seconds than AutoGluon did after 4 hours

Interestingly, none of the final models make use of ECFP4 fingerprints (binary or count-based). In my experiments, count-based fingerprints consistently resulted in poorer model performance.
