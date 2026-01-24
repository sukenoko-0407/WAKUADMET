# ExpansionRx - Model Summary

## Description of the Model

**Algorithm:** ChemProp version 2

**Parameters:** Ensemble of different parameters across ensembles and across endpoints with Bounded-MSE loss, MVE estimators for raw (bounded) data but sometimes delimited data removed on a per endpoint basis

**Data:** Only the data provided for the challenge and curated publicly available data were utilised. No proprietary data was used.

## Additional Steps

- All endpoint predictions consist of an ensemble from multiple different ChemProp approaches – no one model or approach was used per-task. Weighted averaging was applied during the ensembling to certain endpoints.

- Reuse of pre-existing representations: Prior weights are used from ChemMeleon, published work, and our pre-existing in-house ADMET multi-task model trained on large scale ADMET tasks from curated public data sources

- Multitask pretraining on bespoke de novo chemical space: Pretraining was also performed millions of de novo generated molecules produced by a bespoke pipeline: bivalent ExpansionRx molecules were explicitly identified, cut, and the payload/linker fragments combinatorially enumerated; REINVENT was trained via reinforcement learning separately on competition small molecules and the enumerated bivalents; large-scale filtering and cleaning retained only chemically sensible structures. The generative process was intentionally constrained to chemistry relevant to the competition, while avoiding model collapse or drift into irrelevant regions of chemical space. This curated yet diverse corpus of 10M compounds allowed representation learning of the competition space, yielding a starting representation that may generalise better than foundational Chemprop pretraining on random samples (e.g. CheMeleon) or narrow datasets

- Explicit task-level masking during pretraining: For each de novo training batch, a fixed-size random subset of Mordred descriptor and calculated physicochemical property tasks is selected, and all other targets are masked out (set to NaN and excluded from the loss). This is analogous to masked-objective training in large multitask language models, but applied here to Chemprop-style molecular graphs: the same molecular graphs are exposed across the many batches while different task subsets are revealed each time, preventing the network from overfitting to correlated descriptor blocks (e.g. Mordred families) and encouraging the MPNN to learn task-agnostic chemical features

- Training stabilisation via staged freezing: A staged freezing strategy was applied to the subsequent adaptation to the competition dataset. In early training phases, the message-passing architecture (MPNN and associated batch-normalisation layers) was frozen while only the prediction head was optimised. After this warm-up phase, the trunk was unfrozen and jointly fine-tuned with the head, enabling graph representations to adapt to target tasks while avoiding early optimisation instability during transition from foundation to competition-specific training

- Downweighting of auxiliary training tasks: Although ExpansionRx tasks may be trained jointly via multi-task learning, optimisation was performed explicitly per-task: the task(s) of primary interest were given higher loss weights, while auxiliary tasks were down-weighted (on a per-endpoint basis). This allowed auxiliary signals to shape learned representations without overwhelming or diluting the task of intent, ensuring shared features beneficial to the main endpoint dominated learned embeddings

- Auxiliary (side information) data options: Auxiliary data were incorporated to competition-specific training via two frameworks:
  - (i) as additional input features appended to the descriptor matrix, or
  - (ii) as auxiliary prediction targets appended to the label matrix

  After incorporation in (ii), task affinity grouping (TAG) was implemented via a meta-learning procedure. In initial exploration, task combinations were evaluated whether joint optimisation improved primary task of intent loss and used to infer affinity groups. Only tasks within the affinity group were trained jointly, ensuring representations were learnt from chemically and biologically compatible tasks, reducing negative transfer from unrelated endpoints. We observed improvements appending selected in-house ADMET multi-task bioactivity matrices derived from curated public data sources (without overlap to the competition data), as well as predictions from a public predictive head (https://pubs.acs.org/doi/full/10.1021/acsomega.5c04861) to the label matrix.

- Additional featurization: Mordred, Jazzy and ECFP descriptor features were evaluated and applied on a TAG group basis

- De-noising applied to training data: Denoising of the training data was performed once there were sufficiently supported predictions on the training set, as in: https://chemrxiv.org/engage/chemrxiv/article-details/661cd020418a5379b0df1fda

- Reconciled public data into training: Public data for HLM and MLM CLint endpoints were reconciled into the ExpansionRx tasks for some ensemble members

- Method agnostic ensembling: Multiple variants of the above configurations (different masking schemes, auxiliary feature/label combinations, and task weightings) were trained, and ensembles constructed per-endpoint, rather than relying on a single fixed pretraining recipe

- Stabilised learning and freezing: To stabilise optimisation, some networks were trained deliberately slowly via extended learning-rate warmup schedules (10–30 epochs depending on the task). Conservative initial, max and final learning rates and relatively strong weight decay on the predictor head were utilised while keeping lighter trunk regularisation (dropout applied primarily within the FFN). Our aim was to encourage incremental adaptation of the learned molecular representations, allowing signal to accumulate across epochs while suppressing sensitivity to batch-level noise and spurious correlations.

## Observed Performance During Training

Bivalent molecules were treated as a distinct and highly variant chemical class, exhibiting substantially greater structural diversity and conformational flexibility than typical small molecules, with correspondingly high variance in model predictions and performance during training. Their controlled fragmentation and explicit enumeration during foundational pre-training were considered critical to prevent representation collapse toward the dominant small-molecule chemical space, which may be less represented in the final leaderboard test set. During representation learning, particular care was taken with inherently noisy experimental endpoints, which were reflected in volatile leaderboard performance that changed rapidly across submissions and exhibited behaviour consistent with substantial label and evaluation noise. The masked multi-task objective, task subsampling, auxiliary-task regularisation, slowed training, and staged unfreezing of the trunk were collectively used to regularise learning, encouraging robust and transferable representations vs. memorisation of assay noise during training.

In addition, some ensemble members were selected not on lowest validation MAE, but on clearly higher R² performance. This choice was made to favour models capturing stronger explanatory structure, with the potential that this would translate into improved generalisation and superior performance on the final leaderboard.
