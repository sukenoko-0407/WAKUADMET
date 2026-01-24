# Pebble OpenADMET model report

# Description of the Model

The submission represents an ensemble of several graph neural network architectures. The ensemble includes both models trained on only Expansion data, as well as models trained on Expansion data along with curated public and proprietary ADMET data.

Models were trained separately for related task groups, specifically: {LogD}, {KSOL}, {MLM, HLM}, {Caco-2 Papp, Caco-2 ER}, {MPPB, MGMB, MBPB}. For some task-group models, data from other properties was included as additional tasks to improve model performance. For example, many of the models were trained to predict LogD and pKa along with their target tasks.

Models were optimized via Optuna using both temporal (compound-ID based) and clustered splits. Models were ensembled using the [Caruana](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) method using out-of-fold CV predictions.

# Performance comments

We observed that locally trained models tended to have very high variance in their predictions for compounds in chemical space not well-represented in the Expansion training set. Models trained with additional ADMET data had more stable predictions for these compounds.

Validation accuracy on the labeled set, even with use of careful temporal or clustered validation approaches, did not always correlate closely with accuracy on the later-in-time blind test set. In our experience this is typical for drug programs, which frequently undergo distribution shifts as the program evolves over time.  
 

