## Methodology Report
### Description of the Model(s)
We employed a hybrid modelling strategy combining deep learning and classical machine learning approaches to maximise predictive performance and robustness across tasks. For deep learning, we used a multi-task message-passing neural networks (MPNNs) and a graph transformer model. Hyperparameter optimisation was performed using Ray Tune and HyperBand (https://docs.ray.io/en/latest/tune/index.html) which gave some small performance gains. 

In parallel, we trained a suite of classical machine learning models using AutoGluon, covering both CPU- and GPU-based learners with automated model selection and ensembling (https://github.com/autogluon/autogluon). These models were trained with an extensive set of classical molecular representations generated using scikit-fingerprints (https://github.com/scikit-fingerprints/scikit-fingerprints).

Beyond standard supervised training on the provided datasets, several additional steps were applied for some endpoints:
-	Pretraining on selected external datasets augmented with internal proprietary data, with task-specific weighting applied to relevant endpoints.
-	Automated model optimisation via Ray Tune (deep learning) and AutoGluon (classical models).
-	Model evaluation using both 5×5 cross-validation with multiple random seeds to ensure robustness and time-based holdout splits on the training set.
-	Once a model is stable, we retrained the models on the entire provided training data. 

Following evaluation, the best-performing classical and deep learning models were aggregated into an ensemble, inspired by prior work on heterogeneous model aggregation https://arxiv.org/pdf/2006.08573

### Performance comments
Some endpoints benefited from the integration of external datasets and multi-task training, leading to substantial performance gains. For certain endpoints, cross-validation scores were poorly correlated with held-out performance and exhibited higher variance. 