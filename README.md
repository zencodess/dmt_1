# dmt_1

1. To run the code please install all requirements
2. Run `python main.py`

**NOTE** : We already trained RNN model, you need not train it again. So, we commented line 107 
`self.rnn_classifier_run(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=False)` in `main.py` with production_run=False argument. 
If you want to train it again, uncomment the line above and comment below line, 
`self.rnn_classifier_run(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=True)`
