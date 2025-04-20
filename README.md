# dmt_1 ( group 4) 

1. To run the code, please install all requirements using `pip install -r requirements.txt`
2. Run `python main.py`
3. You will be able to observe the results in the order: 

 - Random Forests classifer for mood output prediction, 
 - RNN (LSTM) best model classifier for mood output prediction and then 
 - Regression results for screen activity and activity score predictions. 

**NOTE** : We already trained the RNN classifier model and saved the best model. you need not train it again. So, we commented on line 113 
`self.rnn_classifier_run(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=False)` in `main.py` with production_run=False argument. 

If you want to train it again, uncomment the line above and comment below line, at line number 116
`self.rnn_classifier_run(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=True)`

Random forest algorithm also runs only the testing phase of the model. If you want to train the Random Forest algorithm again then please change the `test_only` value in Line 108 from True to False. 
