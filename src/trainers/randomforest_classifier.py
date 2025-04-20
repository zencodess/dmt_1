import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import os
import joblib
import matplotlib.pyplot as plt

from src.utils.const import EXP_ML_IMPUTE, ML_IMPUTE, MEDIAN_IMPUTE, ZERO_IMPUTE, INTERPOLATE_IMPUTE, LOCF_ROLLING_MEAN_IMPUTE


DATA_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')
MODEL_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','models')
from sklearn.inspection import permutation_importance



class RandomForest():
    def modeltraining(self,clean_data,impute_option=MEDIAN_IMPUTE):
        print("Random Forest Model for Dataset cleaned with impute option: ",impute_option)
        X=clean_data.drop(['mood_output','id','date'], axis=1)
        Y=clean_data[['mood_output']].values.ravel()
        X_train_init,X_test_init,Y_train_init,Y_test_init=train_test_split(X,Y,test_size=0.2,random_state=42)
        #print(clean_data[['mood_output']].value_counts(normalize=True))
        rf=RandomForestClassifier(random_state=2) #A simple random forest that uses all columns
        rf.fit(X_train_init,Y_train_init)
        feature_importances = pd.Series(rf.feature_importances_, index=X_train_init.columns).sort_values(ascending=False) #Get the important features
        #print(feature_importances[:5])
        
        result = permutation_importance(rf, X_train_init, Y_train_init, n_repeats=10, random_state=42)
        importance = pd.Series(result.importances_mean, index=X_train_init.columns).sort_values(ascending=False)
        #print(importance[:5])
        X1=clean_data[feature_importances.index[:5]]
        Y=clean_data[['mood_output']].values.ravel()
        X2=clean_data[importance.index[:5]]
        X_train1,X_test1,Y_train1,Y_test1=train_test_split(X1,Y,test_size=0.2,random_state=42)
        X_train2,X_test2,Y_train2,Y_test2=train_test_split(X2,Y,test_size=0.2,random_state=42)
        hyperparameter_values={
            'n_estimators'      : randint(50,200), #Slower as the value increases
            'max_depth'         : randint(10,20), #Reduce overfitting as it decreases
            'min_samples_leaf'  : randint(1,10), #Reduces overfitting as it increases, more general
            'random_state'      : randint(2,24) #Ensure the same variables are used when running the model, so that there is no ambiguity in the prediction
        }
        rf=RandomForestClassifier() #class_weight='balanced', #This is not required as we have an almost evenly split classification so the number of 0 occurences is relatively similar to the number of 1 occurences. 
        rand_search1=RandomizedSearchCV(rf,param_distributions=hyperparameter_values,n_iter=5,cv=5)
        rand_search1.fit(X_train1,Y_train1)
        best_rf1 = rand_search1.best_estimator_
        #print(best_rf1)
        self.modelsave(best_rf1,'rfginimodel')
        self.testmodel(X_test1,Y_test1,'rfginimodel')
        rf=RandomForestClassifier() #class_weight='balanced', #This is not required as we have an almost evenly split classification so the number of 0 occurences is relatively similar to the number of 1 occurences. 
        rand_search2=RandomizedSearchCV(rf,param_distributions=hyperparameter_values,n_iter=5,cv=5)
        rand_search2.fit(X_train2,Y_train2)
        best_rf2 = rand_search2.best_estimator_
        #print(best_rf2)
        self.modelsave(best_rf2,'rfpermutationmodel')
        self.testmodel(X_test2,Y_test2,'rfpermutationmodel')
            
        
    def modelsave(self,rf,model_name):
        if(model_name=='rfpermutationmodel'):
            joblib.dump(rf,os.path.join(MODEL_PATH,'best_rf_model.joblib'))
        joblib.dump(rf,os.path.join(MODEL_PATH,model_name+'.joblib'))
        
    def testmodel(self,X_test,Y_test,modelname):
        saved_model=joblib.load(os.path.join(MODEL_PATH,modelname+'.joblib'))
        Y_pred=saved_model.predict(X_test)
        accuracy=accuracy_score(Y_test,Y_pred) #Correct answer predicted : happy person is happy and sad person is sad.
        precision = precision_score(Y_test,Y_pred) #When predicted happy how many times was the model right?
        recall = recall_score(Y_test,Y_pred) #Of all happy people who were correctly identified?      
        print("Accuracy: ",accuracy) 
        print("Precision: ",precision)
        print("Recall: ",recall)
        cm=confusion_matrix(Y_test, Y_pred)#Shows the tp,fp,tn,fn
        print(cm)
        #ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        #plt.show()

