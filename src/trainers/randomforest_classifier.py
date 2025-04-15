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


class RandomForest():
    def test(self):
        file_path=os.path.join(os.path.dirname(__file__),'..','data','rf_input_df.csv')
        clean_data=pd.read_csv(file_path)
        
        print(clean_data.shape[0])
                
        X = clean_data[[ 'appCat.weather_sum_hist','appCat.social_sum_hist', 'appCat.finance_sum_hist', 'appCat.entertainment_sum_hist','appCat.communication_sum_hist','circumplex.arousal_mean_hist']]
        Y = clean_data[['mood_output']].values.ravel()
        
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

        print(clean_data[['mood_output']].value_counts(normalize=True))

        rf=RandomForestClassifier(class_weight='balanced')
        rf.fit(X_train,Y_train)

        Y_pred=rf.predict(X_test)

        accuracy=accuracy_score(Y_test,Y_pred)
        
        print(confusion_matrix(Y_test, Y_pred))
#        print(classification_report(Y_test, y_pred, digits=4))

        print("Accuracy: ",accuracy)

