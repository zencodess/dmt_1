import os
import pandas as pd

from src.features.cleaner import DataCleaner
from src.features.feature_builder import FeatureMaker

from src.trainers.rnn_classifier import RNNClassifier
from src.trainers.randomforest_classifier import RandomForest



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')


class PredictMood():
    def __init__(self):
        self.input_data_path = os.path.join(DATA_PATH,'dataset_mood_smartphone.csv')
        self.data = None
        self.clean_df = None
        self.feature_maker = FeatureMaker()
        self.rf_input_df = None
        self.rnn_seqs, self.rnn_labels = None, None
        self.rnn_classifier = None
        self.regression_df = None

    def read_data(self):
        self.data = pd.read_csv(self.input_data_path)

    def clean_data(self):
        self.clean_df = DataCleaner.clean_and_impute(self.data)
        self.clean_df.to_csv(os.path.join(DATA_PATH,"cleaned_dataset.csv"), index=False)
        print("Data cleaning and imputation complete. Cleaned data saved to 'cleaned_dataset.csv'.")

    def feature_engineering(self):
        self.feature_maker.build_features(self.clean_df)

    def rf_data_categorization_preparation(self, impute_option='MEDIAN_IMPUTE'):
        self.rf_input_df = self.feature_maker.build_non_temporal_dataset_from_cleaned(self.clean_df, impute_option)
        #train_df, val_df, test_df = FeatureMaker.train_test_val_split(self.rf_input_df)
        print("Generating data file imputed with impute option: ",impute_option)
        if impute_option == 'ML_IMPUTE':
            file_suffix = "ml_imputed"
        elif impute_option == 'MEDIAN_IMPUTE':
            file_suffix='median_impute'
        elif impute_option=='INTERPOLATE_IMPUTE':
            file_suffix='interpolate_impute'
        else:
            file_suffix = ""
        self.rf_input_df.to_csv(os.path.join(DATA_PATH,f"rf_input_df{file_suffix}.csv"), index=False)
        #train_df.to_csv(os.path.join(DATA_PATH,f"rf_train_df{file_suffix}.csv"), index=False)
        #val_df.to_csv(os.path.join(DATA_PATH,f"rf_val_df{file_suffix}.csv"), index=False)
        #test_df.to_csv(os.path.join(DATA_PATH,f"rf_test_df{file_suffix}.csv"), index=False)
        
    def train_randomforest_classifier(self, impute_option='MEDIAN_IMPUTE'):
        self.rf_input_df = self.feature_maker.build_non_temporal_dataset_from_cleaned(self.clean_df, impute_option)
        self.RandomForestInstance=RandomForest()
        self.RandomForestInstance.modeltraining(self.rf_input_df,impute_option=impute_option)


    def train_rnn_classifier(self, impute_option='MEDIAN_IMPUTE'):
        self.rnn_seqs, self.rnn_labels = self.feature_maker.build_rnn_temporal_dataset(self.clean_df, impute_option)
        train_x, val_x, test_x, train_y, val_y, test_y = FeatureMaker.test_train_split_numpy_data(self.rnn_seqs, self.rnn_labels)
        input_dim_rnn = train_x.shape[2]
        self.rnn_classifier = RNNClassifier(input_dim_rnn)
        self.rnn_classifier.train_rnn_model(train_x, train_y, val_x, val_y)
        # continue to save model and test
        self.rnn_classifier.test_rnn_model(test_x, test_y)

    def train_regression_models(self, impute_option='MEDIAN_IMPUTE'):
        self.regression_df = self.feature_maker.build_non_temporal_dataset_from_cleaned(self.clean_df, impute_option)
        self.regression_df.to_csv(os.path.join(DATA_PATH,"regression_df.csv"), index=False)

    def run(self):
        self.read_data()
        self.clean_data()
        self.rf_data_categorization_preparation(impute_option='ML_IMPUTE')
        # train random forest classifier
        self.train_randomforest_classifier(impute_option='ML_IMPUTE')        
       
        # train rnn classifier
        #self.train_rnn_classifier(impute_option='INTERPOLATE_IMPUTE')

        # train regression models
        # insert code



if __name__ == "__main__":
    mood_predictor = PredictMood()
    mood_predictor.run()
