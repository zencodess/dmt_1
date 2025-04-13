import pandas as pd
from src.features.cleaner import DataCleaner
from src.features.feature_builder import FeatureMaker

from rnn_classifier import RNNClassifier


DATA_PATH = 'data/dataset_mood_smartphone.csv'



class PredictMood():
    def __init__(self):
        self.input_data_path = DATA_PATH
        self.data = None
        self.clean_df = None
        self.feature_maker = FeatureMaker()
        self.df_predictive = None
        self.rnn_classifier = None

    def read_data(self):
        self.data = pd.read_csv(self.input_data_path)

    def clean_data(self):
        self.clean_df = DataCleaner.clean_and_impute(self.data)
        self.clean_df.to_csv("data/cleaned_dataset.csv", index=False)
        print("Data cleaning and imputation complete. Cleaned data saved to 'cleaned_dataset.csv'.")

    def feature_engineering(self):
        self.feature_maker.build_features(self.clean_df)

    def train_random_forest_classifier(self):
        train_df, val_df, test_df = self.feature_maker.train_test_split(self.df_predictive)
        train_df.to_csv("data/train_df.csv", index=False)
        val_df.to_csv("data/val_df.csv", index=False)
        test_df.to_csv("data/test_df.csv", index=False)
        # continue coding

    def train_rnn_classifier(self):
        seqs, labels = self.feature_maker.build_rnn_sequence_dataset(self.df_predictive)
        train_x, val_x, test_x, train_y, val_y, test_y = self.feature_maker.test_train_split_numpy_data(seqs, labels)
        input_dim_rnn = train_x.shape[2]
        self.rnn_classifier = RNNClassifier(input_dim_rnn)
        model_rnn = self.rnn_classifier.train_rnn_model(train_x, train_y, val_x, val_y)
        


    def run(self):
        self.read_data()
        self.clean_data()
        self.df_predictive = self.feature_maker.build_predictive_dataset_from_cleaned(self.clean_df)
        self.df_predictive.to_csv("data/predictive_mood_dataset.csv", index=False)
        # train random forest classifier
        self.train_random_forest_classifier()
        # train rnn classifier
        self.train_rnn_classifier()



if __name__ == "__main__":
    mood_predictor = PredictMood()
    mood_predictor.run()
