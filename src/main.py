import pandas as pd
from src.features.cleaner import DataCleaner
from src.features.feature_builder import FeatureMaker


DATA_PATH = 'data/dataset_mood_smartphone.csv'



class PredictMood():
    def __init__(self):
        self.input_data_path = DATA_PATH
        self.data = None
        self.clean_df = None
        self.feature_maker = FeatureMaker()
        self.df_predictive = None

    def read_data(self):
        self.data = pd.read_csv(self.input_data_path)

    def clean_data(self):
        self.clean_df = DataCleaner.clean_and_impute(self.data)
        self.clean_df.to_csv("data/cleaned_dataset.csv", index=False)
        print("Data cleaning and imputation complete. Cleaned data saved to 'cleaned_dataset.csv'.")

    def feature_engineering(self):
        self.feature_maker.build_features(self.clean_df)

    def run(self):
        self.read_data()
        self.clean_data()
        self.df_predictive = self.feature_maker.build_predictive_dataset_from_cleaned(self.clean_df)
        self.df_predictive.to_csv("data/predictive_mood_dataset.csv", index=False)
        train_df, val_df, test_df = self.feature_maker.train_test_split(self.df_predictive)
        train_df.to_csv("data/train_df.csv", index=False)
        val_df.to_csv("data/val_df.csv", index=False)
        test_df.to_csv("data/test_df.csv", index=False)

if __name__ == "__main__":
    mood_predictor = PredictMood()
    mood_predictor.run()
