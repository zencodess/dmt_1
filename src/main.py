import pandas as pd
from src.features.cleaner import DataCleaner


DATA_PATH = 'data/dataset_mood_smartphone.csv'



class PredictMood():
    def __init__(self):
        self.input_data_path = DATA_PATH
        self.data = None
        self.clean_df = None

    def read_data(self):
        self.data = pd.read_csv(self.input_data_path)

    def clean_data(self):
        self.clean_df = DataCleaner.clean_and_impute(self.data)
        self.clean_df.to_csv("data/cleaned_dataset.csv", index=False)
        print("Data cleaning and imputation complete. Cleaned data saved to 'cleaned_dataset.csv'.")

    def run(self):
        self.read_data()
        self.clean_data()


if __name__ == "__main__":
    mood_predictor = PredictMood()
    mood_predictor.run()
