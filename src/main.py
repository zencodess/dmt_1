import os
import pandas as pd
import numpy as np

from src.features.cleaner import DataCleaner
from src.features.feature_builder import FeatureMaker

from src.trainers.rnn_classifier import AttentionLSTM
from src.trainers.randomforest_classifier import RandomForest, MODEL_PATH
from src.utils.const import EXP_ML_IMPUTE, ML_IMPUTE, MEDIAN_IMPUTE, ZERO_IMPUTE, INTERPOLATE_IMPUTE, IMPUTE_OPTIONS, \
    LOCF_ROLLING_MEAN_IMPUTE, BEST_RNN_MODEL, RF_REGRESSOR, RBF_BAYESIAN_RIDGE
from src.trainers.arima_regression import ARIMARegression
from src.trainers.svr_regression import SVRRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class PredictMood():
    def __init__(self):
        self.input_data_path = os.path.join(DATA_PATH, 'dataset_mood_smartphone.csv')
        self.data = None
        self.clean_df = None
        self.feature_maker = FeatureMaker()
        self.rf_input_df = None
        self.rnn_seqs, self.rnn_labels = None, None
        self.rnn_instance = None
        self.arima_regression = None
        self.svr_regression = None
        self.regression_df = None

    def read_data(self):
        self.data = pd.read_csv(self.input_data_path)

    def clean_data(self):
        self.clean_df = DataCleaner.clean_data_pipe(self.data)
        self.clean_df.to_csv(os.path.join(DATA_PATH, "cleaned_dataset.csv"), index=False)
        print("Data cleaning and imputation complete. Cleaned data saved to 'cleaned_dataset.csv'.")

    def rf_data_categorization_preparation(self, impute_option=ML_IMPUTE, impute_strategy=RF_REGRESSOR):
        self.rf_input_df = self.feature_maker.build_non_temporal_dataset_from_cleaned(self.clean_df, impute_option,
                                                                                      impute_strategy)
        print("Generating RF data file imputed with impute option: ", impute_option, "and impute strategy: ",
              impute_strategy)
        if impute_option in IMPUTE_OPTIONS:
            file_suffix = impute_option.lower()
        else:
            file_suffix = ""
        self.rf_input_df.to_csv(os.path.join(DATA_PATH, f"rf_input_df{file_suffix}.csv"), index=False)

    def train_randomforest_classifier(self, impute_option=MEDIAN_IMPUTE):
        self.RandomForestInstance = RandomForest()
        self.RandomForestInstance.modeltraining(self.rf_input_df, impute_option=impute_option)

    def rnn_classifier_run(self, impute_option=MEDIAN_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=True):
        for seqs, labels in self.feature_maker.build_rnn_temporal_dataset(self.clean_df, impute_option,
                                                                          impute_strategy):
            self.train_or_test_rnn_classifier(seqs, labels, production_run)

    def train_or_test_rnn_classifier(self, rnn_seqs, rnn_labels, production_run):
        train_x, val_x, test_x, train_y, val_y, test_y = FeatureMaker.test_train_split_numpy_data(rnn_seqs, rnn_labels)
        input_dim_rnn = train_x.shape[2]
        if production_run:
            self.rnn_instance = AttentionLSTM(input_dim_rnn)
            self.rnn_instance.load_model(os.path.join(MODEL_PATH, BEST_RNN_MODEL))
            # evaluate model after loading best model
            self.rnn_instance.test_rnn_model(test_x, test_y)
        else:
            self.rnn_instance = AttentionLSTM(input_dim_rnn)
            self.rnn_instance.train_rnn_model(train_x, train_y, val_x, val_y)
            self.rnn_instance.test_rnn_model(test_x, test_y)

    def train_regression_models(self, target_variable, impute_option=MEDIAN_IMPUTE):
        self.regression_df = self.feature_maker.build_non_temporal_dataset_from_cleaned(self.clean_df, impute_option)
        if target_variable == 'screen_t':
            self.regression_df[target_variable + '_log'] = np.log(self.regression_df[target_variable])
            target_variable = target_variable + '_log'

        train_df, test_df = FeatureMaker.train_test_split_regression(self.regression_df)
        # SVR regression
        self.svr_regression = SVRRegression()
        svr_model, y_test_svr, y_pred_svr = self.svr_regression.train_and_evaluate(
            train_df, test_df, target_variable=target_variable)
        svr_plot = self.svr_regression.plot_results(y_test_svr, y_pred_svr)
        # ARIMA regression
        self.arima_regression = ARIMARegression()
        actual_arima, pred_arima = self.arima_regression.train_and_evaluate(self.regression_df,
                                                                            target_variable=target_variable)
        arima_plot = self.arima_regression.plot_results(actual_arima, pred_arima)

    def run(self):
        self.read_data()
        self.clean_data()

        # train random forest classifier
        self.rf_data_categorization_preparation(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE)
        self.train_randomforest_classifier(impute_option=ML_IMPUTE)

        # train rnn classifier
        # NOTE: you do not need to train temporal rnn model, we already did and stored at models/best_rnn_model.pth
        # but just in case, you can try using below function by uncommenting it
        # self.rnn_classifier_run(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=False)

        # use best rnn classifier already trained before
        self.rnn_classifier_run(impute_option=ML_IMPUTE, impute_strategy=RBF_BAYESIAN_RIDGE, production_run=True)

        # train regression models
        self.train_regression_models(target_variable='screen_t')
        self.train_regression_models(target_variable='activity_t')


if __name__ == "__main__":
    mood_predictor = PredictMood()
    mood_predictor.run()
