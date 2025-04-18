import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


class SVRRegression:
    def __init__(self):
        self.model = None
        self.y_scaler = StandardScaler()
        self.important_features = [
            'appCat.communication_sum_hist', 'appCat.entertainment_sum_hist', 'appCat.game_sum_hist',
            'appCat.office_sum_hist', 'appCat.social_sum_hist', 'appCat.finance_sum_hist',
            'screen_sum_hist', 'circumplex.arousal_mean_hist', 'circumplex.valence_mean_hist',
            'call_sum_hist', 'sms_sum_hist', 'appCat.other_sum_hist',
            'appCat.travel_sum_hist', 'appCat.utilities_sum_hist', 'appCat.weather_sum_hist'
]


    def train_and_evaluate(self, train_df, test_df, target_variable='screen_t'):
        print("Running SVR Regression with Y-scaling...")

        X_train = train_df[self.important_features]
        y_train = train_df[target_variable]

        X_test = test_df[self.important_features]
        y_test = test_df[target_variable]

        # Scale target variable
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        pipeline = make_pipeline(StandardScaler(), SVR())
        param_grid = {
            'svr__C': [0.1, 1, 10, 5],
            'svr__epsilon': [0.1, 0.2, 0.5, 0.9],
            'svr__kernel': ['rbf', 'linear'],
            'svr__gamma': ['scale', 'auto']
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train_scaled)

        y_pred_scaled = grid_search.predict(X_test)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print("Best SVR Params:", grid_search.best_params_)
        print(f"SVR Results - MSE: {mse:.2f}, R2: {r2:.2f}, MAE: {mae:.2f}")

        self.model = grid_search.best_estimator_
        return self.model, y_test, y_pred

    def plot_results(self, y_test, y_pred):
        plt.figure(figsize=(7, 4))
        plt.plot(y_test.values, label="True", marker="o")
        plt.plot(y_pred, label="Predicted", marker="x")
        plt.title("SVR: True vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
