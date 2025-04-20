import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
import warnings
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class ARIMARegression:
    def __init__(self):
        self.best_models = {}

    def train_and_evaluate(self, df, target_variable='screen_t'):
        print("Running Temporal ARIMA with Grid Search for all users...")

        actual_values = []
        predicted_values = []

        p_values = range(0, 4)
        d_values = range(0, 3)
        q_values = range(0, 4)
        pdq_combinations = list(product(p_values, d_values, q_values))

        for user_id, user_data in df.groupby('id'):
            user_data = user_data.sort_values('date')
            user_data['date'] = pd.to_datetime(user_data['date'])
            user_data.set_index('date', inplace=True)
            ts = user_data[target_variable]

            if len(ts) < 10:
                continue

            best_order = None
            best_aic = float("inf")
            best_model_fit = None

            for order in pdq_combinations:
                try:
                    model = ARIMA(ts, order=order)
                    model_fit = model.fit()

                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = order
                        best_model_fit = model_fit
                except:
                    continue

            if best_model_fit:
                try:
                    forecast_steps = min(5, len(ts))
                    forecast = best_model_fit.forecast(steps=forecast_steps)
                    actual_values.extend(ts[-forecast_steps:].values)
                    predicted_values.extend(forecast)
                    self.best_models[user_id] = (best_order, best_model_fit)
                except Exception as e:
                    print(f"Forecast failed for user {user_id} with order {best_order}: {e}")
            else:
                print(f"No valid ARIMA model found for user {user_id}")

        print("ARIMA Completed for All Users")

        if actual_values:
            mse = mean_squared_error(actual_values, predicted_values)
            mae = mean_absolute_error(actual_values, predicted_values)
            print(f"\nARIMA Results - MSE: {mse:.4f}, MAE: {mae:.4f}")
        else:
            print("Not enough data for ARIMA evaluation.")

        return np.array(actual_values), np.array(predicted_values)

    def plot_results(self, actual, predicted):
        plt.figure(figsize=(7, 4))
        plt.plot(actual, label="True (ARIMA)", marker="o")
        plt.plot(predicted, label="Predicted (ARIMA)", marker="x")
        plt.title("ARIMA: True vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
