import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.cleaner import DataCleaner
from src.utils.const import EXP_ML_IMPUTE, ML_IMPUTE, MEDIAN_IMPUTE, ZERO_IMPUTE, INTERPOLATE_IMPUTE, DURATION_VARS, \
    CATEGORICAL_VARS, NUMERICAL_VARS, LOCF_ROLLING_MEAN_IMPUTE, RBF_BAYESIAN_RIDGE


class FeatureMaker():
    def __init__(self, df=None):
        self.df = df
        self.categorical_variables = CATEGORICAL_VARS
        self.numerical_variables = NUMERICAL_VARS
        self.duration_variables = DURATION_VARS
        self.categorical_features = []
        self.numerical_features = []
        self.duration_features = []

    def build_daily_average_df(self, df, impute_option, impute_strategy=RBF_BAYESIAN_RIDGE):
        # creating one data row for one day
        df["date"] = df["time"].dt.date
        daily_avg = df.pivot_table(
            index=["id", "date"],
            columns="variable",
            values="value",
            aggfunc="mean"
        ).reset_index()
        daily_avg =  daily_avg.sort_values(by=["id", "date"]).reset_index(drop=True)

        if impute_option == EXP_ML_IMPUTE:
            impute_results = DataCleaner.ml_impute_experiments(daily_avg)
            return impute_results
        elif impute_option == ML_IMPUTE:
            adv_imputed_daily_avg = DataCleaner.advanced_impute_missing_with_ml(daily_avg, impute_strategy)
        elif impute_option == MEDIAN_IMPUTE:
            adv_imputed_daily_avg = DataCleaner.fill_null_values_with_median(daily_avg)
        elif impute_option == INTERPOLATE_IMPUTE:
            adv_imputed_daily_avg = DataCleaner.advanced_impute_linear_interpolator(daily_avg)
        elif impute_option == ZERO_IMPUTE:
            adv_imputed_daily_avg = DataCleaner.fill_null_vars_with_zero(daily_avg)
        elif impute_option == LOCF_ROLLING_MEAN_IMPUTE:
            adv_imputed_daily_avg = DataCleaner.locf_rolling_mean_impute_dataframe(daily_avg)

        # in the end, if any null values still exist, replace them with zero for safety
        adv_imputed_daily_avg = DataCleaner.fill_null_vars_with_zero(adv_imputed_daily_avg)
        return adv_imputed_daily_avg

    def build_non_temporal_dataset_from_cleaned(self, cleaned_df, impute_option, impute_strategy=RBF_BAYESIAN_RIDGE, window_size=5):
        daily_avg = self.build_daily_average_df(cleaned_df, impute_option, impute_strategy)
        instance_rows = []

        # group rows by id, sort by date, build features
        for user_id, group in daily_avg.groupby("id"):
            group = group.sort_values("date").reset_index(drop=True)
            for i in range(window_size, len(group)):
                history_window = group.iloc[i - window_size:i]  # t-5 to t-1
                target_row = group.iloc[i]  # t
                row = {
                    "id": user_id,
                    "date": target_row["date"]
                }
                # categorical variables - sum only
                for var in self.categorical_variables:
                    if var in group.columns:
                        row[f"{var}_sum_hist"] = history_window[var].sum()
                # numerical variables - mean and std
                for var in self.numerical_variables:
                    if var in group.columns:
                        row[f"{var}_mean_hist"] = history_window[var].mean()
                        row[f"{var}_std_hist"] = history_window[var].std()
                # duration variables - sum and std
                for var in self.duration_variables:
                    if var in group.columns:
                        row[f"{var}_sum_hist"] = history_window[var].sum()
                        row[f"{var}_std_hist"] = history_window[var].std()
                # day t values except mood
                for var in self.categorical_variables + self.numerical_variables + self.duration_variables:
                    if var in group.columns and var != "mood":
                        row[f"{var}_t"] = target_row[var]
                # target - mood at day t
                if "mood" in target_row:
                    mood = target_row["mood"]
                    row["mood_output"] = 1 if mood >= 7 else 0  # gives good balance of classes now, Train label balance: [797 546] Val label balance: [114  78]
                instance_rows.append(row)
        df_instances = pd.DataFrame(instance_rows)
        return df_instances

    @staticmethod
    def categorize_mood_col(df):
        mood_median = df["mood"].median()
        df["mood_output"] = df["mood"].apply(lambda x: 1 if x >= mood_median else 0)
        return df

    def build_rnn_temporal_dataset(self, df_instances, impute_option, impute_strategy=RBF_BAYESIAN_RIDGE):
        if impute_option == EXP_ML_IMPUTE:
            results = self.build_daily_average_df(df_instances, impute_option)
            for impute_method, daily_avg in results.items():
                daily_avg = DataCleaner.fill_null_vars_with_zero(daily_avg)
                yield self.daily_avg_to_numpy_seqs(daily_avg)
        else:
            # best ML impute strategy is RBF+BAYESIAN_RIDGE
            daily_avg = self.build_daily_average_df(df_instances, impute_option, impute_strategy)
            yield self.daily_avg_to_numpy_seqs(daily_avg)

    def daily_avg_to_numpy_seqs(self, daily_avg, sequence_length=6):
        daily_avg = FeatureMaker.categorize_mood_col(daily_avg)
        feature_cols = [col for col in daily_avg.columns if col not in ["id", "date"]]
        sequences = []
        labels = []
        for user_id, group in daily_avg.groupby("id"):
            group = group.sort_values("date").reset_index(drop=True)
            for i in range(len(group) - sequence_length + 1):
                window = group.iloc[i:i + sequence_length]
                if len(window) == sequence_length:
                    input_features = window.iloc[:-1][feature_cols].values.astype(np.float32)
                    last_day_features = window.iloc[-1][[col for col in feature_cols if col != "mood_output"]].values.astype(np.float32)
                    last_day_features = last_day_features.reshape(1, -1)
                    if last_day_features.shape[1] < input_features.shape[1]:
                        # pad last_day_features with zeros to match input_features width
                        pad_width = input_features.shape[1] - last_day_features.shape[1] # diff will always be 1, coz mood output of last day is excluded
                        last_day_features = np.pad(last_day_features, ((0, 0), (0, pad_width)), mode='constant')
                    input_features = np.vstack([input_features, last_day_features])
                    output_label = int(window.iloc[-1]["mood_output"])
                    sequences.append(input_features)
                    labels.append(output_label)
        return np.array(sequences), np.array(labels)

    @staticmethod
    def train_test_val_split(df, test_size=0.2, val_size=0.1):
        stratify_col = df['mood_output'] if 'mood_output' in df else None
        # train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=42
        )
        # train vs val - adjusted split from train_val
        val_split_ratio = val_size / (1 - test_size)
        stratify_col_train_val = train_val_df['mood_output'] if 'mood_output' in train_val_df else None
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_split_ratio,
            stratify=stratify_col_train_val,
            random_state=42
        )
        return train_df, val_df, test_df

    @staticmethod
    def test_train_split_numpy_data(X, y, test_size=0.2, val_size=0.1):
        train_val_x, test_x, train_val_y, test_y = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        val_ratio_adjusted = val_size / (1 - test_size)
        train_x, val_x, train_y, val_y = train_test_split(
            train_val_x, train_val_y, test_size=val_ratio_adjusted, stratify=train_val_y, random_state=42
        )
        return train_x, val_x, test_x, train_y, val_y, test_y

    @staticmethod
    def train_test_split_regression(df, test_size=0.2):
        # train vs test
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )
        return train_df, test_df
