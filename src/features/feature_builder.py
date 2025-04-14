import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.cleaner import DataCleaner


class FeatureMaker():
    def __init__(self, df=None):
        self.df = df
        self.categorical_variables = ["call", "sms"]
        self.numerical_variables = ["circumplex.arousal", "circumplex.valence", "activity"]
        self.duration_variables = ["screen", "appCat.builtin", "appCat.communication", "appCat.entertainment",
        "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social",
        "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]
        self.categorical_features = []
        self.numerical_features = []
        self.duration_features = []

    def fill_null_values_with_median(self, df):
        cols_to_fill = list(set(df.columns) - set(["id", "date", "mood_output"]))
        # fill with median values
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df

    def build_daily_average_df(self, df, enable_ml_impute):
        # creating one data row for one day
        df["date"] = df["time"].dt.date
        daily_avg = df.groupby(["id", "date", "variable"])["value"].mean().unstack().reset_index()
        daily_avg =  daily_avg.sort_values(by=["id", "date"]).reset_index(drop=True)
        print('daily_avg', daily_avg.head())

        if enable_ml_impute:
            adv_imputed_daily_avg = DataCleaner.advanced_impute_missing_with_ml(daily_avg)
        else:
            adv_imputed_daily_avg = self.fill_null_values_with_median(daily_avg)
        return adv_imputed_daily_avg

    def build_predictive_dataset_from_cleaned(self, cleaned_df, enable_ml_impute, window_size=5):
        daily_avg = self.build_daily_average_df(cleaned_df, enable_ml_impute)
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
                    row["mood_output"] = 1 if mood >= 5 else 0
                instance_rows.append(row)
        df_instances = pd.DataFrame(instance_rows)
        return df_instances

    def build_rnn_sequence_dataset(self, df_instances, enable_ml_impute, sequence_length=6):
        daily_avg = self.build_daily_average_df(df_instances, enable_ml_impute)
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
    def train_test_split_regression(df, stratify_col='screen_t', test_size=0.2):
        train_df, test_df = train_test_split(
                                    df,
                                    test_size=test_size,
                                    stratify=stratify_col,
                                    random_state=42
                                )
        return train_df, test_df
