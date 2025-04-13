import pandas as pd
from sklearn.model_selection import train_test_split


class FeatureMaker():
    def __init__(self, df=None):
        self.df = df
        self.categorical_variables = ["call", "sms"]
        self.numerical_variables = ["circumplex.arousal", "circumplex.valence"]
        self.duration_variables = ["screen", "appCat.builtin", "appCat.communication", "appCat.entertainment",
        "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social",
        "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"]
        self.categorical_features = []
        self.numerical_features = []
        self.duration_features = []

    def build_predictive_dataset_from_cleaned(self, cleaned_df, window_size=5):
        # creating one data row for one day
        cleaned_df["date"] = cleaned_df["time"].dt.date
        daily_avg = cleaned_df.groupby(["id", "date", "variable"])["value"].mean().unstack().reset_index()
        daily_avg = daily_avg.sort_values(by=["id", "date"]).reset_index(drop=True)
        print('daily_avg', daily_avg.head())
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

    def train_test_split(self, df, test_size=0.2, val_size=0.1):
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
