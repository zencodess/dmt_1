import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class DataCleaner:
    variable_accepted_ranges = {
        "mood": (1, 10),
        "circumplex.arousal": (-2, 2),
        "circumplex.valence": (-2, 2),
        "activity": (0, 1),
        "call": (0, 1),
        "sms": (0, 1)
    }
    duration_vars = [
        "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment",
        "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social",
        "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"
    ]

    locf_variables = ["mood", "circumplex.arousal", "circumplex.valence"]
    rolling_mean_variables = ["activity"] + duration_vars
    all_variables = [
        "mood",
        "circumplex.arousal",
        "circumplex.valence",
        "activity",
        "screen",
        "call",
        "sms",
        "appCat.builtin",
        "appCat.communication",
        "appCat.entertainment",
        "appCat.finance",
        "appCat.game",
        "appCat.office",
        "appCat.other",
        "appCat.social",
        "appCat.travel",
        "appCat.unknown",
        "appCat.utilities",
        "appCat.weather"
    ]
    SIZE_ROLLING_WINDOW = 5
    MIN_PERIODS_IN_WINDOW = 1


    @staticmethod
    def apply_variable_accepted_ranges(df):
        for var, (vmin, vmax) in DataCleaner.variable_accepted_ranges.items():
            if var in ["call", "sms"]:
                df.loc[(df["variable"] == var) & (~df["value"].isin([vmin, vmax])), "value"] = pd.NA
            else:
                df.loc[(df["variable"] == var) & ((df["value"] < vmin) | (df["value"] > vmax)), "value"] = pd.NA
        for var in DataCleaner.duration_vars:
            df.loc[(df["variable"] == var) & (df["value"] < 0), "value"] = pd.NA
        return df

    @staticmethod
    def impute_variable(var_group):
        var = var_group["variable"].iloc[0]
        if var in DataCleaner.locf_variables:
            var_group["value"] = var_group["value"].ffill()
        elif var in DataCleaner.rolling_mean_variables:
            var_group["value"] = var_group["value"].rolling(
                                    window=DataCleaner.SIZE_ROLLING_WINDOW,
                                    min_periods=DataCleaner.MIN_PERIODS_IN_WINDOW,
                                    center=True).mean()
        return var_group

    @staticmethod
    def fill_remaining_na(group, global_means):
        var = group["variable"].iloc[0]
        group["value"] = group["value"].fillna(global_means[var])
        return group

    @classmethod
    def clean_and_impute(cls, df):
        df = cls.apply_variable_accepted_ranges(df)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by=["id", "variable", "time"]).reset_index(drop=True)

        df_imputed = df.groupby(["id", "variable"], group_keys=False).apply(cls.impute_variable)

        global_means = df_imputed.groupby("variable")["value"].mean()
        clean_df = df_imputed.groupby("variable", group_keys=False).apply(lambda g: cls.fill_remaining_na(g, global_means))

        clean_df = clean_df.dropna()
        return clean_df

    @staticmethod
    def fill_null_vars_with_zero(df):
        for col in df.columns:
            if col not in ['id', 'date', 'mood_output']:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)

    @staticmethod
    def fill_null_values_with_median(df, cols_to_fill=None):
        if cols_to_fill is None:
            cols_to_fill = list(set(df.columns) - set(["id", "date", "mood_output", "mood"]))
        # fill with median values
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df

    @staticmethod
    def advanced_impute_missing_with_ml(df, columns=None):
        if columns is None:
            print(df.columns, 'df cols')
            # columns = df.columns - set(["id", "mood_output", "mood", "date"])
            columns = list(set(df.columns) - {"id", "mood_output", "mood", "date"})
        imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42, max_iter=10)
        df[columns] = imputer.fit_transform(df[columns])
        return df