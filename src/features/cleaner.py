import pandas as pd
import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsRegressor

from src.utils.const import VARIABLE_ACCEPTED_RANGES, DURATION_VARS, LOCF_VARS, ALL_VARS, CATEGORICAL_VARS, ROLLING_MEAN_VARS, \
    RBF_BAYESIAN_RIDGE, BAYESIAN_RIDGE, KNN_REGRESSOR, GRADIENT_BOOSTING_REGRESSOR, RF_REGRESSOR


class DataCleaner:
    variable_accepted_ranges = VARIABLE_ACCEPTED_RANGES
    duration_vars = DURATION_VARS
    locf_variables = LOCF_VARS
    rolling_mean_variables = ROLLING_MEAN_VARS
    all_variables = ALL_VARS
    SIZE_ROLLING_WINDOW = 5
    MIN_PERIODS_IN_WINDOW = 1

    @staticmethod
    def apply_variable_accepted_ranges(df):
        for var, (vmin, vmax) in DataCleaner.variable_accepted_ranges.items():
            if var in CATEGORICAL_VARS:
                df.loc[(df["variable"] == var) & (~df["value"].isin([vmin, vmax])), "value"] = pd.NA
            else:
                df.loc[(df["variable"] == var) & ((df["value"] < vmin) | (df["value"] > vmax)), "value"] = pd.NA
        for var in DataCleaner.duration_vars:
            df.loc[(df["variable"] == var) & (df["value"] < 0), "value"] = pd.NA
        return df

    @staticmethod
    def simple_impute_variable(var_group):
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

    @staticmethod
    def locf_rolling_mean_impute_dataframe(df):
        df_imputed = df.groupby(["id", "variable"], group_keys=False).apply(DataCleaner.simple_impute_variable)
        global_means = df_imputed.groupby("variable")["value"].mean()
        clean_df = df_imputed.groupby("variable", group_keys=False).apply(lambda g: DataCleaner.fill_remaining_na(g, global_means))
        clean_df = clean_df.dropna()
        return clean_df

    @classmethod
    def clean_data_pipe(cls, df, scale_and_transform=False):
        df = cls.apply_variable_accepted_ranges(df)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by=["id", "variable", "time"]).reset_index(drop=True)
        if scale_and_transform:
            clean_df = cls.log_transform_duration_columns(df)
            clean_df = cls.scale_arousal_valence(clean_df)
        else:
            clean_df = df
        return clean_df

    @staticmethod
    def fill_null_vars_with_zero(df):
        for col in df.columns:
            if col not in ['id', 'date', 'mood_output']:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
        return df

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
    def advanced_impute_missing_with_ml(df, strategy=RBF_BAYESIAN_RIDGE, columns=None):
        if columns is None:
            columns = list(set(df.columns) - {"id", "mood_output", "mood", "date"})
        print(f"Imputing missing values with ML - {strategy} option..")
        if strategy == RBF_BAYESIAN_RIDGE: # used for temporal rnn model
            imputer = IterativeImputer(estimator=make_pipeline(Nystroem(), BayesianRidge()), random_state=42, verbose=1)
        elif strategy == RF_REGRESSOR:
            imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42, max_iter=25)
        df[columns] = imputer.fit_transform(df[columns])
        return df

    @staticmethod
    def advanced_impute_linear_interpolator(df, columns=None):
        if columns is None:
            columns = list(set(df.columns) - {"id", "mood_output", "mood", "date"})
        for col in columns:
            df[col] = df[col].interpolate(method='cubicspline')
        return df

    @staticmethod
    def ml_impute_experiments(df, columns=None):
        if columns is None:
            columns = list(set(df.columns) - {"id", "mood_output", "mood", "date"})

        strategies = {
            BAYESIAN_RIDGE: make_pipeline(StandardScaler(), BayesianRidge()),
            GRADIENT_BOOSTING_REGRESSOR: HistGradientBoostingRegressor(),
            RF_REGRESSOR: RandomForestRegressor(),
            KNN_REGRESSOR: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
            RBF_BAYESIAN_RIDGE: make_pipeline(Nystroem(), BayesianRidge())
        }

        results = {}
        for name, estimator in strategies.items():
            print(f"\nRunning imputation with: {name}")
            imputer = IterativeImputer(estimator=estimator, random_state=42, verbose=1)
            df_copy = df.copy()
            try:
                df_copy[columns] = imputer.fit_transform(df_copy[columns])
                results[name] = df_copy
                print(f"Imputation with {name} complete.")
            except Exception as e:
                print(f"Imputation with {name} failed: {e}")
        return results

    @classmethod
    def log_transform_duration_columns(cls, df, cols=None):
        if cols is None:
            cols = cls.duration_vars
        for col in cols:
            if col in df.columns:
                print(f"Log transform duration column: {col}")
                df[col] = df[col].apply(lambda x: pd.NA if pd.isna(x) else (0 if x <= 0 else np.log1p(x)))
        return df

    @classmethod
    def scale_arousal_valence(cls, df):
        for col in ["circumplex.arousal", "circumplex.valence"]:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std
        return df