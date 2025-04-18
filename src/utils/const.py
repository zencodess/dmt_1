# feature builder
EXP_ML_IMPUTE = 'EXP_ML_IMPUTE'
ML_IMPUTE = 'ML_IMPUTE'
MEDIAN_IMPUTE = 'MEDIAN_IMPUTE'
ZERO_IMPUTE = 'ZERO_IMPUTE'
INTERPOLATE_IMPUTE = 'INTERPOLATE_IMPUTE'
LOCF_ROLLING_MEAN_IMPUTE = 'LOCF_ROLLING_MEAN_IMPUTE'
IMPUTE_OPTIONS = [EXP_ML_IMPUTE, ML_IMPUTE, ZERO_IMPUTE, INTERPOLATE_IMPUTE, MEDIAN_IMPUTE, LOCF_ROLLING_MEAN_IMPUTE]

CATEGORICAL_VARS =  ["call", "sms"]
NUMERICAL_VARS = ["circumplex.arousal", "circumplex.valence", "activity"]

# impute strategies
RBF_BAYESIAN_RIDGE = "RBF+BayesianRidge"
BAYESIAN_RIDGE = "BayesianRidge"
RF_REGRESSOR = "RandomForest"
GRADIENT_BOOSTING_REGRESSOR = "GradientBoostingRegressor"
KNN_REGRESSOR = "KNeighborsRegressor"

# cleaner
VARIABLE_ACCEPTED_RANGES = {
                                "mood": (1, 10),
                                "circumplex.arousal": (-2, 2),
                                "circumplex.valence": (-2, 2),
                                "activity": (0, 1),
                                "call": (0, 1),
                                "sms": (0, 1)
                            }
DURATION_VARS = [
        "screen", "appCat.builtin", "appCat.communication", "appCat.entertainment",
        "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social",
        "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"
    ]
LOCF_VARS = ["mood", "circumplex.arousal", "circumplex.valence"]
ROLLING_MEAN_VARS = ["activity"] + DURATION_VARS
ALL_VARS = [
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

# rnn model
BEST_RNN_MODEL = "best_rnn_model.pth"