#Imports
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

## == Load Data == 

csv_path = Path.cwd().parent / "data" / "all_fights.csv"
data_frame = pd.read_csv(csv_path)

data_frame = pd.read_csv(csv_path)
data_frame = data_frame[~data_frame["blue_stance"].isin(["Switch ", "Open Stance", "Unknown"])]
data_frame = data_frame[~data_frame["red_stance"].isin(["Switch ", "Open Stance", "Unknown"])]

# Drop Catch Weight from weight_class (72 entries)
data_frame = data_frame[~data_frame["weight_class"].isin(["Catch Weight"])]

# Drop reach diff over 40 (1 entry)
data_frame = data_frame[(data_frame["reach_diff"] > -40) & (data_frame["reach_diff"] < 40)]

# Drop round_diff outlier (1 entry)
data_frame = data_frame[(data_frame["rounds_diff"] > -100) & (data_frame["rounds_diff"] < 100)]

# Convert red_winner from string to bool
data_frame["red_winner"] = (data_frame["red_winner"].astype(str).str.lower() == "t").astype(int)
# Convert title bout from string to bool'
data_frame["title_bout"] = (data_frame["title_bout"].astype(str).str.lower() == "t").astype(int)

data_frame["fight_date"] = pd.to_datetime(data_frame["fight_date"])
data_frame = data_frame.sort_values("fight_date", kind="stable").reset_index(drop=True)

# == Build Features ==
SEED = 42
DATE = "2023-06-01"
SELECTION_METRIC = "roc_auc"

CAT_FEATURES = [
    "gender", "weight_class", "red_stance", "blue_stance"
]


extra_features = CAT_FEATURES + [
    "red_age", "blue_age", "b_match_wc_rank", "r_match_wc_rank"
]

# Build feature dataframe with all extras + all diffs (including odds)
all_features_df = pd.concat([data_frame[extra_features], data_frame.filter(regex="_diff")], axis=1)

# Age gap is more/less impactful depending on where it sits. A 25 v 28 age gap is less significant 
# than it being 32 v 35 and raw age_diff doesn't see that.
all_features_df["avg_age"] = (all_features_df.red_age + all_features_df.blue_age) / 2
all_features_df["avg_age_c"] = all_features_df["avg_age"] - all_features_df["avg_age"].mean()  # center it

# Create a new column that captures the interaction: how much age_diff's
# effect should be amplified or dampened based on how old the pair is on average
all_features_df["age_diff_x_avg"] = all_features_df["age_diff"] * all_features_df["avg_age_c"]

# Drop raw ages since we already have the affect
all_features_df = all_features_df.drop(columns=["red_age", "blue_age"])

# Replace instances of rank 20 (unranked) to 16 so ranking distribution is uniform 
worst_rank = max(all_features_df[["b_match_wc_rank", "r_match_wc_rank"]].max().max(), 16) + 1
all_features_df["b_match_wc_rank"] = all_features_df["b_match_wc_rank"].replace(worst_rank)
all_features_df["r_match_wc_rank"] = all_features_df["r_match_wc_rank"].replace(worst_rank)

## == Benchmarking ==

# Target
y = data_frame["red_winner"]

# Temporal Split
train_mask, test_mask = data_frame['fight_date'] <= DATE, data_frame['fight_date'] > DATE

data_frame_test = data_frame[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

cv = TimeSeriesSplit(n_splits=5)
scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "neg_log_loss": "neg_log_loss"}

majority = max(y_test.mean(), 1 - y_test.mean())

# The lower number is the favorite (-250 beats +215, -150 beats -110). 
# odds_diff = red_odds - blue_odds, so < 0 means red is favored. 
# Whhen equal odds, default to red, the majority class.
red_fav = (data_frame_test["red_odds"] <= data_frame_test["blue_odds"]).astype(int)
n_ties = int((data_frame_test["red_odds"] == data_frame_test["blue_odds"]).sum())
favorite = accuracy_score(y_test, red_fav)

# AUC / log loss for the favorite benchmark: convert odds to implied
# probabilities and remove the bookmaker margin (normalise to sum 1).
def implied(o):
    o = o.astype(float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))

p_red, p_blue = implied(data_frame_test["red_odds"]), implied(data_frame_test["blue_odds"])
p = p_red / (p_red + p_blue)

results = [
    {"model": "Majority class (always red)", "features": "-",
        "test_acc": majority, "test_auc": 0.5,
        "test_logloss": log_loss(y_test, np.full(len(y_test), y_test.mean()))},
    {"model": "Betting favorite", "features": "odds only",
        "test_acc": favorite, "test_auc": roc_auc_score(y_test, p),
        "test_logloss": log_loss(y_test, p)},
]
print(f"Pick'em fights in test set (equal odds, assigned to red): {n_ties}")

# Split into with/without odds
feature_sets = {
    "with odds": list(all_features_df.columns),
    "no odds": [c for c in all_features_df.columns if c != "odds_diff"],
}

cols_with_odds = [c for c in all_features_df.columns]
cols_no_odds = [c for c in cols_with_odds if c != 'odds_diff']

for featureset_name, columns, in feature_sets.items():
    # Collect every non-categorical feature (continuous) into an array
    num_features = [c for c in all_features_df.columns if c not in CAT_FEATURES]
    X_train = all_features_df.loc[train_mask, columns]
    X_test = all_features_df.loc[test_mask, columns]

    ## == Build Models ==

    # --- Preprocessors ---
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

    preprocessor_scaled = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CAT_FEATURES)
    ])

    # --- Param grids ---
    param_grid_lr = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"]
    }

    param_grid_rf = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [5, 10, 20, None],
        "clf__min_samples_leaf": [5, 20, 45],
        "clf__max_features": ["sqrt", "log2", 0.5]
    }

    param_grid_xgb = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [2, 3, 5],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0]
    }

    param_grid_cb = {
        "clf__iterations": [200, 500],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__depth": [4, 6],
        "clf__l2_leaf_reg": [3, 10]
    }

    param_grid_knn = {
        "clf__n_neighbors": [5, 11, 21, 31],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    }

    all_models = {
        # --- Pipelines ---
        "Logistic Regression" : (
            Pipeline([
                ("preprocess", preprocessor_scaled),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
            ]),
            param_grid_lr
        ),

        "Random Forrest" : (
            Pipeline([
                ("preprocess", preprocessor),
                ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
            ]),
            param_grid_rf
        ),

        "XGBClassifier" : (
            Pipeline([
                ("preprocess", preprocessor),
                ("clf", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1))
            ]),
            param_grid_xgb
        ),

        "CatBoost": (
            Pipeline([
                ("clf", CatBoostClassifier(random_seed=SEED,
                                           verbose=0,
                                           thread_count=-1,
                                           allow_writing_files=False))
            ]),
            param_grid_cb,
        ),

        "KNeghborsClassifier" : (
            Pipeline([
                ("preprocess", preprocessor_scaled),
                ("clf", KNeighborsClassifier())
            ]),
            param_grid_knn
        )
    }

    for model, (pipe, param) in all_models.items():
        t0 = time.time()
        grid_search = GridSearchCV(pipe, param, cv=cv, scoring=scoring, 
                                   refit=SELECTION_METRIC, n_jobs=-1, verbose=1,
        )

        fit_params = {}
        if model == "CatBoost":
            fit_params = ({"clf__cat_features": CAT_FEATURES})
        
        grid_search.fit(X_train, y_train, **fit_params)
        elapsed = time.time() - t0

        cvr = pd.DataFrame(grid_search.cv_results_)
        trial_cols = ["params", "mean_test_roc_auc", "std_test_roc_auc",
                        "mean_test_accuracy", "mean_test_neg_log_loss"]
        print(f"\n {model} ({featureset_name}) : {len(cvr)} configs, "f"{elapsed:.1f}s ")
        
        print(cvr[trial_cols].sort_values("mean_test_roc_auc", ascending=False).to_string(index=False))

        # Test set is used once, only for the CV-selected configuration.
        best = grid_search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        results.append({
            "model": model, "features": featureset_name,
            "cv_auc": grid_search.best_score_,
            "cv_acc": cvr.loc[grid_search.best_index_, "mean_test_accuracy"],
            "test_acc": accuracy_score(y_test, (proba >= 0.5).astype(int)),
            "test_auc": roc_auc_score(y_test, proba),
            "test_logloss": log_loss(y_test, proba),
            "best_params": grid_search.best_params_,
            "fit_seconds": round(elapsed, 1),
        })

summary = pd.DataFrame(results)
pd.set_option("display.width", 200)
print("\n Summary (models selected by CV ROC-AUC, then scored once on test) ")
print(summary.drop(columns=["best_params"]).round(4).to_string(index=False))
summary.to_csv("results_summary.csv", index=False)