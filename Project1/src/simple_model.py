#Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

#Target
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

rows = [
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

    all_models = {
        # --- Pipelines ---
        "Logistic Regression" : (
            Pipeline([
                ("preprocess", preprocessor_scaled),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
            ])
        ), 

        "XGBClassifier" : (
            Pipeline([
                ("preprocess", preprocessor),
                ("clf", XGBClassifier(eval_metric="logloss", random_state=SEED))
            ])
        ),

        "Random Forrest" : (
            Pipeline([
                ("preprocess", preprocessor),
                ("clf", RandomForestClassifier(random_state=SEED))
            ])
        ),

        "CatBoost": (
            Pipeline([
                ("clf", CatBoostClassifier(random_seed=SEED, 
                                           verbose=0,
                                           thread_count=-1,
                                           allow_writing_files=False))
            ]),
            {"clf__iterations": [200, 500],
             "clf__learning_rate": [0.03, 0.1],
             "clf__depth": [4, 6],
             "clf__l2_leaf_reg": [3, 10]},
        ),

        "KNeghborsClassifier" : (
            Pipeline([
                ("preprocess", preprocessor_scaled),
                ("clf", KNeighborsClassifier())
            ])
        )
    }





# Train/test with odds
X_train_odds = all_features_df.loc[train_mask, cols_with_odds]
X_test_odds = all_features_df.loc[test_mask, cols_with_odds]

#Train/test without odds
X_train_no_odds = all_features_df.loc[train_mask, cols_no_odds]
X_test_no_odds = all_features_df.loc[test_mask, cols_no_odds]

## == Build Models ==

# --- Preprocessors ---
pre_tree = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

preprocessor_scaled = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_FEATURES)
])

# --- Pipelines ---
pipeline_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42))
])

pipeline_lr = Pipeline([
    ("preprocess", preprocessor_scaled),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_rf = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

pipeline_knn = Pipeline([
    ("preprocess", preprocessor_scaled),
    ("clf", KNeighborsClassifier())
])


# Odds included
X_test_dummy_odds = pd.get_dummies(X_test_odds, columns=CAT_FEATURES)
X_train_dummy_odds = pd.get_dummies(X_train_odds, columns=CAT_FEATURES)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_dummy_odds, y_train)
preds_odds = lr.predict(X_test_dummy_odds)

print('=== With Odds Diff ===')
print(accuracy_score(y_test, preds_odds))
print(classification_report(y_test, preds_odds))

# Odds not included
X_test_dummy_no_odds = pd.get_dummies(X_test_no_odds, columns=CAT_FEATURES)
X_train_dummy_no_odds = pd.get_dummies(X_train_no_odds, columns=CAT_FEATURES)
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_dummy_no_odds, y_train)
preds_no_odds = lr.predict(X_test_dummy_no_odds)

print('=== Without Odds Diff ===')
print(accuracy_score(y_test, preds_no_odds))
print(classification_report(y_test, preds_no_odds))

# your categorical columns (everything that's object/category dtype)


X_train_cat = X_train_odds.copy()
X_test_cat = X_test_odds.copy()

# CatBoost needs cats as string, and no NaNs in cat columns (fill or drop first)
for col in CAT_FEATURES:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

train_pool = Pool(X_train_cat, y_train, columns=CAT_FEATURES)
test_pool = Pool(X_test_cat, y_test, columns=CAT_FEATURES)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="Accuracy",
    random_seed=26,
    verbose=100  # prints progress every 100 iterations
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

preds = model.predict(X_test_cat)
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# feature importance
importances = model.get_feature_importance(train_pool, prettified=True)
print(importances)

X_train_cat_no_odds = X_train_no_odds.copy()
X_test_cat_no_odds = X_test_no_odds.copy()

# CatBoost needs cats as string, and no NaNs in cat columns (fill or drop first)
for col in CAT_FEATURES:
    X_train_cat_no_odds[col] = X_train_cat_no_odds[col].astype(str)
    X_test_cat_no_odds[col] = X_test_cat_no_odds[col].astype(str)

train_pool = Pool(X_train_cat_no_odds, y_train, CAT_FEATURES=CAT_FEATURES)
test_pool = Pool(X_test_cat_no_odds, y_test, CAT_FEATURES=CAT_FEATURES)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="Accuracy",
    random_seed=26,
    verbose=100  # prints progress every 100 iterations
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

preds_no_odds = model.predict(X_test_cat_no_odds)
print(accuracy_score(y_test, preds_no_odds))
print(classification_report(y_test, preds_no_odds))
# feature importance
importances = model.get_feature_importance(train_pool, prettified=True)
print(importances)

# Random Forest
X_train_forest = X_train_odds.copy()
X_test_forest = X_test_odds.copy()

X_train_dummy_odds = pd.get_dummies(X_train_forest, columns=CAT_FEATURES)
X_test_dummy_odds = pd.get_dummies(X_test_forest, columns=CAT_FEATURES)

param_grid = {
    'n_estimators' : [100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_leaf': [1, 5, 10, 20]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=26),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_train_dummy_odds, y_train)

print('=== Best params ===')
print(grid_search.best_params_)
print('=== Best score ===')
print(grid_search.best_score_)

best_rf_odds = grid_search.best_estimator_
preds = best_rf_odds.predict(X_test_dummy_odds)
print('=== Scores ===')
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

print('=== With Odds ===')
importances = pd.Series(best_rf_odds.feature_importances_, index=X_train_dummy_odds.columns)
print(importances.sort_values(ascending=False).head(10))

importances = pd.Series(best_rf_odds.feature_importances_, index=X_train_dummy_odds.columns)
top10 = importances.sort_values(ascending=True).tail(10)


X_train_forest_no_odds = X_train_no_odds.copy()
X_test_forest_no_odds = X_test_no_odds.copy()

X_train_dummy_no_odds = pd.get_dummies(X_train_forest_no_odds, columns=['gender', 'weight_class', 'red_stance', 'blue_stance'])
X_test_dummy_no_odds = pd.get_dummies(X_test_forest_no_odds, columns=['gender', 'weight_class', 'red_stance', 'blue_stance'])

param_grid = {
    'n_estimators' : [100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_leaf': [1, 5, 10, 20]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=26),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_train_dummy_no_odds, y_train)

print('=== Best params ===')
print(grid_search.best_params_)
print('=== Best score ===')
print(grid_search.best_score_)

best_rf_no_odds = grid_search.best_estimator_
preds = best_rf_no_odds.predict(X_test_dummy_no_odds)
print('=== Scores ===')
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

print('=== Importances ===')
importances = pd.Series(best_rf_no_odds.feature_importances_, index=X_train_dummy_no_odds.columns)
print(importances.sort_values(ascending=False).head(10))

importances = pd.Series(best_rf_no_odds.feature_importances_, index=X_train_dummy_no_odds.columns)
top10 = importances.sort_values(ascending=True).tail(10)


numeric_features = [
    "odds_diff", "age_diff", "td_diff", "sig_str_diff", "reach_diff", "losses_diff",
    "sub_att_diff", "rank_diff",
    "win_streak_diff", "longest_win_streak_diff", "lose_streak_diff",
    "title_bout_diff", "wins_diff", "rounds_diff", "height_diff",
    "ko_diff", "submission_diff", "b_match_wc_rank", "r_match_wc_rank", 
    "avg_age", "avg_age_c", "age_diff_x_avg"
]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_FEATURES)
])

pipeline_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("clf", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

param_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.01, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0]
}

grid_search_odds = GridSearchCV(
    pipeline_xgb,
    param_grid,
    cv=5,
    scoring="roc_auc",   # accuracy alone is weak if red_winner is imbalanced — see note below
    n_jobs=-1,
    return_train_score=True
)

grid_search_odds.fit(X_train_odds, y_train)

print("Best parameters:")
print(grid_search_odds.best_params_)
print()
print("Best cross-validation ROC-AUC:")
print(grid_search_odds.best_score_)

best_index = grid_search_odds.best_index_
print("Training score:", grid_search_odds.cv_results_["mean_train_score"][best_index])
print("Validation score:", grid_search_odds.cv_results_["mean_test_score"][best_index])


best_xgb_odds = grid_search_odds.best_estimator_
preds = best_xgb_odds.predict(X_test_odds)
print('=== Test Set ===')
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))


best_model = grid_search_odds.best_estimator_
xgb_clf = best_model.named_steps["clf"]

# Get feature names after one-hot encoding
feature_names = best_model.named_steps["preprocess"].get_feature_names_out()

importances = xgb_clf.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)




numeric_features = [
    "age_diff", "td_diff", "sig_str_diff", "reach_diff", "losses_diff",
    "sub_att_diff", "rank_diff",
    "win_streak_diff", "longest_win_streak_diff", "lose_streak_diff",
    "title_bout_diff", "wins_diff", "rounds_diff", "height_diff",
    "ko_diff", "submission_diff", "b_match_wc_rank", "r_match_wc_rank", 
    "avg_age", "avg_age_c", "age_diff_x_avg"
]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_FEATURES)
])

pipeline_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("clf", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
])

param_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.01, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0]
}

grid_search_no_odds = GridSearchCV(
    pipeline_xgb,
    param_grid,
    cv=5,
    scoring="roc_auc",   # accuracy alone is weak if red_winner is imbalanced — see note below
    n_jobs=-1,
    return_train_score=True
)

grid_search_no_odds.fit(X_train_no_odds, y_train)

print("Best parameters:")
print(grid_search_no_odds.best_params_)
print()
print("Best cross-validation ROC-AUC:")
print(grid_search_no_odds.best_score_)

best_index = grid_search_no_odds.best_index_
print("Training score:", grid_search_no_odds.cv_results_["mean_train_score"][best_index])
print("Validation score:", grid_search_no_odds.cv_results_["mean_test_score"][best_index])

best_xgb_no_odds = grid_search_no_odds.best_estimator_
preds = best_xgb_no_odds.predict(X_test_no_odds)
print('=== Test Set ===')
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))


best_model = grid_search_no_odds.best_estimator_
xgb_clf = best_model.named_steps["clf"]

# Get feature names after one-hot encoding
feature_names = best_model.named_steps["preprocess"].get_feature_names_out()

importances = xgb_clf.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

pre_tree = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

params = {
    "clf__n_estimators": [100, 400],
    "clf__max_depth": [4, 12],
    "clf__min_samples_leaf": [15, 45],
    "clf__bootstrap": [True, False],
}


randforrest = Pipeline([
    ("pre", pre_tree),
    ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))
])

grid_search = GridSearchCV(
    estimator=randforrest,
    param_grid=params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train_odds, y_train)

best_model_odds = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_)

proba = best_model_odds.predict_proba(X_test_odds)[:, 1]
preds = best_model_odds.predict(X_test_odds)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


print("Accuracy Score: ", accuracy_score(y_test, preds))

print("Classification Report \n")
print(classification_report(y_test, preds))

encoded_features = best_model_odds.named_steps["pre"].named_transformers_["cat"] 
encoded_names = list(encoded_features.get_feature_names_out(CAT_FEATURES))
features_names = num_features + encoded_names

importances = (best_model_odds.named_steps["clf"].feature_importances_) * 100
feature_imp_df = pd.DataFrame({"Feature" : features_names, "Importance" : importances}).sort_values("Importance", ascending=False)

print(feature_imp_df)

test_auc = round(roc_auc_score(y_test, proba), 3)
print("Test AUC score: ", test_auc)
print("Log-Loss score: ", log_loss(y_test, proba))
print("ROC AUC score: ", roc_auc_score((y_test == True), proba))

conf_matrix = confusion_matrix(y_test, preds, labels=[False, True])

pre_tree = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

params = {
    "clf__n_estimators": [100, 400],
    "clf__max_depth": [4, 12],
    "clf__min_samples_leaf": [15, 45],
    "clf__bootstrap": [True, False],
}


randforrest = Pipeline([
    ("pre", pre_tree),
    ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))
])

grid_search = GridSearchCV(
    estimator=randforrest,
    param_grid=params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train_odds, y_train)

best_model_odds = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_)

proba = best_model_odds.predict_proba(X_test_odds)[:, 1]
preds = best_model_odds.predict(X_test_odds)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


print("Accuracy Score: ", accuracy_score(y_test, preds))

print("Classification Report \n")
print(classification_report(y_test, preds))

encoded_features = best_model_odds.named_steps["pre"].named_transformers_["cat"] 
encoded_names = list(encoded_features.get_feature_names_out(CAT_FEATURES))
features_names = num_features + encoded_names

importances = (best_model_odds.named_steps["clf"].feature_importances_) * 100
feature_imp_df = pd.DataFrame({"Feature" : features_names, "Importance" : importances}).sort_values("Importance", ascending=False)

print(feature_imp_df)

test_auc = round(roc_auc_score(y_test, proba), 3)
print("Test AUC score: ", test_auc)
print("Log-Loss score: ", log_loss(y_test, proba))
print("ROC AUC score: ", roc_auc_score((y_test == True), proba))

conf_matrix = confusion_matrix(y_test, preds, labels=[False, True])

pre_tree = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

params = {
    "clf__n_estimators": [100, 400],
    "clf__max_depth": [4, 12],
    "clf__min_samples_leaf": [15, 45],
    "clf__bootstrap": [True, False],
}


randforrest = Pipeline([
    ("pre", pre_tree),
    ("clf", RandomForestClassifier(n_jobs=-1, random_state=42))
])

grid_search = GridSearchCV(
    estimator=randforrest,
    param_grid=params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train_no_odds, y_train)

best_model_no_odds = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_)

proba = best_model_no_odds.predict_proba(X_test_no_odds)[:, 1]
preds = best_model_no_odds.predict(X_test_no_odds)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


print("Accuracy Score: ", accuracy_score(y_test, preds))

print("Classification Report \n")
print(classification_report(y_test, preds))

encoded_features = best_model_no_odds.named_steps["pre"].named_transformers_["cat"] 
encoded_names = list(encoded_features.get_feature_names_out(CAT_FEATURES))
features_names = num_features + encoded_names

importances = (best_model_no_odds.named_steps["clf"].feature_importances_) * 100
feature_imp_df = pd.DataFrame({"Feature" : features_names, "Importance" : importances}).sort_values("Importance", ascending=False)

print(feature_imp_df)


test_auc = round(roc_auc_score(y_test, proba), 3)
print("Test AUC score: ", test_auc)
print("Log-Loss score: ", log_loss(y_test, proba))
print("ROC AUC score: ", roc_auc_score((y_test == True), proba))

conf_matrix = confusion_matrix(y_test, preds, labels=[False, True])


x_train = X_train_odds


# --- Feature groups ---
numeric_features = [
    "odds_diff", "age_diff", "td_diff", "sig_str_diff", "reach_diff", "losses_diff",
    "sub_att_diff", "rank_diff", "win_streak_diff", "longest_win_streak_diff",
    "lose_streak_diff", "title_bout_diff", "wins_diff", "rounds_diff",
    "height_diff", "ko_diff", "submission_diff"
]

# --- Preprocessors ---
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_FEATURES)
])

preprocessor_scaled = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), CAT_FEATURES)
])

# --- Pipelines ---
pipeline_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42))
])

pipeline_lr = Pipeline([
    ("preprocess", preprocessor_scaled),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_rf = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

pipeline_knn = Pipeline([
    ("preprocess", preprocessor_scaled),
    ("clf", KNeighborsClassifier())
])

# --- Param grids ---
param_grid_xgb = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.01, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0]
}

param_grid_lr = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"]
}

param_grid_rf = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 5],
    "clf__max_features": ["sqrt", "log2"]
}

param_grid_knn = {
    "clf__n_neighbors": [5, 11, 21, 31],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2]
}

# --- Grid searches ---
grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5,
                                scoring="roc_auc", n_jobs=-1, return_train_score=True)
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5,
                               scoring="roc_auc", n_jobs=-1, return_train_score=True)
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5,
                               scoring="roc_auc", n_jobs=-1, return_train_score=True)
grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5,
                                scoring="roc_auc", n_jobs=-1, return_train_score=True)

grid_search_xgb.fit(x_train, y_train)
print("XGBoost done —", grid_search_xgb.best_params_, grid_search_xgb.best_score_)

grid_search_lr.fit(x_train, y_train)
print("Logistic Regression done —", grid_search_lr.best_params_, grid_search_lr.best_score_)

grid_search_rf.fit(x_train, y_train)
print("Random Forest done —", grid_search_rf.best_params_, grid_search_rf.best_score_)

grid_search_knn.fit(x_train, y_train)
print("kNN done —", grid_search_knn.best_params_, grid_search_knn.best_score_)

# --- Summary table ---
results_summary = pd.DataFrame({
    "model": ["XGBoost", "Logistic Regression", "Random Forest", "kNN"],
    "best_cv_roc_auc": [
        grid_search_xgb.best_score_,
        grid_search_lr.best_score_,
        grid_search_rf.best_score_,
        grid_search_knn.best_score_
    ]
}).sort_values("best_cv_roc_auc", ascending=False)

print()
print(results_summary)



# --- Feature groups ---

numeric_features = [
    "odds_diff", "age_diff", "td_diff", "sig_str_diff", "reach_diff", "losses_diff",
    "sub_att_diff", "rank_diff", "win_streak_diff", "longest_win_streak_diff",
    "lose_streak_diff", "title_bout_diff", "wins_diff", "rounds_diff",
    "height_diff", "ko_diff", "submission_diff"
]

# --- Preprocessing: impute only — CatBoost handles cats and scale itself ---
preprocessor = ColumnTransformer([
    ("cat", SimpleImputer(strategy="constant", fill_value="missing"), CAT_FEATURES),
    ("num", SimpleImputer(strategy="median"), numeric_features)
], verbose_feature_names_out=False).set_output(transform="pandas")

# force categorical columns back to string after imputation (SimpleImputer can upcast dtypes)
def cast_cats_to_str(df):
    df = df.copy()
    df[CAT_FEATURES] = df[CAT_FEATURES].astype(str)
    return df

cast_step = FunctionTransformer(cast_cats_to_str)

# --- Pipeline (no CAT_FEATURES in the constructor — passed via fit() instead) ---
pipeline_cb = Pipeline([
    ("preprocess", preprocessor),
    ("cast", cast_step),
    ("clf", CatBoostClassifier(
        random_seed=26,
        verbose=0
    ))
])

# --- Param grid ---
""" param_grid_cb = {
    "clf__iterations": [300, 500],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__depth": [4, 6, 8],
    "clf__l2_leaf_reg": [1, 3, 5]
} """

param_grid_cb = {
    "clf__iterations": [300, 500],
    "clf__learning_rate": [0.03, 0.05],
    "clf__depth": [4, 6],
    "clf__l2_leaf_reg": [1, 3]
}
# Hyper parameters for CatBoostClassifier


# --- Grid search ---
grid_search_cb = GridSearchCV(
    pipeline_cb, param_grid_cb,
    cv=5, scoring="roc_auc", n_jobs=-1, return_train_score=True
)


# CAT_FEATURES passed here via clf__ prefix, forwarded to CatBoostClassifier.fit() on every fold
grid_search_cb.fit(x_train, y_train, clf__CAT_FEATURES=CAT_FEATURES)

print("Best parameters:")
print(grid_search_cb.best_params_)
print()
print("Best cross-validation ROC-AUC:")
print(grid_search_cb.best_score_)

best_index = grid_search_cb.best_index_
print("Training score:", grid_search_cb.cv_results_["mean_train_score"][best_index])
print("Validation score:", grid_search_cb.cv_results_["mean_test_score"][best_index])


# Plot 1 Model comparison bar chart
results = pd.DataFrame({
    'Model': ['Baseline', 'Linear Regression', 'Random Forest', 'CatBoost', 'XGBoost'],
    'Without Odds': [0.58, 0.603, 0.607, 0.609, 0.605],
    'With Odds':    [0.58, 0.700, 0.688, 0.705, 0.679]
})
