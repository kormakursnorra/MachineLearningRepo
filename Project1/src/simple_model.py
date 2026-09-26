import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=FutureWarning)

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
new_worst_rank = 16
worst_rank = all_features_df[["b_match_wc_rank", "r_match_wc_rank"]].max().max()
all_features_df["b_match_wc_rank"] = all_features_df["b_match_wc_rank"].replace(to_replace=worst_rank, value=new_worst_rank)
all_features_df["r_match_wc_rank"] = all_features_df["r_match_wc_rank"].replace(to_replace=worst_rank ,value=new_worst_rank)

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

# Keep test-set probabilities so the Results section can plot ROC / calibration curves
test_probas = {"Betting favorite": p}

results = [
    {"model": "Majority class (always red)", "features": "-",
        "test_acc": majority, "test_auc": 0.5,
        "test_logloss": log_loss(y_test, np.full(len(y_test), y_test.mean()))},
    {"model": "Betting favorite", "features": "odds only",
        "test_acc": favorite, "test_auc": roc_auc_score(y_test, p),
        "test_logloss": log_loss(y_test, p)},
]
print(f"Coin Flip fights in test set (equal odds, assigned to red): {n_ties}")

# Split into with/without odds
feature_sets = {
    "with odds": list(all_features_df.columns),
    "no odds": [c for c in all_features_df.columns if c != "odds_diff"],
}

importance_rows = []

for featureset_name, columns, in feature_sets.items():
    # Collect every non-categorical feature (continuous) into an array
    num_features = [c for c in columns if c not in CAT_FEATURES]
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
        "clf__l1_ratio": [0, 1],
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

    all_models = {
        # --- Pipelines ---
        "Logistic Regression" : (
            Pipeline([
                ("preprocess", preprocessor_scaled),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED))
            ]),
            param_grid_lr
        ),

        "Random Forest" : (
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

        importance = permutation_importance(
            best, X_test, y_test,
            scoring=["roc_auc", "accuracy", "neg_log_loss"],
            n_repeats=10, random_state=SEED, n_jobs=-1,
        )

        for metric, res in importance.items():
            importance_rows.extend(
                {
                    "model": model,
                    "features": featureset_name,
                    "metric": metric,
                    "feature": feature,
                    "importance": mean,
                    "importance_std": std,
                }
                for feature, mean, std in zip(columns, res.importances_mean, res.importances_std)
            )

        proba = best.predict_proba(X_test)[:, 1]
        test_probas[f"{model} ({featureset_name})"] = proba
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

result_path = Path.cwd().parent / "data" / "results_summary.csv"

summary.to_csv(result_path, index=False)

imp = pd.DataFrame(importance_rows)
imp.to_csv(Path.cwd().parent / "data" / "permutation_importance.csv", index=False)

def plot_importance(imp, metric="roc_auc", features="no odds", top_n=15):
    """Horizontal bars per model, top_n features, error bars = std over repeats."""
    d = imp[(imp["metric"] == metric) & (imp["features"] == features)]
    models = d["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 0.35 * top_n + 1.5), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        dm = d[d["model"] == m].nlargest(top_n, "importance").iloc[::-1]  # biggest on top
        colors = np.where(dm["importance"] > 0, "tab:blue", "tab:red")
        ax.barh(dm["feature"], dm["importance"], xerr=dm["importance_std"], color=colors, capsize=2)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(m)
        ax.grid(axis="x", alpha=0.3)
    fig.supxlabel(f"Drop in test {metric} when feature is shuffled")
    fig.suptitle(f"Permutation importance ({features})")
    plt.tight_layout()
    plt.show()

plot_importance(imp, metric="roc_auc", features="no odds")
plot_importance(imp, metric="roc_auc", features="with odds")

# Heatmap: every feature vs every model, for one metric and feature set
def importance_heatmap(imp, metric="roc_auc", features="no odds"):
    d = imp[(imp["metric"] == metric) & (imp["features"] == features)]
    pivot = d.pivot_table(index="feature", columns="model", values="importance")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(pivot) + 1.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"Permutation importance, {metric} ({features})")
    plt.tight_layout()
    plt.show()

importance_heatmap(imp, "roc_auc", "no odds")

# Test-set AUC and accuracy per model, with and without odds.
# Dot plot rather than bars so the axis doesn't have to start at 0 to be honest.
models_df = summary[summary["features"].isin(["with odds", "no odds"])]
fav = summary.loc[summary["model"] == "Betting favorite"].iloc[0]
maj = summary.loc[summary["model"] == "Majority class (always red)"].iloc[0]
model_names = models_df["model"].unique()
ypos = np.arange(len(model_names))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, metric, label in zip(axes, ["test_auc", "test_acc"], ["Test ROC-AUC", "Test accuracy"]):
    for feats, color, offset in [("with odds", "tab:blue", -0.12), ("no odds", "tab:orange", 0.12)]:
        vals = models_df[models_df["features"] == feats].set_index("model").loc[model_names, metric]
        ax.scatter(vals, ypos + offset, color=color, s=60, label=feats, zorder=3)
    ax.axvline(fav[metric], color="gray", ls="--", label="Betting favorite")
    ax.axvline(maj[metric], color="gray", ls=":", label="Majority class")
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.3)
axes[0].set_yticks(ypos, model_names)
axes[0].invert_yaxis()
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.suptitle("Test performance by model")
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.show()


# ROC curves on the test set: the with-odds models against the bookmaker's implied probabilities
fig, ax = plt.subplots(figsize=(6, 6))
for name, proba in test_probas.items():
    if name == "Betting favorite" or "(with odds)" in name:
        fpr, tpr, _ = roc_curve(y_test, proba)
        style = dict(color="black", ls="--", lw=2) if name == "Betting favorite" else dict(lw=1.5)
        ax.plot(fpr, tpr, label=f"{name}  (AUC {roc_auc_score(y_test, proba):.3f})", **style)
ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (test set)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# Calibration: when a model says "red wins 70%", does red win ~70% of the time?
# Matters more than AUC if these probabilities are ever compared to betting lines.
fig, ax = plt.subplots(figsize=(6, 6))
for name in ["Betting favorite", "Logistic Regression (with odds)", "Logistic Regression (no odds)"]:
    frac_pos, mean_pred = calibration_curve(y_test, test_probas[name], n_bins=10, strategy="quantile")
    style = dict(color="black", ls="--") if name == "Betting favorite" else {}
    ax.plot(mean_pred, frac_pos, marker="o", label=name, **style)
ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="Perfect calibration")
ax.set_xlabel("Predicted P(red wins)")
ax.set_ylabel("Observed fraction red wins")
ax.set_title("Calibration (test set, 10 quantile bins)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()