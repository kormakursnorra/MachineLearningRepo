import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

ufc_dataset_df = pd.read_csv( "Project1/data/all_fights.csv" )
cleaned_ufc_df = ufc_dataset_df.drop( columns=["red_return", "blue_return"] )
ufc_df_sorted = cleaned_ufc_df.sort_values( "fight_date" )

ufc_df_sorted = ufc_df_sorted[~ufc_df_sorted["blue_stance"].isin(
    ["Switch ", "Open Stance", "Unknown"]
)]

ufc_df_sorted = ufc_df_sorted[~ufc_df_sorted["red_stance"].isin(
    ["Switch ", "Open Stance", "Unknown"]
)]

ufc_df_sorted = ufc_df_sorted[~ufc_df_sorted["weight_class"].isin(
    ["Catch Weight"]
)]


#Round_diff - check this one a bit better, only 1 super outlier

ufc_df_sorted = ufc_df_sorted[
    (ufc_df_sorted["reach_diff"] > -40) |
    (ufc_df_sorted["reach_diff"] < 40)
]

ufc_df_sorted = ufc_df_sorted[
    (ufc_df_sorted["rounds_diff"] > -100) |
    (ufc_df_sorted["rounds_diff"] < 100)
]

SPLIT_DATE = r"2023-06-01"

ufc_df_training = ufc_df_sorted.loc[ufc_df_sorted['fight_date'] <= SPLIT_DATE]
ufc_df_testing = ufc_df_sorted.loc[ufc_df_sorted['fight_date'] > SPLIT_DATE]




# dfTrain, dfTest = train_test_split(
#     y=ufc_df_sorted['red_win'],
#     stratify=ufc_df_sorted['fight_date']
# )

# diff_only_df = ufc_df_sorted.filter(regex=r'_diff')


# print(ufc_df_sorted['weight_class' == 'Catch Weight'].info())

# diff_only_df.info()

# x_train, x_test = train_test_split(
#     diff_only_df,
#     train_size=0.75
# )

# model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
# model.fit(x_train, y_train)
# p = model.predict_proba(x_test)[:, 1]