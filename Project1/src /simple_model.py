import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

ufc_dataset_df = pd.read_csv( "Project1/data/all_fights.csv" )
cleaned_ufc_df = ufc_dataset_df.drop( columns=["red_return", "blue_return"] )

dfTrain, dfTest = train_test_split(
    ufc_dataset_df,
    test_size=0.15
)

knc = KNeighborsClassifier()

knc.fit(
    dfTrain['red_fighter'],
    dfTrain['blue_fighter']
)