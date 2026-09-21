import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fights = pd.read_csv("data/all_fights.csv")
print(fights.shape)
print(fights.describe)

print("===Info===")
print(fights.info())
print('===Red Winner===')
print(fights['red_winner'].value_counts())