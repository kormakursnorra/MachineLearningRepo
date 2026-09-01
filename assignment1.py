import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tit = pd.read_csv("titanic.csv")

sns.histplot(data=tit, x="age", hue="alive", bins=30)
plt.title("Age distribution by survival")
plt.show()