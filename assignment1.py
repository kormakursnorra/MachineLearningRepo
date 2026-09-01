import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tit = pd.read_csv("titanic.csv")

'''sns.barplot(data=tit, x="who", y="survived", order=["man", "woman", "child"])
plt.title("Survival Rate by Category")
plt.ylabel("Survival Rate")
plt.show()
'''
sns.barplot(data=tit, x="who", y="survived", hue="pclass",
            order=["man", "woman", "child"])
plt.title("Survival Rate by Category and Class")
plt.ylabel("Survival Rate")
plt.show()
