#### Logistic Regression

For a baseline accuracy for our models to compare with, we will be comparing them to a simple logistic regression model.

Why did we choose logistic regression as our baseline for accuracy?
Since our data is split between \_diff features and other features, and the \_diff features are numeric, that is what logistic regression is built for. So for example, each unit of reach advantage shifts the odds of red winning by some amount.

Another reason for using logistic regression is it's fast to train and does not take a lot of setup.

The main disadvantage of using logistic regression is that it does not handle well when there is a relation between features; for example, maybe reach advantage has different effects based on what weight class the fight is in.

#### Random Forest

The reasons we chose random forest for one of our models are:

- Handles the mix of numeric \_diff and categorical features without needing scaling, and works well with on-hot encoded for categorical features.
- A major benefit of random forests is that they can capture non-linear relationships and interactions automatically, for example, tne takedown advantage matters a lot more when there is a larger weight difference.

#### XGBoost

The reasons we chose XGBoost for one of our models are:

- Gradient boosting is a very standard choice when it comes to mixed numeric categorical data, and also since our data set is not too big the training time for the models is not too long.
- Captures non-linear relationships and interactions like random forest, but often extracts more predictive power from the same features because each new tree learns from the previous ones.

Cons for XGBoost

- There are a lot of hyperparameters to tune, and if you don't tune them well the model can be prone to overfitting.
- Requires one-hot encoding for categorical features, which can cause the numbers of features to blow up quite quickly.

Why include it: it's the strongest general-purpose model for the type of data we have, and comparing it to random forest can tell us if the boosting error correction approach is helpful, over just averaging on the dataset.

#### CatBoost

The reasons we chose CatBoost for one of our models are:

- Handles categorical features natively, instead of one-hot encoding, this is very useful since we have a lot of categorical features.
- Built-in handling of categorical interactions like red_stance vs blue_stance, this has a big inpact on real fight analysis, since orthodox vs southpaw matchups are a known factor for predicting fight outcomes.

Cons for CatBoost

- Is quite similar to XGBoost and is a lot slower to train.

Why include it: this gives a good way to test whether the native category handling matters, and a good way to find that out is compare XGBoost vs CatBoost.

### XGBoost

### CatBoost
