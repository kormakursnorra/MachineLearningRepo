# Reasons for droppping columns/rows in training model

Below are the columns/rows we decided to exclude from the model

## Odds

We decided to exclude the

## Return

The columns `red_return` and `blue_return` are columns who's values reflect the payout a $1 bet would return. Including them would
cause target leakage since they tell the model exactly who won (loser's return being 0), causing a 100% model prediction rate accuracy.

## Stances

We noticed in the `red_stance` and `blue_stance` columns that there were a few rows with faulty classifications ("Southpaw " with a trailing space and "Unknown")
that needed to be excluded. We also found one other classification called "Open Stance", which describes an opposite stance matchup, which was only found
in the `blue_stance` column. Excluding it made sense on the premise that it conflicts with our own stance classifications.

## Catch Weight

In the column `weight_class` there is a value called "Catch Weight" which describes fights where one or both fighters missed their
target weights and therefore fought outside one of the 13 regulartory weight classes. The reason for dropping this is because:

1. It's an irrelevant outlier that might conflict with the standard classification we're planning to use
2. It only consists of 72 rows out of 6000+

## \*\_\_diff Column Outliers

In the `rounds_diff` column we found one extreme outlier where the value was > 440. We excluded this due to it being an impossible data value when
looking at either fighters records (i.e. total fights).

In the `reach_diff` column there is a single outlier where the reach difference was >186 caused by one of the fighters reach being measured 0 cm.
This is an obivious error caused by a missing data value so we naturally excluded it.
