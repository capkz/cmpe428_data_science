# -*- coding: utf-8 -*-
"""Data Task 4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18psjOzkgNqhjH7Lol5WG2gPZvXy_abXm

# CMPE428 Assignment 4
Building Nearest Neighbour Classifiers
by Çağıl Peköz

Imports
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

"""Import CSV"""

df = pd.read_csv('stdData 1.csv')

df.head()

"""## Task

### Split Dataset with Equal Positives and Negatives

We can see the amount of negatives and positives with this command
"""

df.Label.value_counts()

"""#### Replacing Categorical with Binary"""

df["Label"] = df["Label"].replace({"positive":1,"negative":0})

df.head()

"""#### Splitting Dataset Into 2, with Equal Amounts of Negative and Positive

We first get our X and Y values.
"""

X = df.drop(["Label"], axis = 1)
y = df["Label"]

"""We use stratify on y and split our dataframe into 2, so that we get equal distribution of negatives and positives."""

X_train, X_test = train_test_split(df, test_size = 0.5, stratify = y)

y_train.value_counts()

y_test.value_counts()

"""We can see that numbers of 0 and 1 is equal across test and train.

### kNN Classifier and Scores

Here I have built the model and for test purposes I tried k=1.
"""

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)

scores = {}
scores['K Value'] = "1"
scores['Accuracy'] = accuracy_score(y_test, y_pred)
scores['F1'] = f1_score(y_test, y_pred)
scores['Recall'] = recall_score(y_test, y_pred)
scores['Precision'] = precision_score(y_test, y_pred)

scores_df = pd.DataFrame.from_dict(scores, orient='index')
scores_df = scores_df.transpose()
scores_df

"""### Testing Different K Values

Here I will try k=2.
"""

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)

scores = {}
scores['K Value'] = "2"
scores['Accuracy'] = accuracy_score(y_test, y_pred)
scores['F1'] = f1_score(y_test, y_pred)
scores['Recall'] = recall_score(y_test, y_pred)
scores['Precision'] = precision_score(y_test, y_pred)

scores_df = scores_df.append(scores, ignore_index=True)
scores_df

"""Here I will try k=3."""

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)

scores = {}
scores['K Value'] = "3"
scores['Accuracy'] = accuracy_score(y_test, y_pred)
scores['F1'] = f1_score(y_test, y_pred)
scores['Recall'] = recall_score(y_test, y_pred)
scores['Precision'] = precision_score(y_test, y_pred)

scores_df = scores_df.append(scores, ignore_index=True)
scores_df

"""### Trying Out a Few Formulas I Found Online

According to my research, as a general rule of thumb, k-value is determined by k = sqrt(N)/2 or k = sqrt(N) formula. I will use both of these formulas and check the scores.
"""

k1 = sqrt(len(X_train))/2
round(k1)

k2 = sqrt(len(X_train))
round(k2)

"""Now I will try k=6."""

neigh = KNeighborsClassifier(n_neighbors=6)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)

scores = {}
scores['K Value'] = "6"
scores['Accuracy'] = accuracy_score(y_test, y_pred)
scores['F1'] = f1_score(y_test, y_pred)
scores['Recall'] = recall_score(y_test, y_pred)
scores['Precision'] = precision_score(y_test, y_pred)

scores_df = scores_df.append(scores, ignore_index=True)
scores_df

"""Lastly, I will try k=12."""

neigh = KNeighborsClassifier(n_neighbors=12)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)

scores = {}
scores['K Value'] = "12"
scores['Accuracy'] = accuracy_score(y_test, y_pred)
scores['F1'] = f1_score(y_test, y_pred)
scores['Recall'] = recall_score(y_test, y_pred)
scores['Precision'] = precision_score(y_test, y_pred)

scores_df = scores_df.append(scores, ignore_index=True)
scores_df

"""In here, we can see that our Precision and Accuracy is all time high in the k-value 12, however our F1 and Recall went down. In k-value 6, our F1 and Recall numbers were higher."""