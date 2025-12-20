# -*- coding: utf-8 -*-
"""
Ad. Sale Prediction from Existing customer - Logistic Regression
"""

# ---------------------------------
# Importing Libraries
# ---------------------------------
import pandas as pd  # useful for loading the dataset
import numpy as np   # to perform array operations

# ---------------------------------
# Load Dataset (Local Directory)
# ---------------------------------
dataset = pd.read_csv('D:\Materials & Projects\Machine Learning\Day_3 Advertisement Sale Prediction from and Exsisting Customer Using Logistic Rgression\Day_3\DigitalAd_dataset.csv')

# ---------------------------------
# Summarize Dataset
# ---------------------------------
print(dataset.shape)
print(dataset.head(5))

# ---------------------------------
# Segregate Dataset into X & Y
# ---------------------------------
X = dataset.iloc[:, :-1].values
X

Y = dataset.iloc[:, -1].values
Y

# ---------------------------------
# Splitting Dataset into Train & Test
# ---------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

# ---------------------------------
# Feature Scaling
# ---------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------------------
# Training Logistic Regression Model
# ---------------------------------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# ---------------------------------
# Predicting New Customer
# ---------------------------------
age = int(input("Enter New Customer Age: "))
sal = int(input("Enter New Customer Salary: "))

newCust = [[age, sal]]
result = model.predict(sc.transform(newCust))

print(result)

if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")

# ---------------------------------
# Prediction for all Test Data
# ---------------------------------
y_pred = model.predict(X_test)
print(
    np.concatenate(
        (y_pred.reshape(len(y_pred), 1),
         y_test.reshape(len(y_test), 1)),
        1
    )
)

# ---------------------------------
# Evaluating Model - Confusion Matrix
# ---------------------------------
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
