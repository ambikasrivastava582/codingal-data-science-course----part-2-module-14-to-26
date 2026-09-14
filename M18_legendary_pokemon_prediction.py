import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('Pokemon Data.csv')
data.head()

data['Type 2'] = data['Type 2'].fillna('None')
print(data.isnull().sum())

data['Type 1'].value_counts().plot.bar()
plt.show()

data['Type 2'].value_counts().plot.bar()
plt.show()

data['Legendary'].value_counts().plot.bar()
plt.show()

print(data['Type 1'].unique())
print(data['Type 2'].unique())

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['Legendary'] = lb.fit_transform(data['Legendary'])

data.head()

data.drop('Name', axis=1, inplace=True)

data = pd.get_dummies(data)

print(data.shape)

# MISSING: Separate features and target
X = data.drop('Legendary', axis=1)
y = data['Legendary']

# MISSING: Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LogReg = LogisticRegression(max_iter=1000)
LogReg.fit(X_train, y_train)

ypred1 = LogReg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, ypred1))

# KNN
from sklearn.neighbors import KNeighborsClassifier
error_rates = []

for a in range(1, 40):
    k = a
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    # CORRECTED error calculation
    error_rates.append(np.mean(y_test != preds))

plt.figure(figsize=(10, 7))
plt.plot(range(1, 40), error_rates, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knn_model = KNeighborsClassifier(n_neighbors=8)
knn_model.fit(X_train, y_train)

y_predict = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_predict))

# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier()
clf_model.fit(X_train, y_train)

y_predict = clf_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_predict))