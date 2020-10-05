###################################################
##Template for all basic Classification ML models##
###################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')

# Importing the dataset
file = ('../Data/shootings.csv')
dataset = pd.read_csv(file)


x = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values


###
### EDA
###
row_count = len(y)

gender = dataset.iloc[:, 6].values
count = 0
for g in gender:
    if g == 'M': count += 1
print(str((count/row_count) * 100) + ' percect of total deaths were male.')

flee = dataset.iloc[:, 12].values
count = 0
for f in flee:
    if f == 'Not fleeing': count += 1
print(str((count/row_count) * 100) + ' percect of total deaths were people who did not attempt to flee the police.')

armed_cat = dataset.iloc[:, -1].values
count = 0
for a in armed_cat:
    if a == 'Unarmed': count += 1
print(str((count/row_count) * 100) + ' percect of total police shooting victims were unarmed.')
#

races = dataset.iloc[:, 7].values
unique_races, race_frequency = np.unique(races, return_counts=True)
race_distribution = dict(zip(unique_races, race_frequency))
print("Race distribution:" + str(race_distribution))
##need to add pie chart


"""
###
### Encoding categorical data
###
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###
### Logistic Regression
###
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

###
### Support Vector Machine (SVM)
###
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

###
### Kernel SVM
###
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

###
### K-Nearest-Neighbors
###
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

###
### Naive Bayes
###
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

###
### Decision Tree
###
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

###
### Random Forest
###
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
"""
