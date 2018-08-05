from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


np.random.seed(0)
df = pd.read_csv("train_set.csv", delimiter = ",", encoding="utf-8")


#dataset slicing

X = df.values[:, 1:3]
Y = df.values[:,0]

#dataset split


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# decision tree classifier using gini information gain
clf_gini = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')

clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print(y_pred)
print(y_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
