from sklearn.neural_network import MLPClassifier
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import explained_variance_score

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# reading the csv file

df = pd.read_csv("train_set.csv", delimiter = ",", encoding="utf-8")

# removing the na in the dataset

df=df.dropna()


#split the data for training and testing as percentage

train, test = train_test_split(df, test_size=0.2)
# train dataset preprocessing 
train_x = train.drop(columns=['sum'])
train_y = train.drop(columns=['Temperature','Humidity','Pressure'])
test_x = test.drop(columns=['sum'])
test_y = test.drop(columns=['Temperature','Humidity','Pressure'])

np_train_x = np.array(train_x);
np_train_y = np.array(train_y);
np_test_x = np.array(test_x);
np_test_y = np.array(test_y);
scaler.fit(train_x)
X_train = scaler.transform(np_train_x)
X_test = scaler.transform(np_test_x)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,train_y)
predictions = mlp.predict(X_test)


#clf.fit(np_train_x, np_train_y)
#predicted = clf.predict(np_test_x)

print(confusion_matrix(test_y,predictions))
#print(classification_report(test_y,predictions))
print(accuracy_score(test_y, predictions, normalize=False))


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(test_y, predictions, edgecolors=(0, 0, 0))
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()





