import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score




from sklearn.naive_bayes import GaussianNB

# reading the csv file

df = pd.read_csv("decisionTree.csv", delimiter = ",", encoding="utf-8")

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

model = GaussianNB()

# Train the model using the training sets 
model.fit(np_train_x, np_train_y)
predicted= model.predict(np_test_x)

print(predicted)
print(confusion_matrix(predicted,np_test_y))
print(accuracy_score(np_test_y, predicted, normalize=False))


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(np_test_y, predicted, edgecolors=(0, 0, 0))
ax.plot([np_test_y.min(), np_test_y.max()], [np_test_y.min(), np_test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

