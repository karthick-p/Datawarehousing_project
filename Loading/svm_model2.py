import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  





5# reading the csv file

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

svclassifier = SVC(kernel='linear')  
svclassifier.fit(train_x, train_y)  


y_pred = svclassifier.predict(test_x)  


print(confusion_matrix(test_y,y_pred))  
print(classification_report(test_y,y_pred))
print(accuracy_score(test_y, y_pred, normalize=False) * 100)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(test_y, y_pred, edgecolors=(0, 0, 0))
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
