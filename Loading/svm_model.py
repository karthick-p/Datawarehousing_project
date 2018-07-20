import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  





5# reading the csv file

df = pd.read_csv("trial.csv", delimiter = ",", encoding="utf-8")

# removing the na in the dataset

df=df.dropna()


#split the data for training and testing as percentage

train, test = train_test_split(df, test_size=0.2)

# train dataset preprocessing 
train_x = train.drop(columns=['Supply_MW','datetime'])
train_y = train.drop(columns=['datetime','Temp','Humidity','wind_speed'])


test_x = test.drop(columns=['Supply_MW','datetime'])
test_y = test.drop(columns=['datetime','Temp','Humidity','wind_speed'])

svclassifier = SVC(kernel='linear')  
svclassifier.fit(train_x, train_y)  


y_pred = svclassifier.predict(test_x)  


print(confusion_matrix(test_y,y_pred))  
print(classification_report(test_y,y_pred))  

