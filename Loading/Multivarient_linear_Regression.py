import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score



df = pd.read_csv("decisionTree.csv", delimiter = ",", encoding="utf-8")

train, test = train_test_split(df, test_size=0.2)

train_x = train.drop(columns=['sum'])
train_y = train.drop(columns=['Humidity','Temperature','Pressure'])

#this creates the linear regeression model

lm = linear_model.LinearRegression()
lm.fit(train_x,train_y)


test_x = test.drop(columns=['sum'])
test_y = test.drop(columns=['Humidity','Temperature','Pressure'])

#test_y.to_csv("actual.csv", sep='\t', encoding='utf-8')
test_x1 = test['Temperature']
test_x2 = test['Humidity']
test_x3 = test['Pressure']

pred_y = lm.predict(test_x)
#pred_y.to_csv("predicted.csv", sep='\t', encoding='utf-8')
#print(accuracy_score(test_y, pred_y, normalize=False) * 100)


# The coefficients
print('Coefficients: \n', lm.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, pred_y))

# Plot outputs
plt.scatter(test_x1, test_y,  color='black')



plt.plot(np.sort(test_x1, axis=0), pred_y, color='blue', linewidth=3)


plt.xticks(())
plt.yticks(())

plt.show()




