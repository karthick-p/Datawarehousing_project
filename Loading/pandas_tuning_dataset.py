import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model


df = pd.read_csv("trial.csv", delimiter = ",", encoding="utf-8")
df=df.dropna()

train, test = train_test_split(df, test_size=0.2)

train_x = train.drop(columns=['Supply_MW','datetime'])
train_y = train.drop(columns=['datetime','Temp','Humidity','wind_speed'])

#this creates the linear regeression model

lm = linear_model.LinearRegression()
lm.fit(train_x,train_y)


test_x = test.drop(columns=['Supply_MW','datetime'])
test_y = test.drop(columns=['datetime','Temp','Humidity','wind_speed'])

#test_y.to_csv("actual.csv", sep='\t', encoding='utf-8')
test_x1 = test['Temp']
test_x2 = test['Humidity']
test_x3 = test['wind_speed']


print(test_x1)
pred_y = lm.predict(test_x)
#pred_y.to_csv("predicted.csv", sep='\t', encoding='utf-8')


# The coefficients
print('Coefficients: \n', lm.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, pred_y))

# Plot outputs
plt.scatter(test_x1, test_y,  color='black')
plt.scatter(test_x2, test_y,  color='black')
plt.scatter(test_x3, test_y,  color='black')


plt.plot(test_x1, pred_y, color='blue', linewidth=3)
plt.plot(test_x2, pred_y, color='blue', linewidth=3)
plt.plot(test_x3, pred_y, color='blue', linewidth=3)


plt.xticks(())
plt.yticks(())

plt.show()
