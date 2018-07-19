import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model


df = pd.read_csv("onemonth.csv", delimiter = ",", encoding="utf-8")
df=df.dropna()

train, test = train_test_split(df, test_size=0.2)

train_x = train.drop(columns=['Supply_MW','datetime'])
train_y = train.drop(columns=['datetime','Temp'])

#this creates the linear regeression model

lm = linear_model.LinearRegression()
lm.fit(train_x,train_y)

print(df)

test_x = test.drop(columns=['Supply_MW','datetime'])
test_y = test.drop(columns=['datetime','Temp'])

#test_y.to_csv("actual.csv", sep='\t', encoding='utf-8')


pred_y = lm.predict(test_x)
#pred_y.to_csv("predicted.csv", sep='\t', encoding='utf-8')


print("hi")
print(test_y)
print("check")
print(pred_y)

# The coefficients
print('Coefficients: \n', lm.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, pred_y))

# Plot outputs
plt.scatter(test_x, test_y,  color='black')
plt.plot(test_x, pred_y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
