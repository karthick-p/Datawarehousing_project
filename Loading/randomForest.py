import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv('decisionTree.csv')
features2 = pd.read_csv('Forecastlastweek.csv')
features2 = features2.drop('T2', axis = 1)
features2 = features2.drop('Date', axis = 1)

features2 = np.array(features2)


# Labels are the values we want to predict
labels = np.array(features['sum'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('sum', axis = 1)



# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)




train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)



# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# Train the model on training data
rf.fit(train_features, train_labels)


future_consp = rf.predict(features2)
print(future_consp)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

df_1 = pd.DataFrame(future_consp.T)


df_1.to_csv('last_week_demand.csv', sep=',', encoding='utf-8')

# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(test_labels, predictions, edgecolors=(0, 0, 0))
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
plt.title('Random Forest')
importances = list(rf.feature_importances_)



# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()

