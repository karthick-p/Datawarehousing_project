import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

training = pd.read_csv('decisionTree.csv')
input_data = pd.read_csv('predict_input.csv')
input_data_bck = input_data

print(input_data)
input_data = input_data.drop('T2', axis = 1)
input_data = input_data.drop('Date', axis = 1)
input_data = np.array(input_data)


# Labels are the values we want to predict
labels = np.array(training['sum'])
# Remove the labels from the training
# axis 1 refers to the columns
training= training.drop('sum', axis = 1)



# Saving feature names for later use
feature_list = list(training.columns)
# Convert to numpy array
training = np.array(training)




train_training, test_training, train_labels, test_labels = train_test_split(training, labels, test_size = 0.25, random_state = 42)



# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# Train the model on training data
rf.fit(train_training, train_labels)




# Use the forest's predict method on the test data
predictions = rf.predict(test_training)




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



predictions = rf.predict(input_data)

print(predictions)


df_1 = pd.DataFrame(predictions.T)


input_data_bck = pd.concat([input_data_bck, df_1], axis=1)

print(input_data_bck)
input_data_bck.to_csv('two_week_demand.csv', sep=',', encoding='utf-8')

