import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Explore the data
print(data.head())

# Split the data into features (X) and target variable (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Visualize the training set results
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Percentage (Training set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# Visualize the test set results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Percentage (Test set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# Predict the percentage score for a given number of hours
hours = 9.25
predicted_score = regressor.predict([[hours]])
print("Predicted Score for {} hours of study: {:.2f}%".format(hours, predicted_score[0]))