# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and preprocess it by converting categorical variables to numerical.
2. Split the data into features (X) and target variable (y).
3. Divide the dataset into training and testing sets.
4. Train the Decision Tree Regressor on the training data.
5. Predict the target values for the test data.
6. Evaluate the model's performance using Mean Squared Error (MSE).
7. Visualize the decision tree.

## Program:
```py
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: YUVAN SUNDAR S
RegisterNumber:  212223040250
*/
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
file_path = 'Salary.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Preprocessing (Assuming 'Salary' is the target variable)
# Convert categorical variables to numerical if any exist
data = pd.get_dummies(data)

# Split the dataset into features and target variable
# Assuming 'Salary' is the target variable
X = data.drop('Salary', axis=1)  # Drop the 'Salary' column from features
y = data['Salary']  # The target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Regressor
regressor = DecisionTreeRegressor()

# Train the model
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(regressor, feature_names=X.columns, filled=True)
plt.show()

```

## Output:
![Screenshot 2024-09-30 155529](https://github.com/user-attachments/assets/1b060439-a8af-417b-9662-e92e64b6e9a0)
![Screenshot 2024-09-30 155601](https://github.com/user-attachments/assets/19d466c5-544e-439b-984d-e02fd255e6d6)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
