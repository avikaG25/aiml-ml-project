import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('data.csv')

# Separate input and output
X = data[['Hours']]
y = data['Marks']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Take user input
hours = float(input("Enter study hours: "))

# Predict
prediction = model.predict([[hours]])

# Output result
print("Predicted Marks:", round(prediction[0], 2))