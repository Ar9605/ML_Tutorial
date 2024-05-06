import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('C:\\Users\\abhin\\Downloads\\FuelConsumptionCo2.csv')
#print(data.head())

X = data['ENGINESIZE'].values.reshape(-1, 1)  
y = data['CYLINDERS'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

# Add labels and legend
plt.xlabel('ENGINESIZE')
plt.ylabel('CYLINDERS')
plt.title('Linear Regression')
plt.legend()

# Show plot
plt.show()