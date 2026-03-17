Crop Yield Prediction using Linear Regression
---
Author: Blessing Mudarikwa

Role: Aspiring Data Engineer | Data Analyst

Tools: Python, Pandas, Scikit-learn, Matplotlib

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("agricultural_dataset.csv")

# Select feature and target
X = df[['Rainfall']]  # Independent variable
y = df['Standard_yield']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance")
print(f"Coefficient (Slope): {model.coef_[0]:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

print(f"Mean Squared Error: {mse:.4f}")

print(f"R² Score: {r2:.4f}")

# 📈 Visualization
plt.figure()

plt.scatter(X_test, y_test)

plt.plot(X_test, y_pred)

plt.xlabel("Rainfall")

plt.ylabel("Standard Yield")

plt.title("Linear Regression: Rainfall vs Crop Yield")
plt.show()

