import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Data Cleaning
df = df.drop("Unnamed: 0", axis=1)

# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Sales Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["Sales"], bins=10, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Feature Selection
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Sales
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Performance:")
print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 4))

# Future Sales Prediction
new_data = pd.DataFrame({
    "TV": [200],
    "Radio": [30],
    "Newspaper": [40]
})

future_sales = model.predict(new_data)
print("\nPredicted Future Sales:", round(future_sales[0], 2))

# Actual vs Predicted Sales Graph
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, label="Actual Sales", marker="o")
plt.plot(y_pred, label="Predicted Sales", marker="x")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Data")
plt.ylabel("Sales")
plt.legend()
plt.show()