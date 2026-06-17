"""
House Price Prediction
-----------------------
A simple machine learning project that predicts house prices based on
features like area, bedrooms, bathrooms, location, etc.

Author: <Your Name>
Internship Task 1 - AI Internship (CodTech IT Solutions)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("house_prices.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], kde=True, color="steelblue")
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

plt.figure(figsize=(8, 6))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Area_sqft", y="Price", hue="Location", data=df)
plt.title("Area vs Price")
plt.tight_layout()
plt.savefig("area_vs_price.png")
plt.close()

print("\nEDA plots saved: price_distribution.png, correlation_heatmap.png, area_vs_price.png")

# ---------------------------------------------------------
# 3. Data Preprocessing
# ---------------------------------------------------------
df_model = df.copy()

# Encode the categorical "Location" column
le = LabelEncoder()
df_model["Location"] = le.fit_transform(df_model["Location"])

X = df_model.drop("Price", axis=1)
y = df_model["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# 4. Model Training
# ---------------------------------------------------------

# Model 1: Linear Regression (baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model 2: Random Forest Regressor (better performance, handles non-linearity)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R2   : {r2:.4f}")
    return mae, rmse, r2


print("\n" + "=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

evaluate_model("Linear Regression", y_test, lr_preds)
evaluate_model("Random Forest Regressor", y_test, rf_preds)

# ---------------------------------------------------------
# 5. Feature Importance (Random Forest)
# ---------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nFeature importance plot saved: feature_importance.png")
print("\nFeature importances:")
print(importances)

# ---------------------------------------------------------
# 6. Actual vs Predicted plot
# ---------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, rf_preds, alpha=0.6, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nActual vs Predicted plot saved: actual_vs_predicted.png")

# ---------------------------------------------------------
# 7. Sample Prediction on New Data
# ---------------------------------------------------------
sample_house = pd.DataFrame({
    "Area_sqft": [2500],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Stories": [2],
    "Age_years": [5],
    "Garage": [1],
    "Location": le.transform(["Urban"])
})

predicted_price = rf_model.predict(sample_house)
print(f"\nSample Prediction -> Predicted price for the new house: ${predicted_price[0]:,.2f}")

print("\nDone! All outputs and plots have been generated successfully.")
