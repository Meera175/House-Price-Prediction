# House Price Prediction

## 👤 Intern Information

| Field | Details |
|-------|---------|
| **Intern Name** | Meera Rani Jyotishi |
| **Intern ID** | CITS2078 |
| **Company** | CODTECH IT Solutions Pvt. Ltd. |
| **Domain** | Artificial Intelligence |
| **Mentor** | Neela Santhosh Kumar |
| **Duration** | 4 Weeks |

---


## Project Overview
This project builds a machine learning model to predict house prices based on
features such as area, number of bedrooms, bathrooms, stories, age of the
property, garage availability, and location. It covers the full ML workflow:
data loading, exploratory data analysis (EDA), preprocessing, model training,
evaluation, and prediction on new data.

## Dataset
The dataset (`house_prices.csv`) contains 500 records with the following columns:

| Column      | Description                                |
|-------------|---------------------------------------------|
| Area_sqft   | Total area of the house in square feet      |
| Bedrooms    | Number of bedrooms                           |
| Bathrooms   | Number of bathrooms                           |
| Stories     | Number of floors/stories                      |
| Age_years   | Age of the house in years                     |
| Garage      | Number of garage spaces                       |
| Location    | Urban / Suburban / Rural                       |
| Price       | Target variable - sale price of the house     |

> Note: This is a synthetically generated dataset created for learning
> purposes, designed to resemble realistic housing data patterns. You may
> replace it with a real-world dataset (e.g. Kaggle's "House Prices" or
> "Boston Housing") if your task requires it — the code will work the same
> way as long as column names are adjusted accordingly.

## Tech Stack / Libraries Used
- Python 3
- pandas, numpy - data handling
- matplotlib, seaborn - visualization
- scikit-learn - machine learning (Linear Regression, Random Forest)

## Project Structure
```
house-price-prediction/
│
├── house_price_prediction.py   # Main script: EDA + training + evaluation
├── house_prices.csv            # Dataset
├── README.md                   # Project documentation
├── price_distribution.png      # Output plot: price distribution
├── correlation_heatmap.png     # Output plot: feature correlation
├── area_vs_price.png           # Output plot: area vs price
├── feature_importance.png      # Output plot: feature importance
├── actual_vs_predicted.png     # Output plot: model accuracy visual
└── screenshots/                # Folder for task submission screenshots
```

## How to Run
1. Clone this repository
   ```
   git clone <your-repo-link>
   cd house-price-prediction
   ```
2. Install dependencies
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the script
   ```
   python house_price_prediction.py
   ```
4. The script will print EDA results and model evaluation metrics in the
   terminal, and save all plots as PNG files in the project folder.

## Approach
1. **Data Loading & Exploration** - inspected dataset shape, types, missing
   values, and summary statistics.
2. **EDA** - visualized price distribution, correlation between features,
   and the relationship between area and price across locations.
3. **Preprocessing** - encoded the categorical `Location` column using
   Label Encoding, then split data into training (80%) and testing (20%) sets.
4. **Model Training** - trained two models for comparison:
   - Linear Regression (simple baseline)
   - Random Forest Regressor (handles non-linear relationships)
5. **Evaluation** - compared models using MAE, RMSE, and R² score.
6. **Feature Importance** - identified which features most strongly affect
   price predictions (area and location proved most influential).
7. **Prediction** - demonstrated prediction on a new, unseen house example.

## Results
| Model              | MAE      | RMSE     | R² Score |
|--------------------|----------|----------|----------|
| Linear Regression  | ~62,000  | ~80,800  | ~0.94    |
| Random Forest      | ~76,000  | ~94,800  | ~0.91    |

*(Exact values may vary slightly between runs due to randomness in train/test split.)*

## Screenshots
Add screenshots of your terminal output / plots / GitHub submission here:

```
![Terminal Output](screenshots/terminal_output.png)
![Price Distribution](screenshots/price_distribution.png)
```

## Conclusion
Both models performed well, with area and location being the strongest
predictors of house price. Linear Regression slightly outperformed Random
Forest on this dataset, likely because the underlying price relationship
was largely linear by construction. In a real-world dataset with more
complex, non-linear interactions, Random Forest (or gradient boosting
models) typically perform better.

---
*This project was completed as part of Task of the Artificial Intelligence
Internship at CodTech IT Solutions Pvt. Ltd.*
