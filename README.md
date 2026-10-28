# Chicago Taxi Fare Prediction

## Project Overview

This project implements a Linear Regression model to predict taxi fares in Chicago based on trip distance.

The main objective is to explore the relationship between trip distance and fare price using machine learning techniques. This project was developed as a practical introduction to supervised learning and regression analysis.

## Dataset

The dataset contains 31,694 rows and 18 columns.

For this project, the following variables were selected:

- **Feature:** `TRIP_MILES` — The distance traveled during the trip, measured in miles.
- **Target:** `FARE` — The taxi fare.

The model uses trip distance as the independent variable to predict the fare as the dependent variable.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- matplotlib

## Methodology

The project follows these steps:

1. **Data preparation:** Load the dataset and select the relevant variables.
2. **Feature selection:** Use `TRIP_MILES` as the input feature and `FARE` as the target variable.
3. **Train-test split:** Divide the dataset into training and testing sets, using 80% for training and 20% for testing.
4. **Model training:** Train a Linear Regression model using Scikit-learn.
5. **Prediction:** Generate fare predictions using the test dataset.
6. **Model evaluation:** Evaluate performance using Mean Absolute Error (MAE) and the coefficient of determination (R²).

## Model Results

The model achieved the following results:

| Metric | Result |
|---|---:|
| Intercept | 5.0386 |
| Coefficient | 2.2751 |
| Mean Absolute Error (MAE) | 1.5825 |
| R² Score | 0.9499 |

### Regression Equation

The fitted linear regression model is:

`FARE = 5.0386 + 2.2751 × TRIP_MILES`

The coefficient indicates that the predicted fare increases by approximately 2.28 fare units for each additional mile traveled.

The intercept represents the model's predicted fare when the trip distance is zero.

### Model Evaluation

- **MAE:** The model's predictions differ from the actual fares by approximately 1.58 fare units on average.
- **R² Score:** The model explains approximately 94.99% of the variance in the target variable within the test dataset.

These results indicate a strong linear relationship between trip distance and fare in this dataset. However, the model uses only one feature, so it does not account for other factors that may influence taxi fares.


