import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
#Load the Dataset
chicago_taxi_dataset = pd.read_csv('chicago_taxi_train.csv', encoding='utf-8')

#diplay all columns
pd.set_option('display.max_columns', None)

#Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

#View dataset statistics
print(training_df.describe(include='all'))

#Check the correlation between the metrics
print(training_df.corr(numeric_only = True))

#View pairplot and double-checking the correlation
fig = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
fig.show()

#Define the metrics
X = training_df[['TRIP_MILES']]
Y = training_df['FARE']

#Testing the 20% of data from the given dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

#Training the model
model = LinearRegression()

model.fit(X_train, Y_train)

print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_[0])

from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MAE:", mae)
print("R²:", r2)


plt.scatter(X_test["TRIP_MILES"], Y_test, alpha=0.3)
plt.plot(
    X_test["TRIP_MILES"],
    y_pred,
    color="red",
    linewidth=2
)

plt.xlabel("Trip Miles")
plt.ylabel("Fare ($)")
plt.title("Linear Regression: Actual Taxi Fares")
plt.show()
