import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

st.title("Stock Price Prediction")

# Take stock symbol as input
symbol = st.sidebar.text_input("Enter stock symbol (Note: For Indian Stocks add .NS after symbol)", "TATAMOTORS.NS")

# Get historical stock price data up to current date
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
data = yf.download(symbol, start="2010-01-01", end=end_date)

# Create a new DataFrame with only the 'Close' column
df = pd.DataFrame(data["Close"])

# Create a new column with the future stock prices we want to predict
future_days = st.sidebar.slider("Select the number of days to predict future stock prices", 1, 30, 10)
df["Prediction"] = df[["Close"]].shift(-future_days)

# Create a feature dataset and a target dataset
X = np.array(df.drop(["Prediction"], 1))[:-future_days]
y = np.array(df["Prediction"])[:-future_days]

# Split the data into training and testing sets
split_ratio = st.sidebar.slider("Select the training data ratio", 0.1, 0.9, 0.8, step=0.05)
split_point = int(split_ratio * len(X))
X_train, X_test, y_train, y_test = X[:split_point], X[split_point:], y[:split_point], y[split_point:]

# Create and train a Linear Regression model
lr_model = LinearRegression().fit(X_train, y_train)

# Test the accuracy of the model
accuracy = lr_model.score(X_test, y_test)

# Make a prediction for the next 'future_days' days
last_days = np.array(df.drop(["Prediction"], 1))[-future_days:]
prediction = lr_model.predict(last_days)

# Output the accuracy and predicted stock prices
st.write("Accuracy of model:", round(accuracy * 100, 2), "%")
st.write("Predicted stock prices for the next", future_days, "days:")
st.write(pd.DataFrame(prediction, columns=["Prediction"]))

# Plot the historical and predicted stock prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Historical"))
fig.add_trace(go.Scatter(x=data.index[-future_days:], y=prediction, name="Predicted"))
fig.update_layout(title="Stock Price Chart", xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig)
