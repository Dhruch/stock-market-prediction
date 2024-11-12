import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import plotly.graph_objs as go


# Function to fetch stock data
def fetch_stock_data(symbol):
    data = yf.download(symbol, start="2015-01-01")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]


# Feature engineering for time series forecasting
def create_features(data):
    data['Prev Close'] = data['Close'].shift(1)
    data['Daily Return'] = data['Close'].pct_change()
    data['5 Day SMA'] = data['Close'].rolling(window=5).mean()
    data['20 Day SMA'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)
    return data


# Prepare data for model
def prepare_data(data):
    # Define features and target
    features = data[['Prev Close', 'Daily Return', '5 Day SMA', '20 Day SMA']]
    target = data['Close']

    # Handle NaN or infinite values in features
    if features.isnull().values.any():
        features = features.fillna(features.mean())
    if np.isinf(features.values).any():
        features = features.replace([np.inf, -np.inf], 0)

    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler


# Train XGBoost model
def train_model(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model


# Predict the next day price
def predict_next_day(model, last_data, scaler):
    last_data_scaled = scaler.transform([last_data])
    return model.predict(last_data_scaled)[0]


# Streamlit app
def stock_prediction_tool():
    st.title("📈 Stock Price Prediction Tool")
    symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()

    if symbol:
        st.write(f"Fetching data for **{symbol}**...")
        df = fetch_stock_data(symbol)

        # Display historical data
        st.write(f"Historical Close Prices for **{symbol}**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f'{symbol} Historical Close Price'))

        # Add a placeholder if no data is available
        if df.empty:
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Placeholder line"))
            fig.update_layout(title=f'{symbol} Stock Price History (No data available)',
                              xaxis_title='Date', yaxis_title='Close Price')
        else:
            fig.update_layout(title=f'{symbol} Stock Price History', xaxis_title='Date', yaxis_title='Close Price')

        st.plotly_chart(fig)

        # Feature engineering and prepare data
        df = create_features(df)
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)

        # Train model
        model = train_model(X_train, y_train)

        # Get yesterday's closing price (as a scalar value)
        yesterday_close = float(df['Close'].iloc[-1])  # Ensure it's a single value, not a Series

        # Predict today's closing price
        last_data = df[['Prev Close', 'Daily Return', '5 Day SMA', '20 Day SMA']].iloc[-1].values
        predicted_price = predict_next_day(model, last_data, scaler)

        # Display comparison with enhanced formatting
        st.markdown(
            f"<h3 style='color: blue;'>Yesterday's Closing Price for {symbol}: <b>${yesterday_close:.2f}</b></h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: green;'>Predicted Closing Price for Today: <b>${predicted_price:.2f}</b></h3>",
                    unsafe_allow_html=True)

        # Display whether the price is expected to increase or decrease
        if predicted_price > yesterday_close:
            st.success(f"The stock price is predicted to **increase** by ${predicted_price - yesterday_close:.2f}.")
        else:
            st.error(f"The stock price is predicted to **decrease** by ${yesterday_close - predicted_price:.2f}.")

        # Add predictions to the DataFrame for plotting
        df['Predicted Close'] = np.nan
        df.loc[df.index[-len(X_test):], 'Predicted Close'] = model.predict(X_test)

        # Plot predicted vs actual prices
        st.write(f"Predicted vs Actual Close Price for **{symbol}**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Actual Close Price"))
        fig2.add_trace(go.Scatter(x=df.index, y=df['Predicted Close'], mode='lines', name="Predicted Close Price",
                                  line=dict(dash='dash')))
        fig2.update_layout(title=f'{symbol} Predicted vs Actual Close Price', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig2)


# Run the app
if __name__ == '__main__':
    stock_prediction_tool()
