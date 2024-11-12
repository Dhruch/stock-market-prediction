import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# Function to fetch data for a given stock symbol
def fetch_stock_data(symbol):
    data = yf.download(symbol, start="2015-01-01")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]


# Prepare data for the LSTM model
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X_train, y_train = [], []
    for i in range(60, len(scaled_data) - 1):
        X_train.append(scaled_data[i - 60:i, :])
        y_train.append(scaled_data[i, 3])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train, scaler


# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Predict the next day's closing price
def predict_next_day(model, recent_data, scaler):
    scaled_data = scaler.transform(recent_data)
    X_test = np.array([scaled_data[-60:]])
    prediction = model.predict(X_test)
    return scaler.inverse_transform(np.concatenate((np.zeros((1, 4)), prediction), axis=1))[0, -1]


# Streamlit App
def stock_prediction_tool():
    st.title("Stock Price Prediction Tool")
    symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()

    if symbol:
        st.write(f"Fetching data for {symbol}...")
        df = fetch_stock_data(symbol)

        # Display Historical Data
        st.write(f"Historical Close Prices for {symbol}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f'{symbol} Historical Close Price'))
        fig.update_layout(title=f'{symbol} Stock Price History', xaxis_title='Date', yaxis_title='Close Price')
        st.plotly_chart(fig)

        # Prepare data and train the model
        X_train, y_train, scaler = prepare_data(df)
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Predict the next day's closing price
        predicted_price = predict_next_day(model, df.values, scaler)
        st.write(f"\nThe predicted closing price for {symbol} on the next day is: ${predicted_price:.2f}\n")

        # Generate predictions for the dataset to show comparison
        df['Predicted Close'] = np.nan
        for i in range(60, len(df) - 1):
            X_test = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i - 60:i].values
            df['Predicted Close'].iloc[i + 1] = predict_next_day(model, X_test, scaler)

        # Plot actual vs predicted
        st.write(f"Predicted vs Actual Close Price for {symbol}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Actual Close Price"))
        fig2.add_trace(go.Scatter(x=df.index, y=df['Predicted Close'], mode='lines', name="Predicted Close Price",
                                  line=dict(dash='dash')))
        fig2.update_layout(title=f'{symbol} Predicted vs Actual Close Price', xaxis_title='Date', yaxis_title='Price')

