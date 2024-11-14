

# 📈 Stock Price Prediction Tool

This project is a **Stock Price Prediction Tool** that uses **machine learning** to forecast the next day’s closing price of a specified stock. It combines historical stock data from the **Yahoo Finance API** with an **XGBoost regression model** to make predictions based on historical patterns.

## 🚀 Features

- **Historical Stock Data**: Fetches historical stock prices, including `Open`, `High`, `Low`, `Close`, and `Volume`, using the Yahoo Finance API via the `yfinance` library.
- **Feature Engineering**: Creates additional features such as:
  - **Previous Day's Close**: Closing price from the prior day.
  - **Daily Return**: Percentage change from the previous day's close.
  - **5-Day Moving Average**: Average closing price over the past 5 days.
  - **20-Day Moving Average**: Average closing price over the past 20 days.
- **Machine Learning Model**: Uses **XGBoost (eXtreme Gradient Boosting)** to predict the next day's closing price based on engineered features.
- **Data Visualization**: Displays interactive charts for:
  - Historical closing prices.
  - Predicted vs. actual closing prices.
  - Comparison between the previous day's actual closing price and today's predicted price.

## 📂 Project Structure

- `main.py`: The main Streamlit application code.
- `requirements.txt`: Contains the list of dependencies for the project.

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dhruch/stock-price-prediction-tool.git
   cd stock-price-prediction-tool
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run main.py
   ```

2. **Enter a stock symbol** (e.g., `AAPL` for Apple) in the text input field, and the app will:
   - Fetch historical data for the specified stock.
   - Perform feature engineering to generate inputs for the model.
   - Train the XGBoost model on the historical data.
   - Predict the next day's closing price.
   - Display the previous day’s closing price and the predicted price, along with a message indicating whether the stock price is expected to increase or decrease.
   - Plot interactive charts showing:
     - Historical closing prices.
     - Predicted vs. actual closing prices for validation.

## 📊 Example Output

1. **Historical Data Chart**: Displays the past performance of the stock based on historical closing prices.
2. **Prediction Comparison**:
   - Yesterday’s closing price vs. Today’s predicted price.
   - Success or warning message indicating if the price is expected to rise or fall.
3. **Predicted vs Actual Prices**: A chart comparing the model's predictions to actual historical prices, providing a visual evaluation of the model’s performance.

## ⚙️ Technologies Used

- **Python**: Programming language.
- **Streamlit**: For building an interactive web application.
- **Yahoo Finance API (`yfinance`)**: To fetch historical stock data.
- **XGBoost**: A powerful gradient boosting machine learning algorithm for prediction.
- **Scikit-Learn**: For data preprocessing.
- **Plotly**: For interactive data visualization.

## 📈 Model Details

- **Model**: XGBoost Regressor
- **Features**:
  - Previous day's closing price
  - Daily return percentage
  - 5-day moving average of closing prices
  - 20-day moving average of closing prices
- **Target**: The next day’s closing price
- **Objective**: To minimize the mean squared

error in predicting continuous values (i.e., stock prices).

## 🔍 Insights

This project provides an accessible tool to analyze stock price trends using machine learning. While the predictions are not financial advice and should be used with caution, the tool demonstrates how historical data can be leveraged to model potential price movements.

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice. Stock prices are influenced by numerous unpredictable factors, and past performance is not an indicator of future results.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing free stock data.
- The developers of [XGBoost](https://xgboost.ai/) and [Streamlit](https://streamlit.io/) for creating such powerful tools.


