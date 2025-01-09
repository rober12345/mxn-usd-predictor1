"""
modified to predict mxn-usd value

"""

import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ExchangeRatePredictor:
    def __init__(self, ticker="MXN=X", lookback_days=365 * 3):  # 3 years of historical data
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.model = None
        self.last_predictions = None
        
    def fetch_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                           end=end_date.strftime("%Y-%m-%d"), 
                           interval='1d')
        return df[['Close']].dropna()
    
    def prepare_features(self, df):
        df['S_3'] = df['Close'].rolling(window=3).mean()
        df['S_9'] = df['Close'].rolling(window=9).mean()
        df['volatility'] = df['Close'].rolling(window=5).std()
        df['day_of_week'] = df.index.dayofweek  # Day of the week (0=Monday)
        df['month'] = df.index.month  # Month of the year
        df['day_of_year'] = df.index.dayofyear  # Day of the year
        
        df['next_day_rate'] = df['Close'].shift(-1)
        return df.dropna()
    
    def train_model(self, train_size=0.8):
        df = self.fetch_data()
        df = self.prepare_features(df)
        
        X = df[['S_3', 'S_9', 'volatility', 'day_of_week', 'month', 'day_of_year']]
        y = df['next_day_rate']
        
        split_idx = int(train_size * len(df))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        test_predictions = self.model.predict(X_test)
        self.last_predictions = pd.DataFrame(test_predictions, 
                                             index=y_test.index, 
                                             columns=['predicted_rate'])
        
        metrics = {
            'MSE': mean_squared_error(y_test, test_predictions),
            'MAE': mean_absolute_error(y_test, test_predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, test_predictions)),
            'R²': self.model.score(X_test, y_test)
        }
        
        return metrics, df['Close'][-len(self.last_predictions):], self.last_predictions

    def forecast_future(self, months=12):
        last_known_date = datetime.now()
        future_dates = pd.date_range(start=last_known_date, periods=months, freq='M')
        
        future_data = pd.DataFrame(index=future_dates)
        future_data['S_3'] = np.nan
        future_data['S_9'] = np.nan
        future_data['volatility'] = np.nan
        future_data['day_of_week'] = future_data.index.dayofweek
        future_data['month'] = future_data.index.month
        future_data['day_of_year'] = future_data.index.dayofyear
        
        last_close = self.last_predictions['predicted_rate'].iloc[-1]
        
        future_rates = []
        for i in range(months):
            row = future_data.iloc[i]
            features = np.array([
                last_close, last_close, np.std(future_rates[-5:]) if len(future_rates) >= 5 else 0,
                row['day_of_week'], row['month'], row['day_of_year']
            ]).reshape(1, -1)
            
            predicted_rate = self.model.predict(features)[0]
            future_rates.append(predicted_rate)
            last_close = predicted_rate
        
        future_data['predicted_rate'] = future_rates
        return future_data


def main():
    st.set_page_config(page_title="MXN-USD Exchange Rate Prediction", layout="wide")
    
    st.title("MXN-USD Exchange Rate Prediction for 2025")
    st.markdown("This app predicts the MXN-USD exchange rate using a linear regression model. By Rob Ugalde")
    
    lookback_days = 365 * 3
    train_size = 0.8
    
    predictor = ExchangeRatePredictor(lookback_days=lookback_days)
    
    with st.spinner("Training model..."):
        metrics, actual_rates, predictions = predictor.train_model(train_size=train_size)
        future_forecast = predictor.forecast_future(months=12)
    
    # Plot historical vs predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_rates.index, y=actual_rates.values,
        name="Actual Rate", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=predictions.index, y=predictions['predicted_rate'],
        name="Predicted Rate", line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=future_forecast.index, y=future_forecast['predicted_rate'],
        name="Forecasted Rate (2025)", line=dict(color="green", dash='dash')
    ))
    
    fig.update_layout(
        title="MXN-USD Exchange Rate Prediction and 12-Month Forecast",
        xaxis_title="Date",
        yaxis_title="Exchange Rate",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.header("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Squared Error", f"{metrics['MSE']:.2f}")
    col2.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
    col3.metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f}")
    col4.metric("R² Score", f"{metrics['R²']:.4f}")
    
    st.header("12-Month Exchange Rate Forecast")
    st.dataframe(future_forecast[['predicted_rate']].rename(columns={'predicted_rate': 'Forecasted Rate'}))
    
    st.download_button(
        label="Download Forecast Results",
        data=future_forecast.to_csv().encode('utf-8'),
        file_name='mxn_usd_forecast_2025.csv',
        mime='text/csv'
    )


if __name__ == "__main__":
    main()

