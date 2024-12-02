import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Define stock symbols and benchmark symbol
stock_symbols = ['ITC.NS', 'TCS.NS', 'WIPRO.NS']
benchmark_symbol = '^NSEI'

# Fetch data from Yahoo Finance
data = yf.download(stock_symbols + [benchmark_symbol], start="2015-01-01", end="2024-12-01", group_by='ticker')

# Function to calculate returns
def calculate_returns(data, freq='1d'):
    returns = data.resample(freq).ffill().pct_change()  # Adjust to the desired frequency
    return returns

# Function to calculate metrics
def calculate_metrics(returns, benchmark_returns):
    # Annualized Alpha (Alpha = actual return - (Beta * benchmark return))
    covariance_matrix = np.cov(returns, benchmark_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    alpha = np.mean(returns) - beta * np.mean(benchmark_returns)

    # Annualized Volatility (Standard deviation * sqrt(252) for daily data)
    annualized_volatility = returns.std() * np.sqrt(252)

    # Sharpe Ratio (assuming risk-free rate of 0)
    sharpe_ratio = returns.mean() / returns.std()

    # Maximum Drawdown (Peak to trough)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # R-Squared (Squared correlation)
    r_squared = np.corrcoef(returns, benchmark_returns)[0, 1] ** 2

    # Downside Deviation (standard deviation of negative returns)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std()

    # Value at Risk (VaR) at 95% confidence level
    var_95 = np.percentile(returns, 5)  # 5th percentile for VaR at 95%

    return {
        'Alpha': alpha,
        'Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'R-Squared': r_squared,
        'Downside Deviation': downside_deviation,
        'VaR (95%)': var_95
    }

# Get the adjusted closing prices and calculate returns
stock_data = {symbol: data[symbol]['Adj Close'] for symbol in stock_symbols}
benchmark_data = data[benchmark_symbol]['Adj Close']

# Calculate returns for different timeframes
daily_returns = {symbol: calculate_returns(stock_data[symbol], freq='1d') for symbol in stock_symbols}
monthly_returns = {symbol: calculate_returns(stock_data[symbol], freq='M') for symbol in stock_symbols}
minute_returns = {symbol: calculate_returns(stock_data[symbol], freq='T') for symbol in stock_symbols}

benchmark_daily_returns = calculate_returns(benchmark_data, freq='1d')
benchmark_monthly_returns = calculate_returns(benchmark_data, freq='M')
benchmark_minute_returns = calculate_returns(benchmark_data, freq='T')

# Calculate metrics for each stock and display results
metrics = {symbol: calculate_metrics(daily_returns[symbol], benchmark_daily_returns) for symbol in stock_symbols}
monthly_metrics = {symbol: calculate_metrics(monthly_returns[symbol], benchmark_monthly_returns) for symbol in stock_symbols}
minute_metrics = {symbol: calculate_metrics(minute_returns[symbol], benchmark_minute_returns) for symbol in stock_symbols}

# Streamlit Dashboard
st.title("Stock Performance Dashboard")

st.sidebar.header("Select Stock and Frequency")
selected_stock = st.sidebar.selectbox("Select a stock", stock_symbols)
selected_frequency = st.sidebar.selectbox("Select frequency", ['Daily', 'Monthly', 'Minute'])

# Display selected stock metrics based on frequency
if selected_frequency == 'Daily':
    selected_metrics = metrics[selected_stock]
    st.header(f"{selected_stock} Daily Metrics")
elif selected_frequency == 'Monthly':
    selected_metrics = monthly_metrics[selected_stock]
    st.header(f"{selected_stock} Monthly Metrics")
else:
    selected_metrics = minute_metrics[selected_stock]
    st.header(f"{selected_stock} Minute Metrics")

# Display metrics in a table
metrics_df = pd.DataFrame(list(selected_metrics.items()), columns=["Metric", "Value"])
st.dataframe(metrics_df)

# Optional: Plot volatility comparison
if selected_frequency == 'Daily':
    volatility_data = {symbol: metrics[symbol]['Volatility'] for symbol in stock_symbols}
elif selected_frequency == 'Monthly':
    volatility_data = {symbol: monthly_metrics[symbol]['Volatility'] for symbol in stock_symbols}
else:
    volatility_data = {symbol: minute_metrics[symbol]['Volatility'] for symbol in stock_symbols}

volatility_df = pd.DataFrame(volatility_data, index=[selected_frequency])
st.subheader(f"Volatility Comparison ({selected_frequency})")
st.bar_chart(volatility_df.T)

# Plot a comparison of Sharpe Ratios for all stocks
sharpe_ratios = {symbol: selected_metrics['Sharpe Ratio'] for symbol in stock_symbols}
sharpe_df = pd.DataFrame(sharpe_ratios.items(), columns=["Stock", "Sharpe Ratio"])
st.subheader(f"Sharpe Ratio Comparison")
st.bar_chart(sharpe_df.set_index("Stock")["Sharpe Ratio"])

# Optional: Show Value at Risk (VaR)
st.subheader("Value at Risk (VaR) at 95% Confidence Level")
var_data = {symbol: selected_metrics['VaR (95%)'] for symbol in stock_symbols}
var_df = pd.DataFrame(var_data.items(), columns=["Stock", "VaR (95%)"])
st.dataframe(var_df)
