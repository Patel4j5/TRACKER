# Import matplotlib and set non-interactive backend FIRST
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend to avoid GUI issues

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
import io
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
from flask_cors import CORS  # Add CORS support

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create a secret key for the application
app.config['SECRET_KEY'] = 'STARK'  # Change this to a random string

# Create directory for portfolio data if it doesn't exist
os.makedirs('data', exist_ok=True)
PORTFOLIO_FILE = 'data/portfolio.json'

# Initialize portfolio file if it doesn't exist
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump({
            'stocks': {},
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }, f)

# Default parameters
DEFAULT_START_DATE = '2020-01-01'
MOVING_AVERAGE_WINDOW = 5
FUTURE_DAYS = 7
TRAIN_LAST_N_DAYS = 90  # how many days to train on

def download_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        return data, None
    except Exception as e:
        return None, str(e)

def filter_recent(data, days=TRAIN_LAST_N_DAYS):
    """Filter the last days of data."""
    cutoff_date = data.index.max() - pd.Timedelta(days=days)
    return data[data.index >= cutoff_date]

def moving_average_prediction(data, window=MOVING_AVERAGE_WINDOW):
    data['MA_Predicted_Close'] = data['Close'].rolling(window=window).mean()
    return data

def linear_regression_prediction(data):
    data = data.copy()
    data['Days'] = np.arange(len(data))

    recent_data = filter_recent(data)
    recent_data = recent_data.dropna()

    X = recent_data[['Days']]
    y = recent_data['Close']

    model = LinearRegression()
    model.fit(X, y)

    data['LR_Predicted_Close'] = model.predict(data[['Days']])

    return data, model

def polynomial_regression_prediction(data, degree=2):
    data = data.copy()
    data['Days'] = np.arange(len(data))

    recent_data = filter_recent(data)
    recent_data = recent_data.dropna()

    X = recent_data[['Days']]
    y = recent_data['Close']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_all = poly.transform(data[['Days']])
    data['Poly_Predicted_Close'] = model.predict(X_all)

    return data, model

def random_forest_prediction(data):
    data = data.copy()
    data['Days'] = np.arange(len(data))

    recent_data = filter_recent(data)
    recent_data = recent_data.dropna()

    X = recent_data[['Days']]
    y = recent_data['Close']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y.values.ravel())

    data['RF_Predicted_Close'] = model.predict(data[['Days']])

    return data, model

def blended_future_prediction(data, future_days=FUTURE_DAYS):
    cols = ['Close', 'LR_Predicted_Close', 'Poly_Predicted_Close', 'RF_Predicted_Close']
    data['Blended_Average'] = data[cols].mean(axis=1)

    last_14 = data['Blended_Average'].dropna().iloc[-14:]

    X = np.arange(14).reshape(-1, 1)
    y = last_14.values

    model = LinearRegression()
    model.fit(X, y)

    X_future = np.arange(14, 14 + future_days).reshape(-1, 1)
    y_future = model.predict(X_future)

    y_future = np.asarray(y_future).flatten()

    # Fix the float() warning by using .iloc[0]
    last_close = float(data['Close'].dropna().iloc[-1].iloc[0] if isinstance(data['Close'].dropna().iloc[-1], pd.Series) else data['Close'].dropna().iloc[-1])
    adjustment = last_close - float(y_future[0])
    y_future = y_future + adjustment

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'Blended_Future_Predicted_Close': y_future}, index=future_dates)

    return future_df

def add_confidence_bands(future_df):
    days = np.arange(len(future_df))
    percentages = 0.01 + (0.05 - 0.01) * (days / (len(future_df) - 1))  # Linear growth from 1% to 5%

    future_df['Upper_Band'] = future_df['Blended_Future_Predicted_Close'] * (1 + percentages)
    future_df['Lower_Band'] = future_df['Blended_Future_Predicted_Close'] * (1 - percentages)

    return future_df

def evaluate_predictions(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return mae, r2

def create_plot(data, future_df, ticker):
    # Set up the plot based on data
    current_year = datetime.now().year
    start_date = f'{current_year}-01-01'
    
    # Filter data for the current year
    data_plot = data[data.index >= start_date]
    if data_plot.empty:
        data_plot = data.iloc[-90:]  # If no current year data, use last 90 days
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data_plot.index, data_plot['Close'], label='Actual Close', color='blue')
    ax.plot(data_plot.index, data_plot['MA_Predicted_Close'], label=f'{MOVING_AVERAGE_WINDOW}-Day Moving Average', linestyle='--', color='orange')
    ax.plot(data_plot.index, data_plot['LR_Predicted_Close'], label='Linear Regression', linestyle=':', color='green')
    ax.plot(data_plot.index, data_plot['Poly_Predicted_Close'], label='Polynomial Regression', linestyle='-.', color='red')
    ax.plot(data_plot.index, data_plot['RF_Predicted_Close'], label='Random Forest', linestyle='--', color='purple')
    
    # Plot future predictions
    ax.plot(future_df.index, future_df['Blended_Future_Predicted_Close'], label='Blended Future', linestyle=':', color='darkviolet')
    ax.fill_between(future_df.index, future_df['Lower_Band'], future_df['Upper_Band'], color='violet', alpha=0.2, label='Confidence Band')
    
    # Set plot styling
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title(f'{ticker} Stock Prediction', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Price (USD)', fontsize=14)
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)  # Explicitly close the figure to prevent leaks
    
    return plot_url

def get_metrics(data):
    metrics = {}
    
    # Moving Average metrics
    ma_data = data.dropna()
    ma_mae, ma_r2 = evaluate_predictions(ma_data['Close'], ma_data['MA_Predicted_Close'])
    metrics['ma'] = {'mae': round(ma_mae, 2), 'r2': round(ma_r2, 2)}
    
    # Linear Regression metrics
    lr_data = data.dropna()
    lr_mae, lr_r2 = evaluate_predictions(lr_data['Close'], lr_data['LR_Predicted_Close'])
    metrics['lr'] = {'mae': round(lr_mae, 2), 'r2': round(lr_r2, 2)}
    
    # Polynomial Regression metrics
    poly_data = data.dropna()
    poly_mae, poly_r2 = evaluate_predictions(poly_data['Close'], poly_data['Poly_Predicted_Close'])
    metrics['poly'] = {'mae': round(poly_mae, 2), 'r2': round(poly_r2, 2)}
    
    # Random Forest metrics
    rf_data = data.dropna()
    rf_mae, rf_r2 = evaluate_predictions(rf_data['Close'], rf_data['RF_Predicted_Close'])
    metrics['rf'] = {'mae': round(rf_mae, 2), 'r2': round(rf_r2, 2)}
    
    return metrics

def get_price_data(data, future_df):
    # Get the latest price - fixing the floating point issue
    latest_price = data['Close'].iloc[-1]
    if isinstance(latest_price, pd.Series):
        latest_price = latest_price.iloc[0]
    
    # Get the predicted price (15 days ahead)
    future_price = future_df['Blended_Future_Predicted_Close'].iloc[-1]
    future_upper = future_df['Upper_Band'].iloc[-1]
    future_lower = future_df['Lower_Band'].iloc[-1]
    
    # Calculate percentage change
    percent_change = ((future_price / latest_price) - 1) * 100
    
    return {
        'latest_price': round(latest_price, 2),
        'future_price': round(future_price, 2),
        'upper_band': round(future_upper, 2),
        'lower_band': round(future_lower, 2),
        'percent_change': round(percent_change, 2)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', 'AAPL').upper()
    
    # Calculate end date (today + 15 days)
    today = datetime.now()
    end_date = (today + timedelta(days=FUTURE_DAYS + 10)).strftime('%Y-%m-%d')
    
    # Download data
    data, error = download_data(ticker, DEFAULT_START_DATE, end_date)
    
    if error:
        return jsonify({'error': f"Error downloading data: {error}"})
    
    if data.empty:
        return jsonify({'error': "No data found for this ticker."})
    
    try:
        # Apply models
        data = moving_average_prediction(data)
        data, _ = linear_regression_prediction(data)
        data, _ = polynomial_regression_prediction(data)
        data, _ = random_forest_prediction(data)
        
        # Generate future predictions
        future_df = blended_future_prediction(data)
        future_df = add_confidence_bands(future_df)
        
        # Create plot
        plot_url = create_plot(data, future_df, ticker)
        
        # Get metrics
        metrics = get_metrics(data)
        
        # Get price data
        price_data = get_price_data(data, future_df)
        
        return jsonify({
            'plot_url': plot_url,
            'metrics': metrics,
            'price_data': price_data,
            'ticker': ticker
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error processing data: {str(e)}"})

# Portfolio Management Routes
@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    """Get the current portfolio data"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)
        else:
            portfolio = {
                'stocks': {},
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
        
        # Update current prices and calculate growth
        updated_portfolio = update_portfolio_prices(portfolio)
        
        return jsonify(updated_portfolio)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Add to existing portfolio routes
@app.route('/portfolio/analysis', methods=['GET'])
def portfolio_analysis():
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)
        
        # Get updated portfolio with predictions
        portfolio = update_portfolio_prices(portfolio)
        
        # Calculate 7-day portfolio prediction
        total_predicted_value = 0
        individual_predictions = []
        
        for ticker, stock in portfolio['stocks'].items():
            if stock.get('future_price'):
                predicted_value = stock['shares'] * stock['future_price']
                predicted_growth = stock['future_growth']
                
                individual_predictions.append({
                    'ticker': ticker,
                    'current_value': stock['current_value'],
                    'predicted_value': predicted_value,
                    'predicted_growth': predicted_growth,
                    'confidence': 3  # Default confidence level
                })
                
                total_predicted_value += predicted_value
        
        # Calculate portfolio-level metrics
        current_value = portfolio['total_current_value']
        predicted_growth = ((total_predicted_value / current_value) - 1) * 100 if current_value else 0
        
        # Generate suggestions
        suggestions = []
        avg_growth = sum(p['predicted_growth'] for p in individual_predictions) / len(individual_predictions) if individual_predictions else 0
        
        for stock in individual_predictions:
            if stock['predicted_growth'] > (avg_growth + 2):
                suggestions.append({
                    'ticker': stock['ticker'],
                    'action': 'buy',
                    'reason': f"Expected growth ({stock['predicted_growth']:.1f}%) above portfolio average",
                    'confidence': 3
                })
            elif stock['predicted_growth'] < (avg_growth - 1):
                suggestions.append({
                    'ticker': stock['ticker'],
                    'action': 'sell',
                    'reason': f"Expected growth ({stock['predicted_growth']:.1f}%) below portfolio average",
                    'confidence': 2
                })
        
        return jsonify({
            'current_value': round(current_value, 2),
            'predicted_value': round(total_predicted_value, 2),
            'predicted_growth': round(predicted_growth, 2),
            'suggestions': suggestions,
            'predictions': individual_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_rebalancing_suggestions(predictions, portfolio_growth):
    suggestions = []
    avg_growth = sum(p['predicted_growth'] for p in predictions) / len(predictions) if predictions else 0
    
    for stock in predictions:
        # Buy suggestion if predicted growth > portfolio average + threshold
        if stock['predicted_growth'] > (avg_growth + 2):
            suggestions.append({
                'ticker': stock['ticker'],
                'action': 'buy',
                'reason': f"Expected growth ({stock['predicted_growth']:.1f}%) significantly above portfolio average",
                'confidence': stock['confidence']
            })
        # Sell suggestion if predicted growth < portfolio average - threshold
        elif stock['predicted_growth'] < (avg_growth - 1):
            suggestions.append({
                'ticker': stock['ticker'],
                'action': 'sell',
                'reason': f"Expected growth ({stock['predicted_growth']:.1f}%) below portfolio average",
                'confidence': stock['confidence']
            })
    
    return suggestions


@app.route('/portfolio/add', methods=['POST'])
def add_stock():
    """Add a stock to the portfolio"""
    try:
        ticker = request.form.get('ticker', '').upper()
        shares = float(request.form.get('shares', 0))
        equity = float(request.form.get('equity', 0))

        if not ticker or shares <= 0 or equity <= 0:
            return jsonify({'error': 'Invalid input. Ticker, shares, and equity are required.'}), 400

        cost_basis = equity / shares

        # Get current data for the stock
        today = datetime.now()
        end_date = today.strftime('%Y-%m-%d')
        start_date = (today - timedelta(days=5)).strftime('%Y-%m-%d')

        data, error = download_data(ticker, start_date, end_date)

        if error or data.empty:
            return jsonify({'error': f"Could not find data for ticker: {ticker}"}), 404

        # Current price
        current_price = data['Close'].iloc[-1]
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]

        # Load current portfolio
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)

        # Add or update stock
        portfolio['stocks'][ticker] = {
            'ticker': ticker,
            'shares': shares,
            'equity': round(equity, 2),
            'cost_basis': round(cost_basis, 2),
            'current_price': round(current_price, 2),
            'current_value': round(shares * current_price, 2),
            'growth': round(((current_price / cost_basis) - 1) * 100, 2),
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }

        # Update portfolio lastUpdated
        portfolio['last_updated'] = datetime.now().strftime('%Y-%m-%d')

        # Save portfolio
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f)

        return jsonify({'success': True, 'portfolio': portfolio})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/portfolio/remove', methods=['POST'])
def remove_stock():
    """Remove a stock from the portfolio"""
    try:
        ticker = request.form.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required.'}), 400
        
        # Load current portfolio
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)
        
        # Remove stock if it exists
        if ticker in portfolio['stocks']:
            del portfolio['stocks'][ticker]
            
            # Update portfolio lastUpdated
            portfolio['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            
            # Save portfolio
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(portfolio, f)
            
            return jsonify({'success': True, 'portfolio': portfolio})
        else:
            return jsonify({'error': f"Ticker {ticker} not found in portfolio."}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/portfolio/update', methods=['POST'])
def update_stock():
    """Update shares or cost basis for a stock"""
    try:
        ticker = request.form.get('ticker', '').upper()
        shares = request.form.get('shares')
        cost_basis = request.form.get('cost_basis')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required.'}), 400
        
        # Load current portfolio
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)
        
        # Update stock if it exists
        if ticker in portfolio['stocks']:
            stock = portfolio['stocks'][ticker]
            
            if shares:
                stock['shares'] = float(shares)
            
            if cost_basis:
                stock['cost_basis'] = float(cost_basis)
            
            # Recalculate values
            current_price = stock['current_price']
            stock['equity'] = round(stock['shares'] * stock['cost_basis'], 2)
            stock['current_value'] = round(stock['shares'] * current_price, 2)
            stock['growth'] = round(((current_price / stock['cost_basis']) - 1) * 100, 2)
            
            # Update portfolio lastUpdated
            portfolio['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            
            # Save portfolio
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(portfolio, f)
            
            return jsonify({'success': True, 'portfolio': portfolio})
        else:
            return jsonify({'error': f"Ticker {ticker} not found in portfolio."}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_portfolio_prices(portfolio):
    """Update current prices for all stocks in portfolio"""
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=5)).strftime('%Y-%m-%d')
    
    total_equity = 0
    total_current_value = 0
    
    for ticker, stock in portfolio['stocks'].items():
        try:
            data, error = download_data(ticker, start_date, end_date)
            
            if not error and not data.empty:
                current_price = data['Close'].iloc[-1]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
                
                # Try to get future prediction
                try:
                    # Calculate future price
                    future_data, _ = download_data(ticker, DEFAULT_START_DATE, 
                                                 (today + timedelta(days=FUTURE_DAYS + 10)).strftime('%Y-%m-%d'))
                    
                    if not future_data.empty:
                        future_data = moving_average_prediction(future_data)
                        future_data, _ = linear_regression_prediction(future_data)
                        future_data, _ = polynomial_regression_prediction(future_data)
                        future_data, _ = random_forest_prediction(future_data)
                        
                        future_df = blended_future_prediction(future_data)
                        future_df = add_confidence_bands(future_df)
                        
                        future_price = future_df['Blended_Future_Predicted_Close'].iloc[-1]
                        
                        stock['future_price'] = round(future_price, 2)
                        stock['future_growth'] = round(((future_price / current_price) - 1) * 100, 2)
                    else:
                        stock['future_price'] = None
                        stock['future_growth'] = None
                except Exception as e:
                    print(f"Error predicting future price for {ticker}: {str(e)}")
                    stock['future_price'] = None
                    stock['future_growth'] = None
                
                stock['current_price'] = round(current_price, 2)
                stock['current_value'] = round(stock['shares'] * current_price, 2)
                stock['growth'] = round(((current_price / stock['cost_basis']) - 1) * 100, 2)
            
            total_equity += stock['equity']
            total_current_value += stock['current_value']
            
        except Exception as e:
            print(f"Error updating {ticker}: {str(e)}")
    
    # Calculate total portfolio growth
    portfolio_growth = 0
    if total_equity > 0:
        portfolio_growth = round(((total_current_value / total_equity) - 1) * 100, 2)
    
    portfolio['total_equity'] = round(total_equity, 2)
    portfolio['total_current_value'] = round(total_current_value, 2)
    portfolio['portfolio_growth'] = portfolio_growth
    portfolio['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    
    # Save updated portfolio
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f)
    
    return portfolio

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change 5001 to any available port
