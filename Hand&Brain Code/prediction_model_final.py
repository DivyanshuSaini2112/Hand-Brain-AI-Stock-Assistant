import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
import time
import random
import ta

# ===== DATA COLLECTION FUNCTIONS =====

def fetch_stock_data(ticker, period='2y', interval='1d'):
    """Fetch historical price data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None

def fetch_fundamental_data(ticker):
    """Fetch fundamental data for a stock"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        
        # Get financials
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get quarterly results
        quarterly_financials = stock.quarterly_financials
        
        # Extract key fundamental metrics
        fundamentals = {}
        
        # Market metrics
        fundamentals['marketCap'] = info.get('marketCap', np.nan)
        fundamentals['trailingPE'] = info.get('trailingPE', np.nan)
        fundamentals['forwardPE'] = info.get('forwardPE', np.nan)
        fundamentals['priceToBook'] = info.get('priceToBook', np.nan)
        fundamentals['dividendYield'] = info.get('dividendYield', np.nan) if info.get('dividendYield') else 0
        fundamentals['beta'] = info.get('beta', np.nan)
        
        # Calculate additional ratios if available
        if not income_stmt.empty and not balance_sheet.empty:
            try:
                # Get the most recent data
                latest_income = income_stmt.iloc[:, 0]
                latest_balance = balance_sheet.iloc[:, 0]
                
                # Calculate debt to equity
                total_debt = latest_balance.get('Total Debt', np.nan)
                total_equity = latest_balance.get('Total Stockholder Equity', np.nan)
                if not pd.isna(total_debt) and not pd.isna(total_equity) and total_equity != 0:
                    fundamentals['debtToEquity'] = total_debt / total_equity
                
                # Calculate ROE
                net_income = latest_income.get('Net Income', np.nan)
                if not pd.isna(net_income) and not pd.isna(total_equity) and total_equity != 0:
                    fundamentals['ROE'] = net_income / total_equity
                
                # Calculate profit margin
                total_revenue = latest_income.get('Total Revenue', np.nan)
                if not pd.isna(net_income) and not pd.isna(total_revenue) and total_revenue != 0:
                    fundamentals['profitMargin'] = net_income / total_revenue
                
                # Calculate revenue growth if we have at least 2 years of data
                if income_stmt.shape[1] >= 2:
                    current_revenue = income_stmt.iloc[:, 0].get('Total Revenue', np.nan)
                    prev_revenue = income_stmt.iloc[:, 1].get('Total Revenue', np.nan)
                    if not pd.isna(current_revenue) and not pd.isna(prev_revenue) and prev_revenue != 0:
                        fundamentals['revenueGrowth'] = (current_revenue - prev_revenue) / prev_revenue
            except Exception as e:
                print(f"Error calculating ratios: {e}")
        
        return fundamentals
    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return {}

def create_technical_features(df):
    """Add technical indicators to the dataframe"""
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Add volume-based features if volume data is available
    if 'Volume' in data.columns:
        # Volume change
        data['volume_change'] = data['Volume'].pct_change()
        # Volume moving averages
        data['volume_ma5'] = data['Volume'].rolling(window=5).mean()
        data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
        # Relative volume (compared to 20-day average)
        data['relative_volume'] = data['Volume'] / data['volume_ma20']
    
    # Price-based features
    # Moving averages
    data['ma5'] = data['Close'].rolling(window=5).mean()
    data['ma20'] = data['Close'].rolling(window=20).mean()
    data['ma50'] = data['Close'].rolling(window=50).mean()
    data['ma200'] = data['Close'].rolling(window=200).mean()
    
    # Moving average crossovers
    data['ma5_cross_ma20'] = (data['ma5'] > data['ma20']).astype(int)
    data['ma50_cross_ma200'] = (data['ma50'] > data['ma200']).astype(int)
    
    # Price changes
    data['daily_return'] = data['Close'].pct_change()
    data['weekly_return'] = data['Close'].pct_change(5)
    data['monthly_return'] = data['Close'].pct_change(20)
    
    # Volatility measures
    data['volatility_5d'] = data['daily_return'].rolling(window=5).std()
    data['volatility_20d'] = data['daily_return'].rolling(window=20).std()
    
    # Add RSI (Relative Strength Index)
    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # Add MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['bollinger_mavg'] = bollinger.bollinger_mavg()
    data['bollinger_hband'] = bollinger.bollinger_hband()
    data['bollinger_lband'] = bollinger.bollinger_lband()
    data['bollinger_width'] = (data['bollinger_hband'] - data['bollinger_lband']) / data['bollinger_mavg']
    
    # Add ATR (Average True Range) - a volatility indicator
    data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    
    # Distance from 52-week high and low
    data['52w_high'] = data['Close'].rolling(window=252).max()
    data['52w_low'] = data['Close'].rolling(window=252).min()
    data['pct_off_52w_high'] = (data['Close'] - data['52w_high']) / data['52w_high']
    data['pct_off_52w_low'] = (data['Close'] - data['52w_low']) / data['52w_low']
    
    return data

def add_market_data(df, ticker, market_index='^NSEI'):
    """Add market index data for comparison"""
    try:
        # Fetch market index data (default to Nifty 50 for Indian stocks)
        market_data = fetch_stock_data(market_index, period='2y')
        if market_data is None:
            print(f"Could not fetch market data for {market_index}")
            return df
        
        # Rename columns to avoid confusion
        market_data = market_data[['Close']].rename(columns={'Close': 'market_close'})
        
        # Merge with stock data on date index
        merged_data = pd.merge(df, market_data, left_index=True, right_index=True, how='left')
        
        # Calculate relative performance metrics
        merged_data['market_return'] = merged_data['market_close'].pct_change()
        merged_data['relative_return'] = merged_data['daily_return'] - merged_data['market_return']
        merged_data['relative_strength'] = (merged_data['Close'] / merged_data['Close'].iloc[0]) / (merged_data['market_close'] / merged_data['market_close'].iloc[0])
        
        # Calculate correlation with market
        merged_data['market_correlation'] = merged_data['daily_return'].rolling(window=20).corr(merged_data['market_return'])
        
        # Calculate beta (market sensitivity)
        merged_data['beta_20d'] = merged_data['daily_return'].rolling(window=20).cov(merged_data['market_return']) / merged_data['market_return'].rolling(window=20).var()
        
        return merged_data
    except Exception as e:
        print(f"Error adding market data: {e}")
        return df

# ===== ML MODEL DEFINITION =====

class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(StockPredictionModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

# ===== DATA PREPARATION FUNCTIONS =====

def prepare_model_data(df, target_col='Close', sequence_length=20, forecast_horizon=5, fundamental_data=None):
    """Prepare data for the ML model including features and targets"""
    # Make a copy of the dataframe
    data = df.copy()
    
    # Add fundamental data if available
    if fundamental_data:
        for key, value in fundamental_data.items():
            data[key] = value  # Add as constant columns
    
    # Drop rows with NaN values
    data = data.dropna()
    
    if len(data) <= sequence_length + forecast_horizon:
        print("Not enough data after preprocessing")
        return None, None, None
    
    # Save the target column for later
    target_values = data[target_col].values
    
    # Drop the date index temporarily for scaling
    data_values = data.values
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # Create target scaler separately for inverse transform later
    target_scaler = MinMaxScaler()
    target_scaler.fit(target_values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length - forecast_horizon + 1):
        # Input sequence
        X.append(data_scaled[i:i+sequence_length])
        # Target is n days in the future
        y.append(data_scaled[i+sequence_length+forecast_horizon-1, data.columns.get_loc(target_col)])
    
    if not X or not y:
        print("Failed to create sequences")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), (scaler, target_scaler, data.columns.get_loc(target_col))

# ===== VISUALIZATION FUNCTIONS =====

def create_sentiment_gauge(ax, sentiment_score, ticker):
    """Create a semicircle gauge visualization for stock sentiment"""
    # Create semicircle background
    theta1, theta2 = 180, 0
    arc = Arc((0.5, 0), 0.8, 0.8, angle=0, theta1=theta1, theta2=theta2, color='black', lw=2)
    ax.add_patch(arc)
    
    # Create colored regions
    angles = np.linspace(np.pi, 0, 100)
    sentiment_colors = ['darkred', 'red', 'yellow', 'lightgreen', 'darkgreen']
    for i in range(5):
        sector_angles = np.linspace(np.pi - i*np.pi/5, np.pi - (i+1)*np.pi/5, 20)
        x = 0.5 + 0.4 * np.cos(sector_angles)
        y = 0 + 0.4 * np.sin(sector_angles)
        ax.fill(np.append(x, 0.5), np.append(y, 0), color=sentiment_colors[i], alpha=0.7)
    
    # Add labels
    sentiment_labels = ["Strong\nSell", "Sell", "Hold", "Buy", "Strong\nBuy"]
    for i, label in enumerate(sentiment_labels):
        angle = np.pi - (i + 0.5) * np.pi / 5
        x = 0.5 + 0.55 * np.cos(angle)
        y = 0 + 0.55 * np.sin(angle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add needle
    normalized_score = (sentiment_score + 1) / 2  # Convert from [-1, 1] to [0, 1]
    needle_angle = np.pi - normalized_score * np.pi
    x = 0.5 + 0.4 * np.cos(needle_angle)
    y = 0 + 0.4 * np.sin(needle_angle)
    ax.plot([0.5, x], [0, y], 'k-', lw=3)
    
    # Add center circle
    circle = plt.Circle((0.5, 0), 0.03, color='black')
    ax.add_patch(circle)
    
    # Add ticker and prediction labels
    ax.text(0.5, -0.2, f"Sentiment for {ticker}", ha='center', fontsize=12, fontweight='bold')
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.6)
    ax.axis('off')

def visualize_predictions(data, predictions, ticker, target_price, days_to_target, current_price, sentiment_score):
    """Create visualizations for stock predictions"""
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Price prediction chart (top)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    # Plot historical prices
    historical_dates = data.index[-30:]  # Last 30 days
    historical_prices = data['Close'].values[-30:]
    ax1.plot(historical_dates, historical_prices, 'b-', label='Historical Prices')
    
    # Plot prediction
    future_dates = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, len(predictions)+1)]
    ax1.plot(future_dates, predictions, 'r--', label='Predicted Prices')
    
    # Highlight target price
    target_date = data.index[-1] + datetime.timedelta(days=int(days_to_target))
    ax1.scatter([target_date], [target_price], color='green', s=100, zorder=5)
    ax1.annotate(f'Target: {target_price:.2f}', 
                 (target_date, target_price),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=12, fontweight='bold')
    
    # Add labels and formatting
    ax1.set_title(f'{ticker} Price Prediction', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add current price line
    ax1.axhline(y=current_price, color='gray', linestyle='--', alpha=0.7)
    ax1.text(historical_dates[0], current_price, f'Current: {current_price:.2f}', 
             va='bottom', ha='left', fontsize=10)
    
    # Subplot 2: Sentiment gauge (bottom-left)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    create_sentiment_gauge(ax2, sentiment_score, ticker)
    
    # Subplot 3: Key metrics table (bottom-right)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis('off')
    
    # Calculate metrics
    price_change = (target_price - current_price) / current_price * 100
    
    # Create a table of key metrics
    metrics = [
        ['Metric', 'Value'],
        ['Current Price', f'{current_price:.2f}'],
        ['Target Price', f'{target_price:.2f}'],
        ['Projected Change', f'{price_change:.2f}%'],
        ['Time Horizon', f'{days_to_target} days'],
        ['Target Date', f'{target_date.strftime("%Y-%m-%d")}']
    ]
    
    # Add recommendations based on sentiment score
    if sentiment_score > 0.6:
        action = 'Strong Buy'
    elif sentiment_score > 0.2:
        action = 'Buy'
    elif sentiment_score > -0.2:
        action = 'Hold'
    elif sentiment_score > -0.6:
        action = 'Sell'
    else:
        action = 'Strong Sell'
    
    metrics.append(['Recommendation', action])
    
    # Convert metrics to table
    cell_text = [row for row in metrics]
    ax3.table(cellText=cell_text, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    ax3.set_title('Prediction Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    return fig

# ===== MAIN PREDICTION FUNCTION =====

def predict_stock(ticker, forecast_days=30):
    print(f"\nAnalyzing {ticker}...")
    
    # Check if it's an Indian stock and append .NS if needed
    if not (ticker.endswith('.NS') or ticker.endswith('.BO')) and ticker not in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX']:
        ticker_ns = f"{ticker}.NS"
        data_ns = fetch_stock_data(ticker_ns)
        
        if data_ns is None or data_ns.empty:
            ticker_bo = f"{ticker}.BO"
            data_bo = fetch_stock_data(ticker_bo)
            
            if data_bo is None or data_bo.empty:
                data = fetch_stock_data(ticker)
                if data is None or data.empty:
                    print(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
                    return None
                ticker_to_use = ticker
            else:
                data = data_bo
                ticker_to_use = ticker_bo
        else:
            data = data_ns
            ticker_to_use = ticker_ns
    else:
        data = fetch_stock_data(ticker)
        if data is None or data.empty:
            print(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
            return None
        ticker_to_use = ticker
    
    if len(data) < 100:
        print(f"Not enough historical data for {ticker_to_use} to make reliable predictions.")
        return None
    
    current_price = data['Close'].iloc[-1]
    currency = "₹" if '.NS' in ticker_to_use or '.BO' in ticker_to_use else "$"
    print(f"Current Price: {currency}{current_price:.2f}")
    
    print("Fetching fundamental data...")
    fundamental_data = fetch_fundamental_data(ticker_to_use)
    
    print("Calculating technical indicators...")
    data_with_features = create_technical_features(data)
    
    market_index = '^NSEI' if '.NS' in ticker_to_use or '.BO' in ticker_to_use else '^GSPC'
    data_with_features = add_market_data(data_with_features, ticker_to_use, market_index)
    
    print("\nFundamental Metrics:")
    for key, value in fundamental_data.items():
        if not pd.isna(value):
            if key in ['marketCap']:
                formatted_value = f"{value/1e9:.2f}B"
            elif key in ['dividendYield', 'profitMargin', 'ROE', 'revenueGrowth']:
                formatted_value = f"{value*100:.2f}%"
            else:
                formatted_value = f"{value:.2f}"
            print(f"- {key}: {formatted_value}")
    
    sequence_length = 30
    forecast_horizon = 5
    
    print("\nPreparing data for model...")
    X, y, scalers = prepare_model_data(
        data_with_features,
        target_col='Close',
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        fundamental_data=fundamental_data
    )
    
    if X is None or y is None:
        print("Failed to prepare data for modeling.")
        return None
    
    scaler, target_scaler, target_idx = scalers
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    
    input_size = X.shape[2]
    model = StockPredictionModel(input_size=input_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)  # Suppress deprecated warning
    
    num_epochs = 100
    batch_size = 32
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    print("\nTraining model...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        indices = torch.randperm(len(X_train))
        for start_idx in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            batch_indices = indices[start_idx:start_idx+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        last_sequence = X[-1:].to(device)
        future_predictions = []
        curr_seq = last_sequence.clone()  # Initial clone to avoid modifying original
        
        for _ in range(forecast_days):
            next_pred = model(curr_seq).item()
            future_predictions.append(next_pred)
            
            # Create a new sequence for the next iteration
            new_seq = curr_seq.clone()  # Clone to avoid in-place issues
            new_seq[0, :-1, :] = curr_seq[0, 1:, :]  # Shift left
            new_row = curr_seq[0, -1, :].clone()  # Clone the last row
            new_row[target_idx] = next_pred  # Update the target value
            new_seq[0, -1, :] = new_row  # Assign the new row
            curr_seq = new_seq  # Update curr_seq for next iteration
    
    scaled_preds = np.array(future_predictions).reshape(-1, 1)
    dummy_array = np.zeros((len(scaled_preds), input_size))
    dummy_array[:, target_idx] = scaled_preds.flatten()
    unscaled_dummy = scaler.inverse_transform(dummy_array)
    future_prices = unscaled_dummy[:, target_idx]
    
    target_price = future_prices[-1]
    days_to_target = forecast_days
    
    predicted_change = (target_price - current_price) / current_price
    price_sentiment = np.clip(predicted_change * 5, -1, 1)
    
    # Ensure the model can predict both upward and downward trends
    if predicted_change < 0:
        print("The model predicts a downward trend.")
    elif predicted_change > 0:
        print("The model predicts an upward trend.")
    else:
        print("The model predicts no significant change.")
    
    technical_sentiment = 0
    if 'rsi' in data_with_features.columns:
        last_rsi = data_with_features['rsi'].iloc[-1]
        if not pd.isna(last_rsi):
            if last_rsi > 70:
                technical_sentiment -= 0.3
            elif last_rsi < 30:
                technical_sentiment += 0.3
    
    if 'ma50_cross_ma200' in data_with_features.columns:
        last_cross = data_with_features['ma50_cross_ma200'].iloc[-1]
        if not pd.isna(last_cross):
            technical_sentiment += 0.3 if last_cross == 1 else -0.3
    
    fundamental_sentiment = 0
    if 'trailingPE' in fundamental_data and not pd.isna(fundamental_data['trailingPE']):
        pe_ratio = fundamental_data['trailingPE']
        if pe_ratio < 15:
            fundamental_sentiment += 0.2
        elif pe_ratio > 25:
            fundamental_sentiment -= 0.2
    
    if 'profitMargin' in fundamental_data and not pd.isna(fundamental_data['profitMargin']):
        profit_margin = fundamental_data['profitMargin']
        if profit_margin > 0.2:
            fundamental_sentiment += 0.3
        elif profit_margin > 0.1:
            fundamental_sentiment += 0.1
        elif profit_margin < 0:
            fundamental_sentiment -= 0.3
    
    if 'revenueGrowth' in fundamental_data and not pd.isna(fundamental_data['revenueGrowth']):
        revenue_growth = fundamental_data['revenueGrowth']
        if revenue_growth > 0.2:
            fundamental_sentiment += 0.3
        elif revenue_growth > 0.1:
            fundamental_sentiment += 0.1
        elif revenue_growth < 0:
            fundamental_sentiment -= 0.2
    
    if 'debtToEquity' in fundamental_data and not pd.isna(fundamental_data['debtToEquity']):
        debt_to_equity = fundamental_data['debtToEquity']
        if debt_to_equity < 0.5:
            fundamental_sentiment += 0.2
        elif debt_to_equity > 2:
            fundamental_sentiment -= 0.2
    
    overall_sentiment = np.clip((0.5 * price_sentiment + 0.3 * technical_sentiment + 0.2 * fundamental_sentiment), -1, 1)
    
    print("\nPrediction Summary:")
    print(f"Current Price: {currency}{current_price:.2f}")
    print(f"Target Price: {currency}{target_price:.2f}")
    print(f"Predicted Change: {predicted_change*100:.2f}%")
    print(f"Time to Target: {days_to_target} days")
    print(f"Sentiment Score: {overall_sentiment:.2f}")
    
    fig = visualize_predictions(
        data,
        future_prices,
        ticker_to_use,
        target_price,
        days_to_target,
        current_price,
        overall_sentiment
    )
    
    result = {
        'ticker': ticker_to_use,
        'current_price': current_price,
        'target_price': target_price,
        'change_percent': predicted_change * 100,
        'days_to_target': days_to_target,
        'sentiment': overall_sentiment,
        'figure': fig,
        'future_prices': future_prices,
        'fundamental_data': fundamental_data
    }
    return result

# Main execution block with interactive input
if __name__ == "__main__":
    while True:
        ticker_symbol = input("Enter stock ticker symbol (or 'quit' to exit): ").strip().upper()
        
        if ticker_symbol.lower() == 'quit':
            print("Exiting program.")
            break
        
        if not ticker_symbol:
            print("Please enter a valid ticker symbol.")
            continue
        
        result = predict_stock(ticker_symbol, forecast_days=30)
        
        if result:
            plt.figure(result['figure'].number)
            plt.show()
            
            print("\nDetailed Results:")
            print(f"Ticker: {result['ticker']}")
            currency = "₹" if '.NS' in result['ticker'] or '.BO' in result['ticker'] else "$"
            print(f"Current Price: {currency}{result['current_price']:.2f}")
            print(f"Target Price: {currency}{result['target_price']:.2f}")
            print(f"Predicted Change: {result['change_percent']:.2f}%")
            print(f"Days to Target: {result['days_to_target']}")
            print(f"Sentiment Score: {result['sentiment']:.2f}")
            
            result['figure'].savefig(f"{result['ticker']}_prediction.png")
            print(f"Prediction chart saved as {result['ticker']}_prediction.png")
        else:
            print("Analysis failed. Please check the ticker symbol or try again.")