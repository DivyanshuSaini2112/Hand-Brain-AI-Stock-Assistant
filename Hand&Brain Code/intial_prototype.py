import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import datetime
import pandas as pd

# Function to fetch stock market data
def fetch_stock_data(ticker, period='1y', interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None, None, None
        return data['Close'].values.reshape(-1, 1), data.index, data.iloc[-1]['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

# Define an LSTM model for stock prediction
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Prepare data for training
def prepare_data(data, seq_length=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(len(data_scaled) - seq_length):
        x.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    
    return x_tensor, y_tensor, scaler

# Function to create a semicircle gauge
def create_sentiment_gauge(ax, sentiment_score, ticker):
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

# Function to train model and make predictions
def predict_stock(ticker):
    # Check if it's an Indian stock and append .NS if needed
    if not (ticker.endswith('.NS') or ticker.endswith('.BO')) and ticker not in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX']:
        ticker_ns = f"{ticker}.NS"  # Try NSE first
        data_ns, dates_ns, price_ns = fetch_stock_data(ticker_ns)
        
        if data_ns is None:
            ticker_bo = f"{ticker}.BO"  # Try BSE if NSE fails
            data_bo, dates_bo, price_bo = fetch_stock_data(ticker_bo)
            
            if data_bo is None:
                # Try original ticker as a fallback
                data, dates, current_price = fetch_stock_data(ticker)
                if data is None:
                    print(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
                    return
                ticker_to_use = ticker
            else:
                data, dates, current_price = data_bo, dates_bo, price_bo
                ticker_to_use = ticker_bo
        else:
            data, dates, current_price = data_ns, dates_ns, price_ns
            ticker_to_use = ticker_ns
    else:
        # Use the ticker as provided
        data, dates, current_price = fetch_stock_data(ticker)
        if data is None:
            print(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
            return
        ticker_to_use = ticker
    
    print(f"\nAnalyzing {ticker_to_use}...")
    print(f"Current Price: ₹{current_price:.2f}" if '.NS' in ticker_to_use or '.BO' in ticker_to_use else f"Current Price: ${current_price:.2f}")
    
    # Check if we have enough data
    if len(data) < 100:
        print(f"Not enough historical data for {ticker_to_use} to make reliable predictions.")
        return
    
    # Prepare data
    seq_length = 60  # Increased sequence length for better learning
    x_train, y_train, scaler = prepare_data(data, seq_length)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    
    # Initialize model with improved architecture
    model = StockLSTM(input_size=1, hidden_size=64, num_layers=3, dropout=0.2).to(device)
    
    # Train model with early stopping
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    num_epochs = 100
    batch_size = 64
    
    # Create batches
    n_batches = len(x_train) // batch_size
    if n_batches == 0:
        n_batches = 1
        batch_size = len(x_train)
    
    print("Training model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(x_train))
            
            batch_x = x_train[start_idx:end_idx].to(device)
            batch_y = y_train[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model weights if needed
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Predict future prices
    model.eval()
    
    with torch.no_grad():
        # Predict next day
        last_sequence = x_train[-1:].to(device)
        next_day_pred = model(last_sequence).cpu().numpy()
        next_day_price = scaler.inverse_transform(next_day_pred.reshape(-1, 1))[0][0]
        
        # Predict prices for the next 30 days
        future_preds = []
        curr_seq = last_sequence.cpu().numpy()[0]
        
        for _ in range(30):
            pred = model(torch.tensor(curr_seq.reshape(1, seq_length, 1), dtype=torch.float32).to(device)).cpu().numpy()
            future_preds.append(pred[0][0])
            # Update sequence
            curr_seq = np.roll(curr_seq, -1)
            curr_seq[-1] = pred[0][0]
        
        future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    
    # Determine target price and days to reach
    target_price = next_day_price
    days_to_target = 1
    
    # Look for significant price movement
    for i, price in enumerate(future_prices):
        price_change = (price - current_price) / current_price
        if abs(price_change) >= 0.05:  # Significant move (5%)
            target_price = price
            days_to_target = i + 1
            break
    
    # If no significant movement, use the furthest prediction
    if days_to_target == 1 and abs((next_day_price - current_price) / current_price) < 0.03:
        target_price = future_prices[-1]
        days_to_target = 30
    
    # Calculate predicted change and sentiment
    predicted_change = (target_price - current_price) / current_price
    
    # Create sentiment score based on predicted change
    # Adjust these thresholds based on your definition of strong movements
    if predicted_change > 0.1:  # >10% gain: Strong Buy
        sentiment_score = 0.9
        recommendation = "Strong Buy"
    elif predicted_change > 0.05:  # 5-10% gain: Buy
        sentiment_score = 0.5
        recommendation = "Buy"
    elif predicted_change > -0.05:  # -5% to +5%: Hold
        sentiment_score = 0.0
        recommendation = "Hold"
    elif predicted_change > -0.1:  # -5% to -10%: Sell
        sentiment_score = -0.5
        recommendation = "Sell"
    else:  # >10% loss: Strong Sell
        sentiment_score = -0.9
        recommendation = "Strong Sell"
    
    # Display results
    future_date = datetime.datetime.now() + datetime.timedelta(days=days_to_target)
    
    currency = "₹" if '.NS' in ticker_to_use or '.BO' in ticker_to_use else "$"
    print(f"\nCurrent Price: {currency}{current_price:.2f}")
    print(f"Target Price: {currency}{target_price:.2f} ({predicted_change*100:.2f}%)")
    print(f"Expected time to reach target: {days_to_target} days (by {future_date.strftime('%Y-%m-%d')})")
    print(f"Recommendation: {recommendation}")
    
    # Create sentiment gauge plot
    fig, ax = plt.subplots(figsize=(8, 5))
    create_sentiment_gauge(ax, sentiment_score, ticker_to_use)
    
    # Add recommendation and target price
    ax.text(0.5, -0.4, f"Recommendation: {recommendation}", ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.5, f"Target: {currency}{target_price:.2f} in {days_to_target} days ({predicted_change*100:.2f}%)", 
            ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    while True:
        stock_ticker = input("\nEnter stock ticker (or 'q' to quit): ")
        if stock_ticker.lower() == 'q':
            break
        predict_stock(stock_ticker)