import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime
import ta

def fetch_stock_data(ticker, period='6mo', interval='1d'):
    """Fetch historical price data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None
        return data, stock.info
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None, None

def create_technical_features(df):
    """Add technical indicators to the dataframe"""
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price-based features
    # Moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Price changes
    data['Daily_Return'] = data['Close'].pct_change()
    data['Weekly_Return'] = data['Close'].pct_change(5)
    data['Monthly_Return'] = data['Close'].pct_change(20)
    
    # Volatility measures
    data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
    
    # Add RSI (Relative Strength Index)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # Add MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['Bollinger_Middle'] = bollinger.bollinger_mavg()
    data['Bollinger_Upper'] = bollinger.bollinger_hband()
    data['Bollinger_Lower'] = bollinger.bollinger_lband()
    
    # Add ATR (Average True Range)
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    
    return data

def format_currency(value, currency_symbol):
    """Format currency values with appropriate scaling (K, M, B)"""
    if value is None or np.isnan(value):
        return "N/A"
    
    if value >= 1e9:
        return f"{currency_symbol}{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{currency_symbol}{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{currency_symbol}{value/1e3:.2f}K"
    else:
        return f"{currency_symbol}{value:.2f}"

def format_percentage(value):
    """Format percentage values"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value*100:.2f}%"

def visualize_stock_data(df, info, ticker):
    """Create a comprehensive visualization of stock data"""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 4, figure=fig)
    
    # Get currency symbol
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    
    # Extract company name
    company_name = info.get('longName', ticker)
    
    # 1. Price Chart (Top panel spanning all columns)
    ax_price = fig.add_subplot(gs[0:2, :])
    ax_price.plot(df.index, df['Close'], 'b-', linewidth=2, label='Close Price')
    ax_price.plot(df.index, df['MA20'], 'r--', label='20-Day MA')
    ax_price.plot(df.index, df['MA50'], 'g--', label='50-Day MA')
    
    # Add Bollinger Bands
    ax_price.plot(df.index, df['Bollinger_Upper'], 'k--', alpha=0.3)
    ax_price.plot(df.index, df['Bollinger_Lower'], 'k--', alpha=0.3)
    ax_price.fill_between(df.index, df['Bollinger_Upper'], df['Bollinger_Lower'], color='gray', alpha=0.1)
    
    # Format price chart
    current_price = df['Close'].iloc[-1]
    ax_price.set_title(f"{company_name} ({ticker}) - Current Price: {currency}{current_price:.2f}", fontsize=16, fontweight='bold')
    ax_price.set_ylabel('Price', fontsize=12)
    ax_price.legend(loc='upper left')
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    # 2. Volume Chart (Below price chart, spanning all columns but smaller height)
    ax_volume = fig.add_subplot(gs[2, :], sharex=ax_price)
    ax_volume.bar(df.index, df['Volume'], color='blue', alpha=0.5)
    ax_volume.set_ylabel('Volume', fontsize=12)
    plt.setp(ax_volume.get_xticklabels(), visible=False)
    
    # 3. RSI Chart
    ax_rsi = fig.add_subplot(gs[3, 0:2], sharex=ax_price)
    ax_rsi.plot(df.index, df['RSI'], 'purple', linewidth=1.5)
    ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax_rsi.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='red', alpha=0.3)
    ax_rsi.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='green', alpha=0.3)
    ax_rsi.set_ylabel('RSI', fontsize=12)
    ax_rsi.set_ylim(0, 100)
    
    # 4. MACD Chart
    ax_macd = fig.add_subplot(gs[3, 2:], sharex=ax_price)
    ax_macd.plot(df.index, df['MACD'], 'blue', linewidth=1.5, label='MACD')
    ax_macd.plot(df.index, df['MACD_Signal'], 'red', linewidth=1.5, label='Signal')
    ax_macd.bar(df.index, df['MACD_Histogram'], color='gray', alpha=0.5)
    ax_macd.legend(loc='upper left', fontsize=8)
    ax_macd.set_ylabel('MACD', fontsize=12)
    
    # Add key metrics table
    key_metrics = [
        ["Metric", "Value"],
        ["Open", f"{currency}{df['Open'].iloc[-1]:.2f}"],
        ["High", f"{currency}{df['High'].iloc[-1]:.2f}"],
        ["Low", f"{currency}{df['Low'].iloc[-1]:.2f}"],
        ["Close", f"{currency}{df['Close'].iloc[-1]:.2f}"],
        ["Volume", f"{df['Volume'].iloc[-1]:,.0f}"],
        ["Market Cap", format_currency(info.get('marketCap'), currency)],
        ["P/E Ratio", f"{info.get('trailingPE', 'N/A')}"],
        ["52W High", format_currency(info.get('fiftyTwoWeekHigh'), currency)],
        ["52W Low", format_currency(info.get('fiftyTwoWeekLow'), currency)],
        ["Avg Volume", f"{info.get('averageVolume', 0):,.0f}"],
        ["Beta", f"{info.get('beta', 'N/A')}"],
        ["RSI (14)", f"{df['RSI'].iloc[-1]:.2f}"],
        ["1D Change", format_percentage(df['Daily_Return'].iloc[-1])],
        ["1W Change", format_percentage(df['Weekly_Return'].iloc[-1])],
        ["1M Change", format_percentage(df['Monthly_Return'].iloc[-1])],
        ["Volatility", format_percentage(df['Volatility_20d'].iloc[-1])]
    ]
    
    # Add technical indicators summary
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    
    # Technical signals
    signals = []
    
    # RSI signals
    if rsi > 70:
        signals.append("RSI indicates overbought")
    elif rsi < 30:
        signals.append("RSI indicates oversold")
    
    # MACD signals
    if macd > macd_signal:
        signals.append("MACD above signal line (bullish)")
    else:
        signals.append("MACD below signal line (bearish)")
    
    # Moving average signals
    if df['Close'].iloc[-1] > df['MA50'].iloc[-1]:
        signals.append("Price above 50-day MA (bullish)")
    else:
        signals.append("Price below 50-day MA (bearish)")
    
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
        signals.append("20-day MA above 50-day MA (bullish)")
    else:
        signals.append("20-day MA below 50-day MA (bearish)")
    
    # Bollinger Bands signals
    if df['Close'].iloc[-1] > df['Bollinger_Upper'].iloc[-1]:
        signals.append("Price above upper Bollinger Band (overbought)")
    elif df['Close'].iloc[-1] < df['Bollinger_Lower'].iloc[-1]:
        signals.append("Price below lower Bollinger Band (oversold)")
    
    # Add text annotation with signals
    signal_text = "\n".join(signals)
    plt.figtext(0.5, 0.02, "Technical Signals:\n" + signal_text, ha="center", fontsize=12, 
               bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    return fig

def analyze_stock(ticker_symbol):
    print(f"\nAnalyzing {ticker_symbol}...")
    
    # Check if it's an Indian stock and append .NS if needed
    if not (ticker_symbol.endswith('.NS') or ticker_symbol.endswith('.BO')) and ticker_symbol not in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NFLX']:
        ticker_ns = f"{ticker_symbol}.NS"
        data_ns, info_ns = fetch_stock_data(ticker_ns)
        
        if data_ns is None or data_ns.empty:
            ticker_bo = f"{ticker_symbol}.BO"
            data_bo, info_bo = fetch_stock_data(ticker_bo)
            
            if data_bo is None or data_bo.empty:
                data, info = fetch_stock_data(ticker_symbol)
                if data is None or data.empty:
                    print(f"Could not fetch data for {ticker_symbol}. Please check the ticker symbol.")
                    return None
                ticker_to_use = ticker_symbol
            else:
                data = data_bo
                info = info_bo
                ticker_to_use = ticker_bo
        else:
            data = data_ns
            info = info_ns
            ticker_to_use = ticker_ns
    else:
        data, info = fetch_stock_data(ticker_symbol)
        if data is None or data.empty:
            print(f"Could not fetch data for {ticker_symbol}. Please check the ticker symbol.")
            return None
        ticker_to_use = ticker_symbol
    
    if len(data) < 20:  # Need at least 20 days for some indicators
        print(f"Not enough historical data for {ticker_to_use}.")
        return None
    
    print("Calculating technical indicators...")
    data_with_features = create_technical_features(data)
    
    # Display basic information
    currency = "₹" if '.NS' in ticker_to_use or '.BO' in ticker_to_use else "$"
    current_price = data['Close'].iloc[-1]
    company_name = info.get('longName', ticker_to_use)
    
    print(f"\n===== {company_name} ({ticker_to_use}) =====")
    print(f"Current Price: {currency}{current_price:.2f}")
    print(f"Day Range: {currency}{data['Low'].iloc[-1]:.2f} - {currency}{data['High'].iloc[-1]:.2f}")
    
    # Print more details if available
    if info:
        if 'marketCap' in info and info['marketCap']:
            market_cap = info['marketCap']
            if market_cap >= 1e9:
                cap_str = f"{market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                cap_str = f"{market_cap/1e6:.2f}M"
            else:
                cap_str = f"{market_cap/1e3:.2f}K"
            print(f"Market Cap: {currency}{cap_str}")
        
        if 'sector' in info and info['sector']:
            print(f"Sector: {info['sector']}")
        
        if 'industry' in info and info['industry']:
            print(f"Industry: {info['industry']}")
    
    # Create visualization
    fig = visualize_stock_data(data_with_features, info, ticker_to_use)
    
    result = {
        'ticker': ticker_to_use,
        'company_name': company_name,
        'current_price': current_price,
        'data': data_with_features,
        'info': info,
        'figure': fig
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
        
        result = analyze_stock(ticker_symbol)
        
        if result:
            plt.figure(result['figure'].number)
            plt.show()
            
            # Save the figure
            result['figure'].savefig(f"{result['ticker']}_analysis.png")
            print(f"Analysis chart saved as {result['ticker']}_analysis.png")
        else:
            print("Analysis failed. Please check the ticker symbol or try again.")