import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime
import ta
from matplotlib.ticker import FuncFormatter
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

def setup_fonts():
    """Set up font handling for better emoji support"""
    try:
        # Get all available font families
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        # Preferred fonts that might support emojis
        preferred_fonts = ['Noto Sans', 'Segoe UI Emoji', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
        
        for font in preferred_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.family'] = font
                print(f"Using font: {font}")
                return
        # If none found, use the system default sans-serif
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("Warning: No preferred emoji-supporting font found. Falling back to sans-serif.")
    except Exception as e:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print(f"Font setup error: {e}. Falling back to sans-serif.")

def fetch_stock_data(ticker, period='6mo', interval='1d'):
    """Fetch historical price data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None, None
        return data, stock.info
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None, None

def fetch_sector_pe(ticker_info):
    """Attempt to fetch industry PE ratio"""
    try:
        sector = ticker_info.get('sector', '')
        if sector:
            sector_tickers = {
                'Technology': ['INFY.NS', 'TCS.NS', 'WIPRO.NS', 'HCLTECH.NS'],
                'Consumer Cyclical': ['JUBLFOOD.NS', 'DMART.NS', 'TITAN.NS'],
                'Financial Services': ['HDFCBANK.NS', 'SBIN.NS', 'ICICIBANK.NS'],
                'Healthcare': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS'],
                'Energy': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS']
            }
            if sector in sector_tickers:
                pes = []
                for sector_ticker in sector_tickers[sector]:
                    try:
                        info = yf.Ticker(sector_ticker).info
                        if 'trailingPE' in info and info['trailingPE'] is not None:
                            pes.append(info['trailingPE'])
                    except:
                        continue
                if pes:
                    return np.median(pes)
        return None
    except:
        return None

def create_technical_features(df):
    """Add technical indicators to the dataframe"""
    data = df.copy()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Weekly_Return'] = data['Close'].pct_change(5)
    data['Monthly_Return'] = data['Close'].pct_change(20)
    data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['Bollinger_Middle'] = bollinger.bollinger_mavg()
    data['Bollinger_Upper'] = bollinger.bollinger_hband()
    data['Bollinger_Lower'] = bollinger.bollinger_lband()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Support1'] = (2 * data['Pivot']) - data['High']
    data['Resistance1'] = (2 * data['Pivot']) - data['Low']
    return data

def format_currency(value, currency_symbol):
    """Format currency values with appropriate scaling"""
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

def millions_formatter(x, pos):
    """Format y-axis in millions for the volume chart"""
    return f'{int(x/1e6)}M'

def add_panel_styling(ax, title):
    """Add styling to panel charts for a more modern look"""
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.patch.set_facecolor('#f8f9fa')
    return ax

def create_fancy_table(ax, data, title, cmap=None):
    """Create a visually appealing table"""
    ax.axis('tight')
    ax.axis('off')
    ax.text(0.5, 1.05, title, fontsize=10, fontweight='bold', 
            ha='center', va='bottom', transform=ax.transAxes)
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if j == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#ecf0f1')
            else:
                if cmap is not None and len(data[i]) > 1:
                    try:
                        value = data[i][j]
                        if '%' in value and '↑' in value:
                            cell.set_facecolor('#e6f7e9')
                        elif '%' in value and '↓' in value:
                            cell.set_facecolor('#fae9e8')
                    except:
                        pass
        cell.set_edgecolor('#d4d4d4')
    return table

def create_signal_panel(fig, signals, pos):
    """Create an attractive technical signals panel with fallback for missing glyphs"""
    signal_box = fig.add_axes(pos)
    signal_box.axis('off')
    
    patch = FancyBboxPatch((0, 0), 1, 1, 
                         boxstyle=matplotlib.patches.BoxStyle("Round", pad=0.3),
                         facecolor='#f8f9fa', edgecolor='#dee2e6', 
                         alpha=0.95, transform=signal_box.transAxes)
    signal_box.add_patch(patch)
    
    signal_box.text(0.5, 0.95, "TECHNICAL SIGNALS", 
                  ha="center", va="top", fontsize=12, fontweight='bold',
                  transform=signal_box.transAxes)
    
    # Simplistic check: Assume emoji support if using known emoji-supporting fonts
    current_font = matplotlib.rcParams['font.family'][0].lower()
    emoji_fonts = {'noto sans', 'segoe ui emoji', 'dejavu sans'}
    supports_emojis = current_font in emoji_fonts
    
    for i, signal in enumerate(signals):
        y_pos = 0.85 - (i * 0.15)
        if supports_emojis:
            if "Overbought" in signal:
                icon, color = "⚠️ ", 'darkorange'
            elif "Oversold" in signal:
                icon, color = "✅ ", 'green'
            elif "Bullish" in signal:
                icon, color = "📈 ", 'green'
            elif "Bearish" in signal:
                icon, color = "📉 ", 'crimson'
            elif ">" in signal:
                icon, color = "↗️ ", 'green'
            elif "<" in signal:
                icon, color = "↘️ ", 'crimson'
            else:
                icon, color = "⚖️ ", 'darkblue'
        else:
            # Fallback to simple ASCII symbols
            if "Overbought" in signal:
                icon, color = "! ", 'darkorange'
            elif "Oversold" in signal:
                icon, color = "✓ ", 'green'
            elif "Bullish" in signal:
                icon, color = "^ ", 'green'
            elif "Bearish" in signal:
                icon, color = "v ", 'crimson'
            elif ">" in signal:
                icon, color = "> ", 'green'
            elif "<" in signal:
                icon, color = "< ", 'crimson'
            else:
                icon, color = "= ", 'darkblue'
        
        signal_box.text(0.1, y_pos, icon, fontsize=12, ha="center", va="center", 
                      transform=signal_box.transAxes)
        signal_box.text(0.25, y_pos, signal, fontsize=10, ha="left", va="center", 
                      color=color, weight='medium', transform=signal_box.transAxes)
    return signal_box

def visualize_stock_data(df, info, ticker):
    """Create a comprehensive visualization of stock data with modern styling and improved spacing"""
    setup_fonts()
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#e6e6e6'
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.edgecolor'] = '#e6e6e6'
    plt.rcParams['axes.linewidth'] = 1.5
    
    colors = {
        'price': '#1f77b4', 'price_fill': '#c6dcef',
        'ma20': '#ff7f0e', 'ma50': '#2ca02c',
        'volume': '#3498db', 'volume_up': '#2ecc71', 'volume_down': '#e74c3c',
        'rsi': '#9b59b6', 'rsi_overbought': '#e74c3c', 'rsi_oversold': '#2ecc71',
        'macd': '#2980b9', 'signal': '#e74c3c', 
        'histogram_up': '#2ecc71', 'histogram_down': '#e74c3c',
        'bollinger': '#7f7f7f'
    }
    
    # Increase figure size for better spacing
    fig = plt.figure(figsize=(18, 12))  # Increased size for more breathing room
    plt.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.06, hspace=0.35, wspace=0.3)  # More spacing
    
    gs = GridSpec(7, 6, figure=fig)  # Added an extra row for better distribution
    
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    company_name = info.get('longName', ticker)
    
    # Title with reduced font size and better positioning
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.7, f"{company_name} ({ticker})", 
                  fontsize=18, fontweight='bold', ha='center', va='center')  # Reduced font size
    current_price = df['Close'].iloc[-1]
    last_date = df.index[-1].strftime('%d %b %Y')
    price_text = f"Current Price: {currency}{current_price:.2f} | Last Updated: {last_date}"
    title_ax.text(0.5, 0.3, price_text, fontsize=10, ha='center', va='center')  # Reduced font size
    
    # Price chart with more space
    ax_price = fig.add_subplot(gs[1:3, :4])
    ax_price.plot(df.index, df['Close'], color=colors['price'], linewidth=2, label='Close Price')
    ax_price.fill_between(df.index, df['Close'].min()*0.95, df['Close'], 
                         alpha=0.1, color=colors['price_fill'])
    ax_price.plot(df.index, df['MA20'], color=colors['ma20'], linestyle='--', 
                  linewidth=1.5, label='20-Day MA')
    ax_price.plot(df.index, df['MA50'], color=colors['ma50'], linestyle='--', 
                  linewidth=1.5, label='50-Day MA')
    ax_price.plot(df.index, df['Bollinger_Upper'], color=colors['bollinger'], 
                  linestyle='--', alpha=0.5, linewidth=1)
    ax_price.plot(df.index, df['Bollinger_Lower'], color=colors['bollinger'], 
                  linestyle='--', alpha=0.5, linewidth=1)
    ax_price.fill_between(df.index, df['Bollinger_Upper'], df['Bollinger_Lower'], 
                         color=colors['bollinger'], alpha=0.1)
    add_panel_styling(ax_price, "Price Chart")
    ax_price.set_ylabel('Price', fontsize=10)
    ax_price.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.7)  # Reduced font size
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    # Volume chart
    ax_volume = fig.add_subplot(gs[3, :4], sharex=ax_price)
    volume_bars = ax_volume.bar(df.index, df['Volume'], color=colors['volume'], 
                              alpha=0.7, width=0.8)
    for i, bar in enumerate(volume_bars):
        if i > 0 and df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            bar.set_color(colors['volume_up'])
        else:
            bar.set_color(colors['volume_down'])
    add_panel_styling(ax_volume, "Volume")
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax_volume.tick_params(axis='x', labelbottom=False)
    
    # RSI chart
    ax_rsi = fig.add_subplot(gs[4, :2], sharex=ax_price)
    ax_rsi.plot(df.index, df['RSI'], color=colors['rsi'], linewidth=1.5)
    ax_rsi.axhline(70, color=colors['rsi_overbought'], linestyle='--', alpha=0.5)
    ax_rsi.axhline(30, color=colors['rsi_oversold'], linestyle='--', alpha=0.5)
    ax_rsi.fill_between(df.index, df['RSI'], 70, 
                      where=(df['RSI'] >= 70), color=colors['rsi_overbought'], alpha=0.3)
    ax_rsi.fill_between(df.index, df['RSI'], 30, 
                      where=(df['RSI'] <= 30), color=colors['rsi_oversold'], alpha=0.3)
    add_panel_styling(ax_rsi, "Relative Strength Index")
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    
    # MACD chart
    ax_macd = fig.add_subplot(gs[4, 2:4], sharex=ax_price)
    ax_macd.plot(df.index, df['MACD'], color=colors['macd'], linewidth=1.5, label='MACD')
    ax_macd.plot(df.index, df['MACD_Signal'], color=colors['signal'], 
                 linewidth=1.5, label='Signal')
    for i in range(len(df.index)):
        if i < len(df.index) - 1:
            if df['MACD_Histogram'].iloc[i] >= 0:
                ax_macd.bar(df.index[i], df['MACD_Histogram'].iloc[i], 
                          color=colors['histogram_up'], alpha=0.5, width=0.8)
            else:
                ax_macd.bar(df.index[i], df['MACD_Histogram'].iloc[i], 
                          color=colors['histogram_down'], alpha=0.5, width=0.8)
    add_panel_styling(ax_macd, "MACD")
    ax_macd.set_ylabel('MACD', fontsize=10)
    ax_macd.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.7)  # Reduced font size
    
    # Fetch industry PE and prepare metrics tables (unchanged structure, just smaller fonts)
    industry_pe = fetch_sector_pe(info)
    pe_ratio = info.get('trailingPE', None)
    pb_ratio = info.get('priceToBook', None)
    
    price_metrics = [
        ["Price Metrics", "Value"],
        ["Open", f"{currency}{df['Open'].iloc[-1]:.2f}"],
        ["High", f"{currency}{df['High'].iloc[-1]:.2f}"],
        ["Low", f"{currency}{df['Low'].iloc[-1]:.2f}"],
        ["Close", f"{currency}{df['Close'].iloc[-1]:.2f}"],
    ]
    
    volume_metrics = [
        ["Volume Metrics", "Value"],
        ["Volume", f"{df['Volume'].iloc[-1]:,.0f}"],
        ["Avg Volume", f"{info.get('averageVolume', 0):,.0f}"],
    ]
    
    valuation_metrics = [
        ["Valuation", "Value"],
        ["Market Cap", format_currency(info.get('marketCap'), currency)],
        ["P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio and not np.isnan(pe_ratio) else 'N/A'],
        ["P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio and not np.isnan(pb_ratio) else 'N/A'],
        ["Industry P/E", f"{industry_pe:.2f}" if industry_pe else 'N/A'],
    ]
    
    performance_metrics = [
        ["Performance", "Value"],
        ["1D Change", format_percentage(df['Daily_Return'].iloc[-1])],
        ["1W Change", format_percentage(df['Weekly_Return'].iloc[-1])],
        ["1M Change", format_percentage(df['Monthly_Return'].iloc[-1])],
        ["Volatility", format_percentage(df['Volatility_20d'].iloc[-1])],
    ]
    
    # Tables with reduced font sizes and adjusted positioning
    price_table_ax = fig.add_subplot(gs[1, 4:])
    create_fancy_table(price_table_ax, price_metrics, "Price Summary")
    
    volume_table_ax = fig.add_subplot(gs[2, 4:])
    create_fancy_table(volume_table_ax, volume_metrics, "Volume Summary")
    
    valuation_table_ax = fig.add_subplot(gs[3, 4:])
    create_fancy_table(valuation_table_ax, valuation_metrics, "Valuation Summary")
    
    performance_table_ax = fig.add_subplot(gs[4, 4:])
    create_fancy_table(performance_table_ax, performance_metrics, 
                      "Performance Summary", cmap='RdYlGn')
    
    # Technical signals panel with more space and smaller font
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    
    signals = []
    if rsi > 70:
        signals.append(f"RSI: Overbought ({rsi:.1f})")
    elif rsi < 30:
        signals.append(f"RSI: Oversold ({rsi:.1f})")
    else:
        signals.append(f"RSI: Neutral ({rsi:.1f})")
    if macd > macd_signal:
        signals.append("MACD: Bullish Signal")
    else:
        signals.append("MACD: Bearish Signal")
    if df['Close'].iloc[-1] > df['MA50'].iloc[-1]:
        signals.append("Price > 50-Day MA")
    else:
        signals.append("Price < 50-Day MA")
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
        signals.append("20-Day MA > 50-Day MA")
    else:
        signals.append("20-Day MA < 50-Day MA")
    if df['Close'].iloc[-1] > df['Bollinger_Upper'].iloc[-1]:
        signals.append("Price > Upper Bollinger")
    elif df['Close'].iloc[-1] < df['Bollinger_Lower'].iloc[-1]:
        signals.append("Price < Lower Bollinger")
    else:
        signals.append("Price within Bollinger Bands")
    
    # Increase the height of the signal panel and reduce font sizes
    create_signal_panel(fig, signals, [0.07, 0.05, 0.86, 0.18])  # Increased height (0.18 instead of 0.15)
    
    # Footer with smaller font and better positioning
    footer_ax = fig.add_axes([0, 0, 1, 0.02])  # Reduced height
    footer_ax.axis('off')
    footer_ax.text(0.5, 0.5, 
                   "Disclaimer: This analysis is for informational purposes only. Not financial advice.",
                   ha="center", va="center", fontsize=7, style='italic', alpha=0.7)  # Reduced font size
    
    return fig

def analyze_stock(ticker_symbol):
    print(f"Analyzing {ticker_symbol}...")
    
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
    
    if len(data) < 20:
        print(f"Not enough historical data for {ticker_to_use}.")
        return None
    
    print(f"✓ Successfully fetched 6 months of data for {ticker_to_use}")
    print("📊 Calculating technical indicators...")
    print("  ↳ Adding technical features")
    data_with_features = create_technical_features(data)
    print("✓ Technical indicators complete")
    
    currency = "₹" if '.NS' in ticker_to_use or '.BO' in ticker_to_use else "$"
    current_price = data['Close'].iloc[-1]
    company_name = info.get('longName', ticker_to_use)
    
    print(f"\n===== {company_name} ({ticker_to_use}) =====")
    print(f"💰 Current Price: {currency}{current_price:.2f}")
    print(f"📈 Day Range: {currency}{data['Low'].iloc[-1]:.2f} - {currency}{data['High'].iloc[-1]:.2f}")
    
    industry_pe = fetch_sector_pe(info)
    
    if info:
        if 'marketCap' in info and info['marketCap']:
            print(f"🏢 Market Cap: {format_currency(info['marketCap'], currency)}")
        if 'sector' in info and info['sector']:
            print(f"🏭 Sector: {info['sector']}")
        if 'industry' in info and info['industry']:
            print(f"🏬 Industry: {info['industry']}")
        if 'trailingPE' in info and info['trailingPE']:
            print(f"📊 P/E Ratio: {info['trailingPE']:.2f}")
        if 'priceToBook' in info and info['priceToBook']:
            print(f"📚 P/B Ratio: {info['priceToBook']:.2f}")
        if industry_pe:
            print(f"📉 Industry P/E: {industry_pe:.2f}")
    
    print("\n🎨 Creating visualization dashboard...")
    fig = visualize_stock_data(data_with_features, info, ticker_to_use)
    print("✓ Dashboard completed")
    
    result = {
        'ticker': ticker_to_use,
        'company_name': company_name,
        'current_price': current_price,
        'data': data_with_features,
        'info': info,
        'figure': fig
    }
    
    return result

if __name__ == "__main__":
    print("Stock Analysis Tool - Enter 'quit' to exit")
    while True:
        try:
            ticker_symbol = input("\nEnter stock ticker symbol: ").strip().upper()
            
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
                result['figure'].savefig(f"{result['ticker']}_analysis.png", 
                                       dpi=300, bbox_inches='tight')
                print(f"Analysis chart saved as {result['ticker']}_analysis.png")
            else:
                print("Analysis failed. Please check the ticker symbol or try again.")
                
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")