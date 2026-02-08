"""Technical analysis: features, risk metrics, trading signals."""
import numpy as np
import pandas as pd
import ta

from src.data import fetch_stock_data


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe."""
    data = df.copy()
    # Moving averages
    for w in (5, 10, 20, 50, 200):
        data[f"MA{w}"] = data["Close"].rolling(window=w).mean()
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    # Returns
    data["Daily_Return"] = data["Close"].pct_change()
    data["Weekly_Return"] = data["Close"].pct_change(5)
    data["Monthly_Return"] = data["Close"].pct_change(20)
    # Volatility
    data["Volatility_10d"] = data["Daily_Return"].rolling(10).std()
    data["Volatility_20d"] = data["Daily_Return"].rolling(20).std()
    data["Volatility_50d"] = data["Daily_Return"].rolling(50).std()
    # RSI
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["RSI_Smooth"] = data["RSI"].rolling(3).mean()
    # MACD
    macd = ta.trend.MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Histogram"] = macd.macd_diff()
    # Bollinger
    bb = ta.volatility.BollingerBands(data["Close"], window=20, window_dev=2)
    data["Bollinger_Upper"] = bb.bollinger_hband()
    data["Bollinger_Lower"] = bb.bollinger_lband()
    data["Bollinger_Middle"] = bb.bollinger_mavg()
    data["Bollinger_Width"] = (
        (data["Bollinger_Upper"] - data["Bollinger_Lower"]) / data["Bollinger_Middle"]
    )
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(data["High"], data["Low"], data["Close"])
    data["Stoch_K"] = stoch.stoch()
    data["Stoch_D"] = stoch.stoch_signal()
    # ATR, OBV, CCI, MFI
    data["ATR"] = ta.volatility.AverageTrueRange(
        data["High"], data["Low"], data["Close"]
    ).average_true_range()
    data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        data["Close"], data["Volume"]
    ).on_balance_volume()
    data["CCI"] = ta.trend.CCIIndicator(
        data["High"], data["Low"], data["Close"]
    ).cci()
    data["MFI"] = ta.volume.MFIIndicator(
        data["High"], data["Low"], data["Close"], data["Volume"]
    ).money_flow_index()
    # Support / resistance
    data["Support"] = data["Low"].rolling(20).min()
    data["Resistance"] = data["High"].rolling(20).max()
    return data


def calculate_risk_metrics(df: pd.DataFrame) -> dict:
    """Sharpe, max drawdown, VaR, Sortino, win rate."""
    returns = df["Daily_Return"].dropna()
    metrics = {}
    metrics["sharpe_ratio"] = (
        (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    )
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics["max_drawdown"] = drawdown.min() * 100
    metrics["var_95"] = np.percentile(returns, 5) * 100
    downside = returns[returns < 0]
    metrics["sortino_ratio"] = (
        (returns.mean() / downside.std()) * np.sqrt(252) if downside.std() != 0 else 0
    )
    metrics["win_rate"] = (
        (len(returns[returns > 0]) / len(returns)) * 100 if len(returns) > 0 else 0
    )
    return metrics


def generate_trading_signals(df: pd.DataFrame) -> list:
    """List of (text, type, icon, sentiment) from latest row."""
    signals = []
    latest = df.iloc[-1]

    if latest["RSI"] > 70:
        signals.append(("Overbought (RSI > 70)", "danger", "⚠️", "Bearish"))
    elif latest["RSI"] < 30:
        signals.append(("Oversold (RSI < 30)", "success", "✓", "Bullish"))
    else:
        signals.append(("Neutral RSI", "warning", "●", "Neutral"))

    if latest["MACD"] > latest["MACD_Signal"]:
        signals.append(("Bullish MACD Cross", "success", "↑", "Bullish"))
    else:
        signals.append(("Bearish MACD Cross", "danger", "↓", "Bearish"))

    if latest["Close"] > latest["MA50"]:
        signals.append(("Above MA50", "success", "↑", "Bullish"))
    else:
        signals.append(("Below MA50", "danger", "↓", "Bearish"))

    if latest["MA20"] > latest["MA50"]:
        signals.append(("Golden Cross (MA20 > MA50)", "success", "★", "Bullish"))
    elif latest["MA20"] < latest["MA50"]:
        signals.append(("Death Cross (MA20 < MA50)", "danger", "★", "Bearish"))

    if latest["Close"] > latest["Bollinger_Upper"]:
        signals.append(("Above Upper Bollinger", "danger", "⚠️", "Bearish"))
    elif latest["Close"] < latest["Bollinger_Lower"]:
        signals.append(("Below Lower Bollinger", "success", "✓", "Bullish"))

    if latest["Stoch_K"] > 80:
        signals.append(("Stochastic Overbought", "danger", "⚠️", "Bearish"))
    elif latest["Stoch_K"] < 20:
        signals.append(("Stochastic Oversold", "success", "✓", "Bullish"))

    avg_vol = df["Volume"].tail(20).mean()
    if latest["Volume"] > avg_vol * 1.5:
        signals.append(("High Volume Alert", "info", "📊", "Attention"))

    return signals


def analyze_stock(ticker_symbol: str):
    """Fetch data, add features, compute risk and return result dict or None."""
    data, info, final_ticker = fetch_stock_data(ticker_symbol)
    if data is None or data.empty or len(data) < 20:
        return None
    data_with_features = create_advanced_features(data)
    risk_metrics = calculate_risk_metrics(data_with_features)
    return {
        "ticker": final_ticker,
        "data": data_with_features,
        "info": info,
        "risk_metrics": risk_metrics,
    }
