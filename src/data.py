"""Stock and news data fetching."""
import pickle
import time
from datetime import datetime, timedelta

import yfinance as yf

from src.cache import CacheManager
from src.config import CACHE_DIR, CACHE_EXPIRY

US_STOCKS = [
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "TSLA", "META",
    "NFLX", "NVDA", "AMD", "INTC", "IBM", "ORCL", "CSCO", "BA",
    "DIS", "V", "MA", "JPM", "BAC", "WMT", "PG", "JNJ", "UNH",
]


def fetch_from_yfinance(ticker: str, period: str = "6mo", max_retries: int = 3):
    """Fetch OHLCV and info from yfinance with retry and cache."""
    cached_data, cached_info = CacheManager.load_from_cache(ticker, "yfinance")
    if cached_data is not None:
        print(f"[Cache] Loaded {ticker}")
        return cached_data, cached_info

    print(f"Fetching {ticker} from yfinance...")
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(min(2 ** attempt, 5))
                print(f"   Retry {attempt + 1}/{max_retries}...")
            else:
                time.sleep(0.5)

            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval="1d", auto_adjust=True, actions=False)

            if data.empty or len(data) < 10:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)
                data = stock.history(
                    start=start_date, end=end_date, interval="1d",
                    auto_adjust=True, actions=False
                )

            if data.empty or len(data) < 10:
                continue

            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in data.columns for c in required):
                continue

            print(f"   Fetched {len(data)} days")
            info = {"longName": ticker, "symbol": ticker, "currency": "USD"}
            try:
                stock_info = stock.info
                if stock_info and isinstance(stock_info, dict):
                    info.update(stock_info)
            except Exception:
                pass

            CacheManager.save_to_cache(ticker, data, info, "yfinance")
            return data, info

        except Exception as e:
            print(f"   Error attempt {attempt + 1}: {str(e)[:80]}")
    return None, None


def fetch_stock_data(ticker: str, period: str = "6mo"):
    """Resolve symbol (NSE/BSE/US), fetch data, return (data, info, resolved_ticker)."""
    ticker_upper = ticker.strip().upper()
    print(f"\n{'='*60}\nAnalyzing: {ticker_upper}\n{'='*60}")

    if not (ticker_upper.endswith(".NS") or ticker_upper.endswith(".BO")):
        if ticker_upper not in US_STOCKS:
            print("Assuming Indian stock, adding .NS")
            ticker_upper = f"{ticker_upper}.NS"

    data, info = fetch_from_yfinance(ticker_upper, period)

    if (data is None or data.empty) and ticker_upper.endswith(".NS"):
        ticker_bo = ticker_upper.replace(".NS", ".BO")
        print(f"Trying BSE: {ticker_bo}")
        data, info = fetch_from_yfinance(ticker_bo, period)
        if data is not None and not data.empty:
            ticker_upper = ticker_bo

    if (data is None or data.empty) and (
        ticker_upper.endswith(".NS") or ticker_upper.endswith(".BO")
    ):
        clean = ticker_upper.replace(".NS", "").replace(".BO", "")
        print(f"Trying without suffix: {clean}")
        data, info = fetch_from_yfinance(clean, period)
        if data is not None and not data.empty:
            ticker_upper = clean

    if data is None or data.empty:
        print(f"Failed to fetch {ticker}\n{'='*60}\n")
    else:
        print(f"Loaded {len(data)} days for {ticker_upper}\n{'='*60}\n")
    return data, info, ticker_upper


def fetch_stock_news(ticker: str):
    """Fetch news for ticker (cached)."""
    try:
        cache_path = CACHE_DIR / f"{ticker}_news.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if time.time() - cached["timestamp"] < CACHE_EXPIRY:
                    return cached.get("news", [])
            except Exception:
                pass

        news = []
        try:
            time.sleep(1)
            stock = yf.Ticker(ticker)
            raw = stock.news
            if raw and isinstance(raw, list):
                for item in raw[:6]:
                    try:
                        title = item.get("title", "")
                        if title and len(title) > 5:
                            news.append({
                                "title": title[:120],
                                "source": item.get("publisher", "Unknown"),
                                "link": item.get("link", "#"),
                                "timestamp": item.get("providerPublishTime", int(time.time())),
                            })
                    except Exception:
                        continue
        except Exception:
            pass

        if not news:
            clean = ticker.replace(".NS", "").replace(".BO", "")
            news = [{
                "title": f"Market Analysis for {clean}",
                "source": "Financial News",
                "link": f"https://www.google.com/search?q={clean}+stock+news",
                "timestamp": int(time.time()),
            }]

        if news:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({"timestamp": time.time(), "news": news}, f)
            except Exception:
                pass
        return news
    except Exception:
        clean = ticker.replace(".NS", "").replace(".BO", "")
        return [{
            "title": "News temporarily unavailable",
            "source": "System",
            "link": f"https://finance.yahoo.com/quote/{ticker}",
            "timestamp": int(time.time()),
        }]
