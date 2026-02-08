"""Caching layer for stock and news data."""
import pickle
import time
from pathlib import Path

from src.config import CACHE_DIR, CACHE_EXPIRY


class CacheManager:
    """Cache for price and news data with TTL."""

    @staticmethod
    def get_cache_path(ticker: str, data_type: str = "price") -> Path:
        return CACHE_DIR / f"{ticker}_{data_type}.pkl"

    @staticmethod
    def save_to_cache(ticker: str, data, info, data_type: str = "price") -> bool:
        try:
            cache_path = CacheManager.get_cache_path(ticker, data_type)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"timestamp": time.time(), "data": data, "info": info}, f
                )
            return True
        except Exception:
            return False

    @staticmethod
    def load_from_cache(ticker: str, data_type: str = "price"):
        try:
            cache_path = CacheManager.get_cache_path(ticker, data_type)
            if not cache_path.exists():
                return None, None
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            if time.time() - cache_data["timestamp"] > CACHE_EXPIRY:
                return None, None
            return cache_data["data"], cache_data["info"]
        except Exception:
            return None, None
