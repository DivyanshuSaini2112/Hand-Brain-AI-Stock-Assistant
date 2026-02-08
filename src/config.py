"""Application configuration and theme."""
from pathlib import Path

# Cache
CACHE_DIR = Path("./stock_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY = 3600  # 1 hour

# StoxAI Premium Color Scheme
COLORS = {
    # Core Palette
    "background": "#0F1216",      # Deep Charcoal (Primary Background)
    "dark": "#0F1216",             # Same as background
    "surface": "#161B22",          # Surface / Cards
    "card_bg": "#161B22",          # Same as surface
    "dark_secondary": "#161B22",   # Same as surface
    
    # Primary Accent (Muted Emerald - for positive, buy signals, CTAs)
    "primary": "#1DB954",          # Muted Emerald
    "secondary": "#10B981",        # Darker Emerald variant
    "success": "#1DB954",          # Same as primary (for positive changes)
    
    # Warning/Negative (NO RED - use Amber instead)
    "amber": "#F59E0B",            # Amber (for warnings and negative values)
    "danger": "#F59E0B",           # Use amber instead of red
    "warning": "#F59E0B",          # Amber
    
    # Text Colors
    "text": "#E6EAF0",            # Primary Text (Light)
    "text_secondary": "#9AA4B2",  # Secondary Text (Gray)
    
    # Borders & Dividers
    "border": "#232A34",           # Borders / Dividers
    "grid": "#232A34",             # Grid lines
    
    # Special Effects
    "card_glow": "rgba(29, 185, 84, 0.1)",  # Glow effect for cards
    
    # Additional utility
    "info": "#1DB954",             # Info color (same as primary)
}