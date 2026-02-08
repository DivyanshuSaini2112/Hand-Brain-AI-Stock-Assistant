"""
StoxAI – ProStock Analytics Dashboard
Entry point: run with `python app.py`
"""
import warnings
warnings.filterwarnings("ignore")

import dash
import dash_bootstrap_components as dbc
import yfinance as yf

from src.layout import build_layout
from src.callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "StoxAI - ProStock Analytics"
# Favicon (signal icon) - Dash looks for favicon in assets/ or uses _favicon
app._favicon = "favicon.svg"
app.layout = build_layout()
register_callbacks(app)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ProStock Analytics - Professional Trading Dashboard")
    print("=" * 70)
    print("\nFEATURES: Technical Indicators, Risk Analytics, Real-time Data,")
    print("  Dark Theme UI, Caching, News. Markets: NSE/BSE, NYSE/NASDAQ")
    print("\nDashboard: http://127.0.0.1:8050")
    print("Testing yfinance (AAPL)...")
    try:
        t = yf.Ticker("AAPL")
        d = t.history(period="5d")
        print("  Connection OK" if not d.empty else "  No data returned")
    except Exception as e:
        print(f"  Warning: {str(e)[:60]}")
    print("=" * 70 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
