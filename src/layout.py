"""StoxAI - Premium 3D Dashboard Layout with Neural Network Background."""
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.config import COLORS


def build_layout():
    """Build premium StoxAI layout with 3D elements and animations."""
    return html.Div(
        [
            # Animated Neural Network Background
            _neural_background(),
            
            # Custom Loading Animation
            _custom_loading(),
            
            # Main Container
            dbc.Container(
                [
                    _premium_navbar(),
                    _hero_search_section(),
                    
                    # Content Sections (with Dash loading indicator while callback runs)
                    dcc.Loading(
                        id="loading-dashboard",
                        type="circle",
                        color=COLORS["primary"],
                        fullscreen=False,
                        children=html.Div(
                            id="loading-wrapper",
                            children=[
                                html.Div(id="error-message"),
                                html.Div(id="stats-grid"),
                                html.Div(id="risk-metrics-section"),
                                html.Div(id="chart-section"),
                                html.Div(id="signals-section"),
                                html.Div(id="news-section"),
                            ],
                            style={"position": "relative", "zIndex": "10"},
                        ),
                    ),
                    
                    _premium_footer(),
                ],
                fluid=True,
                style={
                    "background": "transparent",
                    "minHeight": "100vh",
                    "padding": "0",
                    "position": "relative",
                    "zIndex": "10",
                },
            ),
        ],
        style={
            "background": COLORS["background"],
            "minHeight": "100vh",
            "position": "relative",
            "overflow": "hidden",
        },
    )


def _neural_background():
    """Animated neural network background using Canvas."""
    return html.Div(
        [
            html.Canvas(
                id="neural-canvas",
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "zIndex": "1",
                    "opacity": "0.15",
                },
            ),
            # Gradient Overlays
            html.Div(
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "background": f"radial-gradient(circle at 20% 20%, {COLORS['card_glow']} 0%, transparent 50%), radial-gradient(circle at 80% 80%, {COLORS['card_glow']} 0%, transparent 50%)",
                    "zIndex": "2",
                    "pointerEvents": "none",
                }
            ),
        ]
    )


def _custom_loading():
    """Premium 3D loading animation."""
    return html.Div(
        id="global-loading",
        children=[
            html.Div(
                [
                    # 3D Cube Loader
                    html.Div(
                        [
                            html.Div(className="cube-face cube-front"),
                            html.Div(className="cube-face cube-back"),
                            html.Div(className="cube-face cube-right"),
                            html.Div(className="cube-face cube-left"),
                            html.Div(className="cube-face cube-top"),
                            html.Div(className="cube-face cube-bottom"),
                        ],
                        className="loading-cube",
                    ),
                    html.Div("Analyzing Market Data...", className="loading-text"),
                    html.Div(className="loading-bar-container", children=[
                        html.Div(className="loading-bar-fill")
                    ]),
                ],
                className="loading-content",
            )
        ],
        style={"display": "none"},
        className="loading-overlay",
    )


def _navbar_logo():
    """Signal icon + typographic StoxAI (Option 3 + Option 1 from logo concepts)."""
    return html.Div(
        [
            html.Img(
                src="/assets/stoxai_signal_icon.svg",
                alt="StoxAI",
                style={"height": "48px", "width": "48px", "flexShrink": "0"},
                className="navbar-signal-icon",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Stox", className="typo-stox", style={"color": COLORS["text_secondary"], "fontWeight": "300", "letterSpacing": "-1px"}),
                            html.Span("AI", className="typo-ai", style={"color": COLORS["primary"], "fontWeight": "500", "letterSpacing": "-2px"}),
                            html.Span(
                                className="typo-dot",
                                style={
                                    "width": "5px",
                                    "height": "5px",
                                    "background": COLORS["primary"],
                                    "borderRadius": "1px",
                                    "display": "inline-block",
                                    "position": "relative",
                                    "top": "2px",
                                    "marginLeft": "2px",
                                    "verticalAlign": "top",
                                },
                            ),
                        ],
                        className="brand-title typo-logo",
                        style={"fontSize": "1.75rem", "fontWeight": "600", "lineHeight": "1", "display": "inline-flex", "alignItems": "baseline"},
                    ),
                    html.Div(
                        [
                            html.Span("●", className="pulse-dot", style={"color": COLORS["primary"], "fontSize": "0.5rem"}),
                            html.Span("AI-Powered Trading Intelligence", style={"marginLeft": "6px"}),
                        ],
                        className="brand-subtitle",
                        style={"fontSize": "0.7rem", "color": COLORS["text_secondary"], "fontWeight": "500", "letterSpacing": "0.5px", "textTransform": "uppercase", "marginTop": "4px", "display": "flex", "alignItems": "center"},
                    ),
                ],
            ),
        ],
        style={"display": "flex", "alignItems": "center", "gap": "16px"},
    )


def _premium_navbar():
    """Premium navbar with signal icon + typographic StoxAI."""
    return html.Nav(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    _navbar_logo(),
                                ],
                                width=12,
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            _status_badge("Neural Engine", COLORS["primary"], "🧠"),
                                            _status_badge("Real-Time", COLORS["primary"], "⚡"),
                                            _status_badge("ML Active", COLORS["primary"], "🎯"),
                                        ],
                                        className="status-badges",
                                    )
                                ],
                                width=12,
                                md=6,
                                className="d-none d-md-block",
                            ),
                        ],
                        align="center",
                    )
                ],
                fluid=True,
            )
        ],
        className="premium-navbar",
    )


def _status_badge(text, color, icon):
    """Create animated status badge."""
    return html.Div(
        [
            html.Span(icon, style={"marginRight": "6px", "fontSize": "0.9rem"}),
            html.Span(text),
        ],
        className="status-badge",
        style={
            "border": f"1px solid {color}",
            "color": color,
        },
    )


def _hero_search_section():
    """Premium hero section with 3D search."""
    return html.Div(
        [
            dbc.Container(
                [
                    # Hero: Typographic StoxAI (Option 1) + tagline
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Stox", className="typo-stox", style={"color": COLORS["text_secondary"], "fontWeight": "300", "letterSpacing": "-1px"}),
                                    html.Span("AI", className="typo-ai", style={"color": COLORS["primary"], "fontWeight": "600", "letterSpacing": "-3px"}),
                                    html.Span(
                                        className="typo-dot",
                                        style={
                                            "width": "8px",
                                            "height": "8px",
                                            "background": COLORS["primary"],
                                            "borderRadius": "2px",
                                            "display": "inline-block",
                                            "position": "relative",
                                            "top": "4px",
                                            "marginLeft": "4px",
                                            "verticalAlign": "top",
                                            "boxShadow": f"0 0 12px {COLORS['primary']}",
                                        },
                                    ),
                                ],
                                className="hero-typo-logo",
                                style={"fontSize": "3rem", "fontWeight": "600", "lineHeight": "1", "display": "inline-flex", "alignItems": "baseline", "marginBottom": "1rem"},
                            ),
                            html.H1(
                                "Predict. Analyze. Profit.",
                                className="hero-title",
                            ),
                            html.P(
                                "Harness the power of AI to make data-driven trading decisions",
                                className="hero-subtitle",
                            ),
                        ],
                        className="hero-text",
                    ),
                    
                    # 3D Search Card
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            # Search Input with 3D effect
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.I(className="fas fa-search search-icon"),
                                                            dbc.Input(
                                                                id="ticker-input",
                                                                placeholder="Enter stock symbol (e.g., AAPL, TSLA, NVDA, RELIANCE.NS)",
                                                                type="text",
                                                                className="premium-input",
                                                                debounce=True,
                                                            ),
                                                        ],
                                                        className="search-input-wrapper",
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.Span("Analyze", className="btn-text"),
                                                            html.I(className="fas fa-arrow-right ms-2"),
                                                        ],
                                                        id="analyze-button",
                                                        className="analyze-button",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="search-container",
                                            ),
                                            
                                            # Quick Access Chips
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Span("🔥", style={"marginRight": "8px", "fontSize": "0.9rem"}),
                                                            html.Span("Trending:", style={"marginRight": "12px", "fontWeight": "600"}),
                                                        ],
                                                        style={"display": "flex", "alignItems": "center"},
                                                    ),
                                                    _quick_chip("AAPL", "quick-aapl"),
                                                    _quick_chip("TSLA", "quick-tsla"),
                                                    _quick_chip("NVDA", "quick-nvda"),
                                                    _quick_chip("RELIANCE", "quick-reliance"),
                                                    _quick_chip("TCS", "quick-tcs"),
                                                    _quick_chip("ITC", "quick-itc"),
                                                ],
                                                className="quick-chips",
                                            ),
                                        ],
                                        width=12,
                                    )
                                ]
                            )
                        ],
                        className="search-card-3d",
                    ),
                ],
                fluid=True,
            )
        ],
        className="hero-section",
    )


def _quick_chip(text, chip_id):
    """Create animated quick access chip."""
    return dbc.Button(
        text,
        id=chip_id,
        n_clicks=0,
        className="quick-chip",
    )


def _premium_footer():
    """Premium footer with 3D elements."""
    return html.Div(
        [
            html.Div(className="footer-glow"),
            dbc.Container(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.I(className="fas fa-shield-alt me-2"),
                                    html.Strong("Risk Disclosure: "),
                                    "StoxAI provides advanced analytics for informational purposes only. ",
                                    "Not financial advice. Trading involves risk. Always conduct thorough research.",
                                ],
                                className="footer-disclaimer",
                            ),
                            html.Div(
                                [
                                    html.Span("© 2024 StoxAI", className="me-3"),
                                    html.Span("•", className="mx-2", style={"color": COLORS["border"]}),
                                    html.Span("Powered by Advanced ML & Real-Time Data", className="ms-3"),
                                ],
                                className="footer-copyright",
                            ),
                        ],
                        className="footer-content",
                    )
                ],
                fluid=True,
            )
        ],
        className="premium-footer",
    )


# Layout 3D/animation CSS: assets/layout_styles.css (Dash auto-loads assets; no html.Style.)
