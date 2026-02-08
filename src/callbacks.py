"""StoxAI - Premium Dashboard Callbacks with 3D Elements."""
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html

from src.analysis import analyze_stock, generate_trading_signals
from src.charts import create_professional_dashboard
from src.config import COLORS
from src.data import fetch_stock_news


def _premium_card_style():
    """Premium 3D card styling."""
    return {
        "background": "linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(35, 42, 52, 0.6))",
        "backdropFilter": "blur(20px) saturate(180%)",
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "20px",
        "padding": "2rem",
        "position": "relative",
        "overflow": "hidden",
        "boxShadow": "0 10px 40px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
        "transition": "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
    }


def _label_style():
    """Premium label styling."""
    return {
        "fontSize": "0.7rem",
        "color": COLORS["text_secondary"],
        "fontWeight": "700",
        "textTransform": "uppercase",
        "letterSpacing": "1.5px",
        "marginBottom": "0.75rem",
        "display": "flex",
        "alignItems": "center",
        "gap": "6px",
    }


def _section_title(icon: str, text: str, subtitle: str = ""):
    """Premium section title with 3D effect."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        icon,
                        className="section-icon",
                        style={
                            "fontSize": "2.5rem",
                            "marginRight": "16px",
                            "filter": "drop-shadow(0 0 20px rgba(29, 185, 84, 0.4))",
                        },
                    ),
                    html.Div(
                        [
                            html.H2(
                                text,
                                style={
                                    "fontSize": "2rem",
                                    "fontWeight": "800",
                                    "color": COLORS["text"],
                                    "margin": "0",
                                    "lineHeight": "1",
                                },
                            ),
                            html.P(
                                subtitle,
                                style={
                                    "fontSize": "0.9rem",
                                    "color": COLORS["text_secondary"],
                                    "margin": "0.5rem 0 0 0",
                                    "fontWeight": "400",
                                },
                            ) if subtitle else None,
                        ]
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "2.5rem",
                },
            )
        ],
        className="section-header-animated",
    )


def register_callbacks(app):
    """Register all dashboard callbacks."""

    @app.callback(
        Output("ticker-input", "value"),
        [
            Input("quick-itc", "n_clicks"),
            Input("quick-reliance", "n_clicks"),
            Input("quick-tcs", "n_clicks"),
            Input("quick-aapl", "n_clicks"),
            Input("quick-tsla", "n_clicks"),
            Input("quick-nvda", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def quick_select(_itc, _rel, _tcs, _aapl, _tsla, _nvda):
        tid = ctx.triggered_id
        if not tid:
            return None
        mapping = {
            "quick-itc": "ITC",
            "quick-reliance": "RELIANCE",
            "quick-tcs": "TCS",
            "quick-aapl": "AAPL",
            "quick-tsla": "TSLA",
            "quick-nvda": "NVDA",
        }
        return mapping.get(tid)

    @app.callback(
        [
            Output("stats-grid", "children"),
            Output("risk-metrics-section", "children"),
            Output("chart-section", "children"),
            Output("signals-section", "children"),
            Output("news-section", "children"),
            Output("error-message", "children"),
        ],
        [Input("analyze-button", "n_clicks")],
        [State("ticker-input", "value")],
        prevent_initial_call=True,
    )
    def update_dashboard(_n_clicks, ticker):
        if not ticker:
            return None, None, None, None, None, None
            
        ticker = ticker.strip().upper()
        result = analyze_stock(ticker)
        
        if result is None:
            err = html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "⚠️",
                                style={
                                    "fontSize": "4rem",
                                    "marginBottom": "1rem",
                                    "filter": "drop-shadow(0 0 20px rgba(245, 158, 11, 0.5))",
                                },
                            ),
                            html.H3(
                                "Unable to Fetch Data",
                                style={
                                    "color": COLORS["text"],
                                    "fontWeight": "700",
                                    "marginBottom": "0.5rem",
                                },
                            ),
                            html.P(
                                f"Could not retrieve market data for '{ticker}'. Please verify the symbol and try again.",
                                style={
                                    "color": COLORS["text_secondary"],
                                    "marginBottom": "1.5rem",
                                },
                            ),
                            html.Div(
                                [
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Try: AAPL, TSLA, NVDA for US stocks or add .NS for Indian stocks (e.g., RELIANCE.NS)",
                                ],
                                style={
                                    "color": COLORS["amber"],
                                    "fontSize": "0.9rem",
                                    "padding": "1rem",
                                    "background": "rgba(245, 158, 11, 0.1)",
                                    "borderRadius": "12px",
                                    "border": f"1px solid rgba(245, 158, 11, 0.3)",
                                },
                            ),
                        ],
                        style={
                            **_premium_card_style(),
                            "textAlign": "center",
                            "maxWidth": "600px",
                            "margin": "4rem auto",
                        },
                    )
                ]
            )
            return None, None, None, None, None, err

        df = result["data"]
        ticker = result["ticker"]
        risk_metrics = result["risk_metrics"]
        currency = "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO")) else "$"

        # Extract metrics
        current_price = df["Close"].iloc[-1]
        daily_change = df["Daily_Return"].iloc[-1] * 100
        daily_change_abs = df["Close"].iloc[-1] - df["Close"].iloc[-2]
        weekly_change = df["Weekly_Return"].iloc[-1] * 100 if not pd.isna(df["Weekly_Return"].iloc[-1]) else 0
        monthly_change = df["Monthly_Return"].iloc[-1] * 100 if not pd.isna(df["Monthly_Return"].iloc[-1]) else 0
        rsi = df["RSI"].iloc[-1]
        volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume"].tail(20).mean()
        volatility = df["Volatility_20d"].iloc[-1] * 100 if not pd.isna(df["Volatility_20d"].iloc[-1]) else 0

        # Market Overview Section
        stats_grid = html.Div(
            [
                dbc.Container(
                    [
                        _section_title("📊", "Market Overview", "Real-time price action and key metrics"),
                        
                        # Main Stats Cards
                        dbc.Row(
                            [
                                # Price Card
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("💰", style={"fontSize": "1rem", "marginRight": "6px"}),
                                                    "CURRENT PRICE",
                                                ],
                                                style=_label_style(),
                                            ),
                                            html.Div(
                                                f"{currency}{current_price:.2f}",
                                                className="metric-value-large",
                                                style={
                                                    "fontSize": "2.5rem",
                                                    "fontWeight": "900",
                                                    "margin": "0.5rem 0",
                                                    "lineHeight": "1",
                                                    "background": f"linear-gradient(135deg, {COLORS['primary']}, #10B981)",
                                                    "WebkitBackgroundClip": "text",
                                                    "WebkitTextFillColor": "transparent",
                                                    "filter": "drop-shadow(0 0 10px rgba(29, 185, 84, 0.3))",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "▲" if daily_change > 0 else "▼",
                                                        style={
                                                            "marginRight": "6px",
                                                            "fontSize": "1rem",
                                                        },
                                                    ),
                                                    html.Span(
                                                        f"{abs(daily_change):.2f}%",
                                                        style={
                                                            "fontWeight": "700",
                                                        },
                                                    ),
                                                    html.Span(
                                                        f" ({currency}{daily_change_abs:+.2f})",
                                                        style={
                                                            "color": COLORS["text_secondary"],
                                                            "fontSize": "0.85rem",
                                                            "marginLeft": "8px",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "fontSize": "1.1rem",
                                                    "color": COLORS["success"] if daily_change > 0 else COLORS["amber"],
                                                    "padding": "0.75rem 1rem",
                                                    "background": f"rgba({'29, 185, 84' if daily_change > 0 else '245, 158, 11'}, 0.15)",
                                                    "borderRadius": "12px",
                                                    "display": "inline-flex",
                                                    "alignItems": "center",
                                                    "border": f"1px solid rgba({'29, 185, 84' if daily_change > 0 else '245, 158, 11'}, 0.3)",
                                                },
                                            ),
                                            # Sparkline placeholder
                                            html.Div(
                                                style={
                                                    "marginTop": "1rem",
                                                    "height": "40px",
                                                    "background": f"linear-gradient(90deg, transparent, rgba(29, 185, 84, 0.1), transparent)",
                                                    "borderRadius": "8px",
                                                    "position": "relative",
                                                    "overflow": "hidden",
                                                },
                                            ),
                                        ],
                                        style={**_premium_card_style(), "minHeight": "280px"},
                                        className="hover-lift",
                                    ),
                                    width=12,
                                    md=6,
                                    lg=4,
                                    className="mb-4",
                                ),
                                
                                # Volume Card
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("📈", style={"fontSize": "1rem", "marginRight": "6px"}),
                                                    "VOLUME",
                                                ],
                                                style=_label_style(),
                                            ),
                                            html.Div(
                                                f"{volume/1e6:.2f}M",
                                                className="metric-value-large",
                                                style={
                                                    "fontSize": "2.5rem",
                                                    "fontWeight": "900",
                                                    "margin": "0.5rem 0",
                                                    "lineHeight": "1",
                                                    "color": COLORS["text"],
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Span("20D Avg: ", style={"color": COLORS["text_secondary"], "fontSize": "0.85rem"}),
                                                            html.Span(
                                                                f"{avg_volume/1e6:.2f}M",
                                                                style={
                                                                    "color": COLORS["primary"],
                                                                    "fontSize": "0.95rem",
                                                                    "fontWeight": "700",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                f"{((volume/avg_volume - 1) * 100):+.1f}%",
                                                                style={
                                                                    "color": COLORS["success"] if volume > avg_volume else COLORS["amber"],
                                                                    "fontSize": "0.9rem",
                                                                    "fontWeight": "600",
                                                                },
                                                            ),
                                                            html.Span(
                                                                " vs avg",
                                                                style={
                                                                    "color": COLORS["text_secondary"],
                                                                    "fontSize": "0.8rem",
                                                                    "marginLeft": "4px",
                                                                },
                                                            ),
                                                        ],
                                                        style={"marginTop": "0.5rem"},
                                                    ),
                                                ],
                                                style={
                                                    "padding": "1rem",
                                                    "background": "rgba(29, 185, 84, 0.08)",
                                                    "borderRadius": "12px",
                                                    "marginTop": "1rem",
                                                    "border": f"1px solid rgba(29, 185, 84, 0.2)",
                                                },
                                            ),
                                        ],
                                        style={**_premium_card_style(), "minHeight": "280px"},
                                        className="hover-lift",
                                    ),
                                    width=12,
                                    md=6,
                                    lg=4,
                                    className="mb-4",
                                ),
                                
                                # RSI Card with Visual Indicator
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("🎯", style={"fontSize": "1rem", "marginRight": "6px"}),
                                                    "RSI INDICATOR",
                                                ],
                                                style=_label_style(),
                                            ),
                                            html.Div(
                                                f"{rsi:.1f}",
                                                className="metric-value-large",
                                                style={
                                                    "fontSize": "2.5rem",
                                                    "fontWeight": "900",
                                                    "margin": "0.5rem 0",
                                                    "lineHeight": "1",
                                                    "color": COLORS["amber"] if rsi > 70 else COLORS["success"] if rsi < 30 else COLORS["text"],
                                                },
                                            ),
                                            # RSI Bar
                                            html.Div(
                                                [
                                                    html.Div(
                                                        style={
                                                            "width": f"{rsi}%",
                                                            "height": "100%",
                                                            "background": f"linear-gradient(90deg, {COLORS['success']}, {COLORS['amber'] if rsi > 70 else COLORS['primary']})",
                                                            "borderRadius": "8px",
                                                            "boxShadow": f"0 0 15px rgba({'245, 158, 11' if rsi > 70 else '29, 185, 84'}, 0.5)",
                                                            "transition": "width 1s ease",
                                                        }
                                                    ),
                                                ],
                                                style={
                                                    "height": "12px",
                                                    "background": COLORS["border"],
                                                    "borderRadius": "8px",
                                                    "overflow": "hidden",
                                                    "marginTop": "1rem",
                                                    "position": "relative",
                                                },
                                            ),
                                            html.Div(
                                                "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                                                style={
                                                    "marginTop": "1rem",
                                                    "padding": "0.75rem 1rem",
                                                    "background": f"rgba({'245, 158, 11' if rsi > 70 else '29, 185, 84' if rsi < 30 else '154, 164, 178'}, 0.15)",
                                                    "color": COLORS["amber"] if rsi > 70 else COLORS["success"] if rsi < 30 else COLORS["text_secondary"],
                                                    "borderRadius": "12px",
                                                    "textAlign": "center",
                                                    "fontWeight": "700",
                                                    "fontSize": "0.9rem",
                                                    "border": f"1px solid rgba({'245, 158, 11' if rsi > 70 else '29, 185, 84' if rsi < 30 else '154, 164, 178'}, 0.3)",
                                                },
                                            ),
                                        ],
                                        style={**_premium_card_style(), "minHeight": "280px"},
                                        className="hover-lift",
                                    ),
                                    width=12,
                                    md=6,
                                    lg=4,
                                    className="mb-4",
                                ),
                            ]
                        ),
                        
                        # Secondary Stats Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div("📅 WEEKLY", style={**_label_style(), "justifyContent": "center"}),
                                            html.Div(
                                                f"{weekly_change:+.2f}%",
                                                style={
                                                    "fontSize": "1.8rem",
                                                    "fontWeight": "800",
                                                    "color": COLORS["success"] if weekly_change > 0 else COLORS["amber"],
                                                    "textAlign": "center",
                                                },
                                            ),
                                        ],
                                        style={**_premium_card_style(), "padding": "1.5rem", "textAlign": "center"},
                                        className="hover-lift",
                                    ),
                                    width=12,
                                    md=4,
                                    className="mb-4",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div("📆 MONTHLY", style={**_label_style(), "justifyContent": "center"}),
                                            html.Div(
                                                f"{monthly_change:+.2f}%",
                                                style={
                                                    "fontSize": "1.8rem",
                                                    "fontWeight": "800",
                                                    "color": COLORS["success"] if monthly_change > 0 else COLORS["amber"],
                                                    "textAlign": "center",
                                                },
                                            ),
                                        ],
                                        style={**_premium_card_style(), "padding": "1.5rem", "textAlign": "center"},
                                        className="hover-lift",
                                    ),
                                    width=12,
                                    md=4,
                                    className="mb-4",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Div("📊 VOLATILITY", style={**_label_style(), "justifyContent": "center"}),
                                            html.Div(
                                                f"{volatility:.2f}%",
                                                style={
                                                    "fontSize": "1.8rem",
                                                    "fontWeight": "800",
                                                    "color": COLORS["primary"],
                                                    "textAlign": "center",
                                                },
                                            ),
                                        ],
                                        style={**_premium_card_style(), "padding": "1.5rem", "textAlign": "center"},
                                        className="hover-lift",
                                    ),
                                    width=12,
                                    md=4,
                                    className="mb-4",
                                ),
                            ]
                        ),
                    ],
                    fluid=True,
                )
            ],
            style={
                "padding": "3rem 2rem",
                "background": COLORS["background"],
                "position": "relative",
            },
            className="section-animated",
        )

        # Risk Analytics Section
        def _get_risk_status(key, val):
            if key == "sharpe_ratio":
                return ("Excellent" if val > 1.5 else "Good" if val > 1 else "Fair", COLORS["success"] if val > 1 else COLORS["amber"])
            if key == "win_rate":
                return ("Strong" if val > 55 else "Moderate" if val > 45 else "Weak", COLORS["success"] if val > 50 else COLORS["amber"])
            return ("High Risk" if abs(val) > 20 else "Moderate Risk", COLORS["amber"])

        risk_cards = []
        risk_data = [
            ("Sharpe Ratio", "sharpe_ratio", "📈", False),
            ("Max Drawdown", "max_drawdown", "📉", True),
            ("VaR (95%)", "var_95", "⚠️", True),
            ("Win Rate", "win_rate", "🎯", True),
        ]

        for label, key, icon, is_percent in risk_data:
            val = risk_metrics[key]
            status, color = _get_risk_status(key, val)
            disp = f"{val:.2f}%" if is_percent else f"{val:.2f}"
            
            risk_cards.append(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(icon, style={"fontSize": "1rem", "marginRight": "6px"}),
                                    label,
                                ],
                                style=_label_style(),
                            ),
                            html.Div(
                                disp,
                                style={
                                    "fontSize": "2rem",
                                    "fontWeight": "800",
                                    "margin": "1rem 0",
                                    "color": color,
                                    "textAlign": "center",
                                },
                            ),
                            html.Div(
                                status,
                                style={
                                    "padding": "0.5rem 1rem",
                                    "background": f"rgba({color.replace('#', '').replace('1DB954', '29, 185, 84').replace('F59E0B', '245, 158, 11')}, 0.15)",
                                    "color": color,
                                    "borderRadius": "10px",
                                    "textAlign": "center",
                                    "fontSize": "0.85rem",
                                    "fontWeight": "600",
                                    "border": f"1px solid rgba({color.replace('#', '').replace('1DB954', '29, 185, 84').replace('F59E0B', '245, 158, 11')}, 0.3)",
                                },
                            ),
                        ],
                        style={**_premium_card_style(), "textAlign": "center"},
                        className="hover-lift",
                    ),
                    width=12,
                    md=6,
                    lg=3,
                    className="mb-4",
                )
            )

        risk_section = html.Div(
            [
                dbc.Container(
                    [
                        _section_title("⚖️", "Risk Analytics", "Portfolio risk assessment and performance metrics"),
                        dbc.Row(risk_cards),
                    ],
                    fluid=True,
                )
            ],
            style={
                "padding": "3rem 2rem",
                "background": COLORS["background"],
            },
            className="section-animated",
        )

        # Technical Analysis Chart
        fig = create_professional_dashboard(result)
        chart_section = html.Div(
            [
                dbc.Container(
                    [
                        _section_title("📈", "Technical Analysis", "Advanced charting with indicators and patterns"),
                        html.Div(
                            dcc.Graph(
                                figure=fig,
                                config={
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                },
                                style={"borderRadius": "16px", "overflow": "hidden"},
                            ),
                            style={
                                **_premium_card_style(),
                                "padding": "1rem",
                            },
                        ),
                    ],
                    fluid=True,
                )
            ],
            style={
                "padding": "3rem 2rem",
                "background": COLORS["background"],
            },
            className="section-animated",
        )

        # Trading Signals
        signals = generate_trading_signals(df)
        signal_elements = []
        
        for signal_text, _type, icon, sentiment in signals:
            if sentiment == "Bullish":
                bg = "rgba(29, 185, 84, 0.15)"
                border = "rgba(29, 185, 84, 0.4)"
                color = COLORS["success"]
            elif sentiment == "Bearish":
                bg = "rgba(245, 158, 11, 0.15)"
                border = "rgba(245, 158, 11, 0.4)"
                color = COLORS["amber"]
            else:
                bg = "rgba(154, 164, 178, 0.15)"
                border = "rgba(154, 164, 178, 0.4)"
                color = COLORS["text_secondary"]
            
            signal_elements.append(
                html.Div(
                    [
                        html.Span(
                            icon,
                            style={
                                "fontSize": "1.5rem",
                                "marginRight": "12px",
                                "filter": f"drop-shadow(0 0 8px {color})",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    sentiment,
                                    style={
                                        "fontSize": "0.7rem",
                                        "fontWeight": "700",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "1px",
                                        "color": color,
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.Div(
                                    signal_text,
                                    style={
                                        "fontSize": "0.95rem",
                                        "fontWeight": "600",
                                        "color": COLORS["text"],
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "padding": "1.25rem 1.5rem",
                        "borderRadius": "16px",
                        "background": bg,
                        "border": f"1px solid {border}",
                        "backdropFilter": "blur(10px)",
                        "transition": "all 0.3s ease",
                        "cursor": "default",
                    },
                    className="signal-card",
                )
            )

        signals_section = html.Div(
            [
                dbc.Container(
                    [
                        _section_title("🎯", "AI Trading Signals", "Machine learning powered market insights"),
                        html.Div(
                            signal_elements,
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                                "gap": "1rem",
                            },
                        ),
                    ],
                    fluid=True,
                )
            ],
            style={
                "padding": "3rem 2rem",
                "background": COLORS["background"],
            },
            className="section-animated",
        )

        # Market News
        news_items = fetch_stock_news(ticker)
        news_cards = []
        
        for idx, item in enumerate(news_items[:6]):
            title = item.get("title", "")
            source = item.get("source", "Unknown")
            link = item.get("link", "#")
            
            if title and len(title) > 5:
                news_cards.append(
                    html.A(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            f"{idx + 1}",
                                            style={
                                                "fontSize": "0.8rem",
                                                "fontWeight": "800",
                                                "color": COLORS["primary"],
                                                "background": "rgba(29, 185, 84, 0.15)",
                                                "width": "28px",
                                                "height": "28px",
                                                "borderRadius": "8px",
                                                "display": "flex",
                                                "alignItems": "center",
                                                "justifyContent": "center",
                                                "border": f"1px solid rgba(29, 185, 84, 0.3)",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    title,
                                                    style={
                                                        "fontSize": "1rem",
                                                        "fontWeight": "600",
                                                        "color": COLORS["text"],
                                                        "marginBottom": "0.5rem",
                                                        "lineHeight": "1.5",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.I(className="fas fa-newspaper me-2", style={"fontSize": "0.8rem"}),
                                                        html.Span(source, style={"fontWeight": "600"}),
                                                        html.Span(" • ", style={"margin": "0 8px", "color": COLORS["border"]}),
                                                        html.Span("Recent", style={"color": COLORS["primary"]}),
                                                    ],
                                                    style={
                                                        "fontSize": "0.8rem",
                                                        "color": COLORS["text_secondary"],
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                    },
                                                ),
                                            ],
                                            style={"flex": "1"},
                                        ),
                                        html.I(
                                            className="fas fa-external-link-alt",
                                            style={
                                                "color": COLORS["primary"],
                                                "fontSize": "1rem",
                                                "opacity": "0.5",
                                                "transition": "all 0.3s ease",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "flex-start",
                                        "gap": "1rem",
                                    },
                                ),
                            ],
                            style={
                                **_premium_card_style(),
                                "padding": "1.5rem",
                                "marginBottom": "1rem",
                                "cursor": "pointer",
                            },
                            className="news-card",
                        ),
                        href=link,
                        target="_blank",
                        style={"textDecoration": "none"},
                    )
                )

        if not news_cards:
            news_cards = [
                html.Div(
                    [
                        html.I(className="fas fa-info-circle me-3", style={"fontSize": "1.5rem", "color": COLORS["primary"]}),
                        html.Div(
                            [
                                html.Div("News temporarily unavailable", style={"fontWeight": "600", "marginBottom": "0.5rem"}),
                                html.A(
                                    "Search on Google →",
                                    href=f"https://www.google.com/search?q={ticker.replace('.NS', '').replace('.BO', '')}+stock+news",
                                    target="_blank",
                                    style={"color": COLORS["primary"], "textDecoration": "none", "fontWeight": "600"},
                                ),
                            ],
                        ),
                    ],
                    style={
                        **_premium_card_style(),
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "1rem",
                    },
                )
            ]

        news_section = html.Div(
            [
                dbc.Container(
                    [
                        _section_title("📰", "Market News", "Latest headlines and market updates"),
                        html.Div(news_cards),
                    ],
                    fluid=True,
                )
            ],
            style={
                "padding": "3rem 2rem",
                "background": COLORS["background"],
            },
            className="section-animated",
        )

        return stats_grid, risk_section, chart_section, signals_section, news_section, None


# Additional CSS for animations and interactions
def inject_interaction_styles():
    """Inject additional interaction styles."""
    return html.Style(
        """
        .hover-lift {
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .hover-lift:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(29, 185, 84, 0.2) !important;
        }
        
        .signal-card:hover {
            transform: translateX(8px);
            box-shadow: 0 8px 24px rgba(29, 185, 84, 0.2);
        }
        
        .news-card:hover {
            transform: translateX(8px);
        }
        
        .news-card:hover .fa-external-link-alt {
            opacity: 1 !important;
            transform: translateX(4px);
        }
        
        .section-animated {
            animation: fadeInSection 0.6s ease-out;
        }
        
        @keyframes fadeInSection {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .section-header-animated {
            animation: slideInLeft 0.6s ease-out;
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .metric-value-large {
            animation: countUp 1s ease-out;
        }
        
        @keyframes countUp {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        """
    )