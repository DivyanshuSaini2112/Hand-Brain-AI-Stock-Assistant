"""Plotly dashboard figure for technical analysis."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS


def create_professional_dashboard(result: dict):
    """Build multi-panel technical analysis figure from analysis result."""
    df = result["data"]
    ticker = result["ticker"]

    fig = make_subplots(
        rows=5,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
        subplot_titles=(
            "Price Action & Technical Indicators",
            "Volume Analysis",
            "RSI (14) & Stochastic Oscillator",
            "Money Flow Index",
            "MACD Indicator",
            "Bollinger Bands Width",
            "CCI & ATR",
            "Support & Resistance",
        ),
        specs=[
            [{"secondary_y": False, "colspan": 2}, None],
            [{"secondary_y": False, "colspan": 2}, None],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
            increasing_fillcolor=COLORS["success"],
            decreasing_fillcolor=COLORS["danger"],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            name="MA20",
            line=dict(color=COLORS["primary"], width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA50"],
            name="MA50",
            line=dict(color=COLORS["secondary"], width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Bollinger_Upper"],
            name="BB Upper",
            line=dict(color=COLORS["text_secondary"], width=1, dash="dot"),
            showlegend=False,
            opacity=0.3,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Bollinger_Lower"],
            name="BB Lower",
            line=dict(color=COLORS["text_secondary"], width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(0, 102, 255, 0.05)",
            showlegend=False,
            opacity=0.3,
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        COLORS["success"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else COLORS["danger"]
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker=dict(color=volume_colors),
            showlegend=False,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color=COLORS["primary"], width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"], opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"], opacity=0.5, row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Stoch_K"],
            name="Stoch %K",
            line=dict(color=COLORS["secondary"], width=1.5),
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Stoch_D"],
            name="Stoch %D",
            line=dict(color=COLORS["warning"], width=1.5),
        ),
        row=3,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MFI"],
            name="MFI",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 102, 255, 0.1)",
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(color=COLORS["primary"], width=2),
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_Signal"],
            name="Signal",
            line=dict(color=COLORS["danger"], width=2),
        ),
        row=4,
        col=2,
    )
    hist_colors = [
        COLORS["success"] if v > 0 else COLORS["danger"]
        for v in df["MACD_Histogram"]
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_Histogram"],
            name="Histogram",
            marker=dict(color=hist_colors),
            showlegend=False,
            opacity=0.5,
        ),
        row=4,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["CCI"],
            name="CCI",
            line=dict(color=COLORS["primary"], width=2),
        ),
        row=5,
        col=1,
    )
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS["danger"], opacity=0.5, row=5, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color=COLORS["success"], opacity=0.5, row=5, col=1)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Support"],
            name="Support",
            line=dict(color=COLORS["success"], width=2, dash="dash"),
        ),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Resistance"],
            name="Resistance",
            line=dict(color=COLORS["danger"], width=2, dash="dash"),
        ),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Price",
            line=dict(color=COLORS["primary"], width=1.5),
        ),
        row=5,
        col=2,
    )

    fig.update_layout(
        height=1400,
        hovermode="x unified",
        plot_bgcolor=COLORS["dark"],
        paper_bgcolor=COLORS["dark"],
        font=dict(family="Inter, sans-serif", size=11, color=COLORS["text"]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=9),
        ),
        margin=dict(l=60, r=30, t=80, b=40),
        xaxis_rangeslider_visible=False,
    )

    for i in range(1, 6):
        for j in range(1, 3):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS["grid"],
                showline=True,
                linewidth=1,
                linecolor=COLORS["border"],
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS["grid"],
                showline=True,
                linewidth=1,
                linecolor=COLORS["border"],
                row=i,
                col=j,
            )
    return fig
