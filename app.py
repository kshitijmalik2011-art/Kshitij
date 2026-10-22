import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Crypto Volatility Forecaster", layout="wide", initial_sidebar_state="expanded")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FNG_URL = "https://api.alternative.me/fng/"
COIN_MAP = {"Bitcoin (BTC)": "bitcoin", "Ethereum (ETH)": "ethereum"}

# ---------------------------------------------------------------------------
# Theme — clean white surface with a bright yellow accent
# ---------------------------------------------------------------------------
BG = "#FFFFFF"
BG_GRADIENT = "radial-gradient(circle at 15% 0%, #FFFDF2 0%, #FFFFFF 45%)"
SURFACE = "#FFFFFF"
SURFACE_2 = "#FFF8DE"
BORDER = "#F2DE8C"
BORDER_SOFT = "#F7EBB8"
TEXT = "#20200F"
TEXT_MUTED = "#8A8262"
BULL = "#2E9E5B"
BEAR = "#E14B3D"
NEUTRAL = "#E8A500"
BRAND = "#F5C518"
BRAND_SOFT = "rgba(245, 197, 24, 0.18)"
GRID = "#F3ECC7"

REGIME_COLOR = {"bull": BULL, "bear": BEAR, "consolidation": NEUTRAL}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {BG_GRADIENT};
    color: {TEXT};
}}

#MainMenu, footer {{ visibility: hidden; }}

section[data-testid="stSidebar"] {{
    background: {SURFACE_2};
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}

section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {TEXT} !important;
}}

.dash-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 10px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 20px;
    margin-bottom: 30px;
}}

.dash-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, {TEXT} 0%, {BRAND} 120%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}

.dash-subtitle {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: {TEXT_MUTED};
    margin-top: 6px;
    letter-spacing: 0.03em;
}}

.dash-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 6px 14px;
    background: {SURFACE};
}}

.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-bottom: 26px;
}}

.kpi-card {{
    background: linear-gradient(160deg, {SURFACE} 0%, {SURFACE_2} 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid var(--accent, {BRAND});
    border-radius: 12px;
    padding: 18px 20px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}

.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(232, 165, 0, 0.18);
    border-color: {BRAND};
}}

.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: {TEXT_MUTED};
    margin-bottom: 10px;
}}

.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    color: {TEXT};
}}

.kpi-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {TEXT_MUTED};
    margin-top: 7px;
}}

.section-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 22px;
    box-shadow: 0 4px 18px rgba(232, 165, 0, 0.10);
}}

.section-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15.5px;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 4px;
}}

.section-caption {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {TEXT_MUTED};
    margin-bottom: 16px;
}}

.stButton > button {{
    background: linear-gradient(90deg, {BRAND} 0%, #E0A700 100%);
    color: #17140A;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    padding: 11px 0;
    transition: filter 0.15s ease, transform 0.15s ease;
}}

.stButton > button:hover {{
    filter: brightness(1.08);
    transform: translateY(-1px);
    color: #17140A;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}

.stTabs [data-baseweb="tab"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 13.5px;
    color: {TEXT_MUTED};
    background: transparent;
    padding: 10px 16px;
}}

.stTabs [aria-selected="true"] {{
    color: {BRAND} !important;
    border-bottom: 2px solid {BRAND} !important;
}}

.badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 12px;
    border-radius: 999px;
}}

.info-note {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: {TEXT_MUTED};
    background: {BRAND_SOFT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 4px;
    line-height: 1.5;
}}

hr {{
    border-color: {BORDER};
}}

::-webkit-scrollbar {{ height: 8px; width: 8px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 8px; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(coin_id, days):
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "inr", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["ts", "price"])
    vols = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
    df = prices.merge(vols, on="ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.floor("D")
    df = df.groupby("date").last()[["price", "volume"]]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed_index(limit=500):
    r = requests.get(FNG_URL, params={"limit": limit, "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.floor("D")
    df["fng_value"] = df["value"].astype(float)
    return df.set_index("date")[["fng_value"]].sort_index()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["sma_20"] = df["price"].rolling(20).mean()
    df["sma_50"] = df["price"].rolling(50).mean()
    df["volatility_14"] = df["return"].rolling(14).std()
    delta = df["price"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df.dropna()


def detect_regime(df):
    vol_threshold = df["volatility_14"].quantile(0.65)
    trend_gap = (df["sma_20"] - df["sma_50"]).abs() / df["sma_50"]

    def classify(v, g, s20, s50):
        if v <= vol_threshold and g < 0.015:
            return "consolidation"
        return "bull" if s20 > s50 else "bear"

    return pd.Series(
        [classify(v, g, s20, s50) for v, g, s20, s50 in
         zip(df["volatility_14"], trend_gap, df["sma_20"], df["sma_50"])],
        index=df.index,
    )


# ---------------------------------------------------------------------------
# LSTM model — epochs are chosen automatically via early stopping, no
# manual "epochs" control is exposed to the user.
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_windows(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def train_lstm(train_prices, window, lr=0.005, max_epochs=300, patience=20, min_delta=1e-5):
    """Trains the LSTM with automatic early stopping — the number of epochs
    is decided on the fly from the loss curve, so nothing needs to be set
    manually."""
    p_min, p_max = train_prices.min(), train_prices.max()
    norm = (train_prices - p_min) / (p_max - p_min + 1e-9)
    X, y = make_windows(norm, window)
    X = X.reshape(-1, window, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    Xt, yt = torch.tensor(X), torch.tensor(y)

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    loss_history = []
    best_loss = float("inf")
    stall_count = 0

    for _ in range(max_epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= patience:
            break

    return model, (p_min, p_max), loss_history[-1], len(loss_history), loss_history


def lstm_forecast(model, history_norm_window, steps, bounds):
    p_min, p_max = bounds
    window = list(history_norm_window)
    preds_norm = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(window[-len(history_norm_window):]).reshape(1, -1, 1).astype(np.float32))
            pred = model(x).item()
            preds_norm.append(pred)
            window.append(pred)
    return np.array(preds_norm) * (p_max - p_min) + p_min


def arima_forecast(train_prices, steps):
    model = ARIMA(train_prices, order=(2, 1, 2)).fit()
    return model.forecast(steps=steps).values


REGIME_SENTIMENT_WEIGHT = {"bull": 0.10, "bear": 0.10, "consolidation": 0.25}


def adaptive_ensemble_forecast(lstm_preds, sentiment_score, regime, volatility):
    w = REGIME_SENTIMENT_WEIGHT.get(regime, 0.15)
    adjustment_pct = w * sentiment_score * min(volatility * 10, 1.0)
    step_multiplier = 1 + adjustment_pct * np.linspace(0.3, 1.0, len(lstm_preds))
    return lstm_preds * step_multiplier


def eval_metrics(actual, pred):
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return {"MAE (\u20b9)": mae, "RMSE (\u20b9)": rmse, "MAPE (%)": mape}


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def plotly_base_layout(fig, height=380):
    fig.update_layout(
        height=height,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="IBM Plex Mono", color=TEXT_MUTED, size=11),
        legend=dict(orientation="h", y=1.14, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT_MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT_MUTED),
        hoverlabel=dict(bgcolor=SURFACE_2, font=dict(color=TEXT, family="IBM Plex Mono")),
    )
    return fig


def kpi_card(label, value, sub, color=TEXT):
    return (
        f'<div class="kpi-card" style="--accent:{color}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def section_open(title, caption=None):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def section_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="dash-header">'
    '<div>'
    '<div class="dash-title">Sentiment-Aware Adaptive Ensemble Forecaster</div>'
    '<div class="dash-subtitle">LSTM (AUTO-TRAINED) &middot; REGIME-CONDITIONED META-LEARNER &middot; ARIMA BASELINE</div>'
    '</div>'
    '<div class="dash-tag">DATA: COINGECKO &middot; ALTERNATIVE.ME</div>'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Configuration")
    coin_label = st.selectbox("Asset", list(COIN_MAP.keys()))
    coin_id = COIN_MAP[coin_label]
    days = st.slider("History window (days)", 120, 365, 200, step=10)
    window = st.slider("LSTM lookback window (days)", 10, 60, 30)
    test_horizon = st.slider("Walk-forward test horizon (days)", 7, 30, 14)

    st.markdown(
        '<div class="info-note">The LSTM trains itself automatically — it stops '
        'as soon as its loss curve stops improving (early stopping), so there '
        "is no epoch count to tune by hand.</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Sentiment source")
    st.caption("Driven entirely by the Crypto Fear & Greed Index — no API key required.")

    st.divider()
    run_btn = st.button("Run pipeline", use_container_width=True)

if run_btn:
    with st.spinner("Fetching price data"):
        try:
            price_df = fetch_price_data(coin_id, days)
        except Exception as e:
            st.error(f"Price fetch failed: {e}")
            st.stop()

    with st.spinner("Fetching Fear & Greed Index"):
        try:
            fng_df = fetch_fear_greed_index(limit=days + 30)
        except Exception:
            fng_df = pd.DataFrame(columns=["fng_value"])

    feat_df = engineer_features(price_df)
    feat_df["regime"] = detect_regime(feat_df)
    feat_df = feat_df.join(fng_df, how="left")
    feat_df["fng_value"] = feat_df["fng_value"].ffill().fillna(50)
    feat_df["fng_sentiment"] = (feat_df["fng_value"] - 50) / 50

    current_sentiment = feat_df["fng_sentiment"].iloc[-1]
    current_regime = feat_df["regime"].iloc[-1]
    current_vol = feat_df["volatility_14"].iloc[-1]

    if len(feat_df) < test_horizon + window + 20:
        st.error("Not enough history for the chosen window/horizon. Increase the history window.")
        st.stop()

    train_df = feat_df.iloc[:-test_horizon]
    test_df = feat_df.iloc[-test_horizon:]

    with st.spinner("Training LSTM (auto epochs via early stopping)"):
        model, bounds, final_loss, epochs_used, loss_history = train_lstm(train_df["price"].values, window)

    p_min, p_max = bounds
    last_window_norm = (train_df["price"].values[-window:] - p_min) / (p_max - p_min + 1e-9)
    lstm_preds = lstm_forecast(model, last_window_norm, test_horizon, bounds)
    ensemble_preds = adaptive_ensemble_forecast(lstm_preds, current_sentiment, current_regime, current_vol)

    with st.spinner("Fitting ARIMA baseline"):
        try:
            arima_preds = arima_forecast(train_df["price"], test_horizon)
        except Exception:
            arima_preds = np.full(test_horizon, np.nan)

    actual = test_df["price"].values
    metrics_lstm = eval_metrics(actual, lstm_preds)
    metrics_ensemble = eval_metrics(actual, ensemble_preds)
    metrics_arima = eval_metrics(actual, arima_preds) if not np.isnan(arima_preds).any() else None

    regime_color = REGIME_COLOR.get(current_regime, TEXT)
    sentiment_color = BULL if current_sentiment > 0.05 else (BEAR if current_sentiment < -0.05 else NEUTRAL)
    improvement = None
    if metrics_arima:
        improvement = (metrics_arima["RMSE (\u20b9)"] - metrics_ensemble["RMSE (\u20b9)"]) / metrics_arima["RMSE (\u20b9)"] * 100
    improvement_color = BULL if (improvement or 0) > 0 else BEAR

    # ---------------- KPI row ----------------
    st.markdown('<div class="kpi-row">' +
        kpi_card("Current price", f"&#8377;{feat_df['price'].iloc[-1]:,.0f}", coin_label, TEXT) +
        kpi_card("Market regime", current_regime.upper(), "SMA20/50 + volatility", regime_color) +
        kpi_card("Sentiment score", f"{current_sentiment:+.2f}", "Fear & Greed Index, -1 to +1", sentiment_color) +
        kpi_card("Ensemble vs ARIMA", f"{improvement:+.1f}%" if improvement is not None else "N/A", "RMSE improvement", improvement_color) +
        kpi_card("Epochs trained", f"{epochs_used}", "auto, early-stopped", BRAND) +
        '</div>', unsafe_allow_html=True)

    # ---------------- Tabs ----------------
    tab_forecast, tab_structure, tab_risk, tab_sentiment, tab_models = st.tabs(
        [" Forecast", " Market Structure", " Risk & Distribution", " Sentiment", " Model Performance"]
    )

    with tab_forecast:
        section_open("Forecast vs actual — walk-forward test window")
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=train_df.index[-60:], y=train_df["price"].iloc[-60:], name="History",
                                  line=dict(color=TEXT_MUTED, width=1.5)))
        fig.add_trace(go.Scatter(x=test_df.index, y=actual, name="Actual",
                                  line=dict(color=TEXT, width=2.5)))
        fig.add_trace(go.Scatter(x=test_df.index, y=lstm_preds, name="LSTM only",
                                  line=dict(color=BRAND, width=1.8, dash="dot")))
        fig.add_trace(go.Scatter(x=test_df.index, y=ensemble_preds, name="Adaptive ensemble",
                                  line=dict(color=BULL, width=2, dash="dash")))
        if metrics_arima:
            fig.add_trace(go.Scatter(x=test_df.index, y=arima_preds, name="ARIMA baseline",
                                      line=dict(color=NEUTRAL, width=1.5, dash="dashdot")))
        fig = plotly_base_layout(fig, height=440)
        fig.update_yaxes(tickprefix="\u20b9")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        section_close()

        section_open("LSTM training curve", "Loss per epoch — training stopped automatically once it plateaued")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(range(1, len(loss_history) + 1)), y=loss_history,
                                       name="Training loss (MSE)", line=dict(color=BRAND, width=2),
                                       fill="tozeroy", fillcolor="rgba(240,185,11,0.10)"))
        fig_loss = plotly_base_layout(fig_loss, height=280)
        fig_loss.update_xaxes(title_text="epoch")
        st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})
        section_close()

    with tab_structure:
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            section_open("Price with SMA20 / SMA50 overlay")
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=feat_df.index, y=feat_df["price"], name="Price",
                                          line=dict(color=TEXT, width=1.6)))
            fig_sma.add_trace(go.Scatter(x=feat_df.index, y=feat_df["sma_20"], name="SMA 20",
                                          line=dict(color=BRAND, width=1.4, dash="dot")))
            fig_sma.add_trace(go.Scatter(x=feat_df.index, y=feat_df["sma_50"], name="SMA 50",
                                          line=dict(color=BEAR, width=1.4, dash="dot")))
            fig_sma = plotly_base_layout(fig_sma, height=340)
            fig_sma.update_yaxes(tickprefix="\u20b9")
            st.plotly_chart(fig_sma, use_container_width=True, config={"displayModeBar": False})
            section_close()

            section_open("Regime timeline")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=feat_df.index, y=feat_df["price"], name="Price",
                                       line=dict(color=TEXT_MUTED, width=1)))
            for regime, color in REGIME_COLOR.items():
                mask = feat_df["regime"] == regime
                fig2.add_trace(go.Scatter(x=feat_df.index[mask], y=feat_df["price"][mask], mode="markers",
                                           name=regime.title(), marker=dict(color=color, size=5)))
            fig2 = plotly_base_layout(fig2, height=320)
            fig2.update_yaxes(tickprefix="\u20b9")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            section_close()

        with col_right:
            section_open("Trading volume")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=feat_df.index, y=feat_df["volume"], name="Volume",
                                      marker_color=BRAND, opacity=0.75))
            fig_vol = plotly_base_layout(fig_vol, height=320)
            st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
            section_close()

            section_open("RSI (14)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=feat_df.index, y=feat_df["rsi_14"], name="RSI 14",
                                          line=dict(color=BRAND, width=1.8)))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color=BEAR)
            fig_rsi.add_hline(y=30, line_dash="dot", line_color=BULL)
            fig_rsi = plotly_base_layout(fig_rsi, height=340)
            fig_rsi.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": False})
            section_close()

    with tab_risk:
        col_left, col_right = st.columns(2)
        with col_left:
            section_open("Rolling volatility (14d)")
            fig_vol14 = go.Figure()
            fig_vol14.add_trace(go.Scatter(x=feat_df.index, y=feat_df["volatility_14"], name="Volatility",
                                            line=dict(color=BEAR, width=1.8), fill="tozeroy",
                                            fillcolor="rgba(240,85,74,0.12)"))
            fig_vol14 = plotly_base_layout(fig_vol14, height=340)
            st.plotly_chart(fig_vol14, use_container_width=True, config={"displayModeBar": False})
            section_close()

        with col_right:
            section_open("Daily return distribution")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=feat_df["return"], nbinsx=40, marker_color=BRAND, opacity=0.8))
            fig_hist = plotly_base_layout(fig_hist, height=340)
            fig_hist.update_xaxes(title_text="daily return")
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
            section_close()

    with tab_sentiment:
        col_left, col_right = st.columns([1.3, 1])
        with col_left:
            section_open("Fear & Greed Index — history")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=feat_df.index, y=feat_df["fng_value"], name="Index (0-100)",
                                       line=dict(color=BRAND, width=1.8), fill="tozeroy",
                                       fillcolor="rgba(240,185,11,0.15)"))
            fig3.add_hline(y=50, line_dash="dot", line_color=TEXT_MUTED)
            fig3 = plotly_base_layout(fig3, height=340)
            fig3.update_yaxes(range=[0, 100])
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            section_close()

        with col_right:
            section_open("Current sentiment gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=feat_df["fng_value"].iloc[-1],
                number=dict(font=dict(color=TEXT, family="Space Grotesk")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=TEXT_MUTED),
                    bar=dict(color=BRAND),
                    bgcolor=SURFACE,
                    bordercolor=BORDER,
                    steps=[
                        {"range": [0, 25], "color": "rgba(240,85,74,0.35)"},
                        {"range": [25, 45], "color": "rgba(240,185,11,0.20)"},
                        {"range": [45, 55], "color": "rgba(137,146,166,0.20)"},
                        {"range": [55, 75], "color": "rgba(240,185,11,0.20)"},
                        {"range": [75, 100], "color": "rgba(46,204,148,0.35)"},
                    ],
                ),
            ))
            fig_gauge.update_layout(height=340, paper_bgcolor=SURFACE, font=dict(family="IBM Plex Mono", color=TEXT_MUTED),
                                     margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
            section_close()

    with tab_models:
        section_open("Error metrics — lower is better")
        rows = [{"Model": "LSTM only", **metrics_lstm}, {"Model": "Adaptive ensemble", **metrics_ensemble}]
        if metrics_arima:
            rows.append({"Model": "ARIMA baseline", **metrics_arima})
        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)
        section_close()

        section_open("RMSE comparison across models")
        fig_bar = go.Figure()
        model_names = metrics_df.index.tolist()
        bar_colors = [BRAND, BULL, NEUTRAL][:len(model_names)]
        fig_bar.add_trace(go.Bar(x=model_names, y=metrics_df["RMSE (\u20b9)"], marker_color=bar_colors))
        fig_bar = plotly_base_layout(fig_bar, height=320)
        fig_bar.update_yaxes(tickprefix="\u20b9")
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        section_close()

else:
    st.markdown(
        '<div class="section-card">Set parameters in the sidebar and click <b>Run pipeline</b>. '
        'The LSTM trains itself automatically — no epoch count to configure.</div>',
        unsafe_allow_html=True,
    )