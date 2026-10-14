"""150-day SMA breakout scanner. Run: streamlit run app.py"""
from __future__ import annotations

from io import StringIO
from datetime import datetime
import time
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SMA 150 Breakout Scanner", page_icon="📈", layout="wide")
st.title("📈 SMA 150 Breakout Scanner")
st.caption("Find stocks that RECENTLY crossed above their 150-session simple moving average — not merely stocks already above it.")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SMA150Scanner/1.0)"}

@st.cache_data(ttl=86400, show_spinner=False)
def get_index_members(index: str) -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" if index == "S&P 500" else "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = requests.get(url, timeout=20, headers=HEADERS)
    html.raise_for_status()
    tables = pd.read_html(StringIO(html.text))
    for table in tables:
        for column in table.columns:
            if str(column).strip().lower() in ("symbol", "ticker"):
                return sorted(set(clean_symbol(x) for x in table[column].dropna().astype(str) if clean_symbol(x)))
    raise ValueError("Could not locate a symbol column on index constituent page")


def clean_symbol(value: str) -> str:
    symbol = str(value).strip().upper().replace(".", "-")
    return symbol if symbol and symbol.replace("-", "").replace("^", "").isalnum() else ""

@st.cache_data(ttl=86400, show_spinner=False)
def get_us_listings() -> list[str]:
    # NASDAQ Trader official symbol directories (US-listed NASDAQ + other exchanges).
    urls = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
    ]
    symbols = set()
    for url, symbol_key in urls:
        r = requests.get(url, timeout=25, headers=HEADERS)
        r.raise_for_status()
        data = pd.read_csv(StringIO(r.text), sep="|", dtype=str)
        if symbol_key not in data.columns:
            raise ValueError(f"Missing {symbol_key} in exchange symbol file")
        for _, row in data.iterrows():
            ticker = clean_symbol(row.get(symbol_key, ""))
            if not ticker or "FILE CREATION" in str(row.get(symbol_key, "")):
                continue
            if row.get("Test Issue", "N") == "Y" or row.get("ETF", "N") == "Y":
                continue
            if any(c in ticker for c in ("$", "/", " ")):
                continue
            symbols.add(ticker)
    return sorted(symbols)


def parse_prices(data: pd.DataFrame, ticker: str, many: bool) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                prices = data[ticker].copy()
            elif ticker in data.columns.get_level_values(1):
                prices = data.xs(ticker, level=1, axis=1).copy()
            else:
                return pd.DataFrame()
        else:
            prices = data.copy()
        if not {"Close", "Volume"}.issubset(prices.columns):
            return pd.DataFrame()
        return prices[["Close", "Volume"]].dropna(subset=["Close"])
    except (KeyError, ValueError):
        return pd.DataFrame()


def evaluate(ticker: str, prices: pd.DataFrame, window: int, slope_days: int,
             min_price: float, min_avg_volume: int, rising_only: bool) -> dict | None:
    if len(prices) < 151 + slope_days:
        return None
    close = prices["Close"].astype(float)
    volume = prices["Volume"].fillna(0).astype(float)
    sma = close.rolling(150, min_periods=150).mean()
    crossing = (close > sma) & (close.shift(1) <= sma.shift(1))
    possible = crossing.iloc[-window:]
    if not possible.any() or close.iloc[-1] <= sma.iloc[-1]:
        return None
    if rising_only and not (sma.iloc[-1] > sma.iloc[-1-slope_days]):
        return None
    avg_volume = volume.iloc[-20:].mean()
    if close.iloc[-1] < min_price or avg_volume < min_avg_volume:
        return None
    breakout = possible[possible].index[-1]
    sessions_ago = len(close) - 1 - close.index.get_loc(breakout)
    return {
        "Ticker": ticker,
        "Last Price": round(float(close.iloc[-1]), 2),
        "SMA 150": round(float(sma.iloc[-1]), 2),
        "% Above SMA": round(float((close.iloc[-1] / sma.iloc[-1] - 1) * 100), 2),
        "Breakout Date": pd.Timestamp(breakout).strftime("%Y-%m-%d"),
        "Sessions Ago": int(sessions_ago),
        "SMA Rising": "Yes" if sma.iloc[-1] > sma.iloc[-1-slope_days] else "No",
        "20d Avg Volume": int(avg_volume),
        "Last Data Date": pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d"),
    }

with st.sidebar:
    st.header("Scan settings")
    universe = st.selectbox("Stock universe", ["S&P 500", "Nasdaq-100", "Custom ticker list", "All U.S.-listed (slow)"])
    custom = st.text_area("Tickers (comma/space separated)", "NVDA, PLTR, AAPL, MSFT, SOUN, AXON", height=90) if universe == "Custom ticker list" else ""
    window = st.slider("Breakout within last N sessions", 1, 20, 5)
    rising_only = st.checkbox("Require SMA 150 to be rising", value=True)
    slope_days = st.slider("Compare SMA against N sessions ago", 1, 20, 5)
    min_price = st.number_input("Minimum share price ($)", min_value=0.0, max_value=10000.0, value=5.0, step=1.0)
    min_avg_volume = st.number_input("Minimum 20-session average volume", min_value=0, max_value=100000000, value=500000, step=100000)
    limit = st.slider("Maximum tickers to scan", 50, 5000, 500, step=50, disabled=(universe == "Custom ticker list"))
    st.caption("Uses publicly accessible Yahoo Finance price history via yfinance. Data may be delayed, incomplete, or rate-limited.")

st.markdown("**Definition:** A breakout occurs when the daily closing price moves from at-or-below the SMA 150 to above it. The stock must **still be above** the SMA 150 today. The 150-day SMA uses trading sessions, not calendar days.")
scan = st.button("🔍 Scan stocks now", type="primary", use_container_width=True)

if scan:
    try:
        if universe == "Custom ticker list":
            import re
            tickers = list(dict.fromkeys(filter(None, (clean_symbol(s) for s in re.split(r"[\s,;]+", custom)))))
        elif universe == "All U.S.-listed (slow)":
            tickers = get_us_listings()[:limit]
            st.info(f"Scanning the first {len(tickers):,} alphabetically sorted eligible symbols. Raise the maximum ticker count to broaden coverage. This is not the entire market unless the cap reaches the listing size.")
        else:
            tickers = get_index_members(universe)[:limit]
        if not tickers:
            st.error("No valid tickers found.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not load the ticker universe: {exc}. Try Custom ticker list instead.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)
    matches, unsuccessful = [], []
    chunk_size = 40
    for start in range(0, len(tickers), chunk_size):
        subset = tickers[start:start+chunk_size]
        status.write(f"Scanning {min(start+len(subset), len(tickers)):,} of {len(tickers):,} symbols…")
        try:
            batch = yf.download(subset, period="1y", interval="1d", auto_adjust=True,
                                group_by="ticker", threads=True, progress=False, timeout=25)
        except Exception:
            unsuccessful.extend(subset)
            progress.progress(min(1.0, (start+len(subset))/len(tickers)))
            continue
        for ticker in subset:
            price_data = parse_prices(batch, ticker, len(subset) > 1)
            if price_data.empty:
                unsuccessful.append(ticker)
                continue
            result = evaluate(ticker, price_data, window, slope_days, min_price, min_avg_volume, rising_only)
            if result:
                matches.append(result)
        progress.progress(min(1.0, (start+len(subset))/len(tickers)))
        time.sleep(0.15)
    status.empty()
    progress.empty()
    st.session_state["results"] = pd.DataFrame(matches)
    st.session_state["scan_summary"] = (len(tickers), len(unsuccessful), universe, datetime.now().strftime("%Y-%m-%d %H:%M"))

if "results" in st.session_state:
    results = st.session_state["results"]
    checked, failed, universe_name, scanned_at = st.session_state["scan_summary"]
    a, b, c = st.columns(3)
    a.metric("Breakout matches", len(results))
    b.metric("Tickers attempted", checked)
    c.metric("Missing / failed quotes", failed)
    st.caption(f"Universe: {universe_name} · Scan run: {scanned_at} (server time) · Most recent price date shown for each match")
    if failed:
        st.warning(f"Price retrieval failed for {failed} ticker(s). Results may be incomplete; retry later or scan a smaller universe.")
    if results.empty:
        st.info("No matching breakouts found with the chosen filters. Try increasing the breakout window or unchecking the rising-SMA requirement.")
    else:
        results = results.sort_values(["Sessions Ago", "% Above SMA"], ascending=[True, False])
        st.dataframe(results, use_container_width=True, hide_index=True,
                     column_config={"Ticker": st.column_config.TextColumn("Ticker"), "% Above SMA": st.column_config.NumberColumn("% Above SMA", format="%.2f%%")})
        st.download_button("⬇ Download matching stocks as CSV", data=results.to_csv(index=False).encode("utf-8"),
                           file_name="sma150_breakouts.csv", mime="text/csv")
        st.caption("Tip: Use the ticker links in your broker or chart site for further review. This scanner is informational, not investment advice.")
else:
    st.info("Choose your filters on the left, then click **Scan stocks now**. No account or premium TradingView subscription required.")
