"""150-day SMA breakout scanner with optional Yahoo Finance analyst data."""
import io
import re
from datetime import date

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="150-Day SMA Breakout Scanner", layout="wide")
st.title("150-Day SMA Breakout Scanner")
st.caption("Daily closing-price crossovers, with optional consensus ratings and targets from Yahoo Finance.")

DEFAULT_TICKERS = "NVDA AAPL MSFT AMZN GOOGL META TSLA PLTR AMD AVGO INTC MU ARM SOUN COIN MSTR VRT AXON NET CRWD PANW ZS TENB HOOD RDDT RKLB IONQ SOFI APP UBER NFLX SNOW SMCI ORCL DELL QCOM MRVL TSM BABA SHOP CELH"
INDEX_URLS = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


def normalize_tickers(text):
    symbols = re.split(r"[\s,;]+", text.upper().strip())
    return list(dict.fromkeys(x.replace(".", "-") for x in symbols if re.fullmatch(r"[A-Z0-9.^=-]+", x)))


@st.cache_data(ttl=86400, show_spinner=False)
def index_tickers(index_name):
    url = INDEX_URLS[index_name]
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MarketScanner/1.0)"}, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    for table in tables:
        cols = [" ".join(str(part) for part in col) if isinstance(col, tuple) else str(col) for col in table.columns]
        for i, col in enumerate(cols):
            if (index_name == "S&P 500" and col.strip() == "Symbol") or (index_name == "Nasdaq-100" and (col.strip() == "Ticker" or col.strip() == "Ticker symbol")):
                return normalize_tickers(" ".join(table.iloc[:, i].astype(str)))
    raise ValueError("Ticker column not found on the index page")


@st.cache_data(ttl=1800, show_spinner=False)
def price_history(symbols):
    """Returns adjusted daily closing prices for a batch of tickers."""
    frame = yf.download(
        tickers=list(symbols), period="2y", interval="1d", group_by="ticker",
        auto_adjust=True, progress=False, threads=True, timeout=15,
    )
    result = {}
    for symbol in symbols:
        try:
            if isinstance(frame.columns, pd.MultiIndex):
                series = frame[symbol]["Close"]
            else:
                series = frame["Close"]
            series = pd.to_numeric(series, errors="coerce").dropna()
            if len(series) >= 155:
                result[symbol] = series
        except (KeyError, TypeError, ValueError):
            continue
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def analyst_info(symbol):
    try:
        info = yf.Ticker(symbol).info or {}
        raw_target = info.get("targetMeanPrice")
        target = float(raw_target) if raw_target is not None else None
        rating = str(info.get("recommendationKey") or "Not available")
        number = info.get("numberOfAnalystOpinions")
        return rating.replace("_", " ").title(), target, number, ""
    except Exception as exc:
        return "Not available", None, None, f"Analyst data unavailable: {type(exc).__name__}"


def breakout_result(symbol, close, lookback, rising_only):
    sma = close.rolling(window=150, min_periods=150).mean()
    above = close > sma
    crossed = above & ~(above.shift(1).fillna(False).astype(bool))
    recent = crossed.iloc[-lookback:]
    if not bool(above.iloc[-1]) or not bool(recent.any()):
        return None
    if rising_only and not bool(sma.iloc[-1] > sma.iloc[-6]):
        return None
    crossover_day = recent[recent].index[-1]
    last_price = float(close.iloc[-1])
    last_sma = float(sma.iloc[-1])
    return {
        "Ticker": symbol,
        "Price date": str(close.index[-1].date()),
        "Breakout date": str(crossover_day.date()),
        "Close": round(last_price, 2),
        "SMA 150": round(last_sma, 2),
        "Above SMA %": round(100 * (last_price / last_sma - 1), 2),
        "SMA rising (5d)": "Yes" if sma.iloc[-1] > sma.iloc[-6] else "No",
    }


with st.sidebar:
    st.header("Scanner settings")
    universe = st.selectbox("Stock universe", ["Starter watchlist", "S&P 500", "Nasdaq-100", "Custom tickers"])
    custom = st.text_area("Custom tickers (used when selected)", value=DEFAULT_TICKERS, height=125)
    max_tickers = st.number_input("Maximum stocks to scan", min_value=10, max_value=600, value=100, step=10)
    lookback = st.slider("Breakout within past trading days", 1, 30, 10)
    rising = st.checkbox("Require rising 150-day SMA", value=True)
    include_analysts = st.checkbox("Get consensus ratings and price targets", value=True)
    filter_buy = st.checkbox("Only Buy / Strong Buy ratings", value=False, disabled=not include_analysts)
    min_upside = st.number_input("Minimum analyst target upside %", min_value=0, max_value=500, value=0, step=10, disabled=not include_analysts)
    scan = st.button("Run scanner", type="primary", use_container_width=True)

st.info("A breakout means a stock closed above its 150-day simple moving average after closing at or below it the previous trading day. Results must still be above the SMA today. Analyst data is not guaranteed for every stock.")

if scan:
    if universe in INDEX_URLS:
        try:
            tickers = index_tickers(universe)
        except Exception as exc:
            st.error(f"Could not load {universe} ticker list: {exc}. Choose Starter watchlist or Custom tickers instead.")
            st.stop()
    elif universe == "Starter watchlist":
        tickers = normalize_tickers(DEFAULT_TICKERS)
    else:
        tickers = normalize_tickers(custom)

    total = len(tickers)
    tickers = tickers[:int(max_tickers)]
    if not tickers:
        st.error("No valid stock symbols found.")
        st.stop()
    if total > len(tickers):
        st.warning(f"Scanning {len(tickers)} of {total} eligible symbols, in source-list order. Raise Maximum stocks to scan for more coverage.")

    matches = []
    missed = []
    scan_progress = st.progress(0, text="Downloading price histories...")
    for start in range(0, len(tickers), 40):
        batch = tickers[start:start + 40]
        try:
            histories = price_history(tuple(batch))
            for symbol in batch:
                if symbol not in histories:
                    missed.append(symbol)
                    continue
                match = breakout_result(symbol, histories[symbol], lookback, rising)
                if match:
                    matches.append(match)
        except Exception as exc:
            missed.extend(batch)
            st.warning(f"Price request for one batch failed ({type(exc).__name__}).")
        scan_progress.progress(min(start + len(batch), len(tickers)) / len(tickers), text="Checking SMA breakouts...")
    scan_progress.empty()

    if not matches:
        st.warning("No qualifying breakouts found in the scanned stocks. Try a longer breakout window, turn off the rising-SMA requirement, or scan more symbols.")
    else:
        if include_analysts:
            analyst_progress = st.progress(0, text="Loading analyst estimates...")
            for i, row in enumerate(matches):
                rating, target, count, note = analyst_info(row["Ticker"])
                row["Consensus rating"] = rating
                row["Analyst count"] = count
                row["Target"] = round(target, 2) if target is not None else None
                row["Target upside %"] = round((target / row["Close"] - 1) * 100, 1) if target is not None and row["Close"] else None
                row["Analyst data notes"] = note
                analyst_progress.progress((i + 1) / len(matches), text=f"Analyst data ({i + 1}/{len(matches)})")
            analyst_progress.empty()
            if filter_buy:
                matches = [r for r in matches if r["Consensus rating"].lower() in ("buy", "strong buy")]
            if min_upside > 0:
                matches = [r for r in matches if r["Target upside %"] is not None and r["Target upside %"] >= min_upside]

        if matches:
            output = pd.DataFrame(matches)
            output = output.sort_values("Target upside %" if include_analysts else "Above SMA %", ascending=False, na_position="last")
            st.success(f"Found {len(output)} matching stocks out of {len(tickers)} scanned.")
            st.dataframe(output, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", output.to_csv(index=False).encode("utf-8"), file_name="sma150_breakouts.csv", mime="text/csv")
        else:
            st.warning("Breakouts were found, but none met your analyst rating/upside filters. Try turning those filters off.")

    if missed:
        with st.expander(f"Missing/insufficient price data ({len(missed)} symbols)"):
            st.write(", ".join(missed))
    st.caption(f"Scan requested {date.today().isoformat()}. Data may be delayed. Prices and analyst targets are sourced from Yahoo Finance through yfinance; no guarantee of completeness or accuracy.")
else:
    st.write("Choose your settings and click **Run scanner**.")
