"""150-day SMA breakout scanner. Run: streamlit run app.py"""
from __future__ import annotations

from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="SMA 150 Breakout Scanner", page_icon="📈", layout="wide")
st.title("📈 SMA 150 Breakout Scanner")
st.caption("Find recent SMA 150 breakouts, then compare analyst consensus ratings and average price targets.")

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

@st.cache_data(ttl=21600, show_spinner=False)
def analyst_snapshot(ticker: str) -> dict:
    """Best-effort third-party analyst snapshot; missing is not equivalent to Buy."""
    result = {"Analyst Consensus": "Unavailable", "Average Target Price": None,
              "Target Upside %": None, "Analyst Count": None}
    try:
        info = yf.Ticker(ticker).info or {}
        raw_rating = str(info.get("recommendationKey") or "").strip().lower().replace("_", " ")
        labels = {"strong buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
                  "sell": "Sell", "strong sell": "Strong Sell"}
        result["Analyst Consensus"] = labels.get(raw_rating, "Unavailable")
        count = info.get("numberOfAnalystOpinions")
        if count is not None and pd.notna(count) and float(count) > 0:
            result["Analyst Count"] = int(count)
        target = info.get("targetMeanPrice")
        if target is not None and pd.notna(target) and float(target) > 0:
            result["Average Target Price"] = round(float(target), 2)
    except Exception:
        pass
    return result


def add_analyst_data(matches: list[dict]) -> list[dict]:
    """Retrieve analyst data only for technical matches to limit provider requests."""
    if not matches:
        return matches
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(analyst_snapshot, row["Ticker"]): row for row in matches}
        for job in as_completed(jobs):
            row = jobs[job]
            try:
                details = job.result()
            except Exception:
                details = {"Analyst Consensus": "Unavailable", "Average Target Price": None,
                           "Target Upside %": None, "Analyst Count": None}
            row.update(details)
            target = row.get("Average Target Price")
            price = row.get("Last Price")
            if target is not None and price is not None and price > 0:
                row["Target Upside %"] = round((target / price - 1) * 100, 2)
    return matches

with st.sidebar:
    st.header("Scan settings")
    universe = st.selectbox("Stock universe", ["S&P 500", "Nasdaq-100", "Custom ticker list", "All U.S.-listed (slow)"])
    custom = st.text_area("Tickers (comma/space separated)", "NVDA, PLTR, AAPL, MSFT, SOUN, AXON", height=90) if universe == "Custom ticker list" else ""
    window = st.slider("Breakout within last N sessions", 1, 20, 5)
    rising_only = st.checkbox("Require SMA 150 to be rising", value=True)
    slope_days = st.slider("Compare SMA against N sessions ago", 1, 20, 5)
    min_price = st.number_input("Minimum share price ($)", min_value=0.0, max_value=10000.0, value=5.0, step=1.0)
    min_avg_volume = st.number_input("Minimum 20-session average volume", min_value=0, max_value=100000000, value=500000, step=100000)
    st.divider()
    st.subheader("Analyst ratings & targets")
    include_analysts = st.checkbox("Add analyst ratings and targets", value=True)
    buy_only = st.checkbox("Only Buy / Strong Buy consensus", value=False, disabled=not include_analysts)
    min_upside = st.number_input("Minimum target upside (%) — optional", min_value=0.0,
                                 max_value=1000.0, value=0.0, step=5.0,
                                 disabled=not include_analysts)
    st.caption("Analyst data comes from Yahoo Finance. Some tickers have missing or stale estimates; missing data never qualifies as Buy.")
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
            st.info(
    f"Scanning the first {len(tickers):,} alphabetically sorted eligible symbols. "
    "Raise the maximum ticker count to broaden coverage."
)
            details.append("No consensus price target returned")
    except Exception as exc:
        details.append(f"Price-target request: {type(exc).__name__}: {str(exc)[:100]}")

    try:
        summary = t.get_recommendations_summary()
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            row = summary.iloc[0]
            # yfinance periods usually order as '0m', '-1m', etc.
            if "period" in summary.columns:
                current = summary.loc[summary["period"].astype(str) == "0m"]
                if not current.empty:
                    row = current.iloc[0]
            names = ["strongBuy", "buy", "hold", "sell", "strongSell"]
            counts = [int(number(row.get(k)) or 0) for k in names]
            analyst_count = sum(counts)
            if analyst_count > 0:
                weighted = sum((i + 1) * c for i, c in enumerate(counts)) / analyst_count
                rating = rating_from_mean(weighted)
        if rating is None:
            details.append("No current analyst recommendation counts returned")
    except Exception as exc:
        details.append(f"Ratings request: {type(exc).__name__}: {str(exc)[:100]}")

    # Additional info keys sometimes fill gaps when analyst endpoints have missing fields.
    if target is None or rating is None or analyst_count is None:
        try:
            info = t.get_info() or {}
            if target is None:
                target = number(info.get("targetMeanPrice"))
            if rating is None:
                rating = rating_from_mean(info.get("recommendationMean"))
            if analyst_count is None:
                analyst_count = number(info.get("numberOfAnalystOpinions"))
        except Exception as exc:
            details.append(f"Ticker info request: {type(exc).__name__}: {str(exc)[:100]}")
    return target, rating or "N/A", analyst_count, "; ".join(details) or "Available"


st.sidebar.header("Screen settings")
ticker_text = st.sidebar.text_area("Ticker symbols (comma, space or newline separated)", DEFAULT, height=160)
lookback = st.sidebar.slider("Breakout window (trading days)", 1, 30, 5)
slope_days = st.sidebar.slider("Rising SMA lookback (trading days)", 1, 30, 5)
require_rising = st.sidebar.checkbox("Require rising 150-day SMA", value=True)
require_above = st.sidebar.checkbox("Require price still above SMA", value=True)
show_all = st.sidebar.checkbox("Show all tickers (diagnostic)", value=False)
run = st.sidebar.button("Run scan", type="primary")

st.info("Analyst data is supplemental. Free Yahoo data can be unavailable or rate-limited for some tickers. N/A means unavailable, not a Hold rating or zero target.")

with st.expander("Test analyst data for one ticker", expanded=True):
    test_symbol = st.text_input("Ticker to test", value="NVDA", key="test_symbol").strip().upper()
    if st.button("Test Analyst Data"):
        with st.spinner(f"Checking Yahoo analyst endpoints for {test_symbol}..."):
            try:
                target, rating, count, note = analyst_data(test_symbol)
                c1, c2, c3 = st.columns(3)
                c1.metric("Consensus target", dollars(target))
                c2.metric("Consensus rating", rating)
                c3.metric("Analysts", str(int(count)) if number(count) is not None else "N/A")
                st.caption(f"Data diagnostics: {note}")
            except Exception as exc:
                st.error(f"Analyst data lookup failed: {type(exc).__name__}: {exc}")

if run:
    symbols = list(dict.fromkeys(t.upper().strip() for t in ticker_text.replace(",", " ").split() if t.strip()))
    if not symbols:
        st.warning("Enter at least one ticker.")
    elif len(symbols) > 100:
        st.error("Use 100 tickers or fewer per scan to reduce rate limits.")
    else:
        rows = []
        errors = []
        progress = st.progress(0, text="Checking price history...")
        for i, sym in enumerate(symbols):
            try:
                df = history(sym)
                if len(df) < 160:
                    errors.append(f"{sym}: insufficient price history")
                    continue
                close = df["Close"].astype(float)
                sma = close.rolling(150).mean()
                cross = (close > sma) & (close.shift(1) <= sma.shift(1))
                positions = [j for j in range(max(1, len(df) - lookback), len(df)) if bool(cross.iloc[j])]
                recent = bool(positions)
                latest = positions[-1] if recent else None
                sma_now = number(sma.iloc[-1])
                price = number(close.iloc[-1])
                rising = sma_now is not None and number(sma.iloc[-1 - slope_days]) is not None and sma_now > float(sma.iloc[-1 - slope_days])
                above = price is not None and sma_now is not None and price > sma_now
                qualified = recent and (rising or not require_rising) and (above or not require_above)
                if not (qualified or show_all):
                    continue
                row = {
                    "Ticker": sym,
                    "Close": price,
                    "150 SMA": sma_now,
                    "Breakout date": str(df.index[latest].date()) if latest is not None else "—",
                    "Still above": "Yes" if above else "No",
                    "Rising SMA": "Yes" if rising else "No",
                    "Qualifies": "Yes" if qualified else "No",
                }
                rows.append(row)
            except Exception as exc:
                errors.append(f"{sym}: {type(exc).__name__}: {str(exc)[:100]}")
            finally:
                progress.progress((i + 1) / len(symbols), text=f"Checking prices ({i + 1}/{len(symbols)})")
        progress.empty()
        if not rows:
            st.warning("No qualifying breakouts in this ticker list. Try a longer lookback or 'Show all tickers' to diagnose.")
        else:
            st.write(f"**{sum(r['Qualifies']=='Yes' for r in rows)}** qualifying breakout(s) from **{len(symbols)}** checked ticker(s).")
            analyst_progress = st.progress(0, text="Retrieving analyst data...")
            for i, row in enumerate(rows):
                target, rating, count, note = analyst_data(row["Ticker"])
                row["Analyst rating"] = rating
                row["Analysts"] = int(count) if number(count) is not None else None
                row["Target"] = target
                row["Target upside %"] = round((target / row["Close"] - 1) * 100, 1) if target is not None and row["Close"] else None
                row["Analyst data notes"] = note
                analyst_progress.progress((i + 1) / len(rows), text=f"Analyst data ({i + 1}/{len(rows)})")
            analyst_progress.empty()
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", out.to_csv(index=False).encode(), "sma150_breakouts.csv", "text/csv")
        if errors:
            with st.expander(f"Price-data warnings ({len(errors)})"):
sting size.")
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
    if include_analysts and matches:
        with st.spinner(f"Retrieving analyst ratings and targets for {len(matches)} breakout matches…"):
            matches = add_analyst_data(matches)
        if buy_only:
            matches = [r for r in matches if r.get("Analyst Consensus") in ("Buy", "Strong Buy")]
        if min_upside > 0:
            matches = [r for r in matches if r.get("Target Upside %") is not None
                       and r["Target Upside %"] >= min_upside]
    st.session_state["results"] = pd.DataFrame(matches)
    st.session_state["analyst_enabled"] = include_analysts
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
        cols = {"Ticker": st.column_config.TextColumn("Ticker"),
                "% Above SMA": st.column_config.NumberColumn("% Above SMA", format="%.2f%%")}
        if st.session_state.get("analyst_enabled"):
            cols.update({"Average Target Price": st.column_config.NumberColumn("Average Target Price", format="$%.2f"),
                         "Target Upside %": st.column_config.NumberColumn("Target Upside %", format="%.2f%%")})
        st.dataframe(results, use_container_width=True, hide_index=True, column_config=cols)
        st.download_button("⬇ Download matching stocks as CSV", data=results.to_csv(index=False).encode("utf-8"),
                           file_name="sma150_breakouts.csv", mime="text/csv")
        st.caption("Analyst consensus and targets are third-party estimates, not guarantees. Upside = (mean target / latest scanned close − 1) × 100. Missing estimates are blank. This scanner is informational, not investment advice.")
else:
    st.info("Choose your filters on the left, then click **Scan stocks now**. No account or premium TradingView subscription required.")

