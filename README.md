# 150-Day SMA Breakout Scanner

A point-and-click browser app that screens a list of stocks for *recent* crossovers above their 150-session simple moving average.

## Start on Windows

1. Install Python 3.11 or newer from https://www.python.org/downloads/ (check **Add Python to PATH**).
2. Unzip the project and open a terminal in the `sma150_scanner` folder.
3. Run `py -m pip install -r requirements.txt`.
4. Run `py -m streamlit run app.py`.
5. The scanner opens in your default browser. Click **Scan stocks now**.

On macOS/Linux, use `python3 -m pip install -r requirements.txt` and `python3 -m streamlit run app.py`.

## Put it on the web

Upload `app.py` and `requirements.txt` to a GitHub repository. At https://share.streamlit.io/ sign in, choose **Create app**, select your GitHub repository, and set the main file path to `app.py`. Deployment requires accounts with GitHub and Streamlit Community Cloud; free tier usage and rate limits may apply.

## How it screens

- Daily adjusted closes from yfinance/Yahoo Finance.
- SMA = average of the most recent **150 trading sessions** including the current close.
- A crossover is recorded when today's close > today's SMA and yesterday's close <= yesterday's SMA.
- Crossovers are accepted within the user-selected trailing N sessions, and price must still be above SMA today.
- Optional rising trend filter: today's SMA > SMA N sessions earlier.
- Minimum closing price and 20-session average volume filters.
- Output includes price, SMA, percent above, breakout date, sessions ago, volume and last data date; CSV export.

## Notes / limits

- Data is unofficial, may be delayed, and Yahoo may rate-limit large scans. Run after market close for completed daily candles; intraday candles can change.
- Index membership is retrieved from Wikipedia and can fail if their pages change.
- The "All U.S.-listed" option is capped by default and samples tickers **alphabetically**, which is **not** a market-wide exhaustive scan unless the cap covers the entire listing universe. This may take a long time and encounter data limits.
- Bulk data includes securities that can have unusual ticker formatting or delisted symbols; unavailable prices are counted.
- Free hosting and third-party data are not guaranteed to remain free/available. No advice to buy or sell.
