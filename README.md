# Evergreen Family Income

**Dividends to live on. Capital to pass on.**
Group BE · Nova SBE 2270 Financial Modeling · Group assignment 2026/27

A rules-based multi-asset income portfolio for a European single-family office: 30% Eurozone dividend stocks selected by our own rule, 30% US dividend aristocrats (SPYD), 20% euro government bonds, 15% euro corporate bonds and 5% gold. All income is paid out to the family, and capital is never sold to fund spending.

## Repository

| Path | Content |
|---|---|
| `notebook/Evergreen_backtest.ipynb` | Full backtest; reproduces every number, table and figure in the prospectus |
| `evergreen_core.py` | Core functions (selection rule, simulation, performance measures), shared by the notebook and the app |
| `data/universe_msci_emu.csv` | Stock universe: MSCI EMU constituents mapped to Yahoo Finance tickers |
| `streamlit_app.py` | Streamlit app: change the strategy's assumptions and compare with the base case and benchmarks |
| `app_data.py` | Data preparation and portfolio set-up for the app, copied from notebook sections 2–7 |
| `requirements.txt` | Python dependencies (notebook and app) |

## Run the notebook

1. Open `notebook/Evergreen_backtest.ipynb` in Google Colab: *File → Open notebook → GitHub*, paste `https://github.com/NiklasWMueller2404/evergreen-family-income`.
2. *Runtime → Run all*. The first run takes about 5 minutes.

All market data is downloaded at run time through APIs: Yahoo Finance via `yfinance` (prices, dividends, ETF distributions, EUR/USD) and FRED (euro area 3-month rate, euro area HICP, US 3-month T-bill). The only file read from this repository is the universe list. The sample runs from the first selection on 31 Dec 2013 to 31 Oct 2026 (monthly), with dates set explicitly in section 2 of the notebook.

## Stock universe

`data/universe_msci_emu.csv` lists the 211 MSCI EMU constituents in the iShares Core MSCI EMU UCITS ETF holdings of 25 Nov 2026, with Yahoo tickers (exchange suffix: .DE, .PA, .MI, .MC, .AS, .BR, .HE, .VI, .LS, .IR), sector and country as classified by the index provider. AerCap and Nebius are left out because they are listed only in New York; Coca-Cola Europacific Partners uses its Amsterdam listing. Because these are today's constituents, the backtest carries survivorship bias; the notebook measures it with a 1/N diagnostic.

## App

`streamlit_app.py` lets the user change the strategy's assumptions and immediately see the backtest, the benchmark comparison and all performance measures. Every setting starts at the **base case** of the prospectus, and *Reset to base case* restores it.

**Adjustable assumptions:** allocation (equity share 40–80% or custom weights), US currency hedge, the Eurozone stock rule (dividend record: base case / strict / lenient, number of stocks, sector and country caps, loser filter, holdings buffer), rebalancing bands, trading costs, and the payout rule (all income or smoothed).

**Tabs:**
- **Performance:** annualised average return, volatility, Sharpe ratio, beta, alpha and information ratio vs. B1 Policy twin, with a short interpretation; cumulative performance vs. B1–B3 and the 1/N diagnostic (start = 1); all measures and drawdowns on request.
- **Income & wealth:** what the family is paid each year, real wealth after payouts, stress years and calendar-year returns.
- **Portfolio:** the yearly Eurozone stock selection and the sleeve attribution (our rule vs. EURO STOXX 50, Select Dividend 30 and 1/N universe; SPYD vs. S&P 500).
- **About:** strategy, planning assumptions, data and limitations.

The app uses the same portfolio names as the notebook (B1 Policy twin, B2 60/40, B3 Dividend ETF, 1/N diagnostic).

**Consistency with the notebook.** All calculations use `evergreen_core.py`, the same module the notebook writes in section 1. `app_data.py` copies the notebook's parameters, data preparation and portfolio set-up (sections 2–7; each part names its notebook cell). With all settings on the base case, the app reproduces the notebook's monthly returns and performance tables exactly. **`evergreen_core.py` must stay identical to the code in section 1 of the notebook, and `app_data.py` must follow sections 2–7. If you change one, change the other.**

### Run the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first start downloads all market data from Yahoo Finance and FRED (about a minute) and stores it in `evergreen_cache/`; after that each new scenario takes about 10 seconds.

### Deploy

Streamlit Community Cloud: *Create app* → this repository → branch `main` → main file `streamlit_app.py`.

## Group BE

Tobias de Maddalena · Nando Baumann · Felix Löhnert · Niklas Müller · Lieke [surname] · Morris [surname]

*For academic purposes only. Not investment advice. Past performance is not a reliable indicator of future results.*
