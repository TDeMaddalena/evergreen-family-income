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
| `requirements.txt` | Python dependencies (notebook and app) |

## Run the notebook

1. Open `notebook/Evergreen_backtest.ipynb` in Google Colab: *File → Open notebook → GitHub*, paste `https://github.com/NiklasWMueller2404/evergreen-family-income`.
2. *Runtime → Run all*. The first run takes about 5 minutes.

All market data is downloaded at run time through APIs: Yahoo Finance via `yfinance` (prices, dividends, ETF distributions, EUR/USD) and FRED (euro area 3-month rate, euro area HICP, US 3-month T-bill). The only file read from this repository is the universe list. The sample runs from the first selection on 31 Dec 2013 to 31 Oct 2026 (monthly), with dates set explicitly in section 2 of the notebook.

## Stock universe

`data/universe_msci_emu.csv` lists the 211 MSCI EMU constituents in the iShares Core MSCI EMU UCITS ETF holdings of 25 Nov 2026, with Yahoo tickers (exchange suffix: .DE, .PA, .MI, .MC, .AS, .BR, .HE, .VI, .LS, .IR), sector and country as classified by the index provider. AerCap and Nebius are left out because they are listed only in New York; Coca-Cola Europacific Partners uses its Amsterdam listing. Because these are today's constituents, the backtest carries survivorship bias; the notebook measures it with a 1/N diagnostic.

## App

The Streamlit app goes in `streamlit_app.py` at the repository root and imports `evergreen_core.py`, so the app and the notebook compute identical numbers. **`evergreen_core.py` must stay identical to the code written in section 1 of the notebook. If you change one, change the other.**

Deploy on Streamlit Community Cloud: *New app* → this repository → branch `main` → main file `streamlit_app.py`.

## Group BE

Tobias de Maddalena · Nando Baumann · Felix Löhnert · Niklas Müller · Lieke [surname] · Morris [surname]

*For academic purposes only. Not investment advice. Past performance is not a reliable indicator of future results.*
