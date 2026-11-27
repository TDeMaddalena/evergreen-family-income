"""
app_data.py - data preparation and portfolio set-up for the Streamlit app.

Everything here is copied from the notebook (notebook/Evergreen_backtest.ipynb) so that the app
computes the same numbers. All calculations themselves live in evergreen_core.py, which is
shared with the notebook and must not be changed here.

    Notebook section 2 (cell 5)   -> parameters: START_DATA, START, END, W0, WEIGHTS, RULE, CFG, INSTR, FRED
    Notebook section 3 (cell 7)   -> ASSUMPTIONS for the planning tables
    Notebook section 4 (cells 9-10) -> load_market_data(): universe, prices, instruments, FRED, hedged US
    Notebook section 6 (cell 14)  -> selection(): yearly stock selection
    Notebook section 7 (cell 16)  -> fixed(), sleeve(), five_sleeves(), portfolio specs

If you change one of these cells in the notebook, change the matching part here as well.
"""
from pathlib import Path

import pandas as pd

import evergreen_core as dc

ROOT = Path(__file__).parent

# ---------- sample (notebook cell 5) ----------
START_DATA = '2008-01-01'                 # stock dividends from 2009 feed the first five-year screen
START = pd.Period('2013-12', 'M')         # first selection: last trading day of December 2013
END = pd.Period('2026-10', 'M')           # last complete month -> monthly returns Jan 2014 - Oct 2026

# ---------- client ----------
W0 = 20_000_000                           # EUR 20 m (illustrative)

# ---------- strategic weights ----------
WEIGHTS = {'EZ': 0.30, 'US': 0.30, 'GOV': 0.20, 'CORP': 0.15, 'GOLD': 0.05}
GROUP = {'EZ': 'EQ', 'US': 'EQ', 'GOV': 'BOND', 'CORP': 'BOND', 'GOLD': 'GOLD', 'LINK': 'BOND'}

# ---------- Eurozone stock selection rule ----------
RULE = dict(n_stocks=20, hist_years=5, min_paid_years=5, cut_tol=0.10, net_growth=True,
            max_yield=0.10, loser_quantile=0.05, buffer_rank=30,
            max_per_sector=4, max_per_country=6, fallback='EXW1.DE')

# ---------- rebalancing, payout and costs ----------
CFG = dict(w0=W0, rebal_month=12, band_months=(3, 6, 9),
           bands={'EQ': 0.05, 'BOND': 0.05, 'GOLD': 0.02},
           payout='income', payout_rate=0.035, smooth_months=36,
           tc_stock=0.0010, tc_etf=0.0005, cost_mult=1.0, fallback=RULE['fallback'])

# ---------- instruments (Xetra, EUR); first ticker with complete data is used ----------
INSTR = {'US': ['SPYD.DE'], 'GOV': ['EUNH.DE'], 'CORP': ['EUN5.DE', 'IEAC.AS'], 'GOLD': ['4GLD.DE'],
         'EZ_ETF': ['EXW1.DE'], 'SP500': ['IUSA.DE', 'VUSA.DE'], 'WORLD': ['EUNL.DE'],
         'DIV30': ['EXSG.DE'], 'LINK': ['IBCI.DE']}
NEEDS_DIST = {'US', 'GOV', 'CORP', 'EZ_ETF', 'SP500', 'DIV30', 'LINK'}
US_PROXY, FX = 'SDY', 'EURUSD=X'
FRED = {'rf': 'IR3TIB01EZM156N', 'hicp': 'CP0000EZ19M086NEST', 'us_rate': 'TB3MS'}

# ---------- planning assumptions (notebook cell 7) ----------
ASSUMPTIONS = dict(
    exp_return={'EZ': 0.065, 'US': 0.060, 'GOV': 0.039, 'CORP': 0.035, 'GOLD': 0.020},
    income={'EZ': 0.045, 'US': 0.028, 'GOV': 0.027, 'CORP': 0.030, 'GOLD': 0.000},
    vol={'EZ': 0.17, 'US': 0.15, 'GOV': 0.05, 'CORP': 0.045, 'GOLD': 0.15},
    corr=[[1.00, 0.65, 0.10, 0.40, 0.10],
          [0.65, 1.00, 0.10, 0.40, 0.10],
          [0.10, 0.10, 1.00, 0.90, 0.10],
          [0.40, 0.40, 0.90, 1.00, 0.10],
          [0.10, 0.10, 0.10, 0.10, 1.00]],
    inflation=0.02, costs=0.0015,
    sources={'Eurozone dividend stocks': {'Expected return': 0.065, 'Income': 0.045},
             'US dividend growers': {'Expected return': 0.060, 'Income': 0.028},
             'Euro government bonds': {'Expected return': 0.039, 'Income': 0.027},
             'Euro corporate bonds': {'Expected return': 0.035, 'Income': 0.030},
             'Euro high yield (not held)': {'Expected return': 0.046, 'Income': 0.058},
             'Gold': {'Expected return': 0.020, 'Income': 0.000}})

# ---------- dividend-record variants (notebook cell 25, RULE_VARIANTS) ----------
DIVIDEND_RECORDS = {
    'Base case': dict(min_paid_years=5, cut_tol=0.10, net_growth=True),
    'Strict': dict(min_paid_years=5, cut_tol=0.05, net_growth=False),        # notebook: "first draft"
    'Lenient': dict(min_paid_years=4, cut_tol=None, net_growth=True),        # notebook: "paid 4 of 5 years"
}

SUFFIX = {'.DE': 'Germany', '.PA': 'France', '.MI': 'Italy', '.MC': 'Spain', '.AS': 'Netherlands',
          '.BR': 'Belgium', '.HE': 'Finland', '.VI': 'Austria', '.LS': 'Portugal', '.IR': 'Ireland'}

CACHE = ROOT / 'evergreen_cache'


def _cached(name, fn):
    """Same disk cache as the notebook (cell 10)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f'{name}_{END}.pkl'
    if path.exists():
        return pd.read_pickle(path)
    obj = fn()
    pd.to_pickle(obj, path)
    return obj


def load_market_data():
    """Notebook cells 9-10: universe, prices and dividends, instruments, FRED series, hedged US sleeve."""
    meta = pd.read_csv(ROOT / 'data' / 'universe_msci_emu.csv').set_index('ticker')
    if 'country' not in meta.columns:
        meta['country'] = [SUFFIX['.' + t.split('.')[-1]] for t in meta.index]
    meta = meta[['name', 'sector', 'country']]

    ALL = list(meta.index) + sorted({t for c in INSTR.values() for t in c}) + [US_PROXY, FX]
    close_raw, div_m = _cached('prices', lambda: dc.download_prices(ALL, START_DATA, END))
    close_raw, div_m = close_raw.copy(), div_m.copy()
    full_idx = pd.period_range(close_raw.index.min(), END, freq='M')
    close_raw, div_m = close_raw.reindex(full_idx), div_m.reindex(full_idx).fillna(0.0)
    failed = [t for t in ALL if t not in close_raw.columns]
    meta = meta.loc[meta.index.intersection(close_raw.columns)]

    def usable(tk, need_dist):
        if tk not in close_raw.columns or close_raw[tk].loc[START:END].notna().mean() < 0.95:
            return False
        if need_dist:
            d = div_m[tk].loc[START + 1:END]
            yearly = d.groupby(d.index.year).sum()
            full_years = [y for y in yearly.index if y < END.year]
            return bool((yearly.loc[full_years] > 0).all())
        return True

    eurusd = close_raw[FX].ffill()
    T, notes = {}, []
    for k, cands in INSTR.items():
        ok = [c for c in cands if usable(c, k in NEEDS_DIST)]
        T[k] = ok[0] if ok else None
    if T['US'] is None and US_PROXY in close_raw.columns:
        close_raw['SDY_EUR'], div_m['SDY_EUR'] = dc.usd_to_eur(close_raw[US_PROXY], div_m[US_PROXY], eurusd)
        T['US'] = 'SDY_EUR'
        notes.append('US sleeve: SPYD.DE distributions incomplete on Yahoo -> SDY (same index) converted to EUR')
    for k in ['US', 'GOV', 'CORP', 'GOLD', 'EZ_ETF', 'SP500', 'WORLD', 'DIV30']:
        if T[k] is None:
            raise RuntimeError(f'{k}: no usable ticker in {INSTR[k]} - Yahoo Finance data incomplete')

    rate, hicp, us_rate = (_cached(f'fred_{v}', lambda v=v: dc.load_fred(v)) for v in FRED.values())
    r_eur_ann = (rate.reindex(full_idx).ffill() / 100).shift(1)
    r_usd_ann = (us_rate.reindex(full_idx).ffill() / 100).shift(1)
    rf = r_eur_ann / 12
    cpi = hicp.reindex(full_idx).ffill()

    close_raw['US_HEDGED'], div_m['US_HEDGED'] = dc.fx_hedged_series(
        close_raw[T['US']], div_m[T['US']], eurusd, r_eur_ann, r_usd_ann)

    PX = close_raw.ffill()
    LAST_VALID = {k: close_raw[k].last_valid_index() for k in meta.index}
    etfs = {v for v in T.values() if v} | {'US_HEDGED', 'SDY_EUR', RULE['fallback']}
    return dict(meta=meta, close_raw=close_raw, div_m=div_m, PX=PX, LAST_VALID=LAST_VALID, T=T,
                notes=notes, failed=failed, rf=rf, cpi=cpi, etfs=etfs, country=meta['country'].to_dict())


# ---------- selection (notebook cell 14) ----------
def rebal_dates(cfg=CFG):
    return [p for p in pd.period_range(START, END, freq='M') if p.month == cfg['rebal_month']]


def selection(D, rule):
    return dc.run_selection(rebal_dates(), D['close_raw'], D['div_m'], D['meta'], rule)


# ---------- portfolio specs (notebook cell 16) ----------
def fixed(tk):
    return lambda t: pd.Series({tk: 1.0})


def sleeve(w, group, fn, fallback=RULE['fallback']):
    return {'weight': w, 'group': group, 'fn': fn, 'fallback': fallback}


def five_sleeves(T, W, ez_fn, us_tk, gov_tk=None, link_tk=None):
    spec = {'EZ': sleeve(W['EZ'], 'EQ', ez_fn),
            'US': sleeve(W['US'], 'EQ', fixed(us_tk)),
            'GOV': sleeve(W['GOV'], 'BOND', fixed(gov_tk or T['GOV'])),
            'CORP': sleeve(W['CORP'], 'BOND', fixed(T['CORP'])),
            'GOLD': sleeve(W['GOLD'], 'GOLD', fixed(T['GOLD']))}
    if W.get('LINK', 0) > 0:
        spec['LINK'] = sleeve(W['LINK'], 'BOND', fixed(link_tk or T['LINK']))
    return spec


def full_cfg(D, **override):
    """CFG plus the data-dependent entries (notebook: CFG.update(etfs=..., country=...))."""
    return {**CFG, 'etfs': D['etfs'], 'country': D['country'], **override}


def run(D, spec, cfg):
    return dc.simulate(spec, D['PX'], D['div_m'], START, END, cfg, D['LAST_VALID'])
