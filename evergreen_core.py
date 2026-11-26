"""
evergreen_core.py - core functions of the Evergreen Family Income strategy.
Imported by the notebook AND by the Streamlit app, so both compute identical numbers.

Sections: 1) data  2) Eurozone stock selection  3) simulation  4) measures  5) planning tables
"""
import time
import numpy as np
import pandas as pd


# =====================================================================
# 1) Data
# =====================================================================
def download_prices(tickers, start, end, chunk=40, retries=3, pause=5):
    """Month-end closes (split-adjusted, NOT dividend-adjusted) and dividends per share per month
    (sum of ex-dates in the month) from Yahoo Finance. Never use Adj Close for yields."""
    import yfinance as yf
    end_excl = (pd.Period(end, 'M') + 1).to_timestamp().strftime('%Y-%m-%d')
    closes, divs = [], []
    for i in range(0, len(tickers), chunk):
        part, df = list(tickers[i:i + chunk]), None
        for _ in range(retries):
            try:
                df = yf.download(part, start=start, end=end_excl, interval='1d', auto_adjust=False,
                                 actions=True, progress=False, threads=True)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                print('retry:', e)
            time.sleep(pause)
        if df is None or df.empty:
            continue
        c = df['Close']
        d = df['Dividends'] if 'Dividends' in df.columns.get_level_values(0) else c * 0
        if isinstance(c, pd.Series):
            c, d = c.to_frame(part[0]), d.to_frame(part[0])
        closes.append(c)
        divs.append(d)
    close = pd.concat(closes, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    div = pd.concat(divs, axis=1)
    div = div.loc[:, ~div.columns.duplicated()].reindex(close.index).fillna(0.0)
    close_m = close.resample('ME').last()
    div_m = div.resample('ME').sum()
    for x in (close_m, div_m):
        x.index = x.index.to_period('M')
    keep = close_m.columns[close_m.notna().any()]
    return close_m[keep], div_m[keep]


def load_fred(series_id):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors='coerce').dropna()
    s.index = s.index.to_period('M')
    return s


def usd_to_eur(close_usd, div_usd, eurusd):
    """Convert a USD-listed series to EUR with month-end EUR/USD (USD per 1 EUR)."""
    fx = eurusd.reindex(close_usd.index).ffill()
    return close_usd / fx, div_usd / fx


def fx_hedged_series(close_eur, div_eur, eurusd, r_eur_ann, r_usd_ann, cost_bps=10):
    """Synthetic EUR-hedged version of a USD asset quoted in EUR (app / sensitivity option).
    hedged return ~ (1 + USD local return) * (1 + (i_EUR - i_USD)/12 - cost/12) - 1  (covered interest parity).
    Distributions keep the unhedged dividend yield, applied to the hedged price."""
    idx = close_eur.index
    p_usd = close_eur * eurusd.reindex(idx).ffill()
    r_loc = p_usd / p_usd.shift(1) - 1
    carry = (r_eur_ann.reindex(idx).ffill() - r_usd_ann.reindex(idx).ffill()) / 12 - cost_bps / 1e4 / 12
    gross = ((1 + r_loc) * (1 + carry.fillna(0))).fillna(1.0)
    first = close_eur.first_valid_index()
    g = gross.loc[first:].copy()
    g.iloc[0] = 1.0
    hp = (g.cumprod() * close_eur.loc[first]).reindex(idx)
    hd = (div_eur / close_eur).fillna(0) * hp
    return hp, hd.fillna(0)


def total_return(px, div):
    """Monthly total return of a single instrument from close + dividends."""
    return (px + div) / px.shift(1) - 1


# =====================================================================
# 2) Eurozone stock selection - uses ONLY data up to and including t
# =====================================================================
FTT = {   # financial transaction tax on share PURCHASES: list of (first month, rate)
    'France': [('2012-08', 0.002), ('2017-01', 0.003), ('2025-04', 0.004)],
    'Italy':  [('2013-03', 0.0012), ('2014-01', 0.001), ('2026-01', 0.002)],
    'Spain':  [('2021-01', 0.002)],
}


def ftt_rate(country, t):
    r = 0.0
    for start, rate in FTT.get(country, []):
        if t >= pd.Period(start, 'M'):
            r = rate
    return r


def dividend_screen(annual, last_year, stocks, R):
    """Dividend-record screen on calendar-year DPS (ex-date year).
    R['hist_years']    : length of the window (5)
    R['min_paid_years']: years in the window with DPS > 0 (5 = every year)
    R['cut_tol']       : max. year-on-year cut (0.10); None = no year-on-year test
    R['net_growth']    : DPS in the latest year >= DPS in the first year of the window"""
    yrs = list(range(last_year - R['hist_years'] + 1, last_year + 1))
    a = annual.reindex(index=yrs, columns=stocks).fillna(0.0)
    ok = ((a > 0).sum() >= R['min_paid_years']) & (a.iloc[-1] > 0)
    if R.get('cut_tol') is not None:
        prev, cur = a.iloc[:-1].values, a.iloc[1:].values
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(prev > 0, cur / prev, np.where(cur > 0, np.inf, 0.0))
        ok &= pd.Series((ratio >= 1 - R['cut_tol']).all(axis=0), index=stocks)
    if R.get('net_growth', True):
        ok &= a.iloc[-1] >= a.iloc[0] * (1 - 1e-9)
    return ok


def select_stocks(t, close_raw, div_m, meta, R, incumbents=()):
    """Eurozone selection on date t. Returns
    w      : weights within the Eurozone sleeve (sum 1; empty slots go to R['fallback'])
    info   : table of the selected stocks
    counts : how many stocks pass each screen (data check / prospectus)"""
    close_h, div_h = close_raw.loc[:t], div_m.loc[:t]          # nothing after t is visible
    assert close_h.index.max() <= t and div_h.index.max() <= t, 'look-ahead'
    stocks = meta.index.intersection(close_h.columns)
    p_t, p_12 = close_h.loc[t, stocks], close_h.loc[t - 12, stocks]
    listed = p_t.notna() & p_12.notna() & (p_t > 0)
    u = listed[listed].index

    annual = div_h[u].groupby(div_h.index.year).sum()
    last_year = t.year if t.month == 12 else t.year - 1
    record = dividend_screen(annual, last_year, u, R)
    y = div_h.loc[t - 11:t, u].sum() / p_t[u]                   # trailing 12-month yield
    r12 = p_t[u] / p_12[u] - 1                                  # 12-month price return
    loser = (r12 <= r12.quantile(R['loser_quantile'])) & (r12 < 0)
    yield_ok = (y > 0) & (y <= R['max_yield'])

    surv = y[record & yield_ok & ~loser].sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(surv) + 1), index=surv.index)
    inc = set(incumbents)
    keep = [s for s in surv.index if s in inc and rank[s] <= R['buffer_rank']]
    order = keep + [s for s in surv.index if s not in set(keep)]
    picked, n_sec, n_cty = [], {}, {}
    for s in order:
        sec, cty = meta.at[s, 'sector'], meta.at[s, 'country']
        if n_sec.get(sec, 0) >= R['max_per_sector'] or n_cty.get(cty, 0) >= R['max_per_country']:
            continue
        picked.append(s)
        n_sec[sec] = n_sec.get(sec, 0) + 1
        n_cty[cty] = n_cty.get(cty, 0) + 1
        if len(picked) == R['n_stocks']:
            break

    w = pd.Series(1.0 / R['n_stocks'], index=picked, dtype=float)
    if len(picked) < R['n_stocks']:
        w[R['fallback']] = (R['n_stocks'] - len(picked)) / R['n_stocks']
    info = pd.DataFrame({'name': meta.loc[picked, 'name'], 'sector': meta.loc[picked, 'sector'],
                         'country': meta.loc[picked, 'country'], 'yield_ttm': y.reindex(picked),
                         'rank': rank.reindex(picked), 'kept_by_buffer': [s in keep for s in picked]})
    counts = {'universe': len(stocks), 'listed_12m': len(u), 'dividend_record': int(record.sum()),
              'yield_in_range': int(yield_ok.sum()), 'losers_dropped': int(loser.sum()),
              'eligible': len(surv), 'selected': len(picked), 'fallback_slots': R['n_stocks'] - len(picked)}
    return w, info, counts


def run_selection(dates, close_raw, div_m, meta, R):
    """Sequential yearly selection (the buffer needs last year's holdings)."""
    out, inc = {}, []
    for t in dates:
        w, info, cnt = select_stocks(t, close_raw, div_m, meta, R, inc)
        out[t] = (w, info, cnt)
        inc = list(info.index)
    return out


def equal_weight_universe(t, close_raw, meta):
    """1/N diagnostic: every universe stock with a 12-month price history at t."""
    c = close_raw.loc[:t]
    stocks = meta.index.intersection(c.columns)
    ok = c.loc[t, stocks].notna() & c.loc[t - 12, stocks].notna()
    sel = ok[ok].index
    return pd.Series(1 / len(sel), index=sel)


# =====================================================================
# 3) Simulation in share counts
# =====================================================================
def simulate(spec, px, div, start, end, cfg, last_valid=None):
    """
    spec : {sleeve: {'weight': w, 'group': 'EQ'|'BOND'|'GOLD', 'fn': f(t) -> weights within the sleeve,
                     'fallback': ticker for delisted holdings (optional)}}
    px   : month-end prices, forward-filled;  div: dividends per share in the month (ex-date)
    cfg  : costs, payout rule, rebalancing (see notebook, section 2)
    Each month: dividends/coupons arrive -> payout -> (December) reselect + reset all sleeves,
    (cfg['band_months']) reset all sleeves if a group leaves its band, keeping current weights inside sleeves.
    ret = (value at month end + payout) / value at previous month end - 1  (time-weighted, net of costs).
    """
    months = pd.period_range(start, end, freq='M')
    W = pd.Series({s: v['weight'] for s, v in spec.items() if v['weight'] > 0}, dtype=float)
    assert abs(W.sum() - 1) < 1e-9, f'sleeve weights sum to {W.sum()}'
    grp = pd.Series({s: spec[s]['group'] for s in W.index})
    tgt_g = W.groupby(grp).sum()
    etfs, country = cfg['etfs'], cfg['country']
    cm = cfg.get('cost_mult', 1.0)
    shares = {}

    def holdings_value(p):
        return pd.Series({s: float((sh * p.reindex(sh.index).fillna(0)).sum()) for s, sh in shares.items()},
                         dtype=float).reindex(W.index).fillna(0.0)

    def by_ticker(d):
        if not d:
            return pd.Series(dtype=float)
        return pd.concat(list(d.values())).groupby(level=0).sum()

    def trade_to(total, sleeve_w, within, p, t):
        old = by_ticker({s: sh * p.reindex(sh.index).fillna(0) for s, sh in shares.items()})
        new = by_ticker({s: total * sleeve_w[s] * within[s] for s in sleeve_w.index if sleeve_w[s] > 0})
        delta = new.subtract(old, fill_value=0.0)
        tc = np.array([cfg['tc_etf'] if k in etfs else cfg['tc_stock'] for k in delta.index])
        ftt = np.array([0.0 if k in etfs else ftt_rate(country.get(k, ''), t) for k in delta.index])
        cost = cm * float((delta.abs().values * tc).sum() + (delta.clip(lower=0).values * ftt).sum())
        net = total - cost
        shares.clear()
        for s in sleeve_w.index:
            if sleeve_w[s] <= 0:
                continue
            ww = within[s]
            shares[s] = (net * sleeve_w[s] * ww / p.reindex(ww.index)).replace([np.inf, -np.inf], 0).fillna(0)
        return cost

    def current_within(p):
        out = {}
        for s, sh in shares.items():
            v = sh * p.reindex(sh.index).fillna(0)
            out[s] = v / v.sum() if v.sum() > 0 else v
        return out

    t0 = months[0]
    p = px.loc[t0]
    cost = trade_to(cfg['w0'], W, {s: spec[s]['fn'](t0) for s in W.index}, p, t0)
    prev = float(holdings_value(p).sum())
    hist_v = [prev]
    rows = [{'period': t0, 'value': prev, 'income': 0.0, 'payout': 0.0, 'cost': cost, 'ret': np.nan,
             'event': 'start', **{f'w_{g}': float(tgt_g[g]) for g in tgt_g.index}}]
    for m in months[1:]:
        p, d = px.loc[m], div.loc[m]
        # takeovers / delistings: proceeds go to the sleeve's fallback ETF until the next review
        if last_valid is not None:
            for s in list(shares):
                sh = shares[s]
                dead = [k for k in sh.index if sh[k] > 0 and k in last_valid and last_valid[k] is not None
                        and m > last_valid[k]]
                if dead:
                    fb = spec[s].get('fallback', cfg['fallback'])
                    val = float((sh[dead] * p.reindex(dead)).sum())
                    sh = sh.drop(dead)
                    sh[fb] = sh.get(fb, 0.0) + val / p[fb]
                    shares[s] = sh
        income = float(sum((sh * d.reindex(sh.index).fillna(0)).sum() for sh in shares.values()))
        vals = holdings_value(p)
        V = float(vals.sum())
        if cfg['payout'] == 'income':
            wd = income
        elif cfg['payout'] == 'smoothed':                       # rate x average value of the last N months
            wd = cfg['payout_rate'] / 12 * float(np.mean((hist_v + [V])[-cfg['smooth_months']:]))
        else:
            raise ValueError(cfg['payout'])
        wd = min(wd, V + income)
        cash = income - wd
        cost, event = 0.0, ''
        gw = (vals.groupby(grp).sum() / V) if V > 0 else tgt_g * 0
        if m.month == cfg['rebal_month']:
            cost = trade_to(V + cash, W, {s: spec[s]['fn'](m) for s in W.index}, p, m)
            event = 'annual'
        elif m.month in cfg['band_months'] and any(abs(gw[g] - tgt_g[g]) > cfg['bands'].get(g, 1.0)
                                                   for g in tgt_g.index):
            cost = trade_to(V + cash, W, current_within(p), p, m)
            event = 'band'
        elif abs(cash) > 1e-6:                                  # smoothed payout: buy / sell pro rata
            cost = trade_to(V + cash, vals / V, current_within(p), p, m)
            event = 'flow'
        end_v = float(holdings_value(p).sum())
        gw_end = holdings_value(p).groupby(grp).sum() / end_v
        rows.append({'period': m, 'value': end_v, 'income': income, 'payout': wd, 'cost': cost,
                     'ret': (end_v + wd) / prev - 1 if prev > 0 else np.nan, 'event': event,
                     **{f'w_{g}': float(gw_end[g]) for g in tgt_g.index}})
        prev = end_v
        hist_v.append(end_v)
    return pd.DataFrame(rows).set_index('period')


# =====================================================================
# 4) Performance measures (monthly data)
# =====================================================================
def standalone_stats(r, rf, periods=12):
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0)
    ex = r - rf
    c = (1 + r).cumprod()
    roll12 = (1 + r).rolling(12).apply(np.prod, raw=True) - 1
    return pd.Series({
        'Ann. average return': r.mean() * periods,
        'CAGR': c.iloc[-1] ** (periods / len(r)) - 1,
        'Volatility': r.std() * np.sqrt(periods),
        'Sharpe ratio': ex.mean() * periods / (ex.std() * np.sqrt(periods)),
        'Max drawdown': (c / c.cummax() - 1).min(),
        'Worst 12-month return': roll12.min(),
    })


def relative_stats(r, bench, rf, periods=12):
    """Beta = slope of strategy excess on benchmark excess returns (not annualised);
    alpha = 12 x intercept, with its OLS t-statistic; IR = 12 x mean active / (sqrt(12) x tracking error)."""
    idx = r.dropna().index.intersection(bench.dropna().index)
    r, b, rf = r.loc[idx], bench.loc[idx], rf.reindex(idx).fillna(0)
    y, x = (r - rf).values, (b - rf).values
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    active = r - b
    te = active.std() * np.sqrt(periods)
    return pd.Series({
        'Beta': coef[1],
        'Alpha (ann.)': coef[0] * periods,
        't(alpha)': coef[0] / se[0],
        'Tracking error': te,
        'Information ratio': active.mean() * periods / te if te > 1e-12 else np.nan,
    })


def client_stats(sim, cpi, w0):
    """What the family gets: income in EUR, its stability, and real wealth after payouts."""
    s = sim.iloc[1:]
    yrs = s.index.year
    full = [y for y, n in pd.Series(yrs).value_counts().items() if n == 12]
    inc = s['income'].groupby(yrs).sum().loc[sorted(full)]
    start_val = sim['value'].shift(1).iloc[1:].groupby(yrs).first().loc[sorted(full)]
    infl = cpi.reindex(sim.index).ffill()
    real = sim['value'] / (infl / infl.iloc[0])
    chg = inc.pct_change()
    return pd.Series({
        'Avg. income p.a. (EUR)': inc.mean(),
        'Avg. payout yield': (inc / start_val).mean(),
        'Weakest income year (EUR)': inc.min(),
        'Largest yearly income drop': chg.min(),
        'Years with income drop > 10%': int((chg < -0.10).sum()),
        'Total paid out (EUR)': s['payout'].sum(),
        'End value, nominal (EUR)': sim['value'].iloc[-1],
        'End value, real (x start)': real.iloc[-1] / w0,
        'Total costs (EUR)': sim['cost'].sum(),
    }), inc, real


# =====================================================================
# 5) Planning tables (prospectus sections 2 and 3.2) from stated assumptions
# =====================================================================
SLEEVES = ['EZ', 'US', 'GOV', 'CORP', 'GOLD']


def mix_for_equity(e, gold=0.05, gov_corp=(4, 3)):
    """Equity split 50/50, bonds split 4:3 government/corporate (rounded to whole %), gold fixed."""
    bonds = 1 - e - gold
    gov = round(bonds * gov_corp[0] / sum(gov_corp) * 100) / 100
    return pd.Series({'EZ': e / 2, 'US': e / 2, 'GOV': gov, 'CORP': round((bonds - gov) * 100) / 100, 'GOLD': gold})


def planning_tables(A, equity_shares=(0.4, 0.5, 0.6, 0.7, 0.8), z=1.645):
    src = pd.DataFrame(A['sources']).T
    src['Capital growth after payout'] = src['Expected return'] - src['Income']
    src['Real capital growth'] = src['Capital growth after payout'] - A['inflation']
    mu = pd.Series(A['exp_return'])[SLEEVES]
    inc = pd.Series(A['income'])[SLEEVES]
    vol = pd.Series(A['vol'])[SLEEVES]
    C = pd.DataFrame(A['corr'], index=SLEEVES, columns=SLEEVES)
    S = np.outer(vol, vol) * C.values
    rows = {}
    for e in equity_shares:
        w = mix_for_equity(e)
        er = float(w @ mu) - A['costs']
        v = float(np.sqrt(w.values @ S @ w.values))
        rows[f'{e:.0%}'] = {'Gov / corp bonds': f"{w['GOV']:.0%} / {w['CORP']:.0%}", 'Expected return': er,
                            'Income': float(w @ inc), 'Real capital growth': er - float(w @ inc) - A['inflation'],
                            'Volatility': v, '1-in-20 year': er - z * v}
    eq = pd.DataFrame(rows).T
    eq.index.name = 'Equity share'
    return src, eq
