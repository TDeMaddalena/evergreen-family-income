"""
Evergreen Family Income - Streamlit app.

Run locally:   pip install -r requirements.txt   then   streamlit run streamlit_app.py

All calculations use evergreen_core.py (shared with the notebook); data preparation and portfolio
set-up are copied from the notebook in app_data.py. With every setting on "base case" the app
reproduces the notebook's numbers.
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import app_data as ad
import evergreen_core as dc

st.set_page_config(page_title='Evergreen Family Income', page_icon='🌲', layout='wide')

NAVY, GOLD, GREY, TEAL, RED, SAND = '#0B3C5D', '#C9A227', '#9AA5B1', '#328CC1', '#B23A48', '#E2B55E'
COLORS = {'Your scenario': NAVY, 'Base case': GOLD, 'B1 Policy twin': SAND, 'B2 60/40': GREY,
          'B3 Dividend ETF': TEAL, '1/N diagnostic': RED}
BENCH = ['B1 Policy twin', 'B2 60/40', 'B3 Dividend ETF', '1/N diagnostic']
SLEEVE_NAMES = {'EZ': 'Eurozone stocks (our rule)', 'US': 'US dividend stocks (SPYD)',
                'GOV': 'Euro government bonds', 'CORP': 'Euro corporate bonds', 'GOLD': 'Gold'}
CUSTOM = ['EZ', 'US', 'GOV', 'CORP', 'GOLD']

# ---------------------------------------------------------------------------------------------
# Defaults = base case (notebook section 2). "Reset to base case" restores exactly these.
# ---------------------------------------------------------------------------------------------
DEFAULTS = dict(alloc_mode='Equity share', equity=60,
                w_EZ=30, w_US=30, w_GOV=20, w_CORP=15, w_GOLD=5, hedge=False,
                record='Base case', n_stocks=20, max_sector=4, max_country=6, loser=5, buffer=True,
                bands_on=True, band=5, cost_mult=1.0,
                payout='All income', payout_rate=3.5, smooth_months=36)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_to_base():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------------------------------
# Data and calculations (cached)
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def market_data():
    return ad.load_market_data()


def weights_from(p):
    if p['alloc_mode'] == 'Equity share':
        return {k: float(v) for k, v in dc.mix_for_equity(p['equity'] / 100).items()}
    return {k: p[f'w_{k}'] / 100 for k in CUSTOM}


def rule_from(p):
    n = p['n_stocks']
    return {**ad.RULE, **ad.DIVIDEND_RECORDS[p['record']], 'n_stocks': n,
            'max_per_sector': p['max_sector'], 'max_per_country': p['max_country'],
            'loser_quantile': p['loser'] / 100,
            'buffer_rank': int(round(1.5 * n)) if p['buffer'] else n}


def cfg_from(D, p):
    b = p['band'] / 100
    return ad.full_cfg(D, band_months=(3, 6, 9) if p['bands_on'] else (),
                       bands={'EQ': b, 'BOND': b, 'GOLD': b * 0.4},      # base: 5% / 5% / 2%
                       cost_mult=p['cost_mult'],
                       payout='income' if p['payout'] == 'All income' else 'smoothed',
                       payout_rate=p['payout_rate'] / 100, smooth_months=p['smooth_months'])


@st.cache_data(show_spinner=False, max_entries=20)
def selection(rule_json):
    D = market_data()
    return ad.selection(D, json.loads(rule_json))


@st.cache_data(show_spinner=False, max_entries=20)
def backtest(p_json):
    """Notebook section 7 for one set of assumptions: Evergreen ('Your scenario') and the benchmarks.
    B1, B3 and 1/N use the scenario's weights, so they differ from it only in the stock sleeves."""
    p = json.loads(p_json)
    D, T = market_data(), market_data()['T']
    W, cfg = weights_from(p), cfg_from(D, p)
    SEL = selection(json.dumps(rule_from(p), sort_keys=True))
    ez_rule = lambda t: SEL[t][0]
    ew = lambda t: dc.equal_weight_universe(t, D['close_raw'], D['meta'])
    us = 'US_HEDGED' if p['hedge'] else T['US']
    f, s = ad.fixed, ad.sleeve
    specs = {'Your scenario': ad.five_sleeves(T, W, ez_rule, us),
             'B1 Policy twin': ad.five_sleeves(T, W, f(T['EZ_ETF']), T['SP500']),
             'B2 60/40': {'WORLD': s(0.60, 'EQ', f(T['WORLD'])), 'GOV': s(0.40, 'BOND', f(T['GOV']))},
             'B3 Dividend ETF': ad.five_sleeves(T, W, f(T['DIV30']), T['SP500']),
             '1/N diagnostic': ad.five_sleeves(T, W, ew, us)}
    sims = {k: ad.run(D, sp, cfg) for k, sp in specs.items()}
    R = pd.DataFrame({k: v['ret'] for k, v in sims.items()}).dropna()
    return sims, R


@st.cache_data(show_spinner=False, max_entries=10)
def attribution(p_json):
    """Notebook section 8: each sleeve on its own (100%), same rules and costs."""
    p = json.loads(p_json)
    D = market_data()
    T = D['T']
    cfg = cfg_from(D, p)
    SEL = selection(json.dumps(rule_from(p), sort_keys=True))
    f = ad.fixed
    us = 'US_HEDGED' if p['hedge'] else T['US']
    tests = {'EZ: our rule': lambda t: SEL[t][0], 'EZ: EURO STOXX 50': f(T['EZ_ETF']),
             'EZ: Select Dividend 30': f(T['DIV30']),
             'EZ: 1/N universe': lambda t: dc.equal_weight_universe(t, D['close_raw'], D['meta']),
             'US: SPYD' + (' (hedged)' if p['hedge'] else ''): f(us), 'US: S&P 500': f(T['SP500'])}
    RS = pd.DataFrame({k: ad.run(D, {'X': ad.sleeve(1.0, 'EQ', fn)}, cfg)['ret'] for k, fn in tests.items()}).dropna()
    names = list(tests)
    pairs = [(names[0], names[1]), (names[0], names[2]), (names[0], names[3]), (names[4], names[5])]
    rf = D['rf']
    return pd.DataFrame({f'{a.split(": ")[1]} vs {b.split(": ")[1]}':
                         pd.concat([dc.standalone_stats(RS[a], rf)[['CAGR', 'Volatility', 'Sharpe ratio']],
                                    pd.Series({'Benchmark CAGR': dc.standalone_stats(RS[b], rf)['CAGR']}),
                                    dc.relative_stats(RS[a], RS[b], rf)]) for a, b in pairs})


# ---------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------
PCT_ROWS = {'Ann. average return', 'CAGR', 'Volatility', 'Max drawdown', 'Worst 12-month return',
            'Alpha (ann.)', 'Tracking error', 'Benchmark CAGR'}


def fmt(df):
    rows = [r for r in df.index if r in PCT_ROWS]
    return df.style.format('{:.2f}').format('{:.2%}', subset=pd.IndexSlice[rows, :])


def cum_frame(R, cols):
    out = []
    for k in cols:
        c = pd.concat([pd.Series([1.0], index=[R.index[0] - 1]), (1 + R[k]).cumprod()])
        out.append(pd.DataFrame({'Date': c.index.to_timestamp(), 'Portfolio': k, 'Value': c.values}))
    return pd.concat(out)


def line_chart(df, y_title, y_format='.2f', rule_at=None, height=380):
    dom = [k for k in COLORS if k in df['Portfolio'].unique()]
    base = alt.Chart(df).encode(
        x=alt.X('Date:T', title=None),
        y=alt.Y('Value:Q', title=y_title, axis=alt.Axis(format=y_format), scale=alt.Scale(zero=False)),
        color=alt.Color('Portfolio:N', scale=alt.Scale(domain=dom, range=[COLORS[k] for k in dom]),
                        legend=alt.Legend(orient='top', title=None)),
        strokeDash=alt.StrokeDash('Portfolio:N', scale=alt.Scale(
            domain=dom, range=[[6, 3] if k == 'Base case' else [2, 2] if k == '1/N diagnostic' else [1, 0]
                               for k in dom]), legend=None),
        strokeWidth=alt.condition(alt.datum.Portfolio == 'Your scenario', alt.value(3), alt.value(1.6)),
        tooltip=['Portfolio', alt.Tooltip('Date:T', format='%b %Y'), alt.Tooltip('Value:Q', format=y_format)])
    chart = base.mark_line()
    if rule_at is not None:
        chart += alt.Chart(pd.DataFrame({'y': [rule_at]})).mark_rule(color='black', strokeWidth=0.6).encode(y='y:Q')
    return chart.properties(height=height).interactive(bind_y=False)


# ---------------------------------------------------------------------------------------------
# Sidebar: assumptions
# ---------------------------------------------------------------------------------------------
sb = st.sidebar
sb.title('🌲 Your scenario')
sb.caption('Every setting starts at the **base case** of the prospectus. Change one and the backtest, '
           'benchmarks and measures update.')
sb.button('↺ Reset to base case', on_click=reset_to_base, width='stretch')

sb.header('Allocation')
sb.radio('How to set the weights', ['Equity share', 'Custom weights'], key='alloc_mode', horizontal=True)
if st.session_state.alloc_mode == 'Equity share':
    sb.slider('Equity share (%)', 40, 80, step=5, key='equity',
              help='Split 50/50 between Eurozone and US stocks; bonds split 4:3 government/corporate; '
                   'gold fixed at 5% (prospectus section 3.2).')
else:
    for k in CUSTOM:
        sb.number_input(f'{SLEEVE_NAMES[k]} (%)', 0, 100, step=5, key=f'w_{k}')
sb.toggle('Hedge the US sleeve to EUR', key='hedge',
          help='Synthetic currency hedge via the EUR-USD interest differential, 0.10% p.a. cost.')

sb.header('Eurozone stock rule')
sb.selectbox('Dividend record', list(ad.DIVIDEND_RECORDS), key='record',
             help='**Base case:** dividend paid every year of the last 5, never cut by more than 10%, latest ≥ first.  \n'
                  '**Strict:** paid every year, never cut by more than 5%; overall growth not checked.  \n'
                  '**Lenient:** paid in at least 4 of 5 years, latest ≥ first; cuts allowed.')
sb.slider('Number of stocks', 10, 30, step=5, key='n_stocks',
          help='Equal weight within the Eurozone sleeve. Empty places go to the EURO STOXX 50 ETF.')
c1, c2 = sb.columns(2)
c1.number_input('Max per sector', 2, 10, key='max_sector')
c2.number_input('Max per country', 2, 15, key='max_country')
sb.slider('Drop worst price performers (%)', 0, 10, key='loser',
          help='Drop the worst x% by 12-month price return (only if that return is negative).')
sb.toggle('Holdings buffer', key='buffer',
          help='Keep current holdings while they rank within 1.5 × the number of stocks (base: top 30 of 20). '
               'Lowers turnover and costs.')

sb.header('Rebalancing & costs')
sb.toggle('Quarterly band check', key='bands_on',
          help='Always: full reset every December. With bands: also reset at March, June and September '
               'quarter-ends if equity or bonds drift more than the band from target (gold: 0.4 × band).')
if st.session_state.bands_on:
    sb.slider('Band width (± percentage points)', 2, 10, key='band')
sb.slider('Trading cost multiplier', 0.0, 3.0, step=0.5, key='cost_mult',
          help='Base: 0.10% per stock trade, 0.05% per ETF trade, plus French, Italian and Spanish '
               'transaction taxes on purchases. 0 = frictionless.')

sb.header('Payout to the family')
sb.radio('Payout rule', ['All income', 'Smoothed'], key='payout', horizontal=True,
         help='**All income:** every dividend and coupon is paid out; capital is never sold.  \n'
              '**Smoothed:** a fixed % of the average portfolio value over the last months.')
if st.session_state.payout == 'Smoothed':
    sb.slider('Payout rate (% p.a.)', 2.0, 5.0, step=0.25, key='payout_rate')
    sb.slider('Averaging window (months)', 12, 60, step=12, key='smooth_months')
sb.caption('Returns are time-weighted, so the payout rule mainly changes the **Income & wealth** tab, '
           'not the return chart.')

P = {k: st.session_state[k] for k in DEFAULTS}
if P['alloc_mode'] == 'Custom weights':
    tot = sum(P[f'w_{k}'] for k in CUSTOM)
    if tot != 100:
        st.sidebar.error(f'Weights add up to {tot}%, they must add up to 100%.')
        st.error(f'Your custom weights add up to {tot}%. Adjust them in the sidebar to 100%.')
        st.stop()


def canonical(p):
    """Drop settings that have no effect, so equivalent scenarios share one cache entry."""
    q, d = dict(p), DEFAULTS
    if q['alloc_mode'] == 'Equity share':
        for k in CUSTOM:
            q[f'w_{k}'] = d[f'w_{k}']
    else:
        q['equity'] = d['equity']
    if not q['bands_on']:
        q['band'] = d['band']
    if q['payout'] == 'All income':
        q['payout_rate'], q['smooth_months'] = d['payout_rate'], d['smooth_months']
    return json.dumps(q, sort_keys=True)


P_JSON, BASE_JSON = canonical(P), canonical(DEFAULTS)
IS_BASE = P_JSON == BASE_JSON

# ---------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------
st.title('Evergreen Family Income')
st.markdown('**Dividends to live on. Capital to pass on.** A rules-based multi-asset income portfolio for a '
            'European single-family office. All income is paid out; capital is never sold to fund spending.')

try:
    with st.spinner('Loading market data from Yahoo Finance and FRED. The first load takes a few minutes; '
                    'after that every scenario runs in seconds.'):
        D = market_data()
except Exception as e:
    st.error(f'Market data could not be loaded ({e}). Yahoo Finance may be rate-limiting; please reload in a few minutes.')
    st.stop()
T, rf, cpi = D['T'], D['rf'], D['cpi']

with st.spinner('Running the backtest for your scenario…'):
    SIMS, R = backtest(P_JSON)
    BSIMS, BR = backtest(BASE_JSON)
SEL = selection(json.dumps(rule_from(P), sort_keys=True))

W = weights_from(P)
cols = st.columns(len(W))
for c, (k, v) in zip(cols, W.items()):
    delta = v - ad.WEIGHTS.get(k, 0)
    c.metric(SLEEVE_NAMES[k], f'{v:.0%}', f'{delta:+.0%} vs base' if abs(delta) > 1e-9 else None, delta_color='off')
if IS_BASE:
    st.info('You are looking at the **base case** of the prospectus. Change an assumption in the sidebar to '
            'compare a scenario with it.', icon='ℹ️')

R_ALL = R.copy()
if not IS_BASE:
    R_ALL.insert(1, 'Base case', BR['Your scenario'])
ABS = pd.DataFrame({k: dc.standalone_stats(R_ALL[k], rf) for k in R_ALL})
REL = pd.DataFrame({f'vs {b}': dc.relative_stats(R['Your scenario'], R_ALL[b], rf)
                    for b in (['Base case'] if not IS_BASE else []) + BENCH})

tabs = st.tabs(['📈 Backtest', '💶 Income & wealth', '🔎 Stock selection', '🧩 Attribution', '🧭 Planner', 'ℹ️ About'])

# ---------------------------------------------------------------------------------------------
# Tab 1: Backtest
# ---------------------------------------------------------------------------------------------
with tabs[0]:
    a, b = ABS['Your scenario'], REL['vs B1 Policy twin']
    ref = ABS['Base case'] if not IS_BASE else None
    m = st.columns(5)
    for col, (label, val, key, pct) in zip(m, [('Ann. average return', a['Ann. average return'], 'Ann. average return', True),
                                               ('Volatility', a['Volatility'], 'Volatility', True),
                                               ('Sharpe ratio', a['Sharpe ratio'], 'Sharpe ratio', False),
                                               ('Max drawdown', a['Max drawdown'], 'Max drawdown', True),
                                               ('Alpha vs B1 (ann.)', b['Alpha (ann.)'], None, True)]):
        d = None
        if ref is not None and key is not None:
            d = f'{val - ref[key]:+.2%} vs base' if pct else f'{val - ref[key]:+.2f} vs base'
        col.metric(label, f'{val:.2%}' if pct else f'{val:.2f}', d,
                   delta_color='inverse' if key == 'Volatility' else 'normal')

    st.subheader(f'Cumulative total return, net of costs (start = 1), {(ad.START + 1).strftime('%b %Y')} – {ad.END.strftime('%b %Y')}')
    show = st.multiselect('Portfolios', list(R_ALL.columns), default=list(R_ALL.columns), label_visibility='collapsed')
    if show:
        st.altair_chart(line_chart(cum_frame(R_ALL, show), 'Growth of 1', rule_at=1.0), width='stretch')
    st.caption('All portfolios use the same rebalancing, payout rule and cost model as your scenario. '
               '**B1 Policy twin** and **B3 Dividend ETF** hold your scenario\'s weights with index funds in the stock '
               'sleeves (EURO STOXX 50 or Select Dividend 30, plus the S&P 500), so they isolate the effect of our stock '
               'choices. **B2** is a fixed 60% MSCI World / 40% euro government bonds. The **1/N diagnostic** equal-weights '
               'every stock in our universe and shows how much of the result comes from survivorship bias.')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Absolute measures** (monthly data, annualised)')
        st.dataframe(fmt(ABS), width='stretch')
    with c2:
        st.markdown('**Your scenario relative to each benchmark**')
        st.dataframe(fmt(REL), width='stretch')

    with st.expander('How to read these measures and how they are calculated'):
        st.markdown("""
- **Data:** monthly total returns in EUR, net of trading costs and taxes, January 2014 – October 2026.
  Risk-free rate: euro area 3-month interbank rate (FRED), known at the start of each month.
- **Ann. average return** = 12 × mean monthly return. **CAGR** is the compound annual growth rate actually achieved.
- **Volatility** = √12 × standard deviation of monthly returns: how much the value swings in a typical year.
- **Sharpe ratio** = annualised mean excess return over the risk-free rate ÷ annualised volatility: return per unit of risk.
- **Beta** = slope of a regression of the scenario's monthly excess returns on the benchmark's. **Not annualised.**
  A beta below 1 means the portfolio moves less than the benchmark.
- **Alpha** = 12 × the intercept of that regression: return not explained by the benchmark exposure.
  **t(alpha)** above about 2 means the alpha is unlikely to be luck.
- **Tracking error** = √12 × standard deviation of the monthly return difference;
  **Information ratio** = annualised mean return difference ÷ tracking error: active return per unit of active risk.
- **Max drawdown** and **worst 12-month return** show the deepest loss a family would have lived through.
""")
    with st.expander('Drawdowns'):
        dd = pd.concat([pd.DataFrame({'Date': R_ALL.index.to_timestamp(), 'Portfolio': k,
                                      'Value': ((1 + R_ALL[k]).cumprod() / (1 + R_ALL[k]).cumprod().cummax() - 1).values})
                        for k in R_ALL if k != '1/N diagnostic'])
        st.altair_chart(line_chart(dd, 'Drawdown', '.0%', height=280), width='stretch')
    st.caption('Not every setting is meant to be optimised: picking the assumptions that look best in hindsight is '
               'data snooping. The base case was fixed in the prospectus before the backtest.')

# ---------------------------------------------------------------------------------------------
# Tab 2: Income & wealth (notebook sections 9-10)
# ---------------------------------------------------------------------------------------------
with tabs[1]:
    w0 = st.number_input('Starting wealth (EUR million)', 1.0, 500.0, ad.W0 / 1e6, step=1.0) * 1e6
    scale = w0 / ad.W0                      # all amounts scale linearly with the starting value
    who = {'Your scenario': SIMS['Your scenario']}
    if not IS_BASE:
        who['Base case'] = BSIMS['Your scenario']
    who.update({k: SIMS[k] for k in ['B1 Policy twin', 'B2 60/40', 'B3 Dividend ETF']})
    CLIENT, INCOME, REAL = {}, {}, {}
    for k, sim in who.items():
        CLIENT[k], INCOME[k], REAL[k] = dc.client_stats(sim, cpi, ad.W0)
    ct = pd.DataFrame(CLIENT)
    ct.loc['Worst 12-month return'] = ABS.loc['Worst 12-month return', ct.columns]
    showt = ct.astype(object)
    for r in ct.index:
        showt.loc[r] = [f'{v * scale:,.0f}' if '(EUR)' in r else (f'{v:.2f}x' if '(x' in r else
                        (f'{v:.0f}' if r.startswith('Years') else f'{v:.2%}')) for v in ct.loc[r]]

    inc = pd.DataFrame(INCOME) * scale
    e = inc['Your scenario']
    m = st.columns(4)
    m[0].metric('Average income per year', f'€{e.mean():,.0f}')
    m[1].metric('Weakest income year', f'€{e.min():,.0f}')
    m[2].metric('Largest yearly income drop', f'{e.pct_change().min():.1%}')
    m[3].metric('Real wealth at the end', f'{CLIENT["Your scenario"]["End value, real (x start)"]:.2f}x start')

    st.dataframe(showt, width='stretch')
    worst = ABS.at['Worst 12-month return', 'Your scenario']
    (st.success if worst >= -0.105 else st.warning)(
        f'Risk budget: the prospectus plans for a 1-in-20 bad year of about −10%. The worst 12-month return of your '
        f'scenario was {worst:.1%}' + (' – within budget.' if worst >= -0.105 else ' – the budget is breached.'))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'**Income paid out per year** (EUR, on €{w0 / 1e6:,.0f} m)')
        ib = inc[[k for k in inc if k != 'B2 60/40']].reset_index(names='Year').melt('Year', var_name='Portfolio', value_name='EUR')
        dom = [k for k in COLORS if k in ib['Portfolio'].unique()]
        st.altair_chart(alt.Chart(ib).mark_bar().encode(
            x=alt.X('Year:O', title=None), xOffset='Portfolio:N', y=alt.Y('EUR:Q', axis=alt.Axis(format=',.0f')),
            color=alt.Color('Portfolio:N', scale=alt.Scale(domain=dom, range=[COLORS[k] for k in dom]),
                            legend=alt.Legend(orient='top', title=None)),
            tooltip=['Portfolio', 'Year', alt.Tooltip('EUR:Q', format=',.0f')]).properties(height=330),
            width='stretch')
    with c2:
        st.markdown('**Real wealth after payouts** (start = 1, deflated by euro area HICP)')
        rw = pd.concat([pd.DataFrame({'Date': s.index.to_timestamp(), 'Portfolio': k, 'Value': s.values / ad.W0})
                        for k, s in REAL.items() if k != 'B2 60/40'])
        st.altair_chart(line_chart(rw, 'Real wealth', rule_at=1.0, height=330), width='stretch')
    st.caption('B2 holds an accumulating MSCI World ETF, so it pays out only bond coupons; its income and real wealth '
               'are not comparable and are left out of the charts.')

    st.markdown('**Stress years and calendar-year returns**')
    cal = (1 + R_ALL).groupby(R_ALL.index.year).prod() - 1
    inc_chg = inc.pct_change()
    stress = pd.concat({'Total return': cal.loc[[2020, 2022], inc.columns],
                        'Income change vs. prior year': inc_chg.reindex([2020, 2021, 2022])})
    stress.index = [f'{a} {b}' for a, b in stress.index]
    st.dataframe(stress.style.format('{:+.1%}', na_rep='–'), width='stretch')
    st.caption('2020: Covid – the ECB asked banks to suspend dividends and many firms cut. '
               '2022: rate shock – stocks and bonds fell together.')
    cal.index = [f"{y} (Jan–{ad.END.strftime('%b')})" if y == ad.END.year and ad.END.month < 12 else str(y) for y in cal.index]
    st.dataframe(cal.style.format('{:+.1%}').background_gradient(cmap='RdYlGn', axis=None, vmin=-0.2, vmax=0.2),
                 width='stretch')

# ---------------------------------------------------------------------------------------------
# Tab 3: Stock selection (notebook section 6)
# ---------------------------------------------------------------------------------------------
with tabs[2]:
    rule = rule_from(P)
    st.markdown(f"Every December, using only prices and dividends known on that date, the rule picks "
                f"**{rule['n_stocks']} stocks** from {len(D['meta'])} MSCI EMU constituents: dividend record "
                f"(**{P['record']}**) → yield between 0 and {rule['max_yield']:.0%} → drop the worst "
                f"{rule['loser_quantile']:.0%} price performers → rank by trailing dividend yield"
                + (f" → keep current holdings ranked up to {rule['buffer_rank']}" if P['buffer'] else '')
                + f" → at most {rule['max_per_sector']} per sector and {rule['max_per_country']} per country. "
                f"Empty places go to the EURO STOXX 50 ETF.")
    counts = pd.DataFrame({t.year: c for t, (w, i, c) in SEL.items()}).T
    counts.index.name = 'Selection (Dec)'
    counts.columns = ['Universe', 'Listed ≥ 12 months', 'Dividend record', 'Yield in range', 'Losers dropped',
                      'Eligible', 'Selected', 'Empty places (ETF)']
    st.markdown('**How many stocks pass each screen**')
    st.dataframe(counts, width='stretch')

    dates = list(SEL)
    c1, c2 = st.columns([1, 1])
    with c1:
        t = st.selectbox('Holdings selected in December', dates[::-1], format_func=lambda p: str(p.year))
        info = SEL[t][1].sort_values('yield_ttm', ascending=False)
        st.dataframe(info.rename(columns={'name': 'Name', 'sector': 'Sector', 'country': 'Country',
                                          'yield_ttm': 'Trailing yield', 'rank': 'Rank',
                                          'kept_by_buffer': 'Kept by buffer'})
                     .style.format({'Trailing yield': '{:.2%}', 'Rank': '{:.0f}'}),
                     width='stretch', height=560)
    with c2:
        sec = pd.DataFrame({p.year: w.groupby(D['meta']['sector'].reindex(w.index)
                                              .fillna('EURO STOXX 50 ETF (empty places)')).sum()
                            for p, (w, i, c) in SEL.items()}).T.fillna(0)
        sl = sec.reset_index(names='Year').melt('Year', var_name='Sector', value_name='Weight')
        st.markdown('**Sector weights inside the Eurozone sleeve**')
        st.altair_chart(alt.Chart(sl).mark_bar().encode(
            x=alt.X('Year:O', title=None), y=alt.Y('Weight:Q', axis=alt.Axis(format='%'), title=None),
            color=alt.Color('Sector:N', scale=alt.Scale(scheme='tableau20'), legend=alt.Legend(orient='bottom', columns=3)),
            tooltip=['Year', 'Sector', alt.Tooltip('Weight:Q', format='.0%')]).properties(height=420),
            width='stretch')
        turnover = pd.Series({b.year: 1 - len(set(SEL[b][1].index) & set(SEL[a][1].index)) / rule['n_stocks']
                              for a, b in zip(dates[:-1], dates[1:])}, name='Share of places replaced')
        st.markdown('**Turnover:** share of the stock places replaced each December')
        st.dataframe(turnover.to_frame().T.style.format('{:.0%}'), width='stretch')

# ---------------------------------------------------------------------------------------------
# Tab 4: Attribution (notebook section 8)
# ---------------------------------------------------------------------------------------------
with tabs[3]:
    st.markdown('Evergreen and B1 share their bonds and gold; they differ only in the two stock sleeves. Here each '
                'stock sleeve is tested **on its own** (100% in that sleeve, same rebalancing and costs), so the effect '
                'of our stock rule is not mixed up with the choice of the US fund.')
    with st.spinner('Testing each sleeve on its own…'):
        ATT = attribution(P_JSON)
    st.dataframe(fmt(ATT), width='stretch')
    st.caption('Read across: if our rule beats the EURO STOXX 50 and Select Dividend 30 but not the 1/N universe, '
               'the edge may come from survivorship bias in today\'s constituent list rather than from the rule.')

# ---------------------------------------------------------------------------------------------
# Tab 5: Planner (notebook section 3)
# ---------------------------------------------------------------------------------------------
with tabs[4]:
    st.markdown('Forward-looking planning from the **stated assumptions** in the prospectus (not the backtest): '
                'expected return, income and a 1-in-20 bad year for each equity share. Net of 0.15% costs; '
                'normal approximation for the bad year.')
    shares = (0.4, 0.5, 0.6, 0.7, 0.8)
    if P['alloc_mode'] == 'Equity share':
        shares = tuple(sorted(set(shares) | {P['equity'] / 100}))
    src, eq = dc.planning_tables(ad.ASSUMPTIONS, equity_shares=shares)
    st.markdown('**Choosing the equity share**')
    mine = f"{P['equity'] / 100:.0%}" if P['alloc_mode'] == 'Equity share' else None
    st.dataframe(eq.style.format({c: '{:.1%}' for c in eq.columns if c != 'Gov / corp bonds'})
                 .apply(lambda r: ['background-color: rgba(201,162,39,0.25)' if r.name == mine else '' for _ in r],
                        axis=1), width='stretch')
    if mine:
        st.caption(f'Highlighted: your equity share ({mine}).')
    else:
        st.info('The planner uses the prospectus\'s equity-share mixes. Switch the allocation to **Equity share** to '
                'see your own row highlighted.')
    st.markdown('**Does each income source pay for itself?**')
    st.dataframe(src.style.format('{:+.1%}'), width='stretch')

# ---------------------------------------------------------------------------------------------
# Tab 6: About
# ---------------------------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"""
**Strategy.** 30% Eurozone dividend stocks selected by our rule, 30% US dividend aristocrats (SPYD), 20% euro
government bonds, 15% euro corporate bonds and 5% gold. Reset to target every December; band checks at the other
quarter-ends. All income is paid out to the family.

**Data.** Yahoo Finance via `yfinance` (month-end closes and dividends, ETF distributions, EUR/USD) and FRED
(euro area 3-month rate, euro area HICP, US 3-month T-bill). Universe: {len(D['meta'])} MSCI EMU constituents
(iShares Core MSCI EMU holdings, 25 Nov 2026). Sample: first selection December 2013, monthly returns
January 2014 – October 2026. Instruments used: {', '.join(f'{k} = {v}' for k, v in T.items() if v)}.
{' '.join(D['notes'])}

**Consistency with the notebook.** The app imports the same `evergreen_core.py` as the notebook; data preparation
is copied from the notebook in `app_data.py`. With every setting on the base case, the app reproduces the
notebook's numbers.

**Limitations.**
- *Survivorship bias:* the universe is today's constituents; firms that failed or were taken over since 2013 are
  missing. The 1/N diagnostic measures this.
- *Concentration:* 20 stocks with sector and country caps; financials and utilities dominate high-yield lists.
- *Look-ahead bias:* each selection uses only data up to its date, which the code checks.
- *Data:* Yahoo dividend histories can contain gaps or special dividends.
- *Costs:* estimates of commissions and spreads plus the transaction taxes in force; no market impact.

**Links.** Source code and notebook: [github.com/NiklasWMueller2404/evergreen-family-income](https://github.com/NiklasWMueller2404/evergreen-family-income)

Group BE · Nova SBE 2270 Financial Modeling · 2026/27. *For academic purposes only. Not investment advice.
Past performance is not a reliable indicator of future results.*
""")
