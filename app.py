"""Attention Pulse — Part 1 Data App for BANA409.

Audience: the artist-marketing lead who allocates the coming week's promotion
          budget (home-page slots, push notifications, pre-sale SMS) across the roster.
Decision: who to push now, who is fading, and whose "buzz" is only a one-day news spike
          that will not survive long enough to be worth paying for.

Run locally with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent / 'data'
TEAM_NAME = 'Fengjun Jiang'
DATA_PERIOD = '5 Sep - 4 Oct 2026 (30 days of Wikipedia pageviews)'
REFRESH_LINE = ('Refresh: daily at 06:00 UTC in production, when Wikimedia publishes the '
                'previous day; the allocation decision is reviewed every Monday.')

# An artist whose Wikipedia page is not the real one produces a near-zero series.
# Below this mean the row is a name-matching failure, not a real measurement.
MIN_MEAN_VIEWS = 50.0
WINDOW = 7               # the "previous week" / "latest week" comparison window

UP = '#1a7f5a'
DOWN = '#b0433f'
FLAT = '#8a8f98'


@st.cache_data
def load_data():
    """Join the four source tables into one tidy artist x day frame, plus artist-level stats."""
    summary = pd.read_csv(DATA_DIR / 'artist_attention_summary.csv')
    views = pd.read_csv(DATA_DIR / 'pageviews.csv')
    artists = pd.read_csv(DATA_DIR / 'artists.csv')
    releases = pd.read_csv(DATA_DIR / 'releases.csv')

    views['date'] = pd.to_datetime(views['pageview_date'], format='%Y%m%d')
    views = views.rename(columns={'pageviews': 'views'})
    views = views[['artist_name', 'date', 'views']]

    # Real career span and output come from the releases table, because the release_count
    # column in artist_attention_summary.csv is capped at 100 per artist and therefore
    # carries no information once an artist has enough releases.
    releases['year'] = pd.to_numeric(
        releases['release_date'].astype(str).str[:4], errors='coerce')
    output = releases.groupby('artist_name').agg(
        releases_seen=('release_mbid', 'size'),
        first_release=('year', 'min'),
        last_release=('year', 'max'),
    ).reset_index()
    output['catalogue_is_capped'] = output['releases_seen'] >= 100

    profile = artists[['artist_name', 'artist_type', 'gender', 'country', 'area',
                       'begin_date', 'tags']].copy()
    profile['debut_year'] = pd.to_numeric(
        profile['begin_date'].astype(str).str[:4], errors='coerce')

    # Some artists carry absent profile fields; the outer joins keep whichever source has them.
    base = profile.merge(output, on='artist_name', how='outer').merge(
        summary[['artist_name', 'top_tags']], on='artist_name', how='outer')
    return views, base


def artist_stats(views, base):
    """One row per artist: recent trend, how far their best day rose, and how long ago it was."""
    daily = views.groupby(['artist_name', 'date'], as_index=False)['views'].sum()

    dates = sorted(daily['date'].unique())
    early_days, late_days = dates[:WINDOW], dates[-WINDOW:]
    last_day = dates[-1]

    def window_median(frame, days):
        part = frame[frame['date'].isin(days)]
        return part.groupby('artist_name')['views'].median()

    stats = pd.DataFrame({
        'prior_week': window_median(daily, early_days),
        'latest_week': window_median(daily, late_days),
        'mean_views': daily.groupby('artist_name')['views'].mean(),
        'peak_day_views': daily.groupby('artist_name')['views'].max(),
        'total_views': daily.groupby('artist_name')['views'].sum(),
        'median_daily': daily.groupby('artist_name')['views'].median(),
    }).reset_index()

    stats['momentum'] = stats['latest_week'] / stats['prior_week'] - 1
    # Baseline = the artist's own first-week median, so "how big was the peak" is measured
    # against where they started rather than against a window already lifted by the event.
    stats['peak_multiple'] = stats['peak_day_views'] / stats['prior_week']
    stats['tracking_is_reliable'] = stats['mean_views'] >= MIN_MEAN_VIEWS

    peak_date = daily.loc[daily.groupby('artist_name')['views'].idxmax(),
                          ['artist_name', 'date']].rename(columns={'date': 'peak_date'})
    stats = stats.merge(peak_date, on='artist_name', how='left')
    # Days between the artist's busiest day and the end of the window: 0 means it happened
    # yesterday, 25 means the wave passed weeks ago and only the trend matters now.
    stats['days_since_peak'] = (last_day - stats['peak_date']).dt.days
    return stats.merge(base, on='artist_name', how='left'), daily


def momentum_bucket(row):
    '''Three action labels the decision maker actually uses.'''
    if not row['tracking_is_reliable']:
        return 'Check data'
    if row['momentum'] >= 0.10:
        return 'Rising - amplify'
    if row['momentum'] <= -0.10:
        return 'Fading - do not fund'
    return 'Steady - hold'


def style_palette(labels):
    return {'Rising - amplify': UP, 'Steady - hold': FLAT,
            'Fading - do not fund': DOWN, 'Check data': '#c9a227'}


def metrics_row(frame, daily):
    '''The four numbers she looks at before anything else.'''
    total_now = int(frame['latest_week'].sum())
    total_then = int(frame['prior_week'].sum())
    shift = total_now / total_then - 1 if total_then else 0.0
    unreliable = frame[~frame['tracking_is_reliable']]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric('Artists tracked', f'{len(frame)}', help='After the filters in the sidebar.')
    col_b.metric('Attention units this week', f'{total_now:,}',
                 f'{shift:+.0%} vs previous week',
                 help='Sum of per-artist median daily views: what the roster draws in total.')
    rising = int((frame['action'] == 'Rising - amplify').sum())
    col_c.metric('Rising artists', f'{rising}',
                 help='Latest week at least 10% above the prior week.')
    if len(unreliable):
        col_d.metric('Rows to verify', f'{len(unreliable)}',
                     help='Wikipedia page probably mismatched. Excluded by default.')
    else:
        col_d.metric('Rows to verify', '0')


def momentum_chart(frame):
    '''Ranked bars: the single chart the Monday allocation meeting is built around.'''
    plot_frame = frame.sort_values('momentum')
    plot_frame = plot_frame.assign(label=plot_frame['artist_name'])
    fig = px.bar(plot_frame, x='momentum', y='label', orientation='h',
                 color='action', color_discrete_map=style_palette(None),
                 custom_data=['latest_week', 'prior_week'],
                 labels={'momentum': 'Latest week vs prior week', 'label': '', 'action': ''})
    fig.update_traces(hovertemplate='<b>%{y}</b><br>%{x:+.0%} change'
                                    '<br>latest week %{customdata[0]:,.0f}/day'
                                    '<br>prior week %{customdata[1]:,.0f}/day<extra></extra>')
    fig.add_vline(x=0, line_width=1, line_color='#333333')
    fig.update_layout(height=max(420, 24 * len(plot_frame)), margin=dict(l=8, r=8, t=8, b=8),
                      showlegend=False, xaxis_tickformat='+.0%')
    return fig


def trajectory_chart(daily, names, event_days):
    '''Daily attention lines, with every artist's peak day marked.'''
    plot_frame = daily[daily['artist_name'].isin(names)].sort_values('date')
    fig = px.line(plot_frame, x='date', y='views', color='artist_name',
                  labels={'date': '', 'views': 'Wikipedia views per day',
                          'artist_name': ''})
    for name in names:
        own = plot_frame[plot_frame['artist_name'] == name]
        if own.empty:
            continue
        top = own.loc[own['views'].idxmax()]
        fig.add_annotation(x=top['date'], y=top['views'], text='peak', showarrow=True,
                           arrowhead=1, ax=0, ay=-22, font=dict(size=9, color='#555555'))
    for day in event_days:
        fig.add_vline(x=day, line_width=1, line_dash='dot', line_color='#c0c0c0')
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    return fig


def timing_chart(frame):
    '''Peak timing against trend: is this wave still happening, or is it over?'''
    fig = px.scatter(frame, x='days_since_peak', y='momentum', color='action',
                     color_discrete_map=style_palette(None), size='peak_multiple',
                     hover_name='artist_name',
                     labels={'days_since_peak': 'Days since the artist\'s busiest day',
                             'momentum': 'Latest week vs prior week',
                             'action': ''},
                     custom_data=['peak_multiple', 'latest_week'])
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>peak %{customdata[0]:.1f}x baseline'
                                    '<br>%{x} days ago<br>trend %{y:+.0%}'
                                    '<br>now %{customdata[1]:,.0f}/day<extra></extra>')
    fig.add_vline(x=WINDOW, line_width=1, line_dash='dash', line_color='#b0433f')
    fig.add_hline(y=0, line_width=1, line_color='#333333')
    fig.add_annotation(xref='paper', x=0.01, yref='paper', y=0.98, showarrow=False,
                       text='still warming: peak in the last week and trending up',
                       font=dict(size=9, color=UP), xanchor='left')
    fig.add_annotation(xref='paper', x=0.99, yref='paper', y=0.02, showarrow=False,
                       text='already over: the wave passed and attention is draining',
                       font=dict(size=9, color=DOWN), xanchor='right')
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=8, b=8),
                      yaxis_tickformat='+.0%',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    return fig


def main():
    st.set_page_config(page_title='Attention Pulse', layout='wide')
    views, base = load_data()
    stats, daily = artist_stats(views, base)
    stats['action'] = stats.apply(momentum_bucket, axis=1)

    st.title('Attention Pulse')
    st.caption(f'{TEAM_NAME}  ·  data period {DATA_PERIOD}  ·  {REFRESH_LINE}')

    # ---------------------------------------------------------------- sidebar
    st.sidebar.header('Filters')
    keep_bad = st.sidebar.toggle(
        'Include rows with a suspected page mismatch', value=False,
        help=('Two artists in the source files point at the wrong Wikipedia title, so their '
              'series is near zero. Off by default.'))
    frame = stats if keep_bad else stats[stats['tracking_is_reliable']]

    countries = sorted([c for c in frame['country'].dropna().unique()])
    picked_countries = st.sidebar.multiselect('Artist country', countries)
    if picked_countries:
        frame = frame[frame['country'].isin(picked_countries)]

    types = sorted([t for t in frame['artist_type'].dropna().unique()])
    picked_types = st.sidebar.multiselect('Artist type', types, default=types)
    if picked_types:
        frame = frame[frame['artist_type'].isin(picked_types)]

    top_attention = int(stats['latest_week'].max()) if len(stats) else 1
    min_attention = st.sidebar.slider(
        'Minimum latest-week attention (views/day)', 0, top_attention, 0,
        help='Defaults to zero so nothing is filtered out silently. Raise it to drop the '
             'long tail she would never fund anyway.')

    frame = frame[frame['latest_week'] >= min_attention]
    if frame.empty:
        st.warning('No artist passes these filters. Loosen them in the sidebar.')
        return

    # ---------------------------------------------------------------- headline
    metrics_row(frame, daily)
    st.divider()

    st.subheader('Who is actually moving')
    st.markdown(
        'Each bar compares the artist\'s median daily views in the **last 7 days** against the '
        '**first 7 days** of the window. This is the ranking the week\'s budget is set from: '
        'a rising artist is amplified with paid support, a fading one is left alone no matter '
        'how big they are.')
    st.plotly_chart(momentum_chart(frame), width='stretch')

    st.divider()
    st.subheader('Is the wave still happening, or is it already over?')
    st.markdown(
        'Every point is one artist. **Horizontal** is how many days ago their busiest day was; '
        '**vertical** is their current trend; **bubble size** is how far that peak rose above '
        'where they started. The chart answers the question she actually has, which a size '
        'ranking cannot: an artist can be enormous and already draining. Fund the artists '
        'low on the horizontal axis **and** above zero — their attention is still building.')
    st.plotly_chart(timing_chart(frame), width='stretch')

    st.divider()
    st.subheader('Daily shape')
    default_names = (frame.sort_values('latest_week', ascending=False)['artist_name']
                     .head(5).tolist())
    choices = sorted(frame['artist_name'].unique())
    picked = st.multiselect('Artists to plot', choices, default=default_names)
    if picked:
        busiest = (daily.groupby('date')['views'].sum())
        busiest_day = busiest.idxmax()
        st.plotly_chart(trajectory_chart(daily, picked, [busiest_day]), width='stretch')
        st.caption(f'Dotted vertical line: {busiest_day:%d %b}, the single busiest day across '
                   'the whole roster. Peaks are labelled.')
    else:
        st.info('Pick at least one artist to see their daily line.')

    st.divider()
    st.subheader('Roster detail')
    table = frame[['artist_name', 'country', 'artist_type', 'prior_week', 'latest_week',
                   'momentum', 'peak_multiple', 'peak_date', 'days_since_peak',
                   'action']].copy()
    table = table.sort_values('momentum', ascending=False)
    table['prior_week'] = table['prior_week'].map('{:,.0f}'.format)
    table['latest_week'] = table['latest_week'].map('{:,.0f}'.format)
    table['momentum'] = table['momentum'].map(lambda v: f'{v:+.0%}')
    table['peak_multiple'] = table['peak_multiple'].map('{:.1f}x'.format)
    table['peak_date'] = table['peak_date'].dt.strftime('%d %b')
    table.columns = ['Artist', 'Country', 'Type', 'Prior week/day', 'Latest week/day',
                     'Change', 'Peak vs baseline', 'Peak date', 'Days since peak',
                     'Recommended move']
    st.dataframe(table, width='stretch', hide_index=True)

    with st.expander('What this app deliberately does not show, and why'):
        st.markdown('''
- **Raw 30-day totals or means.** One news day can double an artist's monthly average and
  buy them a budget they do not deserve. Every headline number here uses a **median**, and the
  trend compares two 7-day windows rather than single days.
- **`release_count` from `artist_attention_summary.csv`.** It is capped at 100 per artist, and
  42 of the 51 artists hit the cap, so the column measures the cap rather than productivity.
  Nothing here uses it.
- **`last_release_year` for judging whether an artist is active.** Same truncation problem:
  the file reports 1982 for Michael Jackson and 2008 for Taylor Swift, which are just the
  hundredth row of their catalogue. Use the daily-traffic trend instead.
- **`track_count` in `releases.csv`.** Every row is 0, so the field carries no information.
- **Anything about YouTube, Spotify or ticket sales.** Not in these files. The app says
  "attention" rather than "revenue" on purpose; mixing them would invite conclusions the data
  cannot support.
        ''')


if __name__ == '__main__':
    main()
