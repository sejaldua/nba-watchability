"""
One-screen "slate board" layout for the NBA Watchability dashboard.

Everything that matters for a given day fits in a single 1600x1200 viewport,
which is what scripts/capture_dashboard.py screenshots for the Slack post:

    header      title, slate summary, day picker
    left        watchability landscape chart, best games + best nights of the next 7 days
    right       every game on the selected day, ranked by watchability

Row HTML keeps the `menu-row` class because the capture script waits on it.
"""
from __future__ import annotations

import datetime as dt
import html as py_html

import altair as alt
import pandas as pd
import streamlit as st
from dateutil import tz

import core.watchability as watch
from app.dashboard_views import (
    _network_logo_html,
    _parse_score,
    _spread_display_parts,
    _to_valid_datetime,
    _w2wn_live_boost,
    _win_prob_html,
)
from core.standings import _normalize_team_name
from core.team_meta import get_team_abbr

ET = tz.gettz("America/New_York")
CHART_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

REGION_ORDER = ["Must Watch", "Strong Watch", "Watchable", "Skippable", "Hard Skip"]
REGION_COLORS = {
    "Must Watch": "#1D6FE3",
    "Strong Watch": "#0EA47A",
    "Watchable": "#E68A00",
    "Skippable": "#8B6DB0",
    "Hard Skip": "#8899AA",
}
REGION_SLUG = {
    "Must Watch": "must",
    "Strong Watch": "strong",
    "Watchable": "watchable",
    "Skippable": "skippable",
    "Hard Skip": "hard",
}

BOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

.stApp, .stApp button, .stApp p, .stApp span, .stApp div { font-family: 'DM Sans', -apple-system, 'Segoe UI', sans-serif; }

:root {
  --wb-bg: #F3F5F9;
  --wb-card: #FFFFFF;
  --wb-line: #E4E8EF;
  --wb-line-soft: #EEF1F6;
  --wb-ink: #0D1B2A;
  --wb-ink-2: #4A5668;
  --wb-ink-3: #8592A6;
  --wb-must: #1D6FE3;
  --wb-strong: #0EA47A;
  --wb-watchable: #E68A00;
  --wb-skippable: #8B6DB0;
  --wb-hard: #8899AA;
  --wb-live: #E5173F;
  --wb-display: 'Barlow Condensed', 'DM Sans', sans-serif;
}

/* Kiosk chrome: no Streamlit header/toolbar, tight page padding. */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stAppDeployButton { display: none !important; }
[data-testid="stAppViewContainer"], .stApp { background: var(--wb-bg); }
[data-testid="stMainBlockContainer"], .block-container {
  padding: 18px 28px 12px 28px !important;
  max-width: 1600px;
}

/* Header */
.wb-head { display: flex; flex-direction: column; gap: 2px; }
.wb-title {
  font-family: var(--wb-display) !important; font-weight: 800; font-size: 34px; line-height: 1;
  letter-spacing: 0.01em; text-transform: uppercase; color: var(--wb-ink);
  display: flex; align-items: center; gap: 10px;
}
.wb-title .wb-ball {
  width: 12px; height: 12px; border-radius: 50%;
  background: linear-gradient(135deg, #F28C28, #C8102E);
}
.wb-sub { font-size: 13px; color: var(--wb-ink-2); }
.wb-sub b { color: var(--wb-ink); font-weight: 700; }
.wb-counts { display: inline-flex; gap: 6px; margin-left: 6px; vertical-align: middle; }
.wb-count {
  font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 999px;
  color: #fff; letter-spacing: 0.02em;
}

/* Day picker sits flush right */
.st-key-wb_day_picker { display: flex; justify-content: flex-end; }
.st-key-wb_day_picker [data-testid="stWidgetLabel"] { display: none; }

/* Cards */
.st-key-wb_chart_card, .st-key-wb_week_card, .st-key-wb_slate_card {
  background: var(--wb-card);
  border: 1px solid var(--wb-line) !important;
  border-radius: 14px;
  box-shadow: 0 1px 2px rgba(13,27,42,0.04), 0 8px 24px rgba(13,27,42,0.05);
}
.wb-card-head {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 2px 2px 8px 2px; border-bottom: 1px solid var(--wb-line-soft); margin-bottom: 4px;
}
.wb-card-title {
  font-family: var(--wb-display) !important; font-weight: 700; font-size: 19px;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--wb-ink);
}
.wb-card-note { font-size: 11.5px; color: var(--wb-ink-3); }

/* Game rows */
.wb-rows { display: flex; flex-direction: column; }
.wb-row {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr) 128px 168px;
  align-items: center; gap: 14px;
  padding: 5px 4px; border-bottom: 1px solid var(--wb-line-soft);
}
.wb-row:last-child { border-bottom: none; }
.wb-row.is-live { background: linear-gradient(90deg, rgba(229,23,63,0.06), transparent 60%); }
.wb-row.is-final { opacity: 0.6; }

.wb-score {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 44px; border-radius: 10px; color: #fff;
}
.wb-score .n { font-family: var(--wb-display) !important; font-weight: 800; font-size: 26px; line-height: 1; }
.wb-score .l { font-size: 8.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.92; }
.wb-must { background: var(--wb-must); }
.wb-strong { background: var(--wb-strong); }
.wb-watchable { background: var(--wb-watchable); }
.wb-skippable { background: var(--wb-skippable); }
.wb-hard { background: var(--wb-hard); }

.wb-teams { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.wb-team { display: flex; align-items: center; gap: 8px; min-width: 0; }
.wb-team img { width: 22px; height: 22px; object-fit: contain; flex: none; }
.wb-team .nm {
  font-weight: 700; font-size: 14px; color: var(--wb-ink);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.wb-team .rec { font-size: 11.5px; color: var(--wb-ink-3); font-variant-numeric: tabular-nums; flex: none; }
.wb-team .pts { margin-left: auto; font-family: var(--wb-display) !important; font-weight: 700; font-size: 18px; color: var(--wb-ink); }
.wb-flags { display: inline-flex; gap: 4px; margin-left: 4px; flex: none; }
.wb-flag {
  font-size: 10px; font-weight: 700; padding: 0 6px; border-radius: 999px; line-height: 16px;
  background: #F1F4F9; color: var(--wb-ink-2); cursor: help;
}
.wb-flag.inj { background: #FDECEF; color: #B0102B; }
.wb-flag.star { background: #FFF5DB; color: #8A5A00; }

.wb-when { display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }
.wb-tip { font-weight: 700; font-size: 13px; color: var(--wb-ink); font-variant-numeric: tabular-nums; }
.wb-tip.live { color: var(--wb-live); display: inline-flex; align-items: center; gap: 6px; }
.wb-tip.live::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--wb-live);
  box-shadow: 0 0 0 3px rgba(229,23,63,0.18);
}
.wb-when .network-logo { height: 18px; max-width: 64px; object-fit: contain; }
.wb-when .network-text { font-size: 11px; font-weight: 600; color: var(--wb-ink-2); }

.wb-line { display: flex; flex-direction: column; gap: 5px; }
.wb-spread { font-size: 12px; color: var(--wb-ink-2); }
.wb-spread b { color: var(--wb-ink); font-variant-numeric: tabular-nums; }
.wb-spread .fc { font-size: 10px; color: var(--wb-ink-3); margin-left: 4px; }
.winprob-row { display: flex; align-items: center; gap: 6px; }
.winprob-label { font-size: 10.5px; font-weight: 600; color: var(--wb-ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
.winprob-bar { flex: 1; display: flex; height: 5px; border-radius: 999px; overflow: hidden; background: var(--wb-line); min-width: 40px; }
.winprob-away { background: #C8102E; }
.winprob-home { background: #1D6FE3; }

/* Watch-now callout */
.wb-now {
  display: flex; align-items: center; gap: 10px; margin: 2px 0 6px 0; padding: 8px 12px;
  border-radius: 10px; background: #FFF0F3; color: #8E0A24; font-size: 13px;
}
.wb-now b { font-weight: 800; }
.wb-now .tag {
  font-family: var(--wb-display) !important; font-weight: 800; font-size: 15px; letter-spacing: 0.04em;
  color: #fff; background: var(--wb-live); padding: 1px 8px; border-radius: 6px;
}

/* Next 7 days */
.wb-week { display: grid; grid-template-columns: minmax(0, 1fr) 200px; column-gap: 0; }
.wb-week > div:first-child { padding-right: 18px; }
.wb-week > div:last-child { padding-left: 18px; border-left: 1px solid var(--wb-line-soft); }
.wb-mini, .wb-night { min-height: 58px; box-sizing: border-box; border-bottom: 1px solid var(--wb-line-soft); }
.wb-mini:last-child, .wb-night:last-child { border-bottom: none; }
.wb-mini { display: grid; grid-template-columns: 44px minmax(0, 1fr) auto; column-gap: 14px; align-items: center; padding: 8px 0; }
.wb-mini .wb-score { height: 40px; border-radius: 8px; }
.wb-mini .wb-score .n { font-size: 21px; }
.wb-mini .wb-teams { gap: 4px; }
.wb-mini .wb-team img { width: 18px; height: 18px; }
.wb-mini .wb-team .nm { font-size: 13px; }
.wb-mini .when { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; font-size: 11.5px; color: var(--wb-ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
.wb-mini .when b { color: var(--wb-ink); font-size: 12.5px; }
.wb-sec { font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--wb-ink-3); margin: 6px 0 2px 0; }
.wb-night { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 0; text-decoration: none !important; }
.wb-night .info { display: flex; flex-direction: column; gap: 3px; }
.wb-night .d { font-weight: 700; font-size: 13.5px; color: var(--wb-ink); }
.wb-night .s { font-size: 11px; color: var(--wb-ink-3); }
.wb-night .k {
  display: inline-flex; align-items: baseline; gap: 4px; padding: 3px 9px; border-radius: 999px;
  background: #E3F6EF; color: #08795A; white-space: nowrap;
}
.wb-night .k b { font-family: var(--wb-display) !important; font-weight: 800; font-size: 17px; line-height: 1; }
.wb-night .k small { font-size: 10px; font-weight: 700; letter-spacing: 0.03em; }

.wb-foot { font-size: 11px; color: var(--wb-ink-3); padding: 4px 2px 0 2px; }
.wb-foot b { color: var(--wb-ink-2); }
.wb-empty { padding: 20px 4px; color: var(--wb-ink-3); font-size: 13px; }
</style>
"""


def inject_board_css() -> None:
    st.html(BOARD_CSS)


# ---------------------------------------------------------------------------
# Small formatters
# ---------------------------------------------------------------------------

def _abbr(team: str) -> str:
    return get_team_abbr(team) or team[:3].upper()


def _tip_et(row) -> dt.datetime | None:
    return _to_valid_datetime(row.get("Tip dt (ET)"))


def _fmt_time(t: dt.datetime | None, row=None) -> str:
    if t is None:
        fallback = str((row.get("Tip display") or row.get("Tip short") or "") if row is not None else "").strip()
        return py_html.escape(fallback) if fallback and fallback.lower() != "nan" else "TBD"
    return t.strftime("%I:%M %p").lstrip("0") + " ET"


def _win_prob_block(row) -> str:
    try:
        ok = all(pd.notna(float(row.get(k))) for k in ("Home win prob", "Away win prob"))
    except (TypeError, ValueError):
        ok = False
    return _win_prob_html(row) if ok else ""


def _status(row) -> str:
    s = str(row.get("Status") or "pre").lower()
    if bool(row.get("Is live", False)):
        return "in"
    return s


def _score_pill(row) -> str:
    region = str(row.get("Region") or "Hard Skip")
    slug = REGION_SLUG.get(region, "hard")
    n = int(round(float(row.get("aWI") or 0.0)))
    label = py_html.escape(region.replace(" Watch", ""))
    return f"<div class='wb-score wb-{slug}'><span class='n'>{n}</span><span class='l'>{label}</span></div>"


def _team_line(row, side: str, *, flags: str = "", show_points: bool = False) -> str:
    cap = side.capitalize()
    team = str(row.get(f"{cap} team") or "")
    logo = str(row.get(f"{cap} logo") or "")
    rec = str(row.get(f"Record ({side})") or "")
    img = f"<img src='{py_html.escape(logo)}' alt=''/>" if logo else ""
    pts = ""
    if show_points:
        p = _parse_score(row.get(f"{cap} score"))
        if p is not None:
            pts = f"<span class='pts'>{p}</span>"
    return (
        f"<div class='wb-team'>{img}<span class='nm'>{py_html.escape(team)}</span>"
        f"<span class='rec'>{py_html.escape(rec)}</span>{flags}{pts}</div>"
    )


def _flags_html(row, side: str, top_stars: set[str]) -> str:
    cap = side.capitalize()
    out = []
    key = _normalize_team_name(str(row.get(f"{cap} team") or ""))
    star = str(row.get(f"{cap} Star Player") or "").strip()
    if key in top_stars and star:
        out.append(f"<span class='wb-flag star' title='{py_html.escape(star)}'>★ {py_html.escape(star.split()[-1])}</span>")
    inj = str(row.get(f"{cap} Key Injuries") or "").strip()
    if inj:
        out.append(f"<span class='wb-flag inj' title='{py_html.escape(inj)}'>Inj</span>")
    return f"<span class='wb-flags'>{''.join(out)}</span>" if out else ""


def _top_star_teams(day_df: pd.DataFrame, n: int = 5) -> set[str]:
    entries: list[tuple[float, str]] = []
    for _, r in day_df.iterrows():
        for side in ("away", "home"):
            try:
                f = float(r.get(f"Star factor ({side})") or 0.0)
            except Exception:
                f = 0.0
            if f > 0:
                entries.append((f, _normalize_team_name(str(r.get(f"{side.capitalize()} team") or ""))))
    entries.sort(key=lambda x: x[0], reverse=True)
    keys: list[str] = []
    for _f, k in entries:
        if k and k not in keys:
            keys.append(k)
        if len(keys) >= n:
            break
    return set(keys)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header(day_df: pd.DataFrame, selected: str | None) -> str:
    try:
        d = dt.date.fromisoformat(str(selected))
        day_text = d.strftime("%A, %B %-d")
    except Exception:
        day_text = "Today"
    n = len(day_df)
    counts = day_df["Region"].astype(str).value_counts() if "Region" in day_df.columns else pd.Series(dtype=int)
    pills = []
    for region in ("Must Watch", "Strong Watch"):
        c = int(counts.get(region, 0))
        if c:
            pills.append(
                f"<span class='wb-count' style='background:{REGION_COLORS[region]}'>{c} {region}</span>"
            )
    updated = dt.datetime.now(tz=ET).strftime("%-I:%M %p ET")
    return (
        "<div class='wb-head'>"
        "<div class='wb-title'><span class='wb-ball'></span>NBA Watchability</div>"
        f"<div class='wb-sub'><b>{py_html.escape(day_text)}</b> · {n} game{'s' if n != 1 else ''}"
        f"<span class='wb-counts'>{''.join(pills)}</span> · updated {updated}</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Landscape chart
# ---------------------------------------------------------------------------

def render_landscape(day_df: pd.DataFrame, *, height: int = 430) -> None:
    q_floor = getattr(watch, "QUALITY_FLOOR", 0.1)
    c_floor = getattr(watch, "CLOSENESS_FLOOR", 0.1)
    x_dom = [q_floor, 1.04]
    y_dom = [c_floor, 1.04]

    step = 0.02
    cells = []
    q = q_floor
    while q < 1.0 - 1e-9:
        c = c_floor
        while c < 1.0 - 1e-9:
            cells.append(
                {
                    "q": q, "q2": min(1.0, q + step), "c": c, "c2": min(1.0, c + step),
                    "Region": watch.awi_label(watch.awi(q + step / 2, c + step / 2)),
                }
            )
            c += step
        q += step
    color_scale = alt.Scale(domain=REGION_ORDER, range=[REGION_COLORS[r] for r in REGION_ORDER])
    x_scale = alt.Scale(domain=x_dom, nice=False)
    y_scale = alt.Scale(domain=y_dom, nice=False)

    regions = alt.Chart(pd.DataFrame(cells)).mark_rect(opacity=0.11, strokeWidth=0).encode(
        x=alt.X(
            "q:Q", scale=x_scale,
            axis=alt.Axis(
                title="TEAM QUALITY  →", values=[0.2, 0.4, 0.6, 0.8, 1.0], format=".1f",
                grid=False, domain=False, ticks=False, labelColor="#8592A6", labelFontSize=11,
                titleColor="#4A5668", titleFontSize=11, titleFontWeight=700, titlePadding=8,
            ),
        ),
        x2="q2:Q",
        y=alt.Y(
            "c:Q", scale=y_scale,
            axis=alt.Axis(
                title="COMPETITIVENESS  →", values=[0.2, 0.4, 0.6, 0.8, 1.0], format=".1f",
                grid=False, domain=False, ticks=False, labelColor="#8592A6", labelFontSize=11,
                titleColor="#4A5668", titleFontSize=11, titleFontWeight=700, titlePadding=8,
            ),
        ),
        y2="c2:Q",
        color=alt.Color("Region:N", scale=color_scale, legend=None),
        tooltip=[],
    )

    labels = alt.Chart(
        pd.DataFrame(
            [
                {"t": "MUST WATCH", "x": 0.86, "y": 0.985},
                {"t": "STRONG", "x": 0.70, "y": 0.90},
                {"t": "WATCHABLE", "x": 0.55, "y": 0.68},
                {"t": "SKIPPABLE", "x": 0.36, "y": 0.42},
                {"t": "HARD SKIP", "x": 0.20, "y": 0.15},
            ]
        )
    ).mark_text(fontSize=15, fontWeight=800, opacity=0.28, color="#4A5668", font="Barlow Condensed").encode(
        x=alt.X("x:Q", scale=x_scale), y=alt.Y("y:Q", scale=y_scale), text="t:N", tooltip=[]
    )

    layers = [regions, labels]
    if not day_df.empty:
        cols = ["Matchup", "Region", "aWI", "Team quality", "Closeness", "Tip display",
                "Away team", "Home team", "Away logo", "Home logo"]
        dfp = day_df[[c for c in cols if c in day_df.columns]].copy()
        dfp["_q"] = pd.to_numeric(dfp["Team quality"], errors="coerce").clip(q_floor, 1.0)
        dfp["_c"] = pd.to_numeric(dfp["Closeness"], errors="coerce").clip(c_floor, 1.0)
        # Games with near-identical scores would draw on top of each other:
        # nudge later ones to the nearest free spot inside the plot so every
        # matchup stays readable.
        dfp = dfp.sort_values("aWI", ascending=False)
        gap_q, gap_c = 0.15, 0.13
        rings = [(0, 0)] + [
            (i, j) for r in range(1, 6) for i in range(-r, r + 1) for j in range(-r, r + 1)
            if max(abs(i), abs(j)) == r
        ]
        placed: list[tuple[float, float]] = []
        new_q, new_c = [], []
        for qv, cv in zip(dfp["_q"], dfp["_c"]):
            spot = (qv, cv)
            for i, j in rings:
                cq, cc = qv + i * gap_q, cv + j * gap_c
                if not (q_floor + 0.04 <= cq <= 1.0 and c_floor + 0.05 <= cc <= 1.0):
                    continue
                if all(abs(cq - pq) >= gap_q or abs(cc - pc) >= gap_c for pq, pc in placed):
                    spot = (cq, cc)
                    break
            placed.append(spot)
            new_q.append(spot[0])
            new_c.append(spot[1])
        dfp["_q"], dfp["_c"] = new_q, new_c
        dfp["_wi"] = pd.to_numeric(dfp["aWI"], errors="coerce").round(0)
        dfp["_label"] = dfp.apply(
            lambda r: f"{_abbr(str(r['Away team']))} @ {_abbr(str(r['Home team']))}", axis=1
        )
        tooltip = [
            alt.Tooltip("Matchup:N"),
            alt.Tooltip("_wi:Q", title="Watchability"),
            alt.Tooltip("Region:N"),
            alt.Tooltip("Tip display:N", title="Tip"),
        ]
        dx = 0.022
        imgs = pd.concat(
            [
                dfp.assign(_x=dfp["_q"] - dx, _logo=dfp["Away logo"]),
                dfp.assign(_x=dfp["_q"] + dx, _logo=dfp["Home logo"]),
            ],
            ignore_index=True,
        )
        imgs = imgs[imgs["_logo"].astype(str).str.len() > 0]
        halo = alt.Chart(dfp).mark_circle(size=1500, opacity=0.9, stroke="#FFFFFF", strokeWidth=2).encode(
            x=alt.X("_q:Q", scale=x_scale), y=alt.Y("_c:Q", scale=y_scale),
            color=alt.Color("Region:N", scale=color_scale, legend=None),
            fillOpacity=alt.value(0.16), tooltip=tooltip,
        )
        logos = alt.Chart(imgs).mark_image(width=26, height=26).encode(
            x=alt.X("_x:Q", scale=x_scale), y=alt.Y("_c:Q", scale=y_scale), url="_logo:N", tooltip=tooltip,
        )
        tags = alt.Chart(dfp).mark_text(dy=26, fontSize=10.5, fontWeight=700, color="#4A5668", font=CHART_FONT).encode(
            x=alt.X("_q:Q", scale=x_scale), y=alt.Y("_c:Q", scale=y_scale), text="_label:N", tooltip=tooltip,
        )
        layers += [halo, logos, tags]

    chart = (
        alt.layer(*layers)
        .properties(height=height, padding={"left": 4, "right": 8, "top": 8, "bottom": 4})
        .configure_view(stroke=None)
        .configure_axis(labelFont=CHART_FONT, titleFont=CHART_FONT)
    )
    st.altair_chart(chart, width="stretch", theme=None)


# ---------------------------------------------------------------------------
# Slate list
# ---------------------------------------------------------------------------

def _slate_row(row, top_stars: set[str]) -> str:
    status = _status(row)
    live = status == "in"
    final = status == "post"
    t = _tip_et(row)
    if live:
        tr = str(row.get("Time remaining") or "").strip()
        when = f"<span class='wb-tip live'>LIVE {py_html.escape(tr)}</span>"
    elif final:
        when = "<span class='wb-tip'>Final</span>"
    else:
        when = f"<span class='wb-tip'>{_fmt_time(t, row)}</span>"
    network = _network_logo_html(
        str(row.get("Where to watch provider") or ""), str(row.get("Where to watch URL") or "")
    )
    label, value = _spread_display_parts(row)
    fc = "<span class='fc'>forecast</span>" if label.startswith("Forecast") else ""
    cls = "wb-row rec-card menu-row" + (" is-live" if live else "") + (" is-final" if final else "")
    return (
        f"<div class='{cls}'>"
        f"{_score_pill(row)}"
        "<div class='wb-teams'>"
        f"{_team_line(row, 'away', flags=_flags_html(row, 'away', top_stars), show_points=live or final)}"
        f"{_team_line(row, 'home', flags=_flags_html(row, 'home', top_stars), show_points=live or final)}"
        "</div>"
        f"<div class='wb-when'>{when}{network}</div>"
        f"<div class='wb-line'><div class='wb-spread'>Spread <b>{py_html.escape(value)}</b>{fc}</div>"
        f"{_win_prob_block(row)}</div>"
        "</div>"
    )


def _watch_now_html(day_df: pd.DataFrame) -> str:
    live = day_df[day_df.apply(lambda r: _status(r) == "in", axis=1)] if not day_df.empty else day_df
    if live.empty:
        return ""

    def _score(r) -> float:
        boost = _w2wn_live_boost(
            r.get("Time remaining"), _parse_score(r.get("Away score")), _parse_score(r.get("Home score"))
        )
        return float(r.get("aWI") or 0.0) + float(boost)

    best = live.loc[live.apply(_score, axis=1).idxmax()]
    a, h = _abbr(str(best["Away team"])), _abbr(str(best["Home team"]))
    sa, sh = _parse_score(best.get("Away score")), _parse_score(best.get("Home score"))
    score = f" {sa}-{sh}" if sa is not None and sh is not None else ""
    tr = str(best.get("Time remaining") or "").strip()
    return (
        f"<div class='wb-now'><span class='tag'>WATCH NOW</span>"
        f"<span><b>{py_html.escape(a)} @ {py_html.escape(h)}</b>{py_html.escape(score)}"
        f"{' · ' + py_html.escape(tr) if tr else ''}</span></div>"
    )


def render_slate(day_df: pd.DataFrame, selected: str | None) -> None:
    today = dt.datetime.now(tz=ET).date()
    try:
        d = dt.date.fromisoformat(str(selected))
    except Exception:
        d = today
    title = "Tonight's slate" if d == today else f"{d.strftime('%a')} {d.month}/{d.day} slate"
    head = (
        f"<div class='wb-card-head'><span class='wb-card-title'>{title}, ranked</span>"
        "<span class='wb-card-note'>Watchability 0-100 · spread from home team · ESPN win prob</span></div>"
    )
    if day_df.empty:
        st.html(head + "<div class='wb-empty'>No games on this day.</div>")
        return
    is_today = day_df["Tip dt (ET)"].apply(lambda x: (_to_valid_datetime(x) or dt.datetime.min).date() == today)
    top_stars = _top_star_teams(day_df[is_today]) if is_today.any() else set()
    rows = day_df.sort_values("aWI", ascending=False)
    body = "".join(_slate_row(r, top_stars) for _, r in rows.iterrows())
    st.html(head + _watch_now_html(day_df) + f"<div class='wb-rows'>{body}</div>")


# ---------------------------------------------------------------------------
# Next 7 days
# ---------------------------------------------------------------------------

def _best_nights(df: pd.DataFrame, n: int = 3) -> list[dict]:
    pre = df[df["Status"].astype(str).str.lower() == "pre"] if "Status" in df.columns else df
    if pre.empty:
        return []
    g = (
        pre.dropna(subset=["Local date"])
        .groupby("Local date")
        .agg(
            strong=("Region", lambda s: int(s.isin(["Must Watch", "Strong Watch"]).sum())),
            games=("Region", "size"),
            avg=("aWI", "mean"),
        )
        .reset_index()
        .sort_values(["strong", "avg"], ascending=[False, False])
        .head(n)
    )
    return g.to_dict("records")


def render_week(df: pd.DataFrame, selected: str | None) -> None:
    others = df[df["Local date"].astype(str) != str(selected)]
    if "Status" in others.columns:
        others = others[others["Status"].astype(str).str.lower() == "pre"]
    top = others.sort_values("aWI", ascending=False).head(4)

    minis = []
    for _, r in top.iterrows():
        t = _tip_et(r)
        day = t.strftime("%a %-m/%-d") if t else ""
        label, value = _spread_display_parts(r)
        time_txt = _fmt_time(t, r)
        if not day and isinstance(r.get("Local date"), dt.date):
            # Forecast rows only carry "Sun 7PM ET"; rebuild the date from Local date.
            ld = r["Local date"]
            day = f"{ld.strftime('%a')} {ld.month}/{ld.day}"
            time_txt = time_txt.split(" ", 1)[1] if time_txt.startswith(ld.strftime("%a")) else time_txt
        when = f"{py_html.escape(day)} · {time_txt}" if day else time_txt
        minis.append(
            "<div class='wb-mini'>"
            f"{_score_pill(r)}"
            f"<div class='wb-teams'>{_team_line(r, 'away')}{_team_line(r, 'home')}</div>"
            f"<div class='when'><b>{when}</b><span>Spread {py_html.escape(value)}</span></div>"
            "</div>"
        )
    nights = []
    for rec in _best_nights(df):
        d = rec["Local date"]
        d_label = d.strftime("%a %-m/%-d") if isinstance(d, dt.date) else str(d)
        href = f"?day={d.isoformat()}" if isinstance(d, dt.date) else "#"
        nights.append(
            f"<a class='wb-night' href='{href}' target='_self'>"
            f"<span class='info'><span class='d'>{py_html.escape(d_label)}</span>"
            f"<span class='s'>{int(rec['games'])} games · avg {int(round(float(rec['avg'] or 0)))}</span></span>"
            f"<span class='k' title='Must + Strong Watch games'><b>{int(rec['strong'])}</b>"
            "<small>Strong+</small></span></a>"
        )
    head = (
        "<div class='wb-card-head'><span class='wb-card-title'>Next 7 days</span>"
        "<span class='wb-card-note'>Best games ahead · nights by Strong+ games</span></div>"
    )
    body = (
        "<div class='wb-week'>"
        f"<div><div class='wb-sec'>Best games</div>{''.join(minis) or '<div class=wb-empty>Nothing scheduled.</div>'}</div>"
        f"<div><div class='wb-sec'>Best nights</div>{''.join(nights)}</div>"
        "</div>"
    )
    st.html(head + body)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_board(df: pd.DataFrame, date_options: list[str], default_day: str | None) -> str | None:
    """Render the one-screen board. Returns the selected slate date (ISO)."""
    inject_board_css()

    def _label(iso: str) -> str:
        d = dt.date.fromisoformat(iso)
        return f"{d.strftime('%a')} {d.month}/{d.day}"

    default = default_day if default_day in date_options else (date_options[0] if date_options else None)
    head_col, pick_col = st.columns([1, 1.25], vertical_alignment="bottom")
    with pick_col:
        with st.container(key="wb_day_picker"):
            selected = st.segmented_control(
                "Day", options=date_options, format_func=_label, default=default, label_visibility="collapsed"
            )
    selected = selected or default
    day_df = df[df["Local date"].astype(str) == str(selected)].copy()
    with head_col:
        st.html(render_header(day_df, selected))

    left, right = st.columns([0.92, 1.08], gap="medium")
    with left:
        with st.container(key="wb_chart_card", border=True):
            st.html(
                "<div class='wb-card-head'><span class='wb-card-title'>Watchability landscape</span>"
                "<span class='wb-card-note'>Up and right = better game</span></div>"
            )
            render_landscape(day_df)
        with st.container(key="wb_week_card", border=True):
            render_week(df, selected)
    with right:
        with st.container(key="wb_slate_card", border=True):
            render_slate(day_df, selected)
        st.html(
            "<div class='wb-foot'><b>How it works:</b> Watchability blends team quality "
            "(injury-adjusted win %, last season's until 20 games played) with competitiveness "
            "(closer spread = higher). Must Watch 90+, Strong 75+, Watchable 50+, Skippable 25+. "
            "Live games update with the score.</div>"
        )
    return selected
