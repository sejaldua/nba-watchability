from __future__ import annotations

import html as py_html
import json
import os
import sys
import datetime as dt
import textwrap
from typing import Tuple, Dict, List

import altair as alt
import pandas as pd
import streamlit as st
import requests
from dateutil import parser as dtparser
from dateutil import tz

from core.odds_api import fetch_nba_spreads_window
from core.schedule_espn import fetch_games_for_date
from core.standings import _normalize_team_name, get_record, get_win_pct
from core.standings_espn import fetch_team_standings_detail_maps
from core.team_meta import get_logo_url, get_team_abbr, get_team_mascot
from core.health_espn import compute_team_player_impacts, injury_weight
from app._network_logos_b64 import NETWORK_LOGO_B64
from core.importance import compute_importance_detail_map
from core.watchability_v2_params import KEY_INJURY_IMPACT_SHARE_THRESHOLD, INJURY_OVERALL_IMPORTANCE_WEIGHT
from core.build_watchability_df import build_watchability_df
from core.build_watchability_forecast_df import build_watchability_forecast_df

import core.watchability as watch


def _merge_live_and_forecast_df(live_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    if live_df is None or live_df.empty:
        return forecast_df.copy() if forecast_df is not None else pd.DataFrame()
    if forecast_df is None or forecast_df.empty:
        return live_df.copy()

    live = live_df.copy()
    fc = forecast_df.copy()

    def _key_cols(df: pd.DataFrame) -> pd.Series:
        return (
            df.get("Local date", pd.Series(index=df.index, dtype=object)).astype(str)
            + "|"
            + df.get("Home team", pd.Series(index=df.index, dtype=object)).astype(str).map(_normalize_team_name)
            + "|"
            + df.get("Away team", pd.Series(index=df.index, dtype=object)).astype(str).map(_normalize_team_name)
        )

    live["_merge_key"] = _key_cols(live)
    fc["_merge_key"] = _key_cols(fc)
    live_keys = set(live["_merge_key"].dropna().tolist())

    # Odds/live rows always win. Keep forecast only for games not present in live dataframe.
    fc_keep = fc[~fc["_merge_key"].isin(live_keys)].copy()

    # Align columns safely.
    for c in live.columns:
        if c not in fc_keep.columns:
            fc_keep[c] = None
    for c in fc_keep.columns:
        if c not in live.columns:
            live[c] = None

    out = pd.concat([live, fc_keep[live.columns]], ignore_index=True)
    if "_merge_key" in out.columns:
        out = out.drop(columns=["_merge_key"], errors="ignore")
    if "Tip dt (ET)" in out.columns:
        out = out.sort_values(["Local date", "Tip dt (ET)", "aWI"], ascending=[True, True, False], na_position="last")
    else:
        out = out.sort_values(["Local date", "aWI"], ascending=[True, False], na_position="last")
    return out.reset_index(drop=True)


def _normalize_dashboard_df_types(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    if "Local date" in d.columns:
        try:
            d["Local date"] = pd.to_datetime(d["Local date"], errors="coerce").dt.date
        except Exception:
            pass
    for c in ["Tip dt (ET)", "Tip dt (UTC)"]:
        if c in d.columns:
            try:
                d[c] = pd.to_datetime(d[c], errors="coerce")
            except Exception:
                continue
    if "Team Quality" not in d.columns and "Team quality" in d.columns:
        d["Team Quality"] = d["Team quality"]
    if "Competitiveness" not in d.columns and "Closeness" in d.columns:
        d["Competitiveness"] = d["Closeness"]
    return d


def _coerce_bool_series(values, *, default: bool = False) -> pd.Series:
    s = pd.Series(values).copy()
    if s.empty:
        return s.astype(bool)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default)
    normalized = s.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", "", "none", "nan"}
    out = pd.Series([default] * len(s), index=s.index, dtype=bool)
    out.loc[normalized.isin(truthy)] = True
    out.loc[normalized.isin(falsy)] = False
    return out


def _filter_displayable_dashboard_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()
    keep = pd.Series(True, index=d.index, dtype=bool)

    if "Status" in d.columns:
        status = d["Status"].astype(str).str.strip().str.lower()
        keep &= status != "post"
    else:
        status = pd.Series("pre", index=d.index, dtype=object)

    is_live = pd.Series(False, index=d.index, dtype=bool)
    if "Is live" in d.columns:
        is_live = _coerce_bool_series(d["Is live"], default=False)
    is_live |= status == "in"

    if "Tip dt (ET)" in d.columns:
        tip_dt = pd.to_datetime(d["Tip dt (ET)"], errors="coerce")
        now_et = dt.datetime.now(tz=tz.gettz("America/New_York"))
        forecast_rows = (
            _coerce_bool_series(d["Forecast"], default=False)
            if "Forecast" in d.columns
            else pd.Series(False, index=d.index, dtype=bool)
        )
        # Forecast rows are placeholders only for pre-tip display. If their tip time has
        # passed, hide them entirely and rely on the live dataframe for any active game.
        expired_forecast = forecast_rows & tip_dt.notna() & (tip_dt <= now_et)
        keep &= ~expired_forecast.fillna(False)

    return d.loc[keep].reset_index(drop=True)


@st.cache_data(ttl=60 * 5)  # 5 min for today's live/odds data
def _load_live_watchability_df(days_ahead: int = 7) -> pd.DataFrame:
    return build_watchability_df(days_ahead=days_ahead)


@st.cache_data(ttl=60 * 60)  # 1h for 7d forecast supplement
def _load_forecast_watchability_df(days_ahead: int = 7) -> pd.DataFrame:
    # Try committed artifact first for speed/stability, then fallback to local build.
    p_parquet = os.path.join("data", "forecast", "latest.parquet")
    p_csv = os.path.join("data", "forecast", "latest.csv")
    try:
        if os.path.exists(p_parquet):
            return pd.read_parquet(p_parquet)
    except Exception:
        pass
    try:
        if os.path.exists(p_csv):
            return pd.read_csv(p_csv)
    except Exception:
        pass
    return build_watchability_forecast_df(days_ahead=days_ahead)


@st.cache_data(ttl=60 * 5)  # combined view refresh cadence
def load_watchability_df(days_ahead: int = 7) -> pd.DataFrame:
    live = _load_live_watchability_df(days_ahead=days_ahead)
    forecast = _load_forecast_watchability_df(days_ahead=days_ahead)
    out = _merge_live_and_forecast_df(live, forecast)
    out = _normalize_dashboard_df_types(out)
    return _filter_displayable_dashboard_rows(out)


def inject_base_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;0,9..40,900;1,9..40,400&family=JetBrains+Mono:wght@500;700&display=swap');

/* ═══════════════════════════════════════════════════════
   NBA WATCHABILITY — PREMIUM SPORTS EDITORIAL DESIGN
   ═══════════════════════════════════════════════════════ */

:root {
  /* Core palette */
  --nba-navy: #0C1E3C;
  --nba-navy-mid: #17325A;
  --nba-blue: #1D6FE3;
  --nba-blue-light: #4A9AF5;
  --nba-red: #C8102E;
  --nba-gold: #F0A500;

  /* Surfaces */
  --surface-primary: #FAFBFD;
  --surface-card: #FFFFFF;
  --surface-elevated: #FFFFFF;
  --surface-muted: #F1F3F8;
  --surface-dark: #0C1E3C;

  /* Text */
  --text-primary: #0D1B2A;
  --text-secondary: #4A5568;
  --text-muted: #8899AA;
  --text-inverse: #FFFFFF;

  /* Region colors — vivid, intentional */
  --region-must-watch: #1D6FE3;
  --region-must-watch-bg: rgba(29,111,227,0.08);
  --region-must-watch-border: rgba(29,111,227,0.25);
  --region-strong: #0EA47A;
  --region-strong-bg: rgba(14,164,122,0.08);
  --region-strong-border: rgba(14,164,122,0.25);
  --region-watchable: #E68A00;
  --region-watchable-bg: rgba(230,138,0,0.07);
  --region-watchable-border: rgba(230,138,0,0.22);
  --region-skippable: #8B6DB0;
  --region-skippable-bg: rgba(139,109,176,0.07);
  --region-skippable-border: rgba(139,109,176,0.20);
  --region-hard-skip: #8899AA;
  --region-hard-skip-bg: rgba(136,153,170,0.06);
  --region-hard-skip-border: rgba(136,153,170,0.18);

  /* Shadows */
  --shadow-sm: 0 1px 3px rgba(12,30,60,0.06), 0 1px 2px rgba(12,30,60,0.04);
  --shadow-md: 0 4px 14px rgba(12,30,60,0.08), 0 2px 6px rgba(12,30,60,0.04);
  --shadow-lg: 0 10px 30px rgba(12,30,60,0.10), 0 4px 10px rgba(12,30,60,0.05);
  --shadow-glow-blue: 0 4px 20px rgba(29,111,227,0.15);

  /* Radii */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-pill: 999px;
}

/* ——— Global Streamlit overrides ——— */
section[data-testid="stSidebar"] {display: none;}
div[data-testid="stSidebarNav"] {display: none;}
div[data-testid="collapsedControl"] {display: none;}

.block-container {
  padding-top: 0rem;
  padding-bottom: 1rem;
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Override Streamlit title/header styles */
.block-container h1 {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 900 !important;
  letter-spacing: -0.03em !important;
  color: var(--text-primary) !important;
}
.block-container h2, .block-container h3 {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  color: var(--text-primary) !important;
}

/* ——— Hero header banner ——— */
.hero-banner {
  background: linear-gradient(135deg, #0C1E3C 0%, #17325A 45%, #1D6FE3 100%);
  border-radius: var(--radius-lg);
  padding: 28px 32px 24px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: "";
  position: absolute;
  top: -50%;
  right: -20%;
  width: 60%;
  height: 200%;
  background: radial-gradient(ellipse, rgba(240,165,0,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-banner::after {
  content: "";
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--nba-red), var(--nba-gold), var(--nba-blue));
}
.hero-title {
  font-family: 'DM Sans', sans-serif;
  font-size: 32px;
  font-weight: 900;
  color: #FFFFFF;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: 0;
}
.hero-subtitle {
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  font-weight: 500;
  color: rgba(255,255,255,0.70);
  margin-top: 6px;
  letter-spacing: 0.01em;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(255,255,255,0.12);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: var(--radius-pill);
  padding: 5px 14px;
  font-size: 11px;
  font-weight: 700;
  color: rgba(255,255,255,0.90);
  margin-top: 12px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.hero-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--nba-gold);
  animation: hero-pulse 2s ease-in-out infinite;
}
@keyframes hero-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

/* ——— Section headings ——— */
.section-head {
  font-family: 'DM Sans', sans-serif;
  font-size: 20px;
  font-weight: 900;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  margin-bottom: 12px;
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-head::before {
  content: "";
  width: 4px;
  height: 22px;
  border-radius: 2px;
  background: linear-gradient(180deg, var(--nba-blue), var(--nba-navy));
  flex-shrink: 0;
}

/* ——— Game cards (menu rows) ——— */
.menu-row {
  display: flex;
  align-items: center;
  gap: 14px;
}
.rec-card.menu-row {
  border: 1px solid var(--region-hard-skip-border);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  background: var(--surface-card);
  box-shadow: var(--shadow-sm);
  margin-bottom: 10px;
  transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.15s ease;
  position: relative;
  overflow: hidden;
}
.rec-card.menu-row:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
/* Region accent stripe on left edge */
.rec-card.menu-row::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  border-radius: 12px 0 0 12px;
  background: var(--region-hard-skip);
}
/* Region-specific card styling via data attribute */
.rec-card.menu-row[data-region="Must Watch"] {
  border-color: var(--region-must-watch-border);
  background: linear-gradient(135deg, var(--region-must-watch-bg) 0%, var(--surface-card) 60%);
}
.rec-card.menu-row[data-region="Must Watch"]::before { background: var(--region-must-watch); }
.rec-card.menu-row[data-region="Must Watch"]:hover { box-shadow: var(--shadow-glow-blue); }

.rec-card.menu-row[data-region="Strong Watch"] {
  border-color: var(--region-strong-border);
  background: linear-gradient(135deg, var(--region-strong-bg) 0%, var(--surface-card) 60%);
}
.rec-card.menu-row[data-region="Strong Watch"]::before { background: var(--region-strong); }

.rec-card.menu-row[data-region="Watchable"] {
  border-color: var(--region-watchable-border);
  background: linear-gradient(135deg, var(--region-watchable-bg) 0%, var(--surface-card) 60%);
}
.rec-card.menu-row[data-region="Watchable"]::before { background: var(--region-watchable); }

.rec-card.menu-row[data-region="Skippable"] {
  border-color: var(--region-skippable-border);
}
.rec-card.menu-row[data-region="Skippable"]::before { background: var(--region-skippable); }

.rec-card.menu-row[data-region="Hard Skip"] {
  border-color: var(--region-hard-skip-border);
  opacity: 0.80;
}
.rec-card.menu-row[data-region="Hard Skip"]::before { background: var(--region-hard-skip); }

/* ——— Watchability score badge ——— */
.menu-awi {width: 115px; flex-shrink: 0;}
.menu-awi .label {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.01em;
}
/* Region-colored labels */
.menu-awi .label.region-must-watch { color: var(--region-must-watch); }
.menu-awi .label.region-strong-watch { color: var(--region-strong); }
.menu-awi .label.region-watchable { color: var(--region-watchable); }
.menu-awi .label.region-skippable { color: var(--region-skippable); }
.menu-awi .label.region-hard-skip { color: var(--region-hard-skip); }

.menu-awi .score-number {
  font-family: 'JetBrains Mono', monospace;
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
  margin-top: 2px;
}
.menu-awi .score-number.region-must-watch { color: var(--region-must-watch); }
.menu-awi .score-number.region-strong-watch { color: var(--region-strong); }
.menu-awi .score-number.region-watchable { color: var(--region-watchable); }
.menu-awi .score-number.region-skippable { color: var(--region-skippable); }
.menu-awi .score-number.region-hard-skip { color: var(--region-hard-skip); }

.menu-awi .score-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 1px;
}
.menu-awi .subscores {
  margin-top: 4px;
  font-size: 10px;
  color: var(--text-secondary);
  line-height: 1.2;
}
.menu-awi .subscore {display: block;}
.menu-awi .subscore-val {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 10px;
}

/* Live badge */
.live-badge {color: var(--nba-red); font-weight: 700; font-size: 11px; margin-top: 3px;}
.live-time {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--nba-red);
  font-size: 11px;
  font-weight: 700;
  line-height: 1.1;
  margin-top: 4px;
}
.live-pulse {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--nba-red);
  animation: live-blink 1.2s ease-in-out infinite;
}
@keyframes live-blink {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.3; transform: scale(0.8); }
}

/* ——— Matchup area ——— */
.menu-teams {flex: 1; display:flex; align-items:center; gap:10px; min-width: 240px;}
.menu-teams .team {display:flex; align-items:center; gap:8px; min-width: 0;}
.menu-teams img {width: 28px; height: 28px;}
.menu-teams .name {font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.menu-teams .at {opacity: 0.6; padding: 0 2px;}
.menu-matchup {flex: 1 1 auto; min-width: 0; display:flex; flex-direction: column; gap: 3px;}
.menu-matchup .teamline {display:flex; align-items:center; gap:8px; min-width: 0; flex-wrap: nowrap;}
.menu-matchup img {width: 30px; height: 30px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.10));}
.menu-matchup .name {
  flex: 1 1 auto;
  min-width: 0;
  font-family: 'DM Sans', sans-serif;
  font-size: 15px;
  font-weight: 800;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: -0.01em;
}
.menu-matchup .name-full {display: inline;}
.menu-matchup .name-short {display: none;}
.menu-matchup .record {flex: 0 0 auto; font-size: 11px; font-weight: 500; color: var(--text-muted); white-space: nowrap;}
.menu-matchup .record-inline {font-size: 11px; font-weight: 500; color: var(--text-muted); white-space: nowrap; margin-left: 6px;}
.menu-matchup .sep {font-size: 10px; font-weight: 400; color: rgba(49,51,63,0.25); white-space: nowrap;}
.menu-matchup .health {font-size: 10px; font-weight: 600; color: var(--text-secondary); white-space: nowrap;}
.menu-matchup .health[data-tooltip] {cursor: pointer; text-decoration: underline dotted rgba(49,51,63,0.30); position: relative;}
.menu-matchup .health[data-tooltip]:hover::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 0;
  top: 125%;
  z-index: 9999;
  max-width: 320px;
  white-space: normal;
  background: var(--surface-dark);
  color: var(--text-inverse);
  border: none;
  box-shadow: var(--shadow-lg);
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  font-weight: 500;
  font-size: 11px;
  line-height: 1.35;
}
.menu-matchup .health[data-tooltip]:hover::before {
  content: "";
  position: absolute;
  left: 12px;
  top: 110%;
  border-width: 6px;
  border-style: solid;
  border-color: transparent transparent var(--surface-dark) transparent;
}

/* Network label */
.menu-network {display:flex; align-items:center; justify-content:center; flex-shrink: 0; width: 56px;}
.network-logo {height: 20px; width: auto; max-width: 56px; object-fit: contain; opacity: 0.85; transition: opacity 0.2s;}
.network-logo:hover {opacity: 1;}
.network-text {font-size: 9px; font-weight: 800; color: var(--nba-navy); white-space: nowrap;}
.network-link {text-decoration: none;}
.network-link:hover .network-text {text-decoration: underline;}

/* Meta column */
.menu-meta {width: 220px; font-size: 11px; color: var(--text-secondary); line-height: 1.4;}
.menu-meta div {margin: 1px 0;}
.menu-meta .tip-label {
  font-weight: 700;
  color: var(--text-primary);
  font-size: 12px;
}

/* ——— Win Probability bar ——— */
.winprob-row {display:flex; align-items:center; gap:6px; margin-top:5px;}
.winprob-label {font-family: 'JetBrains Mono', monospace; font-size:10px; font-weight:700; color:var(--text-secondary); white-space:nowrap; min-width:56px;}
.winprob-label:first-child {text-align:right;}
.winprob-label:last-child {text-align:left;}
.winprob-bar {flex:1; display:flex; height:6px; border-radius:3px; overflow:hidden; background: var(--surface-muted);}
.winprob-away {background: linear-gradient(90deg, #C8102E, #E8394D); height:100%; transition:width 0.4s ease;}
.winprob-home {background: linear-gradient(90deg, #1D6FE3, #4A9AF5); height:100%; transition:width 0.4s ease;}

/* ——— Badges (stars + injuries) ——— */
.matchup-badges {display:flex; flex-wrap: wrap; gap: 5px; margin-left: 42px; margin-top: 3px;}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  border: 1px solid rgba(49,51,63,0.12);
  border-radius: var(--radius-pill);
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 700;
  color: var(--text-secondary);
  background: var(--surface-muted);
  transition: background 0.15s, border-color 0.15s;
}
.badge:hover {background: rgba(29,111,227,0.06); border-color: rgba(29,111,227,0.20);}
.badge[data-tooltip] {cursor: pointer; position: relative;}
.badge[data-tooltip]:hover::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 0;
  top: 125%;
  z-index: 9999;
  max-width: 340px;
  white-space: pre-line;
  background: var(--surface-dark);
  color: var(--text-inverse);
  border: none;
  box-shadow: var(--shadow-lg);
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  font-weight: 500;
  font-size: 11px;
  line-height: 1.35;
}
.badge[data-tooltip]:hover::before {
  content: "";
  position: absolute;
  left: 12px;
  top: 110%;
  border-width: 6px;
  border-style: solid;
  border-color: transparent transparent var(--surface-dark) transparent;
}

/* ——— Recommendations module ——— */
.rec-wrap {margin-bottom: 10px;}
.rec-head {
  font-family: 'DM Sans', sans-serif;
  font-size: 20px;
  font-weight: 900;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  margin-bottom: 12px;
  margin-top: 68px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.rec-head::before {
  content: "";
  width: 4px;
  height: 22px;
  border-radius: 2px;
  background: linear-gradient(180deg, var(--nba-red), var(--nba-gold));
  flex-shrink: 0;
}
.rec-card {
  border: 1px solid rgba(12,30,60,0.08);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  background: var(--surface-card);
  box-shadow: var(--shadow-sm);
  margin-bottom: 12px;
  transition: box-shadow 0.2s ease;
}
.rec-card:hover {box-shadow: var(--shadow-md);}
.rec-title {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: var(--text-muted);
  line-height: 1.1;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.rec-title.now {color: var(--nba-red);}
.rec-title.upcoming {color: var(--nba-blue);}
.rec-title.doubleheader {color: #0EA47A;}
.rec-sub {
  margin-top: 4px;
  font-size: 16px;
  font-weight: 900;
  color: var(--text-primary);
  line-height: 1.15;
  letter-spacing: -0.01em;
}
.rec-row {margin-top: 10px; display:flex; align-items:center; gap:12px;}
.rec-teams {flex:1; display:flex; flex-direction: column; gap:6px; min-width: 0;}
.rec-teamline {display:flex; align-items:center; gap:8px; min-width: 0; flex-wrap: wrap; row-gap: 2px;}
.rec-teamline img {width: 30px; height: 30px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.10));}
.rec-teamline .name {
  flex: 1 1 auto;
  min-width: 0;
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  font-weight: 800;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: -0.01em;
}
.rec-teamline .name-full {display: inline;}
.rec-teamline .name-short {display: none;}
.rec-teamline .record {flex: 0 0 auto; font-size: 10px; font-weight: 500; color: var(--text-muted); white-space: nowrap;}
.rec-teamline .record-inline {font-size: 10px; font-weight: 500; color: var(--text-muted); white-space: nowrap; margin-left: 6px;}
.rec-teamline .sep {font-size: 10px; font-weight: 400; color: rgba(49,51,63,0.25); white-space: nowrap;}
.rec-teamline .health {font-size: 10px; font-weight: 600; color: var(--text-secondary); white-space: nowrap;}
.rec-teamline .health[data-tooltip] {cursor: pointer; text-decoration: underline dotted rgba(49,51,63,0.30); position: relative;}
.rec-teamline .health[data-tooltip]:hover::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 0;
  top: 125%;
  z-index: 9999;
  max-width: 320px;
  white-space: normal;
  background: var(--surface-dark);
  color: var(--text-inverse);
  border: none;
  box-shadow: var(--shadow-lg);
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  font-weight: 500;
  font-size: 11px;
  line-height: 1.35;
}
.rec-teamline .health[data-tooltip]:hover::before {
  content: "";
  position: absolute;
  left: 12px;
  top: 110%;
  border-width: 6px;
  border-style: solid;
  border-color: transparent transparent var(--surface-dark) transparent;
}
.rec-meta {display:flex; flex-direction: column; align-items: flex-end; gap:5px;}
.chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(29,111,227,0.15);
  border-radius: var(--radius-pill);
  padding: 5px 12px;
  font-size: 10px;
  font-weight: 700;
  color: var(--nba-blue);
  background: rgba(29,111,227,0.06);
  transition: background 0.15s;
}
.chip:hover {background: rgba(29,111,227,0.12);}
.chip a {color: var(--nba-blue); text-decoration: none;}
.rec-live {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 900;
  color: var(--nba-red);
}
.rec-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  color: var(--nba-red);
  margin-top: -1px;
}
.rec-wi {font-size: 11px; font-weight: 700; color: var(--text-secondary);}
.rec-tip {font-weight: 800; color: var(--text-primary);}
.rec-menu-row {padding-top: 12px; padding-bottom: 12px;}
.rec-menu-row + .rec-menu-row {border-top: 1px solid var(--surface-muted);}

/* Day rank rows */
.day-rank-row {padding: 10px 0;}
.day-rank-row + .day-rank-row {border-top: 1px solid var(--surface-muted);}
.day-rank-day {line-height: 1.2;}
.day-rank-chip {
  display: inline-flex;
  align-items: center;
  border: 1.5px solid rgba(29,111,227,0.25);
  border-radius: var(--radius-pill);
  padding: 6px 16px;
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  font-weight: 800;
  color: var(--nba-blue);
  background: rgba(29,111,227,0.06);
  text-decoration: none;
  transition: all 0.15s ease;
}
.day-rank-chip:hover {
  background: rgba(29,111,227,0.14);
  border-color: rgba(29,111,227,0.40);
  box-shadow: var(--shadow-sm);
}
.day-rank-count {margin-top: 3px; font-size: 11px; font-weight: 700; color: var(--text-muted); line-height: 1.2;}

/* ——— Mobile responsive ——— */
@media (max-width: 640px) {
  .hero-banner {padding: 20px 20px 18px; border-radius: var(--radius-md);}
  .hero-title {font-size: 24px;}
  .hero-subtitle {font-size: 12px;}
  .menu-row {flex-wrap: wrap; align-items: flex-start; gap: 8px 10px;}
  .menu-awi {width: 90px;}
  .menu-awi .score-number {font-size: 24px;}
  .menu-matchup {min-width: 0; flex: 1 1 calc(100% - 100px);}
  .menu-network {width: auto; margin-left: 42px;}
  .menu-meta {width: 100%; padding-left: 90px; font-size: 11px; line-height: 1.35;}
  .menu-matchup .record {font-size: 10px;}
  .matchup-badges {margin-left: 42px;}
  .menu-matchup .name-full {display: none;}
  .menu-matchup .name-short {display: inline;}
  .rec-teamline .name-full {display: none;}
  .rec-teamline .name-short {display: inline;}
  .day-rank-chip {font-size: 11px; padding: 4px 10px;}
  .day-rank-count {font-size: 10px;}
}

/* Mobile rec visibility toggle */
.recs-mobile {display: none;}
.recs-desktop {display: block;}

/* ——— Day selector — elevated calendar style ——— */
[data-testid="stSegmentedControl"] > label {margin-bottom: 0.25rem;}
[data-testid="stSegmentedControl"] [role="radiogroup"] {
  gap: 6px;
  border-radius: 0;
  overflow: visible;
  border: none;
  width: fit-content;
  background: transparent;
}
[data-testid="stSegmentedControl"] [role="radiogroup"] label {
  min-height: 58px;
  min-width: 58px;
  padding: 6px 12px;
  border: 1.5px solid rgba(12,30,60,0.10);
  border-radius: var(--radius-md);
  background: var(--surface-card);
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}
[data-testid="stSegmentedControl"] [role="radiogroup"] label:hover {
  border-color: var(--nba-blue);
  background: rgba(29,111,227,0.04);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}
[data-testid="stSegmentedControl"] [role="radiogroup"] label p {
  font-family: 'DM Sans', sans-serif;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.3;
  text-align: center;
  white-space: pre-line;
  color: var(--text-secondary);
}
[data-testid="stSegmentedControl"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(135deg, var(--nba-navy) 0%, var(--nba-blue) 100%);
  border-color: var(--nba-blue);
  box-shadow: 0 4px 14px rgba(29,111,227,0.30);
  transform: translateY(-1px);
}
[data-testid="stSegmentedControl"] [role="radiogroup"] label:has(input:checked) p {
  color: #FFFFFF;
  font-weight: 800;
}

@media (max-width: 640px) {
  [data-testid="stSegmentedControl"] [role="radiogroup"] {gap: 4px;}
  [data-testid="stSegmentedControl"] [role="radiogroup"] label {
    min-height: 48px;
    min-width: 48px;
    padding: 4px 8px;
  }
  [data-testid="stSegmentedControl"] [role="radiogroup"] label p {
    font-size: 10px;
  }
}

@media (max-width: 640px) {
  .recs-mobile {display: block;}
  .recs-desktop {display: none;}
}

/* ——— Streamlit expander styling ——— */
details[data-testid="stExpander"] {
  border: 1px solid rgba(12,30,60,0.08) !important;
  border-radius: var(--radius-md) !important;
  background: var(--surface-muted) !important;
}
details[data-testid="stExpander"] summary {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 700 !important;
  color: var(--text-secondary) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def inject_minimal_chrome_css() -> None:
    st.markdown(
        """
<style>
header, footer {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden; height: 0px;}
section[data-testid="stSidebar"] {display: none;}
.block-container {padding-top: 0.25rem; padding-bottom: 0.25rem;}
</style>
""",
        unsafe_allow_html=True,
    )


def inject_autorefresh(ms: int = 3_600_000) -> None:
    st.html(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {int(ms)});
        </script>
        """,
        unsafe_allow_javascript=True,
    )


@st.cache_data(ttl=60 * 10)  # 10 min
def load_games() -> list:
    return fetch_nba_spreads_window(days_ahead=2)


@st.cache_data(ttl=60 * 60)  # 1 hour
def load_standings():
    try:
        return fetch_team_standings_detail_maps()
    except Exception:
        return {}, {}, {}


@st.cache_data(ttl=60 * 60 * 24)  # 24 hours (impact stats stable-ish)
def load_team_impacts(team_names: tuple[str, ...]) -> dict[str, dict]:
    """
    Returns per-team (normalized team name):
      { 'players': [{id,name,raw,share,rel}] }
    """
    out: dict[str, dict] = {}
    for name in team_names:
        key = _normalize_team_name(name)
        try:
            players = compute_team_player_impacts(name)
        except Exception:
            out[key] = {"players": []}
            continue

        out[key] = {
            "players": [
                {
                    "id": p.athlete_id,
                    "name": p.name,
                    "raw": float(p.raw_impact),
                    "share": float(p.impact_share),
                    "rel": float(p.relative_raw_impact),
                }
                for p in players
            ]
        }
    return out


def _parse_score(x):
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _is_forecast_spread_row(row) -> bool:
    mode = str(row.get("Spread mode", "") or "").strip().lower()
    src = str(row.get("Spread source", "") or "").strip().lower()
    if mode == "forecast":
        return True
    if "forecast" in src:
        return True
    return bool(row.get("Forecast", False))


def _round_spread_display_value(spread) -> float | None:
    try:
        if spread is None:
            return None
        if pd.isna(spread):
            return None
        return round(float(spread) * 2.0) / 2.0
    except Exception:
        return None


def _spread_display_parts(row) -> tuple[str, str]:
    spread = _round_spread_display_value(row.get("Home spread"))
    home = str(row.get("Home team", "") or "")
    home_abbr = get_team_abbr(home) or home[:3].upper()
    label = "Forecasted Spread" if _is_forecast_spread_row(row) else "Spread"
    if spread is None:
        return label, "?"
    s = f"{spread:+.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return label, f"{home_abbr} {s}"


def _win_prob_html(row) -> str:
    """Render a compact ESPN win probability bar for a game card row."""
    home_wp = row.get("Home win prob")
    away_wp = row.get("Away win prob")
    if home_wp is None or away_wp is None:
        return ""
    try:
        h = float(home_wp)
        a = float(away_wp)
    except Exception:
        return ""

    home_full = str(row.get("Home team", "") or "")
    away_full = str(row.get("Away team", "") or "")
    home_abbr = py_html.escape(get_team_abbr(home_full) or home_full[:3].upper())
    away_abbr = py_html.escape(get_team_abbr(away_full) or away_full[:3].upper())

    h_pct = max(5.0, min(95.0, h))
    a_pct = 100.0 - h_pct

    return (
        f"<div class='winprob-row'>"
        f"<span class='winprob-label'>{away_abbr} {a:.0f}%</span>"
        f"<div class='winprob-bar'>"
        f"<div class='winprob-away' style='width:{a_pct:.1f}%'></div>"
        f"<div class='winprob-home' style='width:{h_pct:.1f}%'></div>"
        f"</div>"
        f"<span class='winprob-label'>{home_abbr} {h:.0f}%</span>"
        f"</div>"
    )


def _to_valid_datetime(x) -> dt.datetime | None:
    """
    Normalize pandas/stdlib datetime-like values and guard against NaT.
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, dt.datetime):
        return x
    try:
        t = pd.to_datetime(x, errors="coerce")
        if pd.isna(t):
            return None
        return t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
    except Exception:
        return None


def _espn_gamecast_url(game_id) -> str:
    gid = str(game_id or "").strip()
    if not gid:
        return ""
    if not gid.isdigit():
        return ""
    return f"https://www.espn.com/nba/game/_/gameId/{gid}"


def _network_logo_html(provider: str, where_url: str = "") -> str:
    """Return a network logo as an inline image, optionally wrapped in a link."""
    label = str(provider or "").strip() or "League Pass"
    b64_src = NETWORK_LOGO_B64.get(label, "")
    if b64_src:
        img = f"<img class='network-logo' src='{b64_src}' alt='{py_html.escape(label)}' />"
    else:
        img = f"<span class='network-text'>{py_html.escape(label)}</span>"
    url = str(where_url or "").strip()
    if url:
        return (
            f"<a class='network-link' href='{py_html.escape(url)}' target='_blank' rel='noopener noreferrer'>"
            f"{img}</a>"
        )
    return img


def _watch_chip_html(where_url: str, provider: str) -> str:
    url = str(where_url or "").strip()
    if not url:
        return ""
    provider_label = str(provider or "").strip() or "League Pass"
    return (
        f"<span class='chip'><a href='{py_html.escape(url)}' target='_blank' rel='noopener noreferrer'>"
        f"Where to watch: {py_html.escape(provider_label)}</a></span>"
    )


def _follow_chip_html(game_id) -> str:
    follow_url = _espn_gamecast_url(game_id)
    if not follow_url:
        return ""
    return (
        f"<span class='chip'><a href='{py_html.escape(follow_url)}' target='_blank' rel='noopener noreferrer'>"
        "Where to follow: ESPN</a></span>"
    )


def _chips_for_row_html(row, *, wrap_in_divs: bool) -> str:
    chips: list[str] = []
    watch_chip = _watch_chip_html(
        str(row.get("Where to watch URL") or ""),
        str(row.get("Where to watch provider") or "") or "League Pass",
    )
    if watch_chip:
        chips.append(watch_chip)
    if wrap_in_divs:
        return "".join(f"<div>{c}</div>" for c in chips)
    return "\n".join(chips)


def _parse_time_remaining(tr: str | None) -> tuple[int | None, int | None]:
    """
    Returns (quarter, seconds_remaining_in_quarter) if parsable.
    Expected formats include: '5:32 Q3', '12:00 Q4', '0.0 Q2'.
    """
    if tr is None:
        return None, None
    s = str(tr).strip().upper()
    if not s:
        return None, None
    try:
        import re

        m = re.search(r"(\\d{1,2})[:.](\\d{1,2})\\s*Q(\\d)", s)
    except Exception:
        m = None
    if not m:
        return None, None
    mm = int(m.group(1))
    ss = int(m.group(2))
    q = int(m.group(3))
    return q, mm * 60 + ss


def _w2wn_live_boost(time_remaining: str | None, away_score: int | None, home_score: int | None) -> float:
    """
    W2WN live boost:
      - +3 in Q3 if score diff < 10
      - +5 in Q4 if score diff < 10
    """
    q, _sec = _parse_time_remaining(time_remaining)
    if q is None:
        return 0.0
    if away_score is None or home_score is None:
        return 0.0
    try:
        diff = abs(int(away_score) - int(home_score))
    except Exception:
        return 0.0
    if diff >= 10:
        return 0.0
    if q == 3:
        return 3.0
    if q >= 4:
        return 5.0
    return 0.0


def _fmt_wait_time(minutes: float) -> str:
    m = max(0.0, float(minutes))
    if m > 60:
        hrs = m / 60.0
        if hrs >= 2.0:
            return f"{int(round(hrs))} hours"
        return f"{hrs:.1f} hours"
    return f"{int(round(m))} mins"


def _pick_slate_df(df: pd.DataFrame, slate_day: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if slate_day and "Local date" in df.columns:
        out = df[df["Local date"].astype(str) == str(slate_day)].copy()
        if not out.empty:
            return out
    # Fallback: earliest ET date in the window.
    if "Local date" in df.columns:
        dates = sorted({str(x) for x in df["Local date"].dropna().tolist() if str(x).strip()})
        if dates:
            return df[df["Local date"].astype(str) == dates[0]].copy()
    return df.copy()


def _region_css_class(region: str) -> str:
    """Map region label to a CSS class suffix."""
    m = {
        "Must Watch": "must-watch",
        "Strong Watch": "strong-watch",
        "Watchable": "watchable",
        "Skippable": "skippable",
        "Hard Skip": "hard-skip",
    }
    return m.get(region, "hard-skip")


def render_recommendations_module(df: pd.DataFrame, *, slate_day: str | None, wrapper_class: str = "") -> None:
    """
    Recommendations:
      - What to watch now (W2WN)
      - Best upcoming game tonight
      - Best doubleheader tonight
      - Best games to switch between
    """
    if df is None or df.empty:
        return

    now_et = dt.datetime.now(tz=tz.gettz("America/New_York"))
    d = _pick_slate_df(df, slate_day)
    if d is None or d.empty:
        return

    selected_slate_date: dt.date | None = None
    try:
        if slate_day and len(str(slate_day)) == 10:
            y, m, dd = (int(x) for x in str(slate_day).split("-"))
            selected_slate_date = dt.date(y, m, dd)
        elif "Local date" in d.columns and not d["Local date"].dropna().empty:
            v = d["Local date"].dropna().iloc[0]
            if isinstance(v, dt.date):
                selected_slate_date = v
    except Exception:
        selected_slate_date = None

    is_same_day_slate = bool(selected_slate_date) and selected_slate_date == now_et.date()
    is_future_slate = bool(selected_slate_date) and selected_slate_date > now_et.date()

    # Recommendation feature flags (keep infra, but allow disabling specific cards).
    ENABLE_DOUBLEHEADER_REC = False
    ENABLE_SWITCH_BETWEEN_REC = False

    d["_is_live"] = d.get("Is live", False)
    d["_status"] = d.get("Status", "pre").astype(str) if "Status" in d.columns else "pre"

    def _minutes_to_tip_row(r) -> float | None:
        dt_tip = _to_valid_datetime(r.get("Tip dt (ET)"))
        if dt_tip is not None:
            return (dt_tip - now_et).total_seconds() / 60.0
        return None

    d["_minutes_to_tip"] = d.apply(_minutes_to_tip_row, axis=1)

    today_et = now_et.date()

    def _is_today_et_tip(x) -> bool:
        t = _to_valid_datetime(x)
        if t is not None:
            return t.date() == today_et
        return False

    def _top_star_set(day_df: pd.DataFrame) -> set[str]:
        entries: list[tuple[float, str]] = []
        for _, r in day_df.iterrows():
            away_key = _normalize_team_name(str(r.get("Away team", "")))
            home_key = _normalize_team_name(str(r.get("Home team", "")))
            try:
                af = float(r.get("Star factor (away)") or 0.0)
            except Exception:
                af = 0.0
            try:
                hf = float(r.get("Star factor (home)") or 0.0)
            except Exception:
                hf = 0.0
            if af > 0 and away_key:
                entries.append((af, away_key))
            if hf > 0 and home_key:
                entries.append((hf, home_key))

        entries.sort(key=lambda x: x[0], reverse=True)
        top_keys: list[str] = []
        for _v, k in entries:
            if k and k not in top_keys:
                top_keys.append(k)
            if len(top_keys) >= 5:
                break
        return set(top_keys)

    # Populate top-star flags on this slate (used by recommendation row rendering).
    if "Tip dt (ET)" in d.columns:
        d["_is_today_et"] = d["Tip dt (ET)"].apply(_is_today_et_tip)
        eligible = d[d["_is_today_et"]].copy()
    else:
        d["_is_today_et"] = False
        eligible = d.iloc[0:0].copy()
    top_star_set = _top_star_set(eligible)
    d["_away_top_star"] = d.apply(
        lambda r: bool(r.get("_is_today_et", False))
        and _normalize_team_name(str(r.get("Away team", ""))) in top_star_set,
        axis=1,
    )
    d["_home_top_star"] = d.apply(
        lambda r: bool(r.get("_is_today_et", False))
        and _normalize_team_name(str(r.get("Home team", ""))) in top_star_set,
        axis=1,
    )

    def _w2wn_score_row(r) -> float:
        wi = float(r.get("aWI") or 0.0)
        # Treat ESPN 'in' as live even if boolean flag is missing.
        if bool(r.get("_is_live", False)) or str(r.get("_status") or "").lower() == "in":
            away_s = _parse_score(r.get("Away score"))
            home_s = _parse_score(r.get("Home score"))
            return wi + float(_w2wn_live_boost(r.get("Time remaining"), away_s, home_s))
        if str(r.get("_status") or "").lower() == "pre":
            m = r.get("_minutes_to_tip")
            if m is None:
                # Unknown tip time: strongly de-prioritize.
                return wi - 1_000_000.0
            m = max(0.0, float(m))
            return wi - (m / 2.0)
        return wi

    d["_w2wn_score"] = d.apply(_w2wn_score_row, axis=1)

    def _subscores_row(r) -> tuple[str, str]:
        q = r.get("Team quality")
        c = r.get("Closeness")
        q_score = None if q is None else 100.0 * float(q)
        c_score = None if c is None else 100.0 * float(c)
        q_str = "—" if q_score is None else str(int(round(q_score)))
        c_str = "—" if c_score is None else str(int(round(c_score)))
        return c_str, q_str

    def _spread_str(r) -> tuple[str, str]:
        return _spread_display_parts(r)

    def _menu_like_row(row) -> str:
        awi_score = int(round(float(row.get("aWI") or 0.0)))
        region_raw = str(row.get("Region") or "")
        label = py_html.escape(region_raw)
        region_class = _region_css_class(region_raw)

        c_str, q_str = _subscores_row(row)

        live_badge = ""
        is_live = bool(row.get("_is_live", False)) or str(row.get("_status") or "").lower() == "in"
        if is_live:
            away_s = _parse_score(row.get("Away score"))
            home_s = _parse_score(row.get("Home score"))
            tr = str(row.get("Time remaining") or "").strip()
            tr_line = f"<div class='live-time'><span class='live-pulse'></span> LIVE {py_html.escape(tr)}</div>" if tr else "<div class='live-time'><span class='live-pulse'></span> LIVE</div>"
            if away_s is not None and home_s is not None:
                live_badge = f"{tr_line}<div class='live-badge'>{int(away_s)} - {int(home_s)}</div>"
            else:
                live_badge = tr_line

        away_full = str(row.get("Away team") or "")
        home_full = str(row.get("Home team") or "")
        away_mascot = get_team_mascot(away_full) or away_full
        home_mascot = get_team_mascot(home_full) or home_full
        away = py_html.escape(away_full)
        home = py_html.escape(home_full)
        away_short = py_html.escape(str(away_mascot))
        home_short = py_html.escape(str(home_mascot))

        tip_text = _tip_et(row)
        if tip_text:
            tip_line = f"Tip {tip_text}"
        else:
            tip_line = "Tip Unknown"
        tip_line = py_html.escape(tip_line)

        spread_label, spread_value = _spread_str(row)
        spread_str = py_html.escape(spread_value)
        spread_label = py_html.escape(spread_label)
        wp_html = _win_prob_html(row)

        rec_network_html = _network_logo_html(
            str(row.get("Where to watch provider") or ""),
            str(row.get("Where to watch URL") or ""),
        )

        record_away = py_html.escape(str(row.get("Record (away)", "—")))
        record_home = py_html.escape(str(row.get("Record (home)", "—")))

        away_inj = str(row.get("Away Key Injuries", "") or "").strip()
        home_inj = str(row.get("Home Key Injuries", "") or "").strip()

        stars_lines: list[str] = []
        if bool(row.get("_away_top_star", False)):
            n = str(row.get("Away Star Player") or "").strip()
            if n:
                stars_lines.append(f"{away_mascot}: {n}")
        if bool(row.get("_home_top_star", False)):
            n = str(row.get("Home Star Player") or "").strip()
            if n:
                stars_lines.append(f"{home_mascot}: {n}")
        stars_tooltip = py_html.escape("\n".join(stars_lines)) if stars_lines else ""

        inj_lines: list[str] = []
        if away_inj:
            inj_lines.append(f"{away_mascot}: {away_inj}")
        if home_inj:
            inj_lines.append(f"{home_mascot}: {home_inj}")
        inj_tooltip = py_html.escape("\n".join(inj_lines)) if inj_lines else ""

        badges_html = ""
        if stars_tooltip or inj_tooltip:
            badges = []
            if stars_tooltip:
                badges.append(f"<span class='badge' data-tooltip=\"{stars_tooltip}\">⭐ Top Stars</span>")
            if inj_tooltip:
                badges.append(f"<span class='badge' data-tooltip=\"{inj_tooltip}\">❗ Key Injuries</span>")
            badges_html = f"<div class='matchup-badges'>{''.join(badges)}</div>"

        away_logo = py_html.escape(str(row.get("Away logo") or ""))
        home_logo = py_html.escape(str(row.get("Home logo") or ""))
        away_img = f"<img src='{away_logo}'/>" if away_logo else ""
        home_img = f"<img src='{home_logo}'/>" if home_logo else ""

        return (
            f"<div class='menu-row rec-menu-row'>"
            f"<div class='menu-awi'>"
            f"<div class='label region-{region_class}'>{label}</div>"
            f"<div class='score-number region-{region_class}'>{awi_score}</div>"
            f"<div class='score-label'>Watchability</div>"
            f"<div class='subscores'>"
            f"<span class='subscore'>Comp <span class='subscore-val'>{py_html.escape(c_str)}</span></span>"
            f"<span class='subscore'>Quality <span class='subscore-val'>{py_html.escape(q_str)}</span></span>"
            f"</div>"
            f"{live_badge}"
            f"</div>"
            f"<div class='menu-matchup'>"
            f"<div class='teamline'>"
            f"{away_img}"
            f"<div class='name'><span class='name-full'>{away}</span><span class='name-short'>{away_short}</span><span class='record-inline'>{record_away}</span></div>"
            f"</div>"
            f"<div class='teamline'>"
            f"{home_img}"
            f"<div class='name'><span class='name-full'>{home}</span><span class='name-short'>{home_short}</span><span class='record-inline'>{record_home}</span></div>"
            f"</div>"
            f"{badges_html}"
            f"</div>"
            f"<div class='menu-network'>{rec_network_html}</div>"
            f"<div class='menu-meta'>"
            f"<div class='rec-tip'>{tip_line}</div>"
            f"<div>{spread_label}: {spread_str}</div>"
            f"{wp_html}"
            f"</div>"
            f"</div>"
        )

    def _matchup_block(r) -> str:
        away_full = str(r.get("Away team") or "")
        home_full = str(r.get("Home team") or "")
        away_mascot = get_team_mascot(away_full) or away_full
        home_mascot = get_team_mascot(home_full) or home_full
        away = py_html.escape(away_full)
        home = py_html.escape(home_full)
        away_short = py_html.escape(str(away_mascot))
        home_short = py_html.escape(str(home_mascot))
        away_logo = py_html.escape(str(r.get("Away logo") or ""))
        home_logo = py_html.escape(str(r.get("Home logo") or ""))
        away_img = f"<img src='{away_logo}'/>" if away_logo else ""
        home_img = f"<img src='{home_logo}'/>" if home_logo else ""
        record_away = py_html.escape(str(r.get("Record (away)", "—")))
        record_home = py_html.escape(str(r.get("Record (home)", "—")))

        away_inj = str(r.get("Away Key Injuries", "") or "").strip()
        home_inj = str(r.get("Home Key Injuries", "") or "").strip()
        away_inj_tip = py_html.escape(away_inj) if away_inj else ""
        home_inj_tip = py_html.escape(home_inj) if home_inj else ""

        away_star_html = ""
        home_star_html = ""
        if bool(r.get("_away_top_star", False)):
            tip = py_html.escape(str(r.get("Away Star Player") or ""))
            away_star_html = f"<div class='sep'>|</div><div class='health' data-tooltip=\"{tip}\">⭐ Top Star</div>"
        if bool(r.get("_home_top_star", False)):
            tip = py_html.escape(str(r.get("Home Star Player") or ""))
            home_star_html = f"<div class='sep'>|</div><div class='health' data-tooltip=\"{tip}\">⭐ Top Star</div>"

        away_key_html = (
            f"<div class='sep'>|</div><div class='health' data-tooltip=\"{away_inj_tip}\">❗ Key Injuries</div>"
            if away_inj
            else ""
        )
        home_key_html = (
            f"<div class='sep'>|</div><div class='health' data-tooltip=\"{home_inj_tip}\">❗ Key Injuries</div>"
            if home_inj
            else ""
        )

        return (
            f"<div class='rec-teams'>"
            f"<div class='rec-teamline'>{away_img}<div class='name'><span class='name-full'>{away}</span><span class='name-short'>{away_short}</span></div><div class='record'>{record_away}</div>{away_star_html}{away_key_html}</div>"
            f"<div class='rec-teamline'>{home_img}<div class='name'><span class='name-full'>{home}</span><span class='name-short'>{home_short}</span></div><div class='record'>{record_home}</div>{home_star_html}{home_key_html}</div>"
            f"</div>"
        )

    def _tip_et(row) -> str:
        def _fmt_clock(x: dt.datetime) -> str:
            return (
                x.strftime("%I:%M%p")
                .replace("AM", "am")
                .replace("PM", "pm")
                .lstrip("0")
            )

        dt_et = _to_valid_datetime(row.get("Tip dt (ET)"))
        if dt_et is not None:
            dow = dt_et.strftime("%a")
            return f"{dow} {_fmt_clock(dt_et)} ET"

        tip_et = str(row.get("Tip (ET)") or row.get("Tip short") or row.get("Tip display") or "").strip()
        if tip_et:
            clean = tip_et.replace(" ET", "").replace("ET", "").replace(" PT", "").replace("PT", "").strip()
            return f"{clean} ET"

        return ""

    def _rec_card(*, title: str, title_class: str, subtitle: str, row, extra_meta: str = "") -> str:
        tip_display = py_html.escape(_tip_et(row) or str(row.get("Tip display") or row.get("Tip short") or ""))
        wi_score = int(round(float(row.get("aWI") or 0.0)))
        net_logo = _network_logo_html(
            str(row.get("Where to watch provider") or ""),
            str(row.get("Where to watch URL") or ""),
        )
        c_str, q_str = _subscores_row(row)
        spread_label, spread_value = _spread_str(row)
        spread_line = py_html.escape(spread_value)
        spread_label = py_html.escape(spread_label)

        live_html = _live_score_html(row)
        return textwrap.dedent(
            f"""
            <div class="rec-card">
            <div class="rec-title {py_html.escape(title_class)}">{py_html.escape(title)}</div>
            <div class="rec-sub">{py_html.escape(subtitle)}</div>
            <div class="rec-row">
            {_matchup_block(row)}
            <div class="rec-meta">
            <div class="rec-wi">Watchability {wi_score}</div>
            <div class="rec-wi">Competitiveness {py_html.escape(c_str)} · Team Quality {py_html.escape(q_str)}</div>
            <div class="rec-wi">{tip_display}</div>
            <div class="rec-wi">{spread_label}: {spread_line}</div>
            {live_html}
            {net_logo}
            {extra_meta}
            </div>
            </div>
            </div>
            """
        ).strip()

    def _live_score_html(row) -> str:
        is_live = bool(row.get("_is_live", False)) or str(row.get("_status") or "").lower() == "in"
        if not is_live:
            return ""
        live_line = ""
        score_line = ""
        tr = str(row.get("Time remaining") or "").strip()
        live_line = f"<span class='live-pulse'></span> LIVE {py_html.escape(tr)}" if tr else "<span class='live-pulse'></span> LIVE"
        away_s = _parse_score(row.get("Away score"))
        home_s = _parse_score(row.get("Home score"))
        if away_s is not None and home_s is not None:
            score_line = f"{int(away_s)} - {int(home_s)}"

        html = f"<div class='rec-live'>{live_line}</div>"
        if score_line:
            html += f"<div class='rec-score'>{py_html.escape(score_line)}</div>"
        return html

    def _rec_meta_block(row) -> str:
        tip_display = py_html.escape(_tip_et(row) or str(row.get("Tip display") or row.get("Tip short") or ""))
        wi_score = int(round(float(row.get("aWI") or 0.0)))
        c_str, q_str = _subscores_row(row)
        spread_label, spread_value = _spread_str(row)
        spread_line = py_html.escape(spread_value)
        spread_label = py_html.escape(spread_label)
        net_logo = _network_logo_html(
            str(row.get("Where to watch provider") or ""),
            str(row.get("Where to watch URL") or ""),
        )
        live_html = _live_score_html(row)
        return textwrap.dedent(
            f"""
            <div class="rec-meta">
              <div class="rec-wi">Watchability {wi_score}</div>
              <div class="rec-wi">Competitiveness {py_html.escape(c_str)} · Team Quality {py_html.escape(q_str)}</div>
              <div class="rec-wi">{tip_display}</div>
              <div class="rec-wi">{spread_label}: {spread_line}</div>
              {live_html}
              {net_logo}
            </div>
            """
        ).strip()

    def _rec_card_multi(*, title: str, title_class: str, subtitle: str, rows: list) -> str:
        # rows: list of dataframe rows (Series-like)
        inner_rows = "\n".join([_menu_like_row(r) for r in rows])
        return textwrap.dedent(
            f"""
            <div class="rec-card">
              <div class="rec-title {py_html.escape(title_class)}">{py_html.escape(title)}</div>
              <div class="rec-sub">{py_html.escape(subtitle)}</div>
              {inner_rows}
            </div>
            """
        ).strip()

    def _day_rank_card(*, title: str, subtitle: str, day_rows: list[dict[str, str]]) -> str:
        inner_rows = []
        for row in day_rows:
            day_text = py_html.escape(str(row.get("day") or ""))
            href = str(row.get("href") or "").strip()
            if href:
                day_html = f"<a class='day-rank-chip' href='{py_html.escape(href)}'>{day_text}</a>"
            else:
                day_html = f"<span class='day-rank-chip'>{day_text}</span>"
            inner_rows.append(
                textwrap.dedent(
                    f"""
                    <div class="day-rank-row">
                      <div class="day-rank-day">{day_html}</div>
                      <div class="day-rank-count">{py_html.escape(str(row.get("count") or ""))}</div>
                    </div>
                    """
                ).strip()
            )
        return textwrap.dedent(
            f"""
            <div class="rec-card">
              <div class="rec-title upcoming">{py_html.escape(title)}</div>
              <div class="rec-sub">{py_html.escape(subtitle)}</div>
              {' '.join(inner_rows)}
            </div>
            """
        ).strip()

    cards: list[str] = []

    # 1) What to watch now
    show_w2wn = False
    try:
        if is_same_day_slate:
            # Only show when there is a live game.
            any_live = bool(
                (d.get("_is_live", False) == True).any()  # noqa: E712
                or (d.get("_status", "").astype(str).str.lower() == "in").any()
            )
            show_w2wn = any_live
    except Exception:
        show_w2wn = False

    if show_w2wn:
        live_df = d[
            (d.get("_is_live", False) == True)  # noqa: E712
            | (d.get("_status", "").astype(str).str.lower() == "in")
        ].copy()
        if not live_df.empty:
            live_sorted = live_df.sort_values(["_w2wn_score", "aWI"], ascending=False).reset_index(drop=True)
            rows = [live_sorted.iloc[0]]
            if len(live_sorted) > 1:
                r2 = live_sorted.iloc[1]
                if float(r2.get("aWI") or 0.0) > 50.0:
                    rows.append(r2)
            cards.append(_rec_card_multi(title="What to watch now", title_class="now", subtitle="Watch LIVE:", rows=rows))

    # 2) Best upcoming game tonight
    if is_future_slate:
        upcoming = d[
            (d["_status"].astype(str).str.lower() == "pre")
            & (d.get("_is_live", False) == False)  # noqa: E712
        ].copy()
    else:
        upcoming = d[
            (d["_status"].astype(str).str.lower() == "pre")
            & (d.get("_is_live", False) == False)  # noqa: E712
            & (d["_minutes_to_tip"].notna())
            & (d["_minutes_to_tip"].astype(float) > 0)
        ].copy()
    if not upcoming.empty:
        upcoming_sorted = upcoming.sort_values(["aWI", "_minutes_to_tip"], ascending=[False, True]).reset_index(drop=True)
        rows = [upcoming_sorted.iloc[0]]
        if len(upcoming_sorted) > 1:
            r2 = upcoming_sorted.iloc[1]
            if float(r2.get("aWI") or 0.0) > 50.0:
                rows.append(r2)
        if len(upcoming_sorted) > 2:
            r3 = upcoming_sorted.iloc[2]
            if float(r3.get("aWI") or 0.0) > 50.0:
                rows.append(r3)
        if selected_slate_date is not None:
            delta = (selected_slate_date - now_et.date()).days
            if delta <= 0:
                date_label = "tonight"
            elif delta == 1:
                date_label = "tomorrow"
            else:
                date_label = f"on {selected_slate_date.strftime('%A %-m/%-d')}"
        else:
            # Infer from the first upcoming game's tip time
            first_tip = _to_valid_datetime(rows[0].get("Tip dt (ET)")) if rows else None
            if first_tip is not None:
                tip_date = first_tip.date() if hasattr(first_tip, "date") else None
                if tip_date is not None:
                    delta = (tip_date - now_et.date()).days
                    if delta <= 0:
                        date_label = "tonight"
                    elif delta == 1:
                        date_label = "tomorrow"
                    else:
                        date_label = f"on {tip_date.strftime('%A %-m/%-d')}"
                else:
                    date_label = ""
            else:
                date_label = ""
        upcoming_title = f"Best upcoming {date_label}" if date_label else "Best upcoming"
        cards.append(_rec_card_multi(title=upcoming_title, title_class="upcoming", subtitle="", rows=rows))

    # 2b) Best games upcoming next 7 days (always use full df, not just selected slate).
    full_upcoming = df.copy()
    if full_upcoming is not None and not full_upcoming.empty:
        if "Status" in full_upcoming.columns:
            status_series = full_upcoming["Status"].astype(str).str.lower()
        else:
            status_series = pd.Series(["pre"] * len(full_upcoming), index=full_upcoming.index)
        full_upcoming = full_upcoming[status_series == "pre"].copy()
        if "Tip dt (ET)" in full_upcoming.columns:
            full_upcoming = full_upcoming.sort_values(["aWI", "Tip dt (ET)"], ascending=[False, True])
        else:
            full_upcoming = full_upcoming.sort_values("aWI", ascending=False)
        if not full_upcoming.empty:
            top_rows = [full_upcoming.iloc[i] for i in range(min(3, len(full_upcoming)))]
            cards.append(
                _rec_card_multi(
                    title="Best games upcoming next 7 days",
                    title_class="upcoming",
                    subtitle="",
                    rows=top_rows,
                )
            )

    # 2c) Best day of games upcoming next 7 days (rank all available days).
    if df is not None and not df.empty and "Local date" in df.columns:
        day_df = df.copy()
        if "Status" in day_df.columns:
            day_df = day_df[day_df["Status"].astype(str).str.lower() == "pre"].copy()
        day_rank_rows: list[dict[str, str]] = []
        if not day_df.empty:
            # If the first game of the current ET day has already started, exclude that day
            # entirely from the "best upcoming days" ranking to avoid mixing "today" with
            # still-upcoming future slates.
            try:
                if "Tip dt (ET)" in df.columns:
                    today_rows = df[df["Local date"] == now_et.date()].copy()
                    today_tips = today_rows["Tip dt (ET)"].apply(_to_valid_datetime).dropna()
                    if not today_tips.empty:
                        earliest_today_tip = min(today_tips.tolist())
                        if earliest_today_tip <= now_et:
                            day_df = day_df[day_df["Local date"] != now_et.date()].copy()
            except Exception:
                pass

            grouped = (
                day_df.dropna(subset=["Local date"])
                .groupby("Local date", dropna=True)
                .apply(
                    lambda g: pd.Series(
                        {
                            "strong_count": int(g["Region"].isin(["Must Watch", "Strong Watch"]).sum()),
                            "game_count": int(len(g)),
                            "avg_awi": (
                                float(pd.to_numeric(g["aWI"], errors="coerce").mean())
                                if "aWI" in g.columns
                                else 0.0
                            ),
                        }
                    )
                )
                .reset_index()
            )
            if not grouped.empty:
                grouped = grouped.sort_values(
                    ["strong_count", "game_count", "Local date"],
                    ascending=[False, False, True],
                )
                for _, gr in grouped.head(3).iterrows():
                    d_local = gr.get("Local date")
                    if isinstance(d_local, dt.date):
                        day_label = f"{d_local.strftime('%a')} {d_local.month}/{d_local.day}"
                    else:
                        day_label = str(d_local)
                    strong_count = int(gr.get("strong_count") or 0)
                    game_count = int(gr.get("game_count") or 0)
                    avg_awi_raw = gr.get("avg_awi")
                    avg_awi = 0 if pd.isna(avg_awi_raw) else int(round(float(avg_awi_raw)))
                    noun = "game" if strong_count == 1 else "games"
                    day_rank_rows.append(
                        {
                            "label": "",
                            "day": day_label,
                            "href": f"?day={d_local.isoformat()}#top" if isinstance(d_local, dt.date) else "",
                            "count": f"{strong_count} Strong+ {noun}, Avg watchability {avg_awi}, {game_count} total games",
                        }
                    )
        if day_rank_rows:
            cards.append(
                _day_rank_card(
                    title="Best day of games upcoming next 7 days",
                    subtitle="",
                    day_rows=day_rank_rows,
                )
            )

    # 3) Best doubleheader tonight (disabled for now; keep infra).
    if ENABLE_DOUBLEHEADER_REC:
        best_pair = None
        if not upcoming.empty and "Tip dt (ET)" in upcoming.columns:
            upcoming2 = upcoming.dropna(subset=["Tip dt (ET)"]).sort_values("Tip dt (ET)").copy()
            best_sum = None
            for i in range(len(upcoming2)):
                ti = upcoming2.iloc[i]["Tip dt (ET)"]
                for j in range(i + 1, len(upcoming2)):
                    tj = upcoming2.iloc[j]["Tip dt (ET)"]
                    if isinstance(ti, dt.datetime) and isinstance(tj, dt.datetime):
                        if (tj - ti).total_seconds() < 2 * 3600:
                            continue
                    s = float(upcoming2.iloc[i].get("aWI") or 0.0) + float(upcoming2.iloc[j].get("aWI") or 0.0)
                    if best_sum is None or s > best_sum:
                        best_sum = s
                        best_pair = (upcoming2.iloc[i], upcoming2.iloc[j])
        if best_pair is not None:
            g1, g2 = best_pair
            wi_avg = int(round(0.5 * (float(g1.get("aWI") or 0.0) + float(g2.get("aWI") or 0.0))))
            tip1 = py_html.escape(str(g1.get("Tip display") or g1.get("Tip short") or ""))
            tip2 = py_html.escape(str(g2.get("Tip display") or g2.get("Tip short") or ""))
            wi1 = int(round(float(g1.get("aWI") or 0.0)))
            wi2 = int(round(float(g2.get("aWI") or 0.0)))
            c1, q1 = _subscores_row(g1)
            c2, q2 = _subscores_row(g2)
            spread_label1, spread_value1 = _spread_str(g1)
            spread_label2, spread_value2 = _spread_str(g2)
            spread1 = py_html.escape(spread_value1)
            spread2 = py_html.escape(spread_value2)
            spread_label1 = py_html.escape(spread_label1)
            spread_label2 = py_html.escape(spread_label2)
            chip1 = _chips_for_row_html(g1, wrap_in_divs=False)
            chip2 = _chips_for_row_html(g2, wrap_in_divs=False)
            html = textwrap.dedent(
                f"""
                <div class="rec-card">
                <div class="rec-title doubleheader">Best doubleheader {date_label}</div>
                <div class="rec-sub">Two games (≥2h apart)</div>
                <div class="rec-wi">Average Watchability {wi_avg}</div>
                <div class="rec-row">
                  {_matchup_block(g1)}
                  <div class="rec-meta">
                    <div class="rec-wi">Watchability {wi1}</div>
                    <div class="rec-wi">Competitiveness {py_html.escape(c1)} · Team Quality {py_html.escape(q1)}</div>
                    <div class="rec-wi">{py_html.escape(_tip_et(g1) or '') or tip1}</div>
                    <div class="rec-wi">{spread_label1}: {spread1}</div>
                    {chip1}
                  </div>
                </div>
                <div class="rec-row">
                  {_matchup_block(g2)}
                  <div class="rec-meta">
                    <div class="rec-wi">Watchability {wi2}</div>
                    <div class="rec-wi">Competitiveness {py_html.escape(c2)} · Team Quality {py_html.escape(q2)}</div>
                    <div class="rec-wi">{py_html.escape(_tip_et(g2) or '') or tip2}</div>
                    <div class="rec-wi">{spread_label2}: {spread2}</div>
                    {chip2}
                  </div>
                </div>
                </div>
                """
            ).strip()
            cards.append(html)

    # 4) Best games to switch between (disabled for now; keep infra).
    if ENABLE_SWITCH_BETWEEN_REC:
        best_pair_close = None
        if not upcoming.empty and "Tip dt (ET)" in upcoming.columns:
            upcoming2 = upcoming.dropna(subset=["Tip dt (ET)"]).sort_values("Tip dt (ET)").copy()
            best_avg = None
            for i in range(len(upcoming2)):
                ti = upcoming2.iloc[i]["Tip dt (ET)"]
                for j in range(i + 1, len(upcoming2)):
                    tj = upcoming2.iloc[j]["Tip dt (ET)"]
                    if isinstance(ti, dt.datetime) and isinstance(tj, dt.datetime):
                        if (tj - ti).total_seconds() > 45 * 60:
                            break
                    avg = 0.5 * (
                        float(upcoming2.iloc[i].get("aWI") or 0.0)
                        + float(upcoming2.iloc[j].get("aWI") or 0.0)
                    )
                    if best_avg is None or avg > best_avg:
                        best_avg = avg
                        best_pair_close = (upcoming2.iloc[i], upcoming2.iloc[j])

        if best_pair_close is not None:
            g1, g2 = best_pair_close
            wi_avg = int(round(0.5 * (float(g1.get("aWI") or 0.0) + float(g2.get("aWI") or 0.0))))
            tip1 = py_html.escape(str(g1.get("Tip display") or g1.get("Tip short") or ""))
            tip2 = py_html.escape(str(g2.get("Tip display") or g2.get("Tip short") or ""))
            wi1 = int(round(float(g1.get("aWI") or 0.0)))
            wi2 = int(round(float(g2.get("aWI") or 0.0)))
            c1, q1 = _subscores_row(g1)
            c2, q2 = _subscores_row(g2)
            spread_label1, spread_value1 = _spread_str(g1)
            spread_label2, spread_value2 = _spread_str(g2)
            spread1 = py_html.escape(spread_value1)
            spread2 = py_html.escape(spread_value2)
            spread_label1 = py_html.escape(spread_label1)
            spread_label2 = py_html.escape(spread_label2)
            chip1 = _chips_for_row_html(g1, wrap_in_divs=False)
            chip2 = _chips_for_row_html(g2, wrap_in_divs=False)
            html = textwrap.dedent(
                f"""
                <div class="rec-card">
                <div class="rec-title">Best games to switch between</div>
                <div class="rec-sub">Two games (≤45m apart)</div>
                <div class="rec-wi">Average Watchability {wi_avg}</div>
                <div class="rec-row">
                  {_matchup_block(g1)}
                  <div class="rec-meta">
                    <div class="rec-wi">Watchability {wi1}</div>
                    <div class="rec-wi">Competitiveness {py_html.escape(c1)} · Team Quality {py_html.escape(q1)}</div>
                    <div class="rec-wi">{py_html.escape(_tip_et(g1) or '') or tip1}</div>
                    <div class="rec-wi">{spread_label1}: {spread1}</div>
                    {chip1}
                  </div>
                </div>
                <div class="rec-row">
                  {_matchup_block(g2)}
                  <div class="rec-meta">
                    <div class="rec-wi">Watchability {wi2}</div>
                    <div class="rec-wi">Competitiveness {py_html.escape(c2)} · Team Quality {py_html.escape(q2)}</div>
                    <div class="rec-wi">{py_html.escape(_tip_et(g2) or '') or tip2}</div>
                    <div class="rec-wi">{spread_label2}: {spread2}</div>
                    {chip2}
                  </div>
                </div>
                </div>
                """
            ).strip()
            cards.append(html)

    if not cards:
        cards.append(
            textwrap.dedent(
                """
                <div class="rec-card">
                  <div class="rec-sub" style="font-size:13px; font-weight:700; color: rgba(49,51,63,0.72);">
                    No live or upcoming recommendation for this slate yet.
                  </div>
                </div>
                """
            ).strip()
        )

    header_html = "<div class='rec-wrap'><div class='rec-head'>Recommendations</div></div>"
    inner = "\n".join([header_html] + cards)
    if wrapper_class:
        inner = f"<div class='{py_html.escape(wrapper_class)}'>{inner}</div>"
    st.markdown(inner, unsafe_allow_html=True)


@st.cache_data(ttl=60 * 10)  # 10 min (live scores)
def load_espn_game_map(local_dates_iso: tuple[str, ...]) -> dict[tuple[str, str, str], dict]:
    """
    Map (date_iso, home_team_lower, away_team_lower) -> dict with:
      - state ('pre'/'in'/'post')
      - game_id (str)
      - home_score (int|None)
      - away_score (int|None)
      - time_remaining (str|None) e.g. '5:32 Q3'
    """
    out: dict[tuple[str, str, str], dict] = {}
    if not local_dates_iso:
        return out

    targets = set(str(x) for x in local_dates_iso)
    local_tz = tz.gettz("America/New_York")

    # ESPN's scoreboard "dates=" is not always aligned with ET local dates for late games,
    # so fetch an extra day window and then map events back into ET dates.
    candidate_days = set()
    for iso in targets:
        try:
            y, m, d = (int(x) for x in iso.split("-"))
            day = dt.date(y, m, d)
            candidate_days.add(day)
            candidate_days.add(day + dt.timedelta(days=1))
        except Exception:
            continue

    for day in sorted(candidate_days):
        try:
            games = fetch_games_for_date(day)
        except Exception:
            continue
        for g in games:
            try:
                start = g.get("start_time_utc")
                if start:
                    dt_local = dtparser.isoparse(str(start)).astimezone(local_tz)
                    iso_local = dt_local.date().isoformat()
                else:
                    iso_local = None
            except Exception:
                iso_local = None
            if not iso_local or iso_local not in targets:
                continue

            home = _normalize_team_name(str(g.get("home_team", "")))
            away = _normalize_team_name(str(g.get("away_team", "")))
            state = str(g.get("state", ""))
            home_score = _parse_score(g.get("home_score"))
            away_score = _parse_score(g.get("away_score"))
            time_remaining = g.get("time_remaining")
            if home and away and state:
                out[(iso_local, home, away)] = {
                    "state": state,
                    "game_id": str(g.get("game_id") or ""),
                    "home_score": home_score,
                    "away_score": away_score,
                    "time_remaining": time_remaining,
                }
    return out


def _normalize_status_for_display(status: str | None) -> str:
    s = (status or "").strip()
    if not s:
        return "Available"
    if s.upper() == "OUT":
        return "Out"
    if s.upper() == "GTD":
        return "GTD"
    return s


@st.cache_data(ttl=60 * 10)  # 10 min
def load_espn_game_injury_report_map(game_ids: tuple[str, ...]) -> dict[str, dict[str, dict[str, str]]]:
    """
    Returns: game_id -> team_key -> athlete_id -> status
    (team_key is normalized team displayName from ESPN summary)
    """
    out: dict[str, dict[str, dict[str, str]]] = {}
    for gid in game_ids:
        gid_s = str(gid).strip()
        if not gid_s:
            continue
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
            r = requests.get(url, params={"event": gid_s}, timeout=12)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        injuries = data.get("injuries") if isinstance(data, dict) else None
        if not isinstance(injuries, list):
            continue

        by_team: dict[str, dict[str, str]] = {}
        for block in injuries:
            if not isinstance(block, dict):
                continue
            team = block.get("team")
            if not isinstance(team, dict):
                continue
            team_name = team.get("displayName") or team.get("name")
            if not team_name:
                continue
            team_key = _normalize_team_name(str(team_name))
            team_inj = block.get("injuries")
            if not isinstance(team_inj, list):
                continue

            m: dict[str, str] = {}
            for inj in team_inj:
                if not isinstance(inj, dict):
                    continue
                athlete = inj.get("athlete")
                athlete_id = None
                if isinstance(athlete, dict) and athlete.get("id"):
                    athlete_id = str(athlete.get("id"))
                if not athlete_id:
                    continue
                status = inj.get("status")
                details = inj.get("details")
                fs = None
                if isinstance(details, dict):
                    fantasy = details.get("fantasyStatus")
                    if isinstance(fantasy, dict):
                        fs = fantasy.get("displayDescription") or fantasy.get("description") or fantasy.get("abbreviation")

                chosen = _normalize_status_for_display(str(fs) if fs else (str(status) if status else ""))
                m[athlete_id] = chosen
            by_team[team_key] = m

        out[gid_s] = by_team

    return out


def _fmt_m_d(d: dt.date) -> str:
    return f"{d.month}/{d.day}"


def build_dashboard_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    df = load_watchability_df(days_ahead=7)
    if df.empty:
        return df, pd.DataFrame(columns=["Local date", "Day"]), [], {}

    df_dates = (
        df.dropna(subset=["Local date"])
        .sort_values("Local date")
        .loc[:, ["Local date", "Day"]]
        .drop_duplicates()
    )
    date_options = [d.isoformat() for d in df_dates["Local date"].tolist()]
    date_to_label = {
        d.isoformat(): f"{d.strftime('%a')}\n{d.strftime('%-m/%-d')}"
        for d, _day in df_dates.itertuples(index=False, name=None)
    }

    return df, df_dates, date_options, date_to_label


def render_chart(
    df: pd.DataFrame,
    date_options: list[str],
    date_to_label: dict[str, str],
    show_day_selector: bool,
    selected_date: str | None,
    default_day: str | None = None,
) -> str | None:
    def _fmt_m_d_yy_from_iso(iso: str | None) -> str | None:
        if not iso:
            return None
        try:
            y, m, d = (int(x) for x in iso.split("-"))
            return f"{m}/{d}/{str(y)[2:]}"
        except Exception:
            return None

    QUALITY_FLOOR = getattr(watch, "QUALITY_FLOOR", 0.1)
    CLOSENESS_FLOOR = getattr(watch, "CLOSENESS_FLOOR", 0.1)

    df_plot = df.copy()
    if "Team Quality" not in df_plot.columns and "Team quality" in df_plot.columns:
        df_plot["Team Quality"] = df_plot["Team quality"]
    if "Competitiveness" not in df_plot.columns and "Closeness" in df_plot.columns:
        df_plot["Competitiveness"] = df_plot["Closeness"]
    if "Away Key Injuries" in df_plot.columns:
        df_plot["Away Key Injuries"] = df_plot["Away Key Injuries"].fillna("")
    if "Home Key Injuries" in df_plot.columns:
        df_plot["Home Key Injuries"] = df_plot["Home Key Injuries"].fillna("")
    if "Away Star Factor" in df_plot.columns:
        df_plot["Away Star Factor"] = df_plot["Away Star Factor"].fillna("")
    if "Home Star Factor" in df_plot.columns:
        df_plot["Home Star Factor"] = df_plot["Home Star Factor"].fillna("")
    selected: str | None = None
    if date_options:
        if show_day_selector:
            default_value = default_day if (default_day in date_options) else date_options[0]
            selected = st.segmented_control(
                "Day",
                options=date_options,
                format_func=lambda x: date_to_label.get(x, x),
                default=default_value,
            )
        else:
            selected = selected_date if selected_date in date_options else date_options[0]
        df_plot = df[df["Local date"].astype(str) == selected].copy()

    chart_date_str = _fmt_m_d_yy_from_iso(selected)

    # Responsive sizing: keep mobile optimized, slightly larger on desktop web.
    # Vega-Lite exposes a `width` signal we can use to scale mark sizes.
    logo_size = alt.ExprRef(expr="clamp(width*0.06, 40, 50)")  # 40px on mobile, up to ~+24% on desktop
    tip_font_size = alt.ExprRef(expr="clamp(width*0.018, 11, 14)")
    region_label_font_size = alt.ExprRef(expr="clamp(width*0.040, 24, 30)")
    legend_font_size = alt.ExprRef(expr="clamp(width*0.020, 13, 16)")
    circle_size = alt.ExprRef(expr="clamp(width*1.2, 800, 992)")
    hit_target_size = alt.ExprRef(expr="clamp(width*6.3, 4200, 5208)")
    tips_dy = alt.ExprRef(expr="clamp(width*0.050, 32, 40)")
    axis_label_font_size = alt.ExprRef(expr="clamp(width*0.030, 20, 25)")
    axis_sublabel_font_size = alt.ExprRef(expr="clamp(width*0.020, 13, 16)")
    axis_label_dx = alt.ExprRef(expr="clamp(width*-0.110, -74, -90)")
    x_axis_title_dy = alt.ExprRef(expr="clamp(width*0.110, 72, 90)")
    x_axis_subtitle_dy = alt.ExprRef(expr="clamp(width*0.140, 92, 115)")
    chart_title_font_size = alt.ExprRef(expr="clamp(width*0.033, 21, 26)")

    region_order = ["Must Watch", "Strong Watch", "Watchable", "Skippable", "Hard Skip"]
    region_colors = {
        "Must Watch": "#1D6FE3",
        "Strong Watch": "#0EA47A",
        "Watchable": "#E68A00",
        "Skippable": "#8B6DB0",
        "Hard Skip": "#8899AA",
    }

    step = 0.02
    q_vals = [QUALITY_FLOOR + i * step for i in range(int((1.0 - QUALITY_FLOOR) / step) + 1)]
    c_vals = [CLOSENESS_FLOOR + i * step for i in range(int((1.0 - CLOSENESS_FLOOR) / step) + 1)]
    cells = []
    for q in q_vals[:-1]:
        for c in c_vals[:-1]:
            q_mid = q + step / 2
            c_mid = c + step / 2
            a = watch.awi(q_mid, c_mid)
            cells.append(
                {
                    "q": q,
                    "q2": min(1.0, q + step),
                    "c": c,
                    "c2": min(1.0, c + step),
                    "Region": watch.awi_label(a),
                }
            )
    regions_df = pd.DataFrame(cells)

    regions_other = (
        alt.Chart(regions_df)
        .transform_filter(alt.datum.Region != "Hard Skip")
        .mark_rect(opacity=0.10)
        .encode(
            x=alt.X("q:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
            x2="q2:Q",
            y=alt.Y("c:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
            y2="c2:Q",
            color=alt.Color(
                "Region:N",
                sort=region_order,
                scale=alt.Scale(domain=region_order, range=[region_colors[r] for r in region_order]),
                legend=None,
            ),
            tooltip=[],
        )
    )

    regions_bad = (
        alt.Chart(regions_df)
        .transform_filter(alt.datum.Region == "Hard Skip")
        .mark_rect(opacity=0.15, color=region_colors["Hard Skip"])
        .encode(
            x=alt.X("q:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
            x2="q2:Q",
            y=alt.Y("c:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
            y2="c2:Q",
            tooltip=[],
        )
    )

    regions = regions_other + regions_bad

    axes = alt.Chart(df_plot).mark_point(opacity=0).encode(
        x=alt.X(
            "Team Quality:Q",
            scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]),
            axis=alt.Axis(
                title="Team Quality",
                format=".2f",
                titleColor="rgba(0,0,0,0.9)",
                titleFontSize=18,
                titleFontWeight="bold",
                titlePadding=28,
                labelColor="rgba(0,0,0,0.65)",
                labelFontSize=12,
            ),
        ),
        y=alt.Y(
            "Competitiveness:Q",
            scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]),
            axis=alt.Axis(
                title="Competitiveness",
                format=".2f",
                titleColor="rgba(0,0,0,0.9)",
                titleFontSize=18,
                titleFontWeight="bold",
                titlePadding=34,
                labelColor="rgba(0,0,0,0.65)",
                labelFontSize=12,
            ),
        ),
        tooltip=[],
    )

    region_labels_df = pd.DataFrame(
        [
            {"label": "Must Watch", "x": 0.93, "y": 0.93},
            {"label": "Strong", "x": 0.83, "y": 0.82},
            {"label": "Watchable", "x": 0.64, "y": 0.60},
            {"label": "Skippable", "x": 0.40, "y": 0.40},
            {"label": "Hard Skip", "x": 0.20, "y": 0.20},
        ]
    )
    region_text = alt.Chart(region_labels_df).mark_text(
        fontSize=region_label_font_size,
        fontWeight=700,
        opacity=0.2,
        color="rgba(49,51,63,0.75)",
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        text=alt.Text("label:N"),
        tooltip=[],
    )

    # X-axis overlay label: render as two separate text marks (more reliable than newline rendering).
    x_axis_label_df_top = pd.DataFrame(
        [{"text": "Quality of Teams", "x": 0.55, "y": CLOSENESS_FLOOR + 0.035}]
    )
    x_axis_label_top = alt.Chart(x_axis_label_df_top).mark_text(
        dy=x_axis_title_dy,
        fontSize=axis_label_font_size,
        fontWeight=800,
        opacity=0.95,
        color="rgba(0,0,0,0.9)",
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        text=alt.Text("text:N"),
        tooltip=[],
    )

    x_axis_label_df_bottom = pd.DataFrame(
        [{"text": "(Avg Injury-Adjusted Winning Percentages)", "x": 0.55, "y": CLOSENESS_FLOOR + 0.035}]
    )
    x_axis_label_bottom = alt.Chart(x_axis_label_df_bottom).mark_text(
        dy=x_axis_subtitle_dy,
        fontSize=axis_sublabel_font_size,
        fontWeight=500,
        opacity=0.95,
        color="rgba(0,0,0,0.9)",
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        text=alt.Text("text:N"),
        tooltip=[],
    )

    x_axis_label_text = x_axis_label_top + x_axis_label_bottom

    y_axis_label_df_top = pd.DataFrame(
        [{"text": "Competitiveness", "x": QUALITY_FLOOR - 0.05, "y": 0.605}]
    )

    y_axis_label_text_top = alt.Chart(y_axis_label_df_top).mark_text(
        dx=axis_label_dx,
        fontSize=axis_label_font_size,
        fontWeight=800,
        opacity=0.95,
        color="rgba(0,0,0,0.9)",
        angle=270,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        text=alt.Text("text:N"),
        tooltip=[],
    )

    y_axis_label_df_bottom = pd.DataFrame(
        [{"text": "(Absolute Spread)", "x": QUALITY_FLOOR - 0.025, "y": 0.605}]
    )

    y_axis_sublabel_dx = alt.ExprRef(expr="clamp(width*-0.110, -74, -90) + 18")
    y_axis_label_text_bottom = alt.Chart(y_axis_label_df_bottom).mark_text(
        dx=y_axis_sublabel_dx,
        fontSize=axis_sublabel_font_size,
        fontWeight=500,
        opacity=0.95,
        color="rgba(0,0,0,0.9)",
        angle=270,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        text=alt.Text("text:N"),
        tooltip=[],
    )

    y_axis_label_text = y_axis_label_text_top + y_axis_label_text_bottom

    df_plot = df_plot.copy()
    def _spread_display_text_row(r) -> str:
        lbl, val = _spread_display_parts(r)
        return f"{lbl}: {val}"

    df_plot["Spread display"] = df_plot.apply(_spread_display_text_row, axis=1)

    def _win_prob_display_row(r) -> str:
        h = r.get("Home win prob")
        a = r.get("Away win prob")
        if h is None or a is None:
            return ""
        home_abbr = get_team_abbr(str(r.get("Home team", "") or "")) or "HOME"
        away_abbr = get_team_abbr(str(r.get("Away team", "") or "")) or "AWAY"
        return f"{away_abbr} {float(a):.0f}% / {home_abbr} {float(h):.0f}%"

    df_plot["Win Prob"] = df_plot.apply(_win_prob_display_row, axis=1)

    game_tooltip = [
        alt.Tooltip("Matchup:N"),
        alt.Tooltip("aWI:Q", title="Watchability", format=".1f"),
        alt.Tooltip("Region:N"),
        alt.Tooltip("Tip (ET):N"),
        alt.Tooltip("Spread display:N", title="Spread"),
        alt.Tooltip("Win Prob:N", title="ESPN Win Prob"),
        alt.Tooltip("Health (away):Q", title="Away health", format=".2f"),
        alt.Tooltip("Health (home):Q", title="Home health", format=".2f"),
        alt.Tooltip("Away Star Factor:N"),
        alt.Tooltip("Home Star Factor:N"),
        alt.Tooltip("Away Key Injuries:N"),
        alt.Tooltip("Home Key Injuries:N"),
        alt.Tooltip("Record (away):N"),
        alt.Tooltip("Record (home):N"),
    ]

    circles = alt.Chart(df_plot).mark_circle(size=circle_size, opacity=0.10).encode(
        x=alt.X("Team Quality:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("Competitiveness:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        color=alt.Color(
            "Region:N",
            sort=region_order,
            scale=alt.Scale(domain=region_order, range=[region_colors[r] for r in region_order]),
            legend=alt.Legend(title=None),
        ),
        tooltip=game_tooltip,
    )

    hit_targets = alt.Chart(df_plot).mark_circle(size=hit_target_size, opacity=0.001).encode(
        x=alt.X("Team Quality:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("Competitiveness:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        tooltip=game_tooltip,
    )

    dx = 0.03
    away_points = df_plot.assign(_x=(df_plot["Team quality"] - dx).clip(0, 1), _logo=df_plot["Away logo"])
    home_points = df_plot.assign(_x=(df_plot["Team quality"] + dx).clip(0, 1), _logo=df_plot["Home logo"])
    tooltip_cols = [
        "Matchup",
        "Tip short",
        "Tip (ET)",
        "Spread display",
        "Home spread",
        "Record (away)",
        "Record (home)",
        "aWI",
        "Region",
        "Team quality",
        "Closeness",
        "Importance",
        "Health (away)",
        "Health (home)",
        "Away Star Factor",
        "Home Star Factor",
        "Away Key Injuries",
        "Home Key Injuries",
        "Importance (away)",
        "Importance (home)",
        "Seed radius (away)",
        "Seed radius (home)",
        "Playoff radius (away)",
        "Playoff radius (home)",
        "_x",
        "_logo",
    ]
    tooltip_cols = list(dict.fromkeys([c for c in tooltip_cols if c in df_plot.columns] + ["_x", "_logo"]))

    images_df = pd.concat(
        [
            away_points[tooltip_cols].assign(_side="away"),
            home_points[tooltip_cols].assign(_side="home"),
        ],
        ignore_index=True,
    )
    images_df = images_df[images_df["_logo"].astype(bool)]

    images = alt.Chart(images_df).mark_image(width=logo_size, height=logo_size).encode(
        x=alt.X("_x:Q", axis=None),
        y=alt.Y("Closeness:Q", axis=None),
        url=alt.Url("_logo:N"),
        tooltip=game_tooltip,
    )

    tips = alt.Chart(df_plot).mark_text(
        dy=tips_dy,
        fontSize=tip_font_size,
        color="rgba(49,51,63,0.75)",
    ).encode(
        x=alt.X("Team quality:Q", axis=None),
        y=alt.Y("Closeness:Q", axis=None),
        text=alt.Text("Tip display:N"),
        tooltip=game_tooltip,
    )

    chart_legend_df = pd.DataFrame(
        [
            {"text": "↗ More watchable", "x": QUALITY_FLOOR + 0.01, "y": CLOSENESS_FLOOR + 0.04},
            #{"text": "↙ Less watchable", "x": QUALITY_FLOOR + 0.01, "y": CLOSENESS_FLOOR + 0.03},
        ]
    )
    chart_legend = alt.Chart(chart_legend_df).mark_text(
        align="left",
        baseline="top",
        fontSize=legend_font_size,
        fontWeight=700,
        color="rgba(49,51,63,0.70)",
        opacity=0.95,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[QUALITY_FLOOR, 1.0]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[CLOSENESS_FLOOR, 1.0]), axis=None),
        text=alt.Text("text:N"),
        tooltip=[],
    )

    chart = (
        axes
        + regions
        + region_text
        + x_axis_label_text
        + y_axis_label_text
        + circles
        + hit_targets
        + images
        + tips
        + chart_legend
    ).resolve_scale(x="shared", y="shared").properties(
        height=560,
        padding={"left": 0, "right": 0, "top": 10, "bottom": 30},
        title=alt.TitleParams(
            text="Watchability Landscape",
            subtitle=chart_date_str if chart_date_str else "Today",
            anchor="middle",
            fontSize=chart_title_font_size,
            fontWeight=800,
            color="rgba(0,0,0,0.9)",
            subtitleFontSize=alt.ExprRef(expr="clamp(width*0.025, 15, 20)"),
            subtitleFontWeight=400,
            subtitleColor="rgba(0,0,0,0.6)",
            dy=4,
        ),
    )
    st.altair_chart(chart, use_container_width=True)
    return selected


def _render_menu_row(r) -> str:
    awi_score = int(round(float(r["aWI"])))
    region_raw = str(r["Region"])
    label = py_html.escape(region_raw)
    region_class = _region_css_class(region_raw)

    q = r.get("Team quality")
    c = r.get("Closeness")
    q_score = None if q is None else 100.0 * float(q)
    c_score = None if c is None else 100.0 * float(c)
    q_str = "—" if q_score is None else str(int(round(q_score)))
    c_str = "—" if c_score is None else str(int(round(c_score)))

    live_badge = ""
    if bool(r.get("Is live", False)):
        away_s = r.get("Away score")
        home_s = r.get("Home score")
        tr = r.get("Time remaining")
        tr_line = (
            f"<div class='live-time'><span class='live-pulse'></span> LIVE {py_html.escape(str(tr))}</div>"
            if tr
            else "<div class='live-time'><span class='live-pulse'></span> LIVE</div>"
        )
        if away_s is not None and home_s is not None:
            live_badge = f"{tr_line}<div class='live-badge'>{int(away_s)} - {int(home_s)}</div>"
        else:
            live_badge = f"{tr_line}"

    away_full = str(r["Away team"])
    home_full = str(r["Home team"])
    away_mascot = get_team_mascot(away_full) or away_full
    home_mascot = get_team_mascot(home_full) or home_full
    away = py_html.escape(away_full)
    home = py_html.escape(home_full)
    away_short = py_html.escape(str(away_mascot))
    home_short = py_html.escape(str(home_mascot))
    dt_et = _to_valid_datetime(r.get("Tip dt (ET)"))
    if dt_et is not None:
        dow = dt_et.strftime("%a")
        et_time = dt_et.strftime("%I:%M%p").replace("AM", "am").replace("PM", "pm").lstrip("0")
        tip_line = f"Tip {dow} {et_time} ET"
    else:
        tip_et = str(r.get("Tip (ET)", "Unknown"))
        tip_line = f"Tip {tip_et} ET"
    tip_line = py_html.escape(tip_line)
    network_html = _network_logo_html(
        str(r.get("Where to watch provider") or ""),
        str(r.get("Where to watch URL") or ""),
    )
    spread_label, spread_value = _spread_display_parts(r)
    spread_str = py_html.escape(spread_value)
    spread_label = py_html.escape(spread_label)
    wp_html = _win_prob_html(r)
    record_away = py_html.escape(str(r.get("Record (away)", "—")))
    record_home = py_html.escape(str(r.get("Record (home)", "—")))
    health_away = r.get("Health (away)")
    health_home = r.get("Health (home)")
    health_away_str = "—" if health_away is None else f"{float(health_away):.2f}"
    health_home_str = "—" if health_home is None else f"{float(health_home):.2f}"

    away_inj = str(r.get("Away Key Injuries", "") or "").strip()
    home_inj = str(r.get("Home Key Injuries", "") or "").strip()

    stars_lines: list[str] = []
    if bool(r.get("_away_top_star", False)):
        n = str(r.get("Away Star Player") or "").strip()
        if n:
            stars_lines.append(f"{away_mascot}: {n}")
    if bool(r.get("_home_top_star", False)):
        n = str(r.get("Home Star Player") or "").strip()
        if n:
            stars_lines.append(f"{home_mascot}: {n}")
    stars_tooltip = py_html.escape("\n".join(stars_lines)) if stars_lines else ""

    inj_lines: list[str] = []
    if away_inj:
        inj_lines.append(f"{away_mascot}: {away_inj}")
    if home_inj:
        inj_lines.append(f"{home_mascot}: {home_inj}")
    inj_tooltip = py_html.escape("\n".join(inj_lines)) if inj_lines else ""

    badges_html = ""
    if stars_tooltip or inj_tooltip:
        badges = []
        if stars_tooltip:
            badges.append(f"<span class='badge' data-tooltip=\"{stars_tooltip}\">⭐ Top Stars</span>")
        if inj_tooltip:
            badges.append(f"<span class='badge' data-tooltip=\"{inj_tooltip}\">❗ Key Injuries</span>")
        badges_html = f"<div class='matchup-badges'>{''.join(badges)}</div>"
    away_logo = py_html.escape(str(r["Away logo"]))
    home_logo = py_html.escape(str(r["Home logo"]))
    away_img = f"<img src='{away_logo}'/>" if away_logo else ""
    home_img = f"<img src='{home_logo}'/>" if home_logo else ""

    # Avoid leading indentation/newlines: Streamlit Markdown can render it as a code block.
    return f"""<div class="rec-card menu-row" data-region="{label}">
<div class="menu-awi">
<div class="label region-{region_class}">{label}</div>
<div class="score-number region-{region_class}">{awi_score}</div>
<div class="score-label">Watchability</div>
<div class="subscores">
<span class="subscore">Comp <span class="subscore-val">{c_str}</span></span>
<span class="subscore">Quality <span class="subscore-val">{q_str}</span></span>
</div>
{live_badge}
</div>
<div class="menu-matchup">
<div class="teamline">
{away_img}
<div class="name"><span class="name-full">{away}</span><span class="name-short">{away_short}</span><span class="record-inline">{record_away}</span></div>
</div>
<div class="teamline">
{home_img}
<div class="name"><span class="name-full">{home}</span><span class="name-short">{home_short}</span><span class="record-inline">{record_home}</span></div>
</div>
{badges_html}
</div>
<div class="menu-network">{network_html}</div>
<div class="menu-meta">
<div class="tip-label">{tip_line}</div>
<div>{spread_label}: {spread_str}</div>
{wp_html}
</div>
</div>"""


def render_table(
    df: pd.DataFrame,
    df_dates: pd.DataFrame,
    date_options: list[str],
    *,
    selected_day: str | None = None,
) -> None:
    sort_mode = st.segmented_control("Sort ↓", options=["Watchability", "Tip time"], default="Watchability")
    sort_mode = sort_mode or "Watchability"
    today_et = dt.datetime.now(tz=tz.gettz("America/New_York")).date()

    def _is_today_et_tip(x) -> bool:
        t = _to_valid_datetime(x)
        if t is not None:
            return t.date() == today_et
        return False

    def _top_star_sets(day_df: pd.DataFrame) -> tuple[set[str], set[str]]:
        # Top 5 star factors across teams playing that day (inclusive of availability scaling).
        entries: list[tuple[float, str, str]] = []
        for _, r in day_df.iterrows():
            away_key = _normalize_team_name(str(r.get("Away team", "")))
            home_key = _normalize_team_name(str(r.get("Home team", "")))
            try:
                af = float(r.get("Star factor (away)") or 0.0)
            except Exception:
                af = 0.0
            try:
                hf = float(r.get("Star factor (home)") or 0.0)
            except Exception:
                hf = 0.0
            if af > 0:
                entries.append((af, away_key, "away"))
            if hf > 0:
                entries.append((hf, home_key, "home"))

        entries.sort(key=lambda x: x[0], reverse=True)
        top_keys: list[str] = []
        for _, k, _side in entries:
            if k and k not in top_keys:
                top_keys.append(k)
            if len(top_keys) >= 5:
                break
        top_set = set(top_keys)
        return top_set, top_set

    if selected_day and selected_day in (date_options or []) and "Local date" in df.columns:
        day_df = df[df["Local date"].astype(str) == str(selected_day)].copy()
        if day_df.empty:
            return
        if "Tip dt (ET)" in day_df.columns:
            day_df["_is_today_et"] = day_df["Tip dt (ET)"].apply(_is_today_et_tip)
            eligible = day_df[day_df["_is_today_et"]].copy()
        else:
            day_df["_is_today_et"] = False
            eligible = day_df.iloc[0:0].copy()

        top_star_set, _ = _top_star_sets(eligible)
        day_df["_away_top_star"] = day_df.apply(
            lambda r: bool(r.get("_is_today_et", False))
            and _normalize_team_name(str(r.get("Away team", ""))) in top_star_set,
            axis=1,
        )
        day_df["_home_top_star"] = day_df.apply(
            lambda r: bool(r.get("_is_today_et", False))
            and _normalize_team_name(str(r.get("Home team", ""))) in top_star_set,
            axis=1,
        )
        if sort_mode == "Tip time" and "Tip dt (ET)" in day_df.columns:
            day_df = day_df.sort_values("Tip dt (ET)", ascending=True, na_position="last")
        else:
            day_df = day_df.sort_values("aWI", ascending=False)
        for _, row in day_df.iterrows():
            st.markdown(_render_menu_row(row), unsafe_allow_html=True)
        return

    if date_options:
        for local_date, day_name in df_dates.itertuples(index=False, name=None):
            st.markdown(f"**{py_html.escape(str(day_name))}**")
            st.divider()
            day_df = df[df["Local date"] == local_date].copy()
            if "Tip dt (ET)" in day_df.columns:
                day_df["_is_today_et"] = day_df["Tip dt (ET)"].apply(_is_today_et_tip)
                eligible = day_df[day_df["_is_today_et"]].copy()
            else:
                day_df["_is_today_et"] = False
                eligible = day_df.iloc[0:0].copy()

            top_star_set, _ = _top_star_sets(eligible)
            day_df["_away_top_star"] = day_df.apply(
                lambda r: bool(r.get("_is_today_et", False))
                and _normalize_team_name(str(r.get("Away team", ""))) in top_star_set,
                axis=1,
            )
            day_df["_home_top_star"] = day_df.apply(
                lambda r: bool(r.get("_is_today_et", False))
                and _normalize_team_name(str(r.get("Home team", ""))) in top_star_set,
                axis=1,
            )
            if sort_mode == "Tip time" and "Tip dt (ET)" in day_df.columns:
                day_df = day_df.sort_values("Tip dt (ET)", ascending=True, na_position="last")
            else:
                day_df = day_df.sort_values("aWI", ascending=False)
            for _, row in day_df.iterrows():
                st.markdown(_render_menu_row(row), unsafe_allow_html=True)
            st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    else:
        flat = df.copy()
        if "Tip dt (ET)" in flat.columns:
            flat["_is_today_et"] = flat["Tip dt (ET)"].apply(_is_today_et_tip)
            eligible = flat[flat["_is_today_et"]].copy()
        else:
            flat["_is_today_et"] = False
            eligible = flat.iloc[0:0].copy()

        top_star_set, _ = _top_star_sets(eligible)
        flat["_away_top_star"] = flat.apply(
            lambda r: bool(r.get("_is_today_et", False)) and _normalize_team_name(str(r.get("Away team", ""))) in top_star_set,
            axis=1,
        )
        flat["_home_top_star"] = flat.apply(
            lambda r: bool(r.get("_is_today_et", False)) and _normalize_team_name(str(r.get("Home team", ""))) in top_star_set,
            axis=1,
        )
        if sort_mode == "Tip time" and "Tip dt (ET)" in flat.columns:
            flat = flat.sort_values("Tip dt (ET)", ascending=True, na_position="last")
        else:
            flat = flat.sort_values("aWI", ascending=False)
        for _, row in flat.iterrows():
            st.markdown(_render_menu_row(row), unsafe_allow_html=True)


def render_full_dashboard(title: str, caption: str) -> None:
    inject_base_css()
    inject_autorefresh()

    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="hero-banner">
<div class="hero-title">{py_html.escape(title)}</div>
<div class="hero-subtitle">{py_html.escape(caption)}</div>
<div class="hero-badge"><span class="hero-dot"></span> Live data</div>
</div>""",
        unsafe_allow_html=True,
    )
    with st.expander("How it works", icon=":material/info:"):
        st.markdown(
            "- **Competitiveness**: based on the spread (smaller spread = more competitive game).\n"
            "- **Team quality**: average of team winning percentages adjusted for key injuries based on player output.\n"
            "- **Output**: a single Watchability score + simple labels (Must Watch, Strong Watch, Watchable, Skippable, Hard Skip).\n"
            "- **Updates live**: watchability changes as the score changes."
        )

    df, df_dates, date_options, date_to_label = build_dashboard_frames()
    if df.empty:
        st.warning("No NBA regular season or playoff games found. Enjoy the break!")
        try:
            meta = {
                "slate_day": None,
                "tweet_date": dt.date.today().strftime("%b %d").replace(" 0", " "),
                "n_games": 0,
                "counts": {},
                "matchups": {},
            }
            meta_attr = py_html.escape(json.dumps(meta, ensure_ascii=False))
            st.markdown(
                f"<div id='tweet-meta' data-meta='{meta_attr}' style='display:none;'></div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        return

    default_day = None
    try:
        default_day = st.query_params.get("day")
    except Exception:
        default_day = None

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        selected = render_chart(
            df=df,
            date_options=date_options,
            date_to_label=date_to_label,
            show_day_selector=True,
            selected_date=None,
            default_day=default_day,
        )
        render_recommendations_module(df, slate_day=selected, wrapper_class="recs-mobile")
        st.markdown('<div class="section-head">All Games</div>', unsafe_allow_html=True)
        render_table(df=df, df_dates=df_dates, date_options=date_options, selected_day=selected)
    with right:
        render_recommendations_module(df, slate_day=selected, wrapper_class="recs-desktop")

    # Hidden machine-readable metadata so the bot can align tweet text
    # with the exact slate rendered on the deployed dashboard.
    try:
        slate = selected if (selected in (date_options or [])) else (date_options[0] if date_options else None)
        if slate:
            df_slate = df[df["Local date"].astype(str) == slate].copy()
        else:
            df_slate = df.copy()

        counts: dict[str, int] = {}
        if not df_slate.empty and "Region" in df_slate.columns:
            vc = df_slate["Region"].astype(str).value_counts()
            counts = {str(k): int(v) for k, v in vc.items()}

        slate_date = None
        if not df_slate.empty and "Local date" in df_slate.columns:
            try:
                slate_date = df_slate["Local date"].dropna().iloc[0]
            except Exception:
                slate_date = None

        tweet_date = None
        if isinstance(slate_date, dt.date):
            tweet_date = slate_date.strftime("%b %d").replace(" 0", " ")
        elif isinstance(slate, str) and len(slate) == 10:
            try:
                y, m, d = (int(x) for x in slate.split("-"))
                tweet_date = dt.date(y, m, d).strftime("%b %d").replace(" 0", " ")
            except Exception:
                tweet_date = None

        matchups: dict[str, list[str]] = {}
        if not df_slate.empty and {"Region", "Matchup"}.issubset(df_slate.columns):
            for region, grp in df_slate.groupby(df_slate["Region"].astype(str)):
                matchups[str(region)] = [str(x) for x in grp["Matchup"].astype(str).tolist()]

        meta = {
            "slate_day": slate,
            "tweet_date": tweet_date,
            "n_games": int(len(df_slate)) if df_slate is not None else 0,
            "counts": counts,
            "matchups": matchups,
        }
        meta_attr = py_html.escape(json.dumps(meta, ensure_ascii=False))
        st.markdown(f"<div id='tweet-meta' data-meta='{meta_attr}' style='display:none;'></div>", unsafe_allow_html=True)
    except Exception:
        pass


def render_chart_page() -> None:
    inject_base_css()
    df, _, date_options, date_to_label = build_dashboard_frames()
    if df.empty:
        st.warning("No NBA regular season or playoff games found. Enjoy the break!")
        return
    selected = st.query_params.get("day")
    render_chart(
        df=df,
        date_options=date_options,
        date_to_label=date_to_label,
        show_day_selector=False,
        selected_date=selected,
        default_day=None,
    )


def render_table_page() -> None:
    inject_base_css()
    df, df_dates, date_options, _ = build_dashboard_frames()
    if df.empty:
        st.warning("No NBA regular season or playoff games found. Enjoy the break!")
        return
    render_table(df=df, df_dates=df_dates, date_options=date_options, selected_day=None)
