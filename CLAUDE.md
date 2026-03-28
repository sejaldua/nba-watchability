
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Watchability is a Python Streamlit app that ranks NBA games by a **Watchability Index (WI)** combining team quality and game closeness. It pulls live data from ESPN, The Odds API, and NBA.com, and forecasts spreads 7 days ahead.

**Live app:** https://nba-watchability.streamlit.app/

## Commands

```bash
# Run the Streamlit dashboard locally
streamlit run app/streamlit_app.py

# Build the 7-day forecast artifact (saves to data/forecast/latest.{csv,json,parquet})
python scripts/build_forecast_7d.py

# Capture dashboard screenshots (requires Playwright)
python scripts/capture_dashboard.py

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (needed for dashboard screenshots)
playwright install chromium
```

No test suite, linter, or formatter is configured.

## Architecture

### Data Pipeline

```
ESPN / Odds API / NBA.com  →  core/ data fetchers  →  WI computation  →  Streamlit dashboard
                                                                       →  Forecast artifacts
                                                                       →  Twitter bot
```

### Core Modules (`core/`)

- **Data fetching:** `standings_espn.py`, `schedule_espn.py`, `odds_api.py`, `health_espn.py`, `results_espn.py` — each wraps an external API with caching via `http_cache.py`
- **WI computation:** `watchability.py` (CES utility function), `importance.py` (playoff seeding weight), `health_espn.py` (injury impact)
- **DataFrame builders:** `build_watchability_df.py` (live/today), `build_watchability_forecast_df.py` (7-day ahead)
- **Forecasting:** `forecast_features.py` (feature extraction), `forecast_spread.py` (spread prediction using learned params from `config/forecast.yml`)
- **Team metadata:** `team_meta.py` (abbreviations, mascots, logos for all 30 teams)
- **Algorithm params:** `watchability_v2_params.py` (CES sigma, injury weights, thresholds)

### Frontend (`app/`)

- `streamlit_app.py` — entry point, renders via `dashboard_views.py`
- `dashboard_views.py` — all dashboard rendering logic (chart + table views)
- `app/pages/chart.py`, `app/pages/table.py` — Streamlit multi-page wrappers

### Scripts (`scripts/`)

- `capture_dashboard.py` — Playwright screenshots of the running Streamlit app
- `compose_tweet.py` — slate summary text generation
- `build_forecast_7d.py` — daily forecast generation
- `log_*.py` — logging scripts for scores, results, injury reports

### Automation (`.github/workflows/`)

- `build_forecast_7d.yml` — Daily 09:00 UTC: build forecast, commit artifact
- `log_injury_report.yml` — 4x daily: log ESPN injury data

## Key Concepts

**Watchability Index (WI):** CES utility combining team quality (normalized win%) and closeness (based on spread). Ranges 0-100 with labels: Must Watch (90+), Strong Watch (75-89), Watchable (50-74), Skippable (25-49), Hard Skip (0-24).

**Caching:** HTTP responses cached in `.cache/http/{namespace}/` with TTL per endpoint. Streamlit uses 5-min in-memory cache for the live dataframe.

## Environment Variables

Stored in `.env` (gitignored):
- `ODDS_API_KEY` — The Odds API key
<!-- Uncomment when Slack integration is added:
- `SLACK_WEBHOOK_URL` or similar Slack credentials
-->
<!-- Previously used for Twitter/X (removed):
- `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_SECRET`
-->
