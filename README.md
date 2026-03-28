# NBA Watchability

**Which NBA games are actually worth watching tonight?**

NBA Watchability ranks every game on the schedule using a **Watchability Index (WI)** that combines team quality and expected game closeness into a single 0-100 score. No more guessing which game to put on — just check the dashboard.

**[Live Dashboard](https://nba-watchability.streamlit.app/)**

---

## How It Works

The Watchability Index is built on two signals:

| Signal | Source | What it measures |
|--------|--------|-----------------|
| **Team Quality** | Win percentages from ESPN | Are these good teams? |
| **Closeness** | Point spreads from [The Odds API](https://the-odds-api.com) | Is the game expected to be competitive? |

These are combined using a **CES (Constant Elasticity of Substitution) utility function** — a formula from economics that ensures both inputs matter. A game between two great teams that's expected to be a blowout still scores lower than a tight matchup between good teams.

### WI Labels

| Score | Label | Meaning |
|-------|-------|---------|
| 90-100 | **Must Watch** | Elite matchup, don't miss it |
| 75-89 | **Strong Watch** | Great game, worth your time |
| 50-74 | **Watchable** | Solid, tune in if you're free |
| 25-49 | **Skippable** | Probably not worth it |
| 0-24 | **Hard Skip** | Find something else to do |

### Adjustments

- **Player injuries** reduce a team's effective quality based on injury severity and player impact share (PPG + APG + RPG relative to team total)
- **Star player boost** gives a small win% bump for teams with dominant statistical performers
- **7-day forecast** predicts spreads for upcoming games using recent team spread history and a home-court adjustment

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/YOUR_USER/nba-watchability.git
cd nba-watchability
pip install -r requirements.txt

# Set your Odds API key (free tier: https://the-odds-api.com)
export ODDS_API_KEY=your_key_here

# Run the dashboard
streamlit run app/streamlit_app.py
```

### Other Commands

```bash
# Build 7-day forecast (saves to data/forecast/latest.{csv,json,parquet})
python scripts/build_forecast_7d.py

# Capture dashboard screenshots (requires Playwright)
pip install playwright && playwright install chromium
python scripts/capture_dashboard.py

# Generate summary text for the current slate
python -c "from scripts.compose_tweet import compose_tweet_text; print(compose_tweet_text())"
```

---

## Project Structure

```
app/
  streamlit_app.py          # Dashboard entry point
  dashboard_views.py        # All rendering logic (chart + table views)
  pages/                    # Streamlit multi-page views

core/
  watchability.py           # WI formula (CES utility function)
  watchability_v2_params.py # Algorithm parameters (sigma, injury weights, thresholds)
  build_watchability_df.py  # Assembles live watchability DataFrame
  build_watchability_forecast_df.py  # Assembles 7-day forecast DataFrame
  odds_api.py               # Fetches point spreads from The Odds API
  standings_espn.py          # Fetches team records from ESPN
  schedule_espn.py           # Fetches game schedule from ESPN
  health_espn.py             # Fetches injury reports + computes health impact
  importance.py              # Playoff seeding importance scoring
  forecast_features.py       # Feature extraction for spread prediction
  forecast_spread.py         # Spread prediction model
  http_cache.py              # TTL-based HTTP response caching
  team_meta.py               # Team abbreviations, mascots, logo URLs

scripts/
  build_forecast_7d.py       # Daily forecast generation
  capture_dashboard.py       # Playwright-based dashboard screenshots
  compose_tweet.py           # Generates summary text from current slate
  log_daily_scores.py        # Logs daily WI scores
  log_previous_day_results.py # Logs game results
  log_espn_injury_report.py  # Logs ESPN injury data

config/
  forecast.yml               # Forecast model parameters

data/
  forecast/                  # Forecast artifacts (CSV, JSON, Parquet)
```

---

## Key Definitions

| Term | Definition |
|------|-----------|
| **WI (Watchability Index)** | 0-100 score combining team quality and game closeness via CES utility |
| **Team Quality** | Normalized average win% of both teams, scaled to [0, 1] where 0.200 maps to 0 and 0.700 maps to 1 |
| **Closeness** | How tight the spread is — a pick'em (0 spread) scores ~1.0, a 15+ point spread bottoms out near 0 |
| **CES Utility** | `(0.7 * quality^ρ + 0.3 * closeness^ρ)^(1/ρ)` where `ρ = (σ-1)/σ` and `σ = 0.4` — ensures both inputs are needed for a high score |
| **Spread** | Median consensus point spread across all available sportsbooks |
| **Health Score** | `1 - 0.6 * Σ(injury_weight × impact_share)` — reduces team quality when key players are injured |
| **Impact Share** | A player's (PPG + RPG + APG) as a fraction of their team's total |
| **Injury Weights** | Available: 0.0, Probable: 0.1, Questionable: 0.4, Doubtful: 0.7, Out: 1.0 |
| **Star Boost** | Small additive win% increase for teams with high-usage stars, based on per-game stats |
| **Slate** | The set of games on a given date (in Pacific Time) |

---

## Data Sources

All data sources are free/public:

- **[ESPN API](https://site.api.espn.com)** — Standings, schedule, scores, injury reports
- **[The Odds API](https://the-odds-api.com)** — Betting spreads from multiple sportsbooks (free tier: 500 requests/month)
- **[NBA.com](https://www.nba.com)** — Schedule data via `nba_api`

---

## Automation

GitHub Actions workflows handle scheduled tasks:

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| `build_forecast_7d.yml` | Daily 09:00 UTC | Builds 7-day forecast, commits artifact |
| `log_injury_report.yml` | 4x daily | Logs ESPN injury data |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ODDS_API_KEY` | Yes | API key from [The Odds API](https://the-odds-api.com) |
