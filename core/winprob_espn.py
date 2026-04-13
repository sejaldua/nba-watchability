"""Fetch ESPN win probabilities for NBA games.

Uses the game summary endpoint's ``predictor`` block to extract
pregame / live win‑probability percentages for each team.

Also supports the Core API ``probabilities`` endpoint for granular
play‑by‑play win‑probability timelines.
"""
from __future__ import annotations

from typing import Any

from core.http_cache import get_json_cached


ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
)

ESPN_PROBABILITIES_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
    "/events/{event_id}/competitions/{event_id}/probabilities"
)


def extract_predictor_win_prob(summary_data: dict[str, Any]) -> float | None:
    """Return the **home** team win probability (0-100) from a summary response.

    ESPN embeds a ``predictor`` block in the game summary payload:
        ``summary_data["predictor"]["homeTeam"]["gameProjection"]``

    Returns ``None`` when the predictor block is missing or unparseable.
    """
    try:
        predictor = summary_data.get("predictor")
        if not isinstance(predictor, dict):
            return None
        home = predictor.get("homeTeam")
        if not isinstance(home, dict):
            return None
        proj = home.get("gameProjection")
        if proj is None:
            return None
        return float(proj)
    except Exception:
        return None


def fetch_win_prob_for_game(
    game_id: str,
    *,
    ttl_seconds: int = 5 * 60,
) -> float | None:
    """Fetch the home‑team win probability for a single ESPN game ID.

    Returns a percentage (0-100) or ``None`` on failure.
    """
    try:
        resp = get_json_cached(
            ESPN_SUMMARY_URL,
            params={"event": str(game_id)},
            namespace="espn",
            cache_key=f"summary:{game_id}",
            ttl_seconds=ttl_seconds,
            timeout_seconds=12,
        )
        return extract_predictor_win_prob(resp.data)
    except Exception:
        return None
