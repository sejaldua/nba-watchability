from pathlib import Path
import html as html_lib
import json
import os
import subprocess
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright
import requests


# When LOCAL_STREAMLIT=1 (default), launch a local Streamlit server and
# screenshot that instead of the deployed app.  Set LOCAL_STREAMLIT=0
# to revert to the old behaviour of screenshotting the deployed URL.
USE_LOCAL = os.getenv("LOCAL_STREAMLIT", "1") == "1"

DASHBOARD_URL = os.getenv(
    "DASHBOARD_URL", "https://nba-watchability.streamlit.app/"
)

LOCAL_PORT = int(os.getenv("LOCAL_STREAMLIT_PORT", "8502"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"

OUT_DIR = Path("output")
FULL_IMG = OUT_DIR / "full.png"
TWEET_META = OUT_DIR / "tweet_meta.json"


def _start_local_streamlit():
    """Launch a local Streamlit server in the background and return the process."""
    env = {**os.environ, "BROWSER": "none"}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(STREAMLIT_APP),
            "--server.port", str(LOCAL_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for the server to become ready (up to 120s).
    url = f"http://localhost:{LOCAL_PORT}/_stcore/health"
    for _ in range(120):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"Local Streamlit server ready on port {LOCAL_PORT}")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Local Streamlit server failed to start within 120s")


def capture_dashboard():
    print("Starting screenshot capture...")
    sys.stdout.flush()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    local_proc = None
    if USE_LOCAL:
        local_proc = _start_local_streamlit()
        target_url = f"http://localhost:{LOCAL_PORT}"
    else:
        target_url = DASHBOARD_URL

    print(f"Screenshotting: {target_url}")
    sys.stdout.flush()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1600, "height": 1200},
                device_scale_factor=2.0,
            )
            page = context.new_page()

            page.goto(target_url, timeout=120_000)
            page.wait_for_load_state("networkidle", timeout=120_000)

            # Wait for game cards to render (rec-card or menu-row class).
            # Falls back to a generous sleep if no cards appear.
            try:
                page.wait_for_selector(".rec-card, .menu-row", timeout=30_000)
                print("Game cards detected, waiting for images to load...")
                sys.stdout.flush()
            except Exception:
                print("No game cards found within 30s, proceeding anyway.")
                sys.stdout.flush()

            time.sleep(8)  # allow logos and remaining async content to render

            # Capture hidden tweet metadata (if available)
            try:
                page.wait_for_selector("#tweet-meta", timeout=10_000)
                meta_attr = page.locator("#tweet-meta").get_attribute("data-meta")
                if meta_attr:
                    meta = json.loads(html_lib.unescape(meta_attr))
                    TWEET_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"Saved tweet metadata to {TWEET_META}")
            except Exception:
                pass

            page.screenshot(path=str(FULL_IMG), full_page=True)
            print(f"Saved screenshot to {FULL_IMG}")
            sys.stdout.flush()
            browser.close()
    finally:
        if local_proc is not None:
            local_proc.terminate()
            local_proc.wait(timeout=10)
            print("Local Streamlit server stopped.")
            sys.stdout.flush()

    return str(FULL_IMG)
