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
CHART_IMG = OUT_DIR / "chart.png"
TABLE_IMG = OUT_DIR / "table.png"
TWEET_META = OUT_DIR / "tweet_meta.json"

# --- Chart crop (historical defaults) ---
CHART_LEFT_PAD = int(os.getenv("CHART_LEFT_PAD", "125"))
CHART_TOP_PAD = int(os.getenv("CHART_TOP_PAD", "580"))
CHART_RIGHT_PAD = int(os.getenv("CHART_RIGHT_PAD", "1900"))
CHART_BOTTOM_PAD = int(os.getenv("CHART_BOTTOM_PAD", "350"))

# --- Table crop (tweak as needed) ---
TABLE_LEFT_PAD = int(os.getenv("TABLE_LEFT_PAD", "2000"))
TABLE_TOP_PAD = int(os.getenv("TABLE_TOP_PAD", "580"))
TABLE_RIGHT_PAD = int(os.getenv("TABLE_RIGHT_PAD", "125"))
TABLE_BOTTOM_PAD = int(os.getenv("TABLE_BOTTOM_PAD", "350"))


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    local_proc = None
    if USE_LOCAL:
        local_proc = _start_local_streamlit()
        target_url = f"http://localhost:{LOCAL_PORT}"
    else:
        target_url = DASHBOARD_URL

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1600, "height": 1000},
                device_scale_factor=2.5,  # sharp
            )
            page = context.new_page()

            page.goto(target_url, timeout=120_000)
            page.wait_for_load_state("networkidle", timeout=120_000)
            time.sleep(6)  # allow Altair + logos to render

            # Capture hidden tweet metadata (if available)
            try:
                page.wait_for_selector("#tweet-meta", timeout=30_000)
                meta_attr = page.locator("#tweet-meta").get_attribute("data-meta")
                if meta_attr:
                    meta = json.loads(html_lib.unescape(meta_attr))
                    TWEET_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"Saved tweet metadata to {TWEET_META}")
            except Exception:
                pass

            page.screenshot(path=str(FULL_IMG), full_page=False)
            browser.close()
    finally:
        if local_proc is not None:
            local_proc.terminate()
            local_proc.wait(timeout=10)
            print("Local Streamlit server stopped.")

    img = Image.open(FULL_IMG)
    width, height = img.size

    # --- Crop left side (chart area) ---
    chart_box = (
        CHART_LEFT_PAD,
        CHART_TOP_PAD,
        width - CHART_RIGHT_PAD,
        height - CHART_BOTTOM_PAD,
    )
    img.crop(chart_box).save(CHART_IMG)

    # --- Crop right side (table area) ---
    table_box = (
        TABLE_LEFT_PAD,
        TABLE_TOP_PAD,
        width - TABLE_RIGHT_PAD,
        height - TABLE_BOTTOM_PAD,
    )
    img.crop(table_box).save(TABLE_IMG)

    print(f"Saved chart screenshot to {CHART_IMG}")
    print(f"Saved table screenshot to {TABLE_IMG}")
    return [CHART_IMG, TABLE_IMG]
