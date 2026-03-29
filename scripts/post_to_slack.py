"""Capture the NBA Watchability dashboard and post to Slack.

Usage:
    python scripts/post_to_slack.py

Environment variables:
    SLACK_BOT_TOKEN   — Bot token (xoxb-...) with chat:write and files:write scopes
    SLACK_CHANNEL_ID  — Channel ID to post to (e.g. C0123456789)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from scripts.capture_dashboard import capture_dashboard
from scripts.compose_message import compose_message_text


SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")

HEADERS = {
    "Authorization": f"Bearer {SLACK_TOKEN}",
    "Content-Type": "application/json",
}


def send_message(text, channel=CHANNEL_ID):
    """Send a plain-text message to Slack."""
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers=HEADERS,
        data=json.dumps({"channel": channel, "text": text}),
    )
    result = resp.json()
    if not result.get("ok"):
        print(f"[WARN] Slack send failed: {result.get('error')}")
    return result


def upload_image(filepath, title="", initial_comment="", channel=CHANNEL_ID):
    """Upload an image file to a Slack channel using files.uploadV2 flow."""
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)

    # Step 1: get an upload URL
    resp1 = requests.get(
        "https://slack.com/api/files.getUploadURLExternal",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
        params={"filename": filename, "length": file_size},
    )
    data1 = resp1.json()
    if not data1.get("ok"):
        print(f"[WARN] files.getUploadURLExternal failed: {data1.get('error')}")
        return data1

    upload_url = data1["upload_url"]
    file_id = data1["file_id"]

    # Step 2: upload the file bytes to the URL
    with open(filepath, "rb") as f:
        resp2 = requests.post(upload_url, files={"file": (filename, f)})
    if resp2.status_code != 200:
        print(f"[WARN] File upload POST failed: {resp2.status_code}")
        return {"ok": False, "error": f"upload status {resp2.status_code}"}

    # Step 3: complete the upload and share to channel
    resp3 = requests.post(
        "https://slack.com/api/files.completeUploadExternal",
        headers={
            "Authorization": f"Bearer {SLACK_TOKEN}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "files": [{"id": file_id, "title": title or filename}],
            "channel_id": channel,
            "initial_comment": initial_comment,
        }),
    )
    data3 = resp3.json()
    if not data3.get("ok"):
        print(f"[WARN] files.completeUploadExternal failed: {data3.get('error')}")
    return data3


def post_to_slack():
    if not SLACK_TOKEN:
        print("Error: SLACK_BOT_TOKEN not set")
        sys.exit(1)
    if not CHANNEL_ID:
        print("Error: SLACK_CHANNEL_ID not set")
        sys.exit(1)

    # Step 1: Capture dashboard screenshots + metadata
    print("Capturing dashboard...")
    capture_dashboard()

    # Step 2: Compose message text (reuses the message composer)
    text = compose_message_text()
    if text is None:
        print("No games on today's slate — skipping Slack post.")
        return

    print(f"Message text:\n{text}\n")

    # Step 3: Upload screenshot with message to Slack
    full_img = Path("output") / "full.png"
    if not full_img.exists():
        print("No screenshot found — sending text only.")
        send_message(text)
        return

    result = upload_image(
        filepath=str(full_img),
        title="NBA Watchability Dashboard",
        initial_comment=text,
    )
    if result.get("ok"):
        print("Posted to Slack successfully.")
    else:
        print(f"Slack post failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    post_to_slack()
