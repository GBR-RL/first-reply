"""Record docs/assets/demo.gif from the running Compose stack with a headless browser.

A customer email goes in, the reviewer opens the review page, approves the draft, and the reply
arrives in the customer's mailbox. Run after `docker compose up` (the Stack workflow does).
"""

from __future__ import annotations

import argparse
import html
import io
from pathlib import Path
from typing import Any

import httpx

from first_reply.service.smoke import SERVICE, deliver, inbox, wait_for

WIDTH, HEIGHT = 1280, 860
CAPTION_H = 56
CAPTIONS = (
    "1  A customer email arrives: n8n posts it to the triage service",
    "2  Routing, a cited draft and the evidence it rests on, held for review",
    "3  The reviewer approves; n8n resumes and sends the reply",
    "4  The customer's mailbox: the reply the reviewer approved",
)


def _caption(png: bytes, text: str) -> object:
    from PIL import Image, ImageDraw, ImageFont

    shot = Image.open(io.BytesIO(png)).convert("RGB")
    frame = Image.new("RGB", (shot.width, shot.height + CAPTION_H), "#1d2330")
    frame.paste(shot, (0, CAPTION_H))
    draw = ImageDraw.Draw(frame)
    font: Any
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text((24, 14), text, fill="#ffffff", font=font)
    return frame


def record(out: Path) -> Path:
    from playwright.sync_api import sync_playwright

    http = httpx.Client(timeout=60)
    # Its own subject: the smoke test sends the same question first, and its reply must not be
    # mistaken for the one approved here.
    subject = "UDX compile error: argument list too long"
    deliver(subject)
    case = wait_for(
        "a pending case",
        lambda: [c for c in http.get(f"{SERVICE}/cases?status=pending").json()
                 if c["subject"] == subject],
        1500,
    )[0]  # fmt: skip
    frames: list[tuple[object, int]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(f"{SERVICE}/review")
        page.wait_for_selector(f"#c-{case['id']}")
        card = page.locator(f"#c-{case['id']}")
        card.scroll_into_view_if_needed()
        frames.append((_caption(page.screenshot(), CAPTIONS[0]), 3500))
        page.mouse.wheel(0, 380)
        frames.append((_caption(page.screenshot(), CAPTIONS[1]), 4500))
        page.click(f"#c-{case['id']} button.primary")
        page.wait_for_timeout(1500)
        frames.append((_caption(page.screenshot(), CAPTIONS[2]), 2500))
        msg = wait_for(
            "the reply in the customer's mailbox",
            lambda: [m for m in inbox("customer", "customer") if m["Subject"] == f"Re: {subject}"],
            300,
        )[0]
        body = msg.get_payload(decode=True)
        text = body.decode(errors="replace") if isinstance(body, bytes) else str(msg.get_payload())
        page.set_content(
            "<html><body style='margin:0;font:16px/1.55 system-ui,sans-serif;background:#f6f7f9'>"
            "<div style='max-width:860px;margin:40px auto;background:#fff;border:1px solid #dfe3ea;"
            "border-radius:10px;padding:28px'>"
            f"<div style='color:#5d6675'>From: {html.escape(str(msg['From']))}<br>"
            f"To: {html.escape(str(msg['To']))}</div>"
            f"<h2 style='margin:12px 0'>{html.escape(str(msg['Subject']))}</h2>"
            f"<div style='white-space:pre-wrap'>{html.escape(text)}</div></div></body></html>"
        )
        frames.append((_caption(page.screenshot(), CAPTIONS[3]), 4500))
        browser.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = (f for f, _ in frames)
    first.save(  # type: ignore[attr-defined]
        out, save_all=True, append_images=rest, duration=[d for _, d in frames], loop=0,
        optimize=True,
    )  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/assets/demo.gif"))
    print(f"wrote {record(parser.parse_args().out)}")


if __name__ == "__main__":
    main()
