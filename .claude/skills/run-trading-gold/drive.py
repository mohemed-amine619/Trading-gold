"""Drive the running dashboard in a real browser (system Chrome via Playwright).

Subcommands (run from the repo root, with serve_dashboard.py already up):
    shot  [PNG]   screenshot of the dashboard Overview   (default: scratchpad-free path below)
    estop         click EMERGENCY STOP, accept the confirm, report the alert text
    writes        check the write gate over HTTP: 503 when unset, 401 without token,
                  200 with the right token. Needs DASHBOARD_TOKEN in this environment.

Playwright is not in .venv. Install it once into a target dir and point PYTHONPATH at it:
    python3 -m pip install --target ~/.cache/trading-gold-browser playwright
    PYTHONPATH=~/.cache/trading-gold-browser python3 .claude/skills/run-trading-gold/drive.py shot
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:%s" % os.environ.get("DASHBOARD_PORT", "8765"))
CHROME = os.environ.get("CHROME_PATH", "/usr/bin/google-chrome")
SHOT_DIR = os.environ.get("SHOT_DIR", "/tmp")


def _browser(p):
    return p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])


def shot(out):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = _browser(p)
        page = b.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE + "/", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=out, full_page=False)
        b.close()
    print("screenshot:", out)


def estop():
    from playwright.sync_api import sync_playwright
    messages = []
    with sync_playwright() as p:
        b = _browser(p)
        page = b.new_page()

        def on_dialog(d):
            messages.append(d.message)
            # The page asks for DASHBOARD_TOKEN on a 401; answer it so the
            # success path is exercised, not just the refusal.
            d.accept(os.environ.get("DASHBOARD_TOKEN", "")) if d.type == "prompt" else d.accept()

        page.on("dialog", on_dialog)
        page.goto(BASE + "/", wait_until="networkidle")
        page.get_by_role("button", name="EMERGENCY STOP").first.click()
        page.wait_for_timeout(2000)
        b.close()
    # Dialogs in order: confirm(), then the token prompt if the page needed one,
    # then the result alert().
    print("dialogs:", json.dumps(messages, ensure_ascii=False))


def _req(method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("X-Dashboard-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def writes():
    token = os.environ.get("DASHBOARD_TOKEN", "")
    if not token:
        sys.exit("set DASHBOARD_TOKEN to the value serve_dashboard.py was started with")
    body = {"max_open_pos": 3}
    print("GET  /api/config/schema      ->", _req("GET", "/api/config/schema"))
    print("POST /api/config (no token)  ->", _req("POST", "/api/config", body))
    print("POST /api/config (wrong tok) ->", _req("POST", "/api/config", body, token="nope"))
    print("POST /api/config (good tok)  ->", _req("POST", "/api/config", body, token=token))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "shot"
    if cmd == "shot":
        shot(sys.argv[2] if len(sys.argv) > 2 else os.path.join(SHOT_DIR, "dashboard.png"))
    elif cmd == "estop":
        estop()
    elif cmd == "writes":
        writes()
    else:
        sys.exit("unknown command: " + cmd)
