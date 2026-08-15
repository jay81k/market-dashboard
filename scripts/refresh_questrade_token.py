#!/usr/bin/env python3
"""
Refreshes the Questrade access/refresh token and writes the result into the
same Cloudflare KV key (`qt_token`) the Worker already reads from. This
script is meant to be the primary writer of that key going forward, running
on a fixed schedule, one instance at a time — unlike concurrent Worker
requests, a scheduled job can't race itself.

The Worker's own refresh logic (with its KV lock) stays in place as a
fallback, untouched. This is additive, not a replacement — if this script
keeps the token proactively fresh, the Worker's own refresh path should
rarely need to fire at all, which is what actually reduces exposure to the
race, not a claim that the race is now impossible.

Required environment variables:
  CLOUDFLARE_API_TOKEN               Cloudflare API token with
                                      "Workers KV Storage Write" permission.
  QUESTRADE_REFRESH_TOKEN_BOOTSTRAP  Only used the very first time this
                                      script ever runs, when KV has nothing
                                      stored yet. After that, KV's own
                                      stored refresh_token takes over — same
                                      bootstrap-then-self-sustaining pattern
                                      the Worker itself already uses for its
                                      QUESTRADE_REFRESH_TOKEN secret.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

CF_ACCOUNT_ID = "09dacb4ab7050bedff69c9434e206816"
CF_KV_NAMESPACE_ID = "ca292c3fdaf847a68b03424e4ea8e0cc"
CF_KV_KEY = "qt_token"

CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
BOOTSTRAP_REFRESH_TOKEN = os.environ.get("QUESTRADE_REFRESH_TOKEN_BOOTSTRAP", "")

KV_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
    f"/storage/kv/namespaces/{CF_KV_NAMESPACE_ID}/values/{CF_KV_KEY}"
)


def cf_headers():
    return {"Authorization": f"Bearer {CF_API_TOKEN}"}


def read_current_token():
    """Returns the currently stored token dict, or None if the key doesn't
    exist yet (first-ever run) or can't be parsed."""
    req = urllib.request.Request(KV_URL, headers=cf_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        body = e.read().decode("utf-8", errors="replace")
        print(f"Cloudflare KV read failed: HTTP {e.code} {body}", file=sys.stderr)
        raise
    except (json.JSONDecodeError, ValueError):
        return None


def write_new_token(token_state):
    # No expiration/expiration_ttl set — this key should persist
    # indefinitely, same as how the Worker's own code writes it. KV
    # enforces a 60s minimum on expiration_ttl if you do set one (a real
    # bug we hit once already on the Worker side) — simplest to just not
    # set one here at all.
    body = json.dumps(token_state).encode("utf-8")
    req = urllib.request.Request(KV_URL, headers=cf_headers(), data=body, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"Cloudflare KV write failed: HTTP {e.code} {err_body}", file=sys.stderr)
        raise


def refresh_questrade(refresh_token):
    # GET, not POST — Questrade's own docs disagree with themselves on this,
    # but GET is what actually worked reliably in testing.
    url = (
        "https://login.questrade.com/oauth2/token"
        f"?grant_type=refresh_token&refresh_token={refresh_token}"
    )
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Questrade refresh failed: {e.code} {body}") from e

    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "api_server": data["api_server"],
        "expires_at": int(time.time() * 1000) + data["expires_in"] * 1000,
    }


def main():
    current = read_current_token()
    refresh_token = (current or {}).get("refresh_token") or BOOTSTRAP_REFRESH_TOKEN

    if not refresh_token:
        print(
            "No refresh token available — KV has nothing stored and "
            "QUESTRADE_REFRESH_TOKEN_BOOTSTRAP isn't set.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        new_state = refresh_questrade(refresh_token)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    write_new_token(new_state)
    print(f"Token refreshed OK. api_server={new_state['api_server']}, "
          f"expires_at={new_state['expires_at']}")


if __name__ == "__main__":
    main()
