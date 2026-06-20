"""Diagnose the ANGEL_TOTP_SECRET without revealing it.

Reads ANGEL_TOTP_SECRET from the environment and reports whether it is a valid
base32 TOTP secret. It never prints the secret itself -- only its length, any
invalid character types, and (if valid) the current 6-digit code so you can
confirm it matches your authenticator app.

    python scripts/check_totp.py
"""

from __future__ import annotations

import base64
import os
import re

from nifty_quant.data.providers.angelone import (
    _clean_totp_secret,
    _normalize_totp_secret,
)
from nifty_quant.dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    raw = os.environ.get("ANGEL_TOTP_SECRET")
    if not raw:
        print("ANGEL_TOTP_SECRET is not set in this shell session.")
        print("Load your .env first, then re-run.")
        return

    cleaned = _clean_totp_secret(raw)
    had_spaces = (" " in raw) or (raw != raw.strip())
    invalid = sorted(set(re.findall(r"[^A-Za-z2-7=]", cleaned)))

    print(f"length (after cleaning): {len(cleaned)}")
    print(f"contained spaces/whitespace: {had_spaces}")
    print(f"invalid base32 characters present: {invalid if invalid else 'none'}")

    if invalid:
        print("\nDIAGNOSIS: the secret has characters outside base32 (A-Z, 2-7).")
        print("You likely pasted the 6-digit code, the otpauth URL, or the")
        print("placeholder. Re-copy the SECRET STRING from Angel 'Enable TOTP'.")
        return

    normalized = _normalize_totp_secret(raw)
    try:
        base64.b32decode(normalized, casefold=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\nDIAGNOSIS: not decodable as base32 even after padding ({exc}).")
        print("The secret length may be invalid; re-copy it from Angel.")
        return

    import pyotp  # type: ignore
    import time

    totp = pyotp.TOTP(normalized)
    remaining = 30 - int(time.time()) % 30
    print(f"\nVALID base32 secret. Current code: {totp.now()} "
          f"(valid ~{remaining}s)")
    print("If this 6-digit code matches your authenticator app, you're set.")


if __name__ == "__main__":
    main()
