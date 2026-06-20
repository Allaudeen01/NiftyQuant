"""Tests for the minimal .env loader (the inline-comment bug fix)."""

import os

from nifty_quant.dotenv import load_dotenv, _clean_value


def test_clean_value_strips_inline_comment():
    assert _clean_value("ABCDEF234567  # the secret string") == "ABCDEF234567"
    assert _clean_value("ABCDEF234567\t# note") == "ABCDEF234567"


def test_clean_value_strips_quotes():
    assert _clean_value('"quoted value"') == "quoted value"
    assert _clean_value("'single'") == "single"


def test_clean_value_plain():
    assert _clean_value("  JBSWY3DPEHPK3PXP  ") == "JBSWY3DPEHPK3PXP"


def test_load_dotenv_parses_and_strips(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment line\n"
        "\n"
        "ANGEL_API_KEY=abc123\n"
        "ANGEL_TOTP_SECRET=JBSWY3DPEHPK3PXP  # the secret string, not the code\n"
        "ANGEL_CLIENT_CODE=R123456   # your login id\n",
        encoding="utf-8",
    )
    for k in ("ANGEL_API_KEY", "ANGEL_TOTP_SECRET", "ANGEL_CLIENT_CODE"):
        monkeypatch.delenv(k, raising=False)

    keys = load_dotenv(env)
    assert set(keys) == {"ANGEL_API_KEY", "ANGEL_TOTP_SECRET", "ANGEL_CLIENT_CODE"}
    assert os.environ["ANGEL_API_KEY"] == "abc123"
    # Inline comment must be stripped -> valid base32 secret remains.
    assert os.environ["ANGEL_TOTP_SECRET"] == "JBSWY3DPEHPK3PXP"
    assert os.environ["ANGEL_CLIENT_CODE"] == "R123456"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_load_dotenv_no_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "fromenv")
    load_dotenv(env, override=False)
    assert os.environ["FOO"] == "fromenv"
