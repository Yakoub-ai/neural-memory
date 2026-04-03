"""Tests for neural_memory.redactor — secret detection and redaction."""

import pytest

from neural_memory.config import RedactionConfig
from neural_memory.redactor import Redactor


def _redactor(**kwargs) -> Redactor:
    config = RedactionConfig(**kwargs)
    return Redactor(config)


# ── Disabled redaction ─────────────────────────────────────────────────────────

def test_disabled_passes_through():
    r = _redactor(enabled=False)
    text = 'api_key = "sk-supersecret12345678"'
    result = r.redact(text)
    assert result.text == text
    assert result.had_redactions is False
    assert result.redaction_count == 0


# ── Builtin patterns ───────────────────────────────────────────────────────────

def test_detects_api_key():
    r = _redactor()
    result = r.redact('api_key = "sk-abc123456789abcdef"')
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_aws_access_key():
    r = _redactor()
    result = r.redact("AKIAIOSFODNN7EXAMPLE123456")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_jwt_token():
    r = _redactor()
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = r.redact(jwt)
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_postgres_connection_string():
    r = _redactor()
    result = r.redact("postgres://user:secretpass@localhost:5432/mydb")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_mongodb_connection_string():
    r = _redactor()
    result = r.redact("mongodb://admin:password123@cluster0.example.mongodb.net/mydb")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_private_key_header():
    r = _redactor()
    result = r.redact("-----BEGIN RSA PRIVATE KEY-----")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_bearer_token():
    r = _redactor()
    result = r.redact("Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.abcdefghijklmnop.signature")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_detects_internal_ip():
    r = _redactor()
    result = r.redact("host = '192.168.1.100'")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


# ── Custom patterns ────────────────────────────────────────────────────────────

def test_custom_pattern():
    r = _redactor(custom_patterns=[r"MYCOMPANY_SECRET_\w+"])
    result = r.redact("key = MYCOMPANY_SECRET_ABCDEF")
    assert result.had_redactions is True
    assert "REDACTED" in result.text


# ── Sensitive variable detection ───────────────────────────────────────────────

def test_sensitive_variable_password():
    r = _redactor()
    # Layer 1 builtin regex catches password assignments (value >= 8 chars)
    result = r.redact('password = "mysecretpassword"')
    assert result.had_redactions is True
    assert "REDACTED" in result.text


def test_sensitive_variable_token():
    r = _redactor()
    # Layer 1 builtin regex catches token assignments (value >= 8 chars)
    result = r.redact('token = "my_access_token_value"')
    assert result.had_redactions is True
    assert "REDACTED" in result.text


# ── Whitelist ──────────────────────────────────────────────────────────────────

def test_whitelist_marks_variable():
    # Short value ("abc") avoids Layer 1 (requires 8+ chars); Layer 2 catches it.
    r = _redactor(whitelist_vars=["token"], builtin_patterns=False)
    result = r.redact('token = "abc"')
    assert "[WHITELISTED]" in result.text


def test_default_sensitive_var_pattern_fires():
    # Isolate Layer 2 by disabling Layer 1; verify default patterns work after fix.
    r = _redactor(builtin_patterns=False)
    result = r.redact('mytoken = "abc"')
    assert "[REDACTED]" in result.text


def test_invalid_custom_pattern_is_skipped():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = _redactor(custom_patterns=["[invalid"])
    assert len(w) == 1
    assert "Invalid redaction pattern" in str(w[0].message)
    # Redactor still works on valid text
    result = r.redact("normal text")
    assert result.had_redactions is False


# ── redact_node_content ────────────────────────────────────────────────────────

def test_redact_node_content_both_fields():
    r = _redactor()
    code = 'password = "secretpassword123"'
    summary = "Connects to postgres://user:pass@db/prod"
    redacted_code, redacted_summary, had = r.redact_node_content(code, summary)
    assert had is True
    assert "[REDACTED]" in redacted_code or "REDACTED" in redacted_code
    assert "REDACTED" in redacted_summary


def test_redact_node_content_no_secrets():
    r = _redactor()
    code = "def add(x, y):\n    return x + y"
    summary = "Adds two numbers"
    redacted_code, redacted_summary, had = r.redact_node_content(code, summary)
    assert had is False
    assert redacted_code == code
    assert redacted_summary == summary


def test_clean_text_no_redactions():
    r = _redactor()
    result = r.redact("def add(x, y):\n    return x + y")
    assert result.had_redactions is False
    assert result.redaction_count == 0
