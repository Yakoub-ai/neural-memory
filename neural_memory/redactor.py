"""Redaction engine for sensitive content in code."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import RedactionConfig

# Built-in regex patterns for common secrets
BUILTIN_PATTERNS = [
    # API keys / tokens (generic long hex/base64 strings in assignments)
    (r"""(?i)(?:api[_-]?key|token|secret|password|passwd|pwd|auth|credential|private[_-]?key|access[_-]?key|client[_-]?secret)\s*[:=]\s*['\"]([^'\"]{8,})['\"]""", "REDACTED_SECRET"),
    # AWS access keys
    (r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}", "REDACTED_AWS_KEY"),
    # JWT tokens
    (r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "REDACTED_JWT"),
    # Connection strings
    (r"""(?i)(?:mongodb|postgres|mysql|redis|amqp|mssql)://[^\s'\"]+""", "REDACTED_CONN_STRING"),
    # Private keys
    (r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----", "REDACTED_PRIVATE_KEY"),
    # Generic env-style secrets
    (r"""(?i)(?:DATABASE_URL|SECRET_KEY|ENCRYPTION_KEY|SIGNING_KEY)\s*=\s*['\"]([^'\"]+)['\"]""", "REDACTED_ENV_SECRET"),
    # Bearer tokens in strings
    (r"""Bearer\s+[A-Za-z0-9_\-\.]{20,}""", "REDACTED_BEARER_TOKEN"),
    # IP addresses (internal ranges)
    (r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b", "REDACTED_INTERNAL_IP"),
]


@dataclass
class RedactionResult:
    """Result of redacting content."""
    text: str
    had_redactions: bool
    redaction_count: int


class Redactor:
    """Multi-layer redaction engine."""

    def __init__(self, config: RedactionConfig):
        self.config = config
        self._compiled_patterns: list[tuple[re.Pattern, str]] = []
        self._sensitive_var_re: list[re.Pattern] = []
        self._build_patterns()

    def _build_patterns(self) -> None:
        """Compile all redaction patterns."""
        if self.config.builtin_patterns:
            for pattern, replacement in BUILTIN_PATTERNS:
                self._compiled_patterns.append(
                    (re.compile(pattern), replacement)
                )

        for pattern in self.config.custom_patterns:
            self._compiled_patterns.append(
                (re.compile(pattern), "REDACTED_CUSTOM")
            )

        for pattern in self.config.sensitive_var_patterns:
            self._sensitive_var_re.append(re.compile(pattern))

    def redact(self, text: str) -> RedactionResult:
        """Apply all redaction layers to text."""
        if not self.config.enabled:
            return RedactionResult(text=text, had_redactions=False, redaction_count=0)

        count = 0
        result = text

        # Layer 1: Regex patterns
        for pattern, replacement in self._compiled_patterns:
            new_result, n = pattern.subn(f"[{replacement}]", result)
            count += n
            result = new_result

        # Layer 2: AST-aware variable value redaction
        # Redact values assigned to sensitive-named variables
        for var_re in self._sensitive_var_re:
            # Match: var_name = "value" or var_name = 'value'
            # Strip inline flags from the source pattern and apply as compile flags
            raw_pattern = var_re.pattern.replace("(?i)", "")
            assign_pattern = re.compile(
                rf"({raw_pattern})\s*[:=]\s*(['\"])(.+?)\2",
                re.IGNORECASE
            )
            def _mask_value(m):
                nonlocal count
                count += 1
                return f'{m.group(1)} = "[REDACTED]"'
            result = assign_pattern.sub(_mask_value, result)

        # Layer 3: Check whitelist — un-redact known safe patterns
        for safe_var in self.config.whitelist_vars:
            # Restore whitelisted variable assignments
            result = result.replace(
                f'{safe_var} = "[REDACTED]"',
                # We can't restore the original, so mark as reviewed
                f'{safe_var} = "[WHITELISTED]"'
            )

        return RedactionResult(
            text=result,
            had_redactions=count > 0,
            redaction_count=count
        )

    def redact_node_content(self, raw_code: str, summary: str) -> tuple[str, str, bool]:
        """Redact both raw code and summary. Returns (redacted_code, redacted_summary, had_redactions)."""
        code_result = self.redact(raw_code)
        summary_result = self.redact(summary)
        had_any = code_result.had_redactions or summary_result.had_redactions
        return code_result.text, summary_result.text, had_any
