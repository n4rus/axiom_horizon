"""Security Hardening and Input Validation for Kai AGI System.

Production-quality security module with input validation,
sanitization, and protection against common attacks.
"""

from __future__ import annotations

import re
import html
import time
import logging
import hashlib
import secrets
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any = None
    error_message: str = ""


class InputValidator:
    """Comprehensive input validation and sanitization."""

    # Patterns for validation
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    URL_PATTERN = re.compile(r'^https?://[^\s<>\"\'{}|\\^`\[\]]+$')
    SAFE_STRING_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_.:;/@+=,]+$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9]+$')

    @staticmethod
    def validate_string(
        value: Any,
        max_length: int = 1000,
        min_length: int = 0,
        allow_empty: bool = False,
        pattern: Optional[re.Pattern] = None,
    ) -> ValidationResult:
        """Validate and sanitize a string input."""
        if value is None:
            if allow_empty:
                return ValidationResult(True, "")
            return ValidationResult(False, error_message="Value is None")

        if not isinstance(value, str):
            value = str(value)

        # HTML escape to prevent XSS
        sanitized = html.escape(value)

        # Strip control characters
        sanitized = ''.join(c for c in sanitized if ord(c) >= 32 or c in '\n\r\t')

        if not allow_empty and len(sanitized.strip()) == 0:
            return ValidationResult(False, error_message="String is empty")

        if len(sanitized) < min_length:
            return ValidationResult(False, error_message=f"String too short (min {min_length})")

        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        if pattern and not pattern.match(sanitized):
            return ValidationResult(False, error_message="String doesn't match required pattern")

        return ValidationResult(True, sanitized)

    @staticmethod
    def validate_number(
        value: Any,
        min_value: float = float('-inf'),
        max_value: float = float('inf'),
        allow_none: bool = False,
    ) -> ValidationResult:
        """Validate a numeric input."""
        if value is None:
            if allow_none:
                return ValidationResult(True, None)
            return ValidationResult(False, error_message="Value is None")

        try:
            num = float(value)
            if not (-1e308 <= num <= 1e308):  # Check for inf/nan
                return ValidationResult(False, error_message="Invalid number (inf/nan)")
            if num < min_value:
                return ValidationResult(False, error_message=f"Value too small (min {min_value})")
            if num > max_value:
                return ValidationResult(False, error_message=f"Value too large (max {max_value})")
            return ValidationResult(True, num)
        except (ValueError, TypeError):
            return ValidationResult(False, error_message="Not a valid number")

    @staticmethod
    def validate_email(value: str) -> ValidationResult:
        """Validate an email address."""
        result = InputValidator.validate_string(value, max_length=254)
        if not result.is_valid:
            return result

        if not InputValidator.EMAIL_PATTERN.match(result.sanitized_value):
            return ValidationResult(False, error_message="Invalid email format")

        return result

    @staticmethod
    def validate_url(value: str) -> ValidationResult:
        """Validate a URL."""
        result = InputValidator.validate_string(value, max_length=2048)
        if not result.is_valid:
            return result

        if not InputValidator.URL_PATTERN.match(result.sanitized_value):
            return ValidationResult(False, error_message="Invalid URL format")

        return result

    @staticmethod
    def validate_dict(
        value: Any,
        required_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None,
        max_keys: int = 100,
    ) -> ValidationResult:
        """Validate a dictionary input."""
        if value is None:
            return ValidationResult(False, error_message="Value is None")

        if not isinstance(value, dict):
            return ValidationResult(False, error_message="Not a dictionary")

        if len(value) > max_keys:
            return ValidationResult(False, error_message=f"Too many keys (max {max_keys})")

        if required_keys:
            missing = set(required_keys) - set(value.keys())
            if missing:
                return ValidationResult(False, error_message=f"Missing required keys: {missing}")

        if allowed_keys:
            extra = set(value.keys()) - set(allowed_keys)
            if extra:
                return ValidationResult(False, error_message=f"Unexpected keys: {extra}")

        return ValidationResult(True, value)

    @staticmethod
    def validate_list(
        value: Any,
        min_length: int = 0,
        max_length: int = 10000,
        item_validator: Optional[Callable] = None,
    ) -> ValidationResult:
        """Validate a list input."""
        if value is None:
            return ValidationResult(False, error_message="Value is None")

        if not isinstance(value, (list, tuple)):
            return ValidationResult(False, error_message="Not a list")

        if len(value) < min_length:
            return ValidationResult(False, error_message=f"List too short (min {min_length})")

        if len(value) > max_length:
            return ValidationResult(False, error_message=f"List too long (max {max_length})")

        if item_validator:
            for i, item in enumerate(value):
                result = item_validator(item)
                if not result.is_valid:
                    return ValidationResult(False, error_message=f"Item {i}: {result.error_message}")

        return ValidationResult(True, value)


class SecuritySanitizer:
    """Security-focused sanitization utilities."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize a filename to prevent path traversal."""
        # Remove path separators
        filename = filename.replace('/', '').replace('\\', '')
        # Remove null bytes
        filename = filename.replace('\x00', '')
        # Remove dots (prevent ../ attacks)
        filename = filename.replace('.', '')
        # Remove control characters
        filename = ''.join(c for c in filename if ord(c) >= 32)
        # Limit length
        filename = filename[:255]
        return filename

    @staticmethod
    def sanitize_sql(value: str) -> str:
        """Basic SQL injection prevention (use parameterized queries in production)."""
        if not isinstance(value, str):
            return str(value)
        # Escape single quotes
        value = value.replace("'", "''")
        # Remove comment markers
        value = value.replace("--", "")
        value = value.replace("/*", "")
        value = value.replace("*/", "")
        return value

    @staticmethod
    def sanitize_command(value: str) -> str:
        """Sanitize a command string to prevent injection."""
        # Remove shell metacharacters
        dangerous_chars = ['|', '&', ';', '`', '$', '(', ')', '{', '}', '<', '>', '\n', '\r']
        for char in dangerous_chars:
            value = value.replace(char, '')
        return value

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_hex(length)

    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256(f"{salt}{data}".encode())
        return f"{salt}:{hash_obj.hexdigest()}"


class RateLimiter:
    """Simple rate limiter for API protection."""

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        if client_id not in self.requests:
            self.requests[client_id] = []

        # Remove old requests
        self.requests[client_id] = [t for t in self.requests[client_id] if t > cutoff]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_usage(self, client_id: str) -> Dict[str, int]:
        """Get current usage for a client."""
        now = time.time()
        cutoff = now - self.window_seconds

        if client_id not in self.requests:
            return {'current': 0, 'max': self.max_requests}

        recent = [t for t in self.requests[client_id] if t > cutoff]
        return {'current': len(recent), 'max': self.max_requests}


if __name__ == "__main__":
    print("=== Security Module Test ===\n")

    # Test input validation
    print("String validation:")
    print(f"  Valid: {InputValidator.validate_string('hello world')}")
    print(f"  Too long: {InputValidator.validate_string('x' * 2000, max_length=100)}")
    print(f"  Empty: {InputValidator.validate_string('', allow_empty=False)}")

    print("\nNumber validation:")
    print(f"  Valid: {InputValidator.validate_number(42, min_value=0, max_value=100)}")
    print(f"  Too large: {InputValidator.validate_number(200, max_value=100)}")

    print("\nEmail validation:")
    print(f"  Valid: {InputValidator.validate_email('test@example.com')}")
    print(f"  Invalid: {InputValidator.validate_email('not-an-email')}")

    print("\nSecurity sanitization:")
    print(f"  Filename: {SecuritySanitizer.sanitize_filename('../../etc/passwd')}")
    print(f"  Token: {SecuritySanitizer.generate_secure_token()[:16]}...")
