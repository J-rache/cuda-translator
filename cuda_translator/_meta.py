"""Package metadata for cuda-translator."""

VERSION = "1.0.0"
SERVER_NAME = "cuda-translator"
MCP_PROTOCOL_VERSION = "2024-11-05"
SCHEMA_VERSION = "cuda-translator.snapshot/1"
DEFAULT_REST_PORT = 8377


class InputError(Exception):
    """Raised when an input payload cannot be analyzed."""


class UsageError(Exception):
    """Raised for CLI/API usage mistakes (bad target, missing argument)."""
