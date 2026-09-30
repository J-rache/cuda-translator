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


class TranslationError(Exception):
    """Fail-closed translation abort (IR rule 1).

    Carries the kernel name, opcode, and PTX source line of the first
    reachable unsupported instruction or unprovable control-flow shape."""

    def __init__(self, message: str, kernel: str = "",
                 opcode: str = "", line: int = 0):
        self.kernel = kernel
        self.opcode = opcode
        self.line = line
        super().__init__(message)
