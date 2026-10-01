"""Package metadata for cuda-translator."""

VERSION = "0.1.1"
SERVER_NAME = "cuda-translator"
MCP_PROTOCOL_VERSION = "2024-11-05"
SCHEMA_VERSION = "cuda-translator.snapshot/1"
DEFAULT_REST_PORT = 8377

# Public resource bounds. These are deliberately generous relative to the
# checked-in corpus (the flagship House Field PTX is < 1 MiB) while preventing
# hostile metadata from turning a small request into an unbounded allocation.
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_ENTRY_BYTES = 64 * 1024 * 1024
MAX_FATBIN_ENTRIES = 4096


class InputError(Exception):
    """Raised when an input payload cannot be analyzed."""


def ensure_input_size(data: bytes, label: str = "input") -> bytes:
    size = len(data)
    if size > MAX_INPUT_BYTES:
        raise InputError(
            f"{label} too large: {size} bytes exceeds {MAX_INPUT_BYTES}-byte limit"
        )
    return data


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
