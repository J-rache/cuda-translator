"""cuda-translator: translate CUDA binaries into useful, portable artifacts.

Public API:
    analyze_input(data)              -> lossless structural report
    translate_input(data)            -> fail-closed OpenCL C translation
    translate_ptx_text(text)         -> same, from PTX text
    fatbin / elf / pe                -> container parsers
    ptx.parse_ptx                    -> PTX frontend (House IR)
    opencl.emit_program              -> OpenCL C backend
    interpreter.run_kernel           -> CPU debug oracle
"""
from ._meta import VERSION, SERVER_NAME, MCP_PROTOCOL_VERSION, DEFAULT_REST_PORT
from ._meta import InputError, UsageError, TranslationError
from .pipeline import analyze_input, translate_input, translate_ptx_text, detect_input
from .ir import TranslationResult, Kernel, Instruction, Operand, Address

__version__ = VERSION
__all__ = [
    "VERSION", "SERVER_NAME", "MCP_PROTOCOL_VERSION", "DEFAULT_REST_PORT",
    "InputError", "UsageError", "TranslationError",
    "analyze_input", "translate_input", "translate_ptx_text", "detect_input",
    "TranslationResult", "Kernel", "Instruction", "Operand", "Address",
]
