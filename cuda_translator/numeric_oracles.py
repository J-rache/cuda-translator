"""Exact numeric oracles for f32 semantics (integer arithmetic, no f64 shortcuts).

fma_rn_f32(a, b, c) implements IEEE-754 fused multiply-add with
roundToNearest-even EXACTLY: the product and addend are combined exactly as
integers, then rounded ONCE to f32. Python float64 ("f32(a*b+c)") is NOT a
valid fma oracle because f64 evaluation can double-round; this module exists
so tests never rely on it.

All functions operate on raw uint32 bit patterns. Also provided: f32_mul_rn
and f32_add_rn (the unfused model) so tests can find inputs where fused and
unfused results genuinely differ.
"""
from __future__ import annotations

import struct
from typing import Tuple


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _decompose(bits: int) -> Tuple[int, int, int]:
    """u32 -> (sign, E, F): value = sign * (2^23+F or F) * 2^(E-150 or -149)."""
    sign = (bits >> 31) & 1
    e = (bits >> 23) & 0xFF
    f = bits & 0x7FFFFF
    return sign, e, f


def _value_scaled(bits: int) -> Tuple[int, int]:
    """Finite f32 bits -> (signed integer mantissa m, exponent e): v = m * 2^e."""
    sign, e, f = _decompose(bits)
    if e == 0:  # subnormal (or zero)
        return (-f if sign else f), -149
    return (-(0x800000 + f) if sign else (0x800000 + f)), e - 150


def _round_to_f32(m: int, e: int) -> int:
    """Round exact value m * 2^e (m signed int, e int) to f32 bits, RN-even."""
    if m == 0:
        return 0
    sign = 0
    if m < 0:
        sign = 1
        m = -m
    hi = m.bit_length() - 1  # value in [2^hi, 2^(hi+1))

    # normal target: value = M * 2^(E-150), M in [2^23, 2^24), E in [1, 254]
    # choose E so M has exactly 24 significant bits: E = e + hi - 23 + 150
    E = e + hi + 127
    if E >= 1:
        shift = hi - 23
        if shift >= 1:
            kept = m >> shift
            rem = m & ((1 << shift) - 1)
            half = 1 << (shift - 1)
            if rem > half or (rem == half and (kept & 1)):
                kept += 1
        elif shift == 0:
            kept = m  # exact, no rounding possible
        else:
            kept = m << (-shift)
        if kept >= (1 << 24):  # rounding carried out of range
            kept >>= 1
            E += 1
        if E > 254:
            return (sign << 31) | 0x7F800000  # overflow -> inf
        if E >= 1:
            return (sign << 31) | (E << 23) | (kept - 0x800000)

    # subnormal target: value = F * 2^-149, F in [0, 2^23)
    shift = -149 - e  # F = m / 2^shift rounded
    if shift >= 1:
        kept = m >> shift
        rem = m & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (kept & 1)):
            kept += 1
    elif shift == 0:
        kept = m
    else:
        kept = m << (-shift)
    if kept >= (1 << 23):  # rounded up into normal range
        return (sign << 31) | (1 << 23) | (kept - 0x800000)
    return (sign << 31) | kept


def fma_rn_f32(a_bits: int, b_bits: int, c_bits: int) -> int:
    """Exact fused multiply-add, round-to-nearest-even, on u32 patterns."""
    for bits in (a_bits, b_bits, c_bits):
        e = (bits >> 23) & 0xFF
        f = bits & 0x7FFFFF
        if e == 0xFF:
            raise ValueError("fma oracle: inf/nan inputs not supported (keep tests finite)")
    ma, ea = _value_scaled(a_bits)
    mb, eb = _value_scaled(b_bits)
    mc, ec = _value_scaled(c_bits)
    # exact product, then exact sum on a common power-of-two scale
    if ea >= eb:
        prod = ma * mb
        eprod = ea + eb
    else:
        # same product regardless; keep scale math simple
        prod = ma * mb
        eprod = ea + eb
    emin = min(eprod, ec)
    total = prod * (1 << (eprod - emin)) + mc * (1 << (ec - emin))
    return _round_to_f32(total, emin)


def f32_mul_rn(a_bits: int, b_bits: int) -> int:
    """Exact unfused multiply (round after multiply)."""
    for bits in (a_bits, b_bits):
        if ((bits >> 23) & 0xFF) == 0xFF:
            raise ValueError("f32_mul_rn: inf/nan not supported")
    ma, ea = _value_scaled(a_bits)
    mb, eb = _value_scaled(b_bits)
    return _round_to_f32(ma * mb, ea + eb)


def f32_add_rn(a_bits: int, b_bits: int) -> int:
    """Exact unfused add (round after add)."""
    for bits in (a_bits, b_bits):
        if ((bits >> 23) & 0xFF) == 0xFF:
            raise ValueError("f32_add_rn: inf/nan not supported")
    ma, ea = _value_scaled(a_bits)
    mb, eb = _value_scaled(b_bits)
    emin = min(ea, eb)
    total = ma * (1 << (ea - emin)) + mb * (1 << (eb - emin))
    return _round_to_f32(total, emin)


def make_f32(x: float) -> int:
    """Nearest f32 pattern for a Python float (used only to build test inputs)."""
    return f32_bits(x)
