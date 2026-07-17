#!/usr/bin/env python3
"""Generate a minimal, valid MATLAB Level-5 .mat (pure stdlib) for testing the
reader. Writes `tools/sample.mat` with a 2x3 double matrix `A` and a char array
`greeting`. Format follows the MAT 5.0 spec (little-endian, column-major).
"""

import struct
import sys

OUT = "tools/sample.mat"


def mat_tag(dtype, nbytes):
    return struct.pack("<ii", dtype, nbytes)


miINT8, miINT32, miDOUBLE, miMATRIX, miUTF16 = 1, 5, 9, 14, 17


def write_matrix(name, data_dtype, raw_bytes, rows, cols, cls):
    # array flags (miINT32, 8 bytes): [class, nzmax]
    flags = struct.pack("<ii", cls, 0)
    # dimensions (miINT32, 8 bytes for 2D)
    dims = struct.pack("<ii", rows, cols)
    # name (miINT8)
    name_bytes = name.encode("utf-8")
    name_tag = mat_tag(miINT8, len(name_bytes)) + name_bytes
    if len(name_bytes) % 8 != 0:
        name_tag += b"\x00" * (_align(len(name_bytes)) - len(name_bytes))
    # real part
    data_tag = mat_tag(data_dtype, len(raw_bytes)) + raw_bytes
    if len(raw_bytes) % 8 != 0:
        data_tag += b"\x00" * (_align(len(raw_bytes)) - len(raw_bytes))
    body = (
        mat_tag(miINT32, 8) + flags
        + mat_tag(miINT32, 8) + dims
        + name_tag
        + data_tag
    )
    return mat_tag(miMATRIX, len(body)) + body


def _align(n):
    return (n + 7) & ~7


def main():
    # double 2x3, column-major: [[1,2,3],[4,5,6]] -> [1,4,2,5,3,6]
    vals = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    A_raw = b"".join(struct.pack("<d", v) for v in vals)
    A = write_matrix("A", miDOUBLE, A_raw, 2, 3, 6)  # 6 = mxDOUBLE_CLASS

    # char array "hello matlab" as UTF-16LE
    greeting = "hello matlab"
    g_raw = greeting.encode("utf-16-le")
    G = write_matrix("greeting", miUTF16, g_raw, 1, len(greeting), 4)  # 4 = mxCHAR_CLASS

    header = b"MATLAB 5.0 MAT-file, Platform: python, Created: test".ljust(116, b" ")
    version = struct.pack("<H", 0x0100)
    endian = b"MI"
    with open(OUT, "wb") as f:
        f.write(header + version + endian + A + G)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
