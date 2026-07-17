#!/usr/bin/env python3
"""Read MATLAB Level-5 .mat files using only the Python standard library.

Emits a JSON array on stdout: one object per top-level variable, each with
`name`, `shape`, `class`, and either numeric `stats` (count/min/max/mean) or a
`value` string for char arrays. Failures are reported as `{"error": ...}` so the
caller can ingest whatever it can and skip the rest.

This deliberately avoids scipy/h5py (not installed, and we stay local/offline).
It handles the common numeric (double/single/int*) and char array classes.
"""

import json
import struct
import sys

# MATLAB numeric data-type codes (MAT version 5).
MI = {
    "miINT8": 1, "miUINT8": 2, "miINT16": 3, "miUINT16": 4, "miINT32": 5,
    "miUINT32": 6, "miSINGLE": 7, "miDOUBLE": 9, "miINT64": 12, "miUINT64": 13,
    "miMATRIX": 14, "miUTF8": 16, "miUTF16": 17, "miUTF32": 18,
}
INV_MI = {v: k for k, v in MI.items()}

# mx array class IDs.
MX = {
    "mxCELL": 1, "mxSTRUCT": 2, "mxLOGICAL": 3, "mxCHAR": 4, "mxSPARSE": 5,
    "mxDOUBLE": 6, "mxSINGLE": 7, "mxINT8": 8, "mxUINT8": 9, "mxINT16": 10,
    "mxUINT16": 11, "mxINT32": 12, "mxUINT32": 13, "mxINT64": 14, "mxUINT64": 15,
}

# data-type code -> (struct fmt, byte width, signed)
NUMERIC = {
    1: ("<b", 1, True), 2: ("<B", 1, False), 3: ("<h", 2, True),
    4: ("<H", 2, False), 5: ("<i", 4, True), 6: ("<I", 4, False),
    7: ("<f", 4, False), 9: ("<d", 8, False), 12: ("<q", 8, True),
    13: ("<Q", 8, False),
}


def _align(n):
    return (n + 7) & ~7


def _read_tag(f):
    tag = f.read(8)
    if len(tag) < 8:
        return None
    dtype, nbytes = struct.unpack("<ii", tag)
    return dtype, nbytes


def _parse_matrix(f, nbytes):
    """Parse one miMATRIX element of `nbytes` bytes into a summary dict."""
    end = f.tell() + nbytes
    cls = None
    dims = None
    name = None
    real = None
    int32_seen = 0
    while f.tell() < end:
        tag = _read_tag(f)
        if tag is None:
            break
        dtype, nb = tag
        data = f.read(nb)
        if len(data) < nb:
            break
        if nb % 8 != 0:
            f.read(_align(nb) - nb)

        if dtype == MI["miINT32"]:
            int32_seen += 1
            vals = struct.unpack("<%di" % (nb // 4), data)
            if int32_seen == 1:  # array flags
                cls = vals[0] & 0xFF
            elif int32_seen == 2:  # dimensions
                dims = list(vals[:2]) if len(vals) >= 2 else [vals[0], 1]
        elif dtype in (MI["miINT8"], MI["miUINT8"], MI["miUTF8"]):
            name = data.rstrip(b"\x00").decode("utf-8", "replace")
        elif dtype in NUMERIC:
            fmt, width, signed = NUMERIC[dtype]
            count = nb // width
            vals = struct.unpack("<%d%s" % (count, fmt[-1]), data[: count * width])
            real = vals
        elif dtype == MI["miUTF16"]:
            chars = data[: nb - (nb % 2)].decode("utf-16-le", "replace")
            name = name or None
            real = ("char", chars)
        elif dtype == MI["miUTF32"]:
            chars = data[: nb - (nb % 4)].decode("utf-32-le", "replace")
            real = ("char", chars)
        # ignore other subelements (e.g. imaginary part) for the summary

    if cls is None:
        return None

    shape = dims or [len(real) if real is not None else 0, 1]
    mx_name = next((k for k, v in MX.items() if v == cls), "mxUNKNOWN")

    if isinstance(real, tuple) and real[0] == "char":
        return {"name": name, "shape": shape, "class": mx_name, "value": real[1][:200]}
    if real is None:
        return {"name": name, "shape": shape, "class": mx_name}
    nums = [float(x) for x in real]
    if nums:
        return {
            "name": name, "shape": shape, "class": mx_name,
            "stats": {
                "count": len(nums),
                "min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums),
            },
        }
    return {"name": name, "shape": shape, "class": mx_name}


def read_mat(path):
    results = []
    try:
        with open(path, "rb") as f:
            header = f.read(116)
            if b"MATLAB" not in header:
                return [{"error": "not a MATLAB 5.0 MAT-file"}]
            f.read(4)  # version (uint16) + endian indicator (uint16)
            while True:
                tag = _read_tag(f)
                if tag is None:
                    break
                dtype, nb = tag
                if dtype == MI["miMATRIX"]:
                    m = _parse_matrix(f, nb)
                    if m is not None:
                        results.append(m)
                else:
                    # skip unknown top-level element
                    f.seek(_align(nb), 1) if False else f.read(_align(nb))
    except Exception as e:  # noqa: BLE001 - report, never crash the ingestor
        results.append({"error": str(e)})
    return results


def main():
    out = []
    for path in sys.argv[1:]:
        out.append({"file": path, "vars": read_mat(path)})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
