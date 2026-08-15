import mmap
import struct
import sys

TYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}

def read_string(data, offset):
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    s = data[offset:offset + length].decode("utf-8", errors="replace")
    offset += length
    return s, offset

def read_value(data, offset, vtype):
    if vtype == 0:
        val = struct.unpack_from("<B", data, offset)[0]; offset += 1
    elif vtype == 1:
        val = struct.unpack_from("<b", data, offset)[0]; offset += 1
    elif vtype == 2:
        val = struct.unpack_from("<H", data, offset)[0]; offset += 2
    elif vtype == 3:
        val = struct.unpack_from("<h", data, offset)[0]; offset += 2
    elif vtype == 4:
        val = struct.unpack_from("<I", data, offset)[0]; offset += 4
    elif vtype == 5:
        val = struct.unpack_from("<i", data, offset)[0]; offset += 4
    elif vtype == 6:
        val = struct.unpack_from("<f", data, offset)[0]; offset += 4
    elif vtype == 7:
        val = bool(struct.unpack_from("<B", data, offset)[0]); offset += 1
    elif vtype == 8:
        val, offset = read_string(data, offset)
    elif vtype == 10:
        val = struct.unpack_from("<Q", data, offset)[0]; offset += 8
    elif vtype == 11:
        val = struct.unpack_from("<q", data, offset)[0]; offset += 8
    elif vtype == 12:
        val = struct.unpack_from("<d", data, offset)[0]; offset += 8
    else:
        raise ValueError(f"Unknown value type {vtype} at offset {offset}")
    return val, offset

def parse_gguf(path):
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            offset = 0
            if data[offset:offset+4] != b"GGUF":
                raise ValueError(f"Bad magic: {data[offset:offset+4]}")
            offset += 4
            version = struct.unpack_from("<I", data, offset)[0]; offset += 4
            tensor_count = struct.unpack_from("<Q", data, offset)[0]; offset += 8
            kv_count = struct.unpack_from("<Q", data, offset)[0]; offset += 8

            print(f"GGUF version: {version}")
            print(f"Tensor count: {tensor_count}")
            print(f"KV pairs: {kv_count}")
            print("-" * 60)

            for i in range(kv_count):
                key, offset = read_string(data, offset)
                vtype = struct.unpack_from("<I", data, offset)[0]; offset += 4
                type_name = TYPE_NAMES.get(vtype, f"unknown({vtype})")

                if vtype == 9:
                    arr_type = struct.unpack_from("<I", data, offset)[0]; offset += 4
                    arr_len = struct.unpack_from("<Q", data, offset)[0]; offset += 8
                    items = []
                    for _ in range(arr_len):
                        item, offset = read_value(data, offset, arr_type)
                        items.append(item)
                    print(f"[{i}] {key} ({type_name}):")
                    for item in items:
                        print(f"      - {item}")
                else:
                    value, offset = read_value(data, offset, vtype)
                    print(f"[{i}] {key} ({type_name}): {value}")

if __name__ == "__main__":
    parse_gguf(sys.argv[1])
