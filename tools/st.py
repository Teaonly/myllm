#!/usr/bin/env python3
"""st.py —— 最小 safetensors 读取器(kernel 开发期读 verify/full_out 的 golden 用)。

系统 python3 无 numpy,需用 Strata 的 venv:
  /home/teaonly/workspace/all-infer/Strata/.venv/bin/python tools/st.py <file.safetensors> [key ...]

不带 key:列出全部张量(shape/dtype/bytes);带 key:打印前 8 个值。
也可 as module:from st import load_st; d = load_st(path)  # {key: ndarray}
dtype 支持 BF16/F32/F16/I64/I32/I16/I8/BOOL;BF16 高 16 位直落 f32。
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np

_DTYPES = {
    "BF16": ("u16→f32", 2),
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "F64": (np.float64, 8),
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


def _conv(raw: bytes, dt: str) -> np.ndarray:
    if dt == "BF16":  # 高 16 位直落 f32
        a = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return a.view(np.float32).copy()
    return np.frombuffer(raw, dtype=_DTYPES[dt][0]).copy()


def load_st(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8 : 8 + n])
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = _conv(raw[base + s : base + e], v["dtype"]).reshape(v["shape"])
    return out


def main() -> None:
    path = sys.argv[1]
    d = load_st(path)
    keys = sys.argv[2:]
    if not keys:
        total = 0
        for k in sorted(d):
            a = d[k]
            print(f"{k:70s} {str(a.shape):20s} {a.dtype} {a.nbytes}B")
            total += a.nbytes
        print(f"---- {len(d)} tensors, {total} bytes")
    else:
        for k in keys:
            a = d[k]
            print(f"{k}: shape={a.shape} dtype={a.dtype}")
            print("  first 8:", a.ravel()[:8])


if __name__ == "__main__":
    main()
