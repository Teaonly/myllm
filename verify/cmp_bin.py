#!/usr/bin/env python3
"""cmp_bin.py —— myllm 与 Strata 层 0 对拍：两个 dump 目录逐张量逐字节比。

两边的 dump 入口（myllm: backend/build/parity；Strata: dump 补丁）都按同一
规约落盘：<dir>/<NN>_<name>_<dtype>.bin + manifest.json（name/dtype/shape/bytes）。
本脚本按 (name, dtype, shape) 配对（不按文件名——序号两边可能漂），判据
**逐位一致**：任何一字节不同即 FAIL（对拍的目标是 bit-exact，不做容差；
容差制是 verify/ PyTorch 金标那一路的事，别串）。

用法：
  python3 cmp_bin.py <dirA> <dirB> [--only 正则] [--ignore-missing]
退出码：0 全部一致 / 1 有差异或缺量 / 2 用法错。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys

# dtype → (每元素字节数, python struct 码)
DT = {"f32": (4, "f"), "i32": (4, "i"), "bf16": (2, None), "f16": (2, "e")}


def load(dirp: pathlib.Path) -> dict[str, dict]:
    mf = dirp / "manifest.json"
    if not mf.is_file():
        sys.exit(f"cmp_bin: {dirp}/manifest.json 不存在（dump 目录不完整？）")
    out: dict[str, dict] = {}
    for e in json.loads(mf.read_text()):
        key = (e["name"], e["dtype"], tuple(e["shape"]))
        if key in out:
            sys.exit(f"cmp_bin: {dirp} 里 {key} 重复")
        out[key] = {"file": dirp / e["file"], "bytes": e["bytes"], "meta": e}
    return out


def to_f32(raw: bytes, dtype: str) -> list[float]:
    """bf16 手工扩位（<<16 再按 f32 解）；f16 走 struct 'e'；f32 直解。"""
    if dtype == "f32":
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if dtype == "bf16":
        n = len(raw) // 2
        u16 = struct.unpack(f"<{n}H", raw)
        return struct.unpack(f"<{n}f", struct.pack(f"<{n}I", *(u << 16 for u in u16)))
    if dtype == "f16":
        n = len(raw) // 2
        return list(struct.unpack(f"<{n}e", raw))
    return []


def ulp_f32(a: float, b: float) -> int:
    """两个 f32 的 ULP 距离（按位模式差的绝对值；仅同号时严格成立）。"""
    ua = struct.unpack("<I", struct.pack("<f", a))[0]
    ub = struct.unpack("<I", struct.pack("<f", b))[0]
    return abs(ua - ub)


def flat_coords(i: int, shape: tuple) -> str:
    coords = []
    for d in reversed(shape):
        coords.append(i % d)
        i //= d
    return "(" + ",".join(str(c) for c in reversed(coords)) + ")"


def diff_summary(a: bytes, b: bytes, dtype: str, shape: tuple, limit: int = 3) -> str:
    """首差 + 差异规模 + 前几个差异位置（定位用，不是容差借口）。"""
    n = min(len(a), len(b)) // 2 if dtype in ("bf16", "f16") else min(len(a), len(b)) // 4
    n = max(n, 1)
    per = DT[dtype][0]
    total = min(len(a), len(b)) // per
    firsts, ndiff = [], 0
    va, vb = to_f32(a, dtype), to_f32(b, dtype)
    for i in range(total):
        if a[i * per:(i + 1) * per] != b[i * per:(i + 1) * per]:
            ndiff += 1
            if len(firsts) < limit:
                fa, fb = va[i], vb[i]
                extra = f" A={fa:.6g} B={fb:.6g}"
                if dtype in ("f32", "bf16", "f16") and fa == fa and fb == fb:
                    extra += f" ulp={ulp_f32(fa, fb)}"
                firsts.append(f"#{i}{flat_coords(i, shape)}{extra}")
    pct = 100.0 * ndiff / max(total, 1)
    head = f"{ndiff}/{total} 个元素不同（{pct:.2f}%）"
    return head + ("；首差 " + " | ".join(firsts) if firsts else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA")
    ap.add_argument("dirB")
    ap.add_argument("--only", default=None, help="只比 name 匹配正则的量")
    ap.add_argument("--ignore-missing", action="store_true", help="单侧缺席不算 FAIL")
    args = ap.parse_args()

    A, B = load(pathlib.Path(args.dirA)), load(pathlib.Path(args.dirB))
    if args.only:
        pat = re.compile(args.only)
        A = {k: v for k, v in A.items() if pat.search(k[0])}
        B = {k: v for k, v in B.items() if pat.search(k[0])}
    if not A and not B:
        print("cmp_bin: 两边都空（--only 太窄？）")
        return 2

    keys = sorted(set(A) | set(B), key=lambda k: (k[0], k[1]))
    n_pass = n_fail = n_missing = 0
    for k in keys:
        name, dtype, shape = k
        tag = f"{name}[{dtype}{'x'.join(map(str, shape))}]"
        if k not in A or k not in B:
            side = "B 缺" if k in A else "A 缺"
            n_missing += 1
            print(f"MISSING  {tag}  ({side})")
            continue
        a, b = A[k]["file"].read_bytes(), B[k]["file"].read_bytes()
        if len(a) != len(b):
            n_fail += 1
            print(f"FAIL     {tag}  字节数 {len(a)} vs {len(b)}")
            continue
        if a == b:
            n_pass += 1
            print(f"PASS     {tag}  ({len(a)} B)")
            continue
        n_fail += 1
        print(f"FAIL     {tag}  {diff_summary(a, b, dtype, shape)}")

    verdict = "一致" if n_fail == 0 and (n_missing == 0 or args.ignore_missing) else "不一致"
    print(f"\n== {n_pass} PASS / {n_fail} FAIL / {n_missing} MISSING → {verdict} ==")
    return 0 if verdict == "一致" else 1


if __name__ == "__main__":
    sys.exit(main())
