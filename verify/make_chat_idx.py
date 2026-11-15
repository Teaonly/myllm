#!/usr/bin/env python3
"""make_chat_idx.py —— 生成两边对照用的静态 ndjson 输入（Template + Tokenizer 全量）。

目的：myllm vt 引擎与 Strata 的 prefill 输入必须逐 token 一致。本脚本用
Strata 自己的参考实现（serve/frontend.py ChatTemplate + tools/strata_tokenizer.py
Tokenizer，即 strata 服务进程运行时的同一段代码路径）把一条用户消息渲染成
prompt 再编码成 idx，落成一个静态 ndjson 文件，喂给 backend/test/feed.sh。

输入两边一致的含义：
  - myllm 侧：feed.sh 把本文件的 UserMsg.input_ids 直接送进 backend；
  - Strata 侧：向 /v1/chat/completions 发同样的消息（"content": "Hello"），
    服务端 render_prompt→encode_prompt 走的正是本脚本调用的同一实现，
    渲染结果另与 serve/chat_golden.json 的 HF 金标（case 'single user'）
    逐字节对照，所以两边 idx 相同。

运行（依赖 jinja2 + regex，用 Strata 的 .venv）：
  /home/teaonly/workspace/all-infer/Strata/.venv/bin/python make_chat_idx.py \
      [--text "Hello"] [--out ../backend/test/chat_hello.ndjson]

ndjson 里的 UserMsg 行除协议字段外带两个元数据字段（backend 的解析策略
是忽略未知字段，protocol.cpp 注释明示）：prompt = 渲染后的完整字符串，
source = 生成参数。文件是静态的：改输入重跑脚本即可，勿手编。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

STRATA = pathlib.Path("/home/teaonly/workspace/all-infer/Strata")
DATA = pathlib.Path("/home/teaonly/workspace/all-infer/Strata-data")   # 数据目录在仓库旁（NVMe）
PACK_TOK = DATA / "packs/iq3_s/tokenizer"   # 与 strata-iq3_s.json 的 "tokenizer" 一致
# GGUF 仓里有四个量化子目录，pack 是 setup 从 IQ3_S 抽出来的（与 strata-iq3_s.json 同源）
sys.path.insert(0, str(STRATA))          # serve.frontend
sys.path.insert(0, str(STRATA / "tools"))  # strata_tokenizer

from serve.frontend import ChatTemplate   # noqa: E402
import strata_tokenizer as ST             # noqa: E402


def build_tokenizer(pack_dir: pathlib.Path) -> ST.Tokenizer:
    """server.py main 的逐行照抄：从 pack tokenizer/ 目录建参考 Tokenizer。"""
    vocab = json.loads((pack_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (pack_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((pack_dir / "token_type.json").read_text())
    cfg = json.loads((pack_dir / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)


def check_golden(rendered: str) -> None:
    """与 serve/chat_golden.json 的 HF 对照金标比（transformers apply_chat_template 产物）。"""
    golden = json.loads((STRATA / "serve/chat_golden.json").read_text(encoding="utf-8"))
    case = next((c for c in golden if c.get("name") == "single user"), None)
    if case is None:
        print("golden: serve/chat_golden.json 无 'single user' 用例，跳过")
        return
    if rendered == case["hf"]:
        print("golden: 渲染与 HF 金标（'single user'）逐字节一致 ✓")
    else:
        print("golden: *** 渲染与 HF 金标不一致 ***")
        print("  mine: %r" % rendered)
        print("  hf:   %r" % case["hf"])
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello", help="用户消息原文（默认 Hello，金标用例）")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent
                                         / "backend/test/chat_hello.ndjson"))
    args = ap.parse_args()

    tok = build_tokenizer(PACK_TOK)
    template = ChatTemplate(PACK_TOK / "chat_template.jinja")

    # 与一条裸 /v1/chat/completions 请求等价：add_generation_prompt=True，
    # 不传 enable_thinking / reasoning_effort（strata-iq3_s.json 无 reasoning_effort，
    # with_shared 不注入）→ 思考开、effort 默认 xhigh。
    messages = [{"role": "user", "content": args.text}]
    prompt = template.render(messages, tools=None, add_generation_prompt=True)

    if args.text == "Hello":
        check_golden(prompt)

    ids = tok.encode(prompt, parse_special=True)   # encode_prompt 的最终路径

    back = tok.decode(ids)
    assert back == prompt, "round trip 失败: %r != %r" % (back, prompt)
    print("round trip: decode(encode(prompt)) == prompt ✓")

    # --------------------------------------------------------------- 报告
    print("\n=== 渲染后的 prompt（repr，换行可见）===")
    print(repr(prompt))
    print("\n=== idx（%d 个 token）===" % len(ids))
    print(ids)
    print("\n=== 逐 token 表（pos 0 起；S=special（3 CONTROL/4 USER_DEFINED））===")
    for i, t in enumerate(ids):
        ty = tok.token_types[t] if tok.token_types and t < len(tok.token_types) else 0
        mark = "S%d" % ty if ty in (3, 4) else "  "
        print("%4d  %6d %s %r" % (i, t, mark, tok.tokens[t]))

    # --------------------------------------------------------------- ndjson
    row = {
        "__type__": "UserMsg",
        "uid": 1,
        "input_ids": ids,
        "sampling_params": {"max_tokens": 8},
        "prompt": prompt,                       # 元数据：backend 忽略，供人核对
        "source": "make_chat_idx.py: ChatTemplate+Tokenizer (Strata ref) from pack iq3_s, text=%r" % args.text,
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nwrote %s（%d bytes, %d input_ids）" % (out, out.stat().st_size, len(ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
