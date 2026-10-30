#!/usr/bin/env python
"""Full-architecture, layer-by-layer golden data generator for Qwen3.8-Flash-Next.

Keeps the ORIGINAL 48-layer config untouched. No weights are downloaded: every
component is materialized on the GPU from a fixed per-component seed, run, and
then discarded -- so the full ~180B-parameter model is verified end-to-end with
only ~12GB of live GPU memory.

The one exception is the PLE n-gram embedding table (~95GiB, too big for any
single GPU even alone): its rows are generated on demand from a per-row seed
(see LazyNGramRows below), which is deterministic and reproducible on any
machine, including your inference engine.

Reference dtype: fp32. Reference implementation: standard Transformers
(local main checkout), driven module-by-module exactly like
Qwen4ExpTextModel.forward / Qwen4ExpForCausalLM.forward.

Usage:
  venv/bin/python gen_golden_full.py [--decode-steps 4] [--prefill-only] ...
"""

import argparse
import hashlib
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig, Qwen4ExpForCausalLM, DynamicCache
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

BASE_CONFIG = "/home/teaonly/workspace/myllm/Qwen3.8-Flash-Next"
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextNGramEmbedding,
)

# ---------------------------------------------------------------------------
# Seed scheme (documented so the engine can reproduce every weight bit):
#   component c = ("embed",), ("layer", i), ("mixer",), ("lm_head",) ...
#   seed(c) = (SEED_BASE * 1_000_003 + component_index) % 2**31
# Every parameter of the component is first filled i.i.d. N(0, initializer_range)
# in named_parameters() order, then the standard Qwen4Exp _init_weights special
# cases are applied on top (zero RMSNorm weights, dt_bias=1, A_log=log U(0.01,16),
# PLE conv1d = 0, ngram hash buffers from config.seed=1234).
# PLE n-gram embedding row r: torch.Generator(seed=PLE_ROW_SEED_BASE ^ r),
# row = randn(head_dim) * initializer_range  (matches nn.Embedding init).
# ---------------------------------------------------------------------------
SEED_BASE = 20240917
INIT_RANGE = 0.02  # tc.initializer_range of the real config


def tensorize(obj, prefix, out, max_elems_fp32=8_000_000):
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.numel() > max_elems_fp32:  # keep huge tensors comparable: bf16 copy
            out[prefix] = t.detach().to(torch.bfloat16).cpu()
        else:
            out[prefix] = t.detach().float().cpu()
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            tensorize(v, f"{prefix}[{i}]", out)


def capture(comp, prefix_root, store):
    handles = []

    def make_hook(name):
        def hook(mod, args, kwargs, output):
            data = {}
            for i, a in enumerate(args):
                tensorize(a, f"{name}/in[{i}]", data)
            for k, v in kwargs.items():
                tensorize(v, f"{name}/in[{k}]", data)
            tensorize(output, f"{name}/out", data)
            store.update({f"{prefix_root}/{k}": v for k, v in data.items()})
        return hook

    for name, mod in comp.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make_hook(name), with_kwargs=True))
    return handles


class LazyNGramRows(nn.Module):
    """Deterministic on-demand replacement for the ~95GiB PLE n-gram embedding.

    Row r is torch.randn(head_dim, generator=Generator(ple_seed_base ^ r)) * std,
    i.e. exactly what nn.Embedding.reset_parameters would give a fresh row
    (transformers inits embeddings with N(0, initializer_range)).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, seed_base: int, std: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.seed_base = seed_base
        self.std = std
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)  # placeholder, 0 bytes

    def _row(self, r: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.seed_base ^ r)
        return torch.randn(self.embedding_dim, generator=g) * self.std

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        dev = ids.device
        flat = ids.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = torch.stack([self._row(int(r)) for r in uniq.cpu().tolist()]).to(dev)
        return rows[inv.to(dev)].view(*ids.shape, self.embedding_dim)


def materialize(comp: nn.Module, seed: int, init_fn, dev, exclude=()):
    """comp lives on meta; move to dev and init deterministically.

    `exclude`: set of submodules (by identity) to leave on meta (already
    replaced by LazyNGramRows before calling).
    """
    torch.manual_seed(seed)
    comp.to_empty(device=dev)
    with torch.no_grad():
        for p in comp.parameters():
            if p.numel():
                p.normal_(0.0, INIT_RANGE)
        comp.apply(init_fn)  # standard Qwen4Exp special cases on top


def free(comp: nn.Module):
    comp.to("meta")
    torch.cuda.empty_cache()


def dump_cache(cache: DynamicCache, prefix: str, out: dict):
    for i, layer in enumerate(cache.layers):
        for attr, val in vars(layer).items():
            if isinstance(val, torch.Tensor):
                out[f"{prefix}/layer{i}/{attr}"] = val.detach().float().cpu()
            elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                for j, t in enumerate(val):
                    out[f"{prefix}/layer{i}/{attr}[{j}]"] = t.detach().float().cpu()
            elif isinstance(val, dict) and val and any(isinstance(v, torch.Tensor) for v in val.values()):
                for k, t in val.items():
                    if isinstance(t, torch.Tensor):
                        out[f"{prefix}/layer{i}/{attr}[{k}]"] = t.detach().float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/teaonly/workspace/myllm/verify/full_out")
    ap.add_argument("--seed-input", type=int, default=123)
    ap.add_argument("--prefill-len", type=int, default=64)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    cfg = AutoConfig.from_pretrained(BASE_CONFIG)
    tc = cfg.text_config
    assert tc.num_hidden_layers == 48 and tc.num_experts == 512

    # -- build the full-architecture model on the meta device (0 bytes of RAM)
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(tc)
    text = model.model
    init_fn = model._init_weights

    # replace the giant PLE n-gram embedding with the deterministic lazy rows
    ple_layer_idx = next(i for i, l in enumerate(text.layers) if getattr(l, "ple", None) is not None)
    ngram = text.layers[ple_layer_idx].ple.ple_embedding.ngram_embedding
    lazy = LazyNGramRows(ngram.num_embeddings, ngram.embedding_dim,
                         seed_base=SEED_BASE * 31 + 7, std=INIT_RANGE).to(dev)
    text.layers[ple_layer_idx].ple.ple_embedding.ngram_embedding = lazy

    # real-device rotary (cheap, deterministic; never materialized from meta)
    rotary = Qwen4ExpTextRotaryEmbedding(config=tc).to(dev)

    # fixed input
    g = torch.Generator().manual_seed(args.seed_input)
    input_ids = torch.randint(0, tc.vocab_size, (1, args.prefill_len), generator=g).to(dev)
    torch.save(input_ids.cpu(), os.path.join(args.out_dir, "input_ids.pt"))

    def component_seed(name: str, idx: int | None = None):
        # stable across processes (builtin hash() is randomized!)
        h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        h = (SEED_BASE * 1_000_003 + h) & 0x7FFFFFFF
        if idx is not None:
            h = (h * 1_000_033 + idx * 7919) & 0x7FFFFFFF
        return h

    manifest = {
        "seed_scheme": {
            "base": SEED_BASE,
            "init_range": INIT_RANGE,
            "description": (
            "component seed = f(SEED_BASE, name, idx); params first filled i.i.d. "
            "N(0, initializer_range) in named_parameters order, then standard "
            "Qwen4Exp _init_weights specials applied. PLE ngram row r = "
            "randn(dim, Generator(lazy_seed_base ^ r)) * initializer_range."
        ),
        },
        "ple_layer_idx": ple_layer_idx,
        "prefill_len": args.prefill_len,
        "decode_steps": args.decode_steps,
        "layer_types": list(tc.layer_types),
        "files": {},
    }

    # ------------------------------------------------------------------
    # one streaming pass over the whole model. `run_pass` materializes each
    # component, forwards the carried hidden state, captures I/O, frees it.
    # ------------------------------------------------------------------
    def run_pass(tok_ids, cache, prefix, capture_submodules=True):
        store = {}
        B, L = tok_ids.shape

        materialize(text.embed_tokens, component_seed("embed"), init_fn, dev)
        h_embed = capture(text.embed_tokens, f"{prefix}/model.embed_tokens", store) if capture_submodules else []
        with torch.no_grad():
            inputs_embeds = text.embed_tokens(tok_ids)
        for h in h_embed:
            h.remove()
        store[f"{prefix}/model.embed_tokens/out"] = inputs_embeds.float().cpu()

        ple_input_ids = tok_ids
        past_seen = cache.get_seq_length() if cache is not None else 0
        position_ids = torch.arange(L, device=dev) + past_seen
        position_ids = position_ids.view(1, 1, -1).expand(4, B, -1)
        text_position_ids = position_ids[0]
        pos3 = position_ids[1:]

        if cache is not None and "indexed_attention" in tc.layer_types:
            indexer_cache = cache.layers[tc.layer_types.index("indexed_attention")]
            if indexer_cache.indexer_position_ids is not None:
                pos3 = torch.cat([indexer_cache.indexer_position_ids, pos3], dim=-1)
            indexer_cache.indexer_position_ids = pos3

        mask_kwargs = {
            "config": tc,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": cache,
            "position_ids": text_position_ids,
            "allow_is_causal_skip": False,
        }
        causal_mask_mapping = {
            "indexed_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        conv_mask = causal_mask_mapping["linear_attention"]

        position_embeddings = rotary(inputs_embeds, pos3)
        store[f"{prefix}/rotary/cos"] = position_embeddings[0].float().cpu()
        store[f"{prefix}/rotary/sin"] = position_embeddings[1].float().cpu()

        hidden = inputs_embeds.repeat(1, 1, tc.hc_count)
        for i, layer in enumerate(text.layers[: tc.num_hidden_layers]):
            materialize(layer, component_seed("layer", i), init_fn, dev)
            hs = capture(layer, f"{prefix}/model.layers.{i}", store) if capture_submodules else []
            with torch.no_grad():
                out = layer(
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask_mapping["indexed_attention"],
                    conv_mask=conv_mask,
                    past_key_values=cache,
                    ple_input_ids=ple_input_ids,
                )
            for h in hs:
                h.remove()
            hidden = out[0] if isinstance(out, tuple) else out
            store[f"{prefix}/model.layers.{i}/out"] = hidden.float().cpu()
            free(layer)

        materialize(text.hyper_connection_mixer, component_seed("mixer"), init_fn, dev)
        hs = capture(text.hyper_connection_mixer, f"{prefix}/model.hyper_connection_mixer", store) if capture_submodules else []
        with torch.no_grad():
            hidden = text.hyper_connection_mixer(hidden)
        for h in hs:
            h.remove()
        free(text.hyper_connection_mixer)

        # lm_head is tied to embed_tokens (standard Qwen4Exp behaviour)
        lm = model.lm_head
        lm.weight = text.embed_tokens.weight  # tie
        with torch.no_grad():
            logits = lm(hidden)
        store[f"{prefix}/lm_head/in"] = hidden.float().cpu()
        store[f"{prefix}/lm_head/out_full_bf16"] = logits.detach().to(torch.bfloat16).cpu()
        store[f"{prefix}/lm_head/out_last_fp32"] = logits[:, -1].float().cpu()
        free(text.embed_tokens)
        return store, hidden, logits

    # ---- prefill ----
    cache = DynamicCache(config=tc)
    store, hidden, logits = run_pass(input_ids, cache, "prefill")
    save_file(store, os.path.join(args.out_dir, "prefill.safetensors"))
    manifest["files"]["prefill.safetensors"] = len(store)
    print(f"[prefill] logits {tuple(logits.shape)} tensors={len(store)}", flush=True)

    # ---- decode ----
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    for step in range(args.decode_steps):
        store, hidden, logits = run_pass(next_tok, cache, f"decode{step:02d}", capture_submodules=False)
        store[f"decode{step:02d}/next_token"] = logits[:, -1].argmax(-1).cpu()
        dump_cache(cache, f"decode{step:02d}/cache", store)
        save_file(store, os.path.join(args.out_dir, f"decode{step:02d}.safetensors"))
        manifest["files"][f"decode{step:02d}.safetensors"] = len(store)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        print(f"[decode {step}] tensors={len(store)}", flush=True)

    final_cache = {}
    dump_cache(cache, "final_cache", final_cache)
    save_file(final_cache, os.path.join(args.out_dir, "cache_states.safetensors"))
    manifest["files"]["cache_states.safetensors"] = len(final_cache)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
