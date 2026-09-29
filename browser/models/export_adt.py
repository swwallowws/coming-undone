"""Export ADT_STR (the drum model stemscribe pins) to ONNX for the browser engine.

  python browser/models/export_adt.py --out browser/models-out

Writes adt_encoder.onnx and adt_decoder.onnx (float32); quantize them with
browser/models/quantize.mjs (browser/models/finish.mjs does all of it).

  encoder: log-mel (B, 246, 128) -> cross_k0, cross_v0, ... (B, 6, 246, 128): the
           encoder memory already projected to each decoder layer's cross-attention K/V
  decoder: token prefix (B, L) + cross K/V -> logits of the last position (B, 1400)

The log-mel front end, chunking and token decoding run in JS (browser/src/drums.js).
Two things the export has to respect:
- torch's fused nn.TransformerEncoderLayer path does not export, and a traced
  nn.MultiheadAttention bakes batch and length into its Reshapes, so attention is
  written out by hand over the same weights.
- ADT_STR's causal mask is additive -1e4 (for bf16) and its attention logits are large
  enough that future tokens leak through it; the model was trained that way. A KV cache
  would change the output, so the decoder re-runs the whole prefix each step with the
  same mask. That is exact: the parity check below compares against ADTModel.sample.

ADT_STR: Melucci, Merialdo, Akama 2026, code and weights CC BY-SA 4.0. The exported
files are adaptations of the weights and stay CC BY-SA 4.0.
Needs the drums extra (pip install '.[drums]'), no onnx package.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch
import torch.nn as nn

from stemscribe.backends import ADT_STR_REPO, ADT_STR_REVISION, ADT_STR_VARIANT

# torch 2.8's legacy exporter imports `onnx` only to splice onnxscript functions in;
# these graphs have none, so skip that step and no onnx package is needed.
from torch.onnx._internal import onnx_proto_utils  # noqa: E402
onnx_proto_utils._add_onnxscript_fn = lambda proto, custom_opsets: proto

H, D = 6, 128                      # heads, head dim (d_model 768)
F = torch.nn.functional


def heads(x):
    return x.reshape(x.shape[0], -1, H, D).transpose(1, 2)


def merge(x):
    return x.transpose(1, 2).reshape(x.shape[0], -1, H * D)


def attend(q, k, v, mask=None):
    s = q @ k.transpose(-1, -2) / math.sqrt(D)
    if mask is not None:
        s = s + mask
    return torch.softmax(s, dim=-1) @ v


class Enc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, logmel):
        e = self.m.encoder
        x = e.dense_layer(self.m.project_to_mel(logmel))
        x = x + e.positional_encoding.pos_embedding[:, : x.shape[1], :]
        for lyr in e.encoder.layers:
            W, b = lyr.self_attn.in_proj_weight, lyr.self_attn.in_proj_bias
            q, k, v = (heads(F.linear(x, W[i * 768:(i + 1) * 768], b[i * 768:(i + 1) * 768])) for i in range(3))
            x = lyr.norm1(x + lyr.self_attn.out_proj(merge(attend(q, k, v))))
            x = lyr.norm2(x + lyr.linear2(F.gelu(lyr.linear1(x))))
        mem = e.layer_norm(x)
        outs = []
        for lyr in self.m.decoder.decoder.layers:
            W, b = lyr.multihead_attn.in_proj_weight, lyr.multihead_attn.in_proj_bias
            outs.append(heads(F.linear(mem, W[768:1536], b[768:1536])))
            outs.append(heads(F.linear(mem, W[1536:], b[1536:])))
        return tuple(outs)


class DecFull(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.dec = m.decoder

    def forward(self, tokens, *cross):
        d = self.dec
        L = tokens.shape[1]
        idx = torch.arange(L)
        mask = (idx.unsqueeze(0) > idx.unsqueeze(1)).float() * -1e4
        x = d.positional_encoding(d.tgt_tok_emb(tokens))
        for i, lyr in enumerate(d.decoder.layers):
            W, b = lyr.self_attn.in_proj_weight, lyr.self_attn.in_proj_bias
            q, k, v = (heads(F.linear(x, W[j * 768:(j + 1) * 768], b[j * 768:(j + 1) * 768])) for j in range(3))
            x = lyr.norm1(x + lyr.self_attn.out_proj(merge(attend(q, k, v, mask))))
            Wc, bc = lyr.multihead_attn.in_proj_weight, lyr.multihead_attn.in_proj_bias
            qc = heads(F.linear(x, Wc[:768], bc[:768]))
            x = lyr.norm2(x + lyr.multihead_attn.out_proj(merge(attend(qc, cross[2 * i], cross[2 * i + 1]))))
            x = lyr.norm3(x + lyr.linear2(F.gelu(lyr.linear1(x))))
        return d.generator(x[:, -1, :])


def test_clip(sr=24000, seconds=2.56):
    """A synthetic kit pattern (kick, snare, hats) so the parity check has hits to agree on."""
    n = int(sr * seconds)
    t = torch.arange(n) / sr
    y = torch.zeros(n)
    g = torch.Generator().manual_seed(0)
    for k in range(0, 13):
        at = int(k * 0.2 * sr)
        seg = torch.arange(n - at) / sr
        if k % 4 == 0:
            y[at:] += 0.8 * torch.sin(2 * math.pi * (50 + 80 * torch.exp(-seg * 30)) * seg) * torch.exp(-seg * 8)
        elif k % 4 == 2:
            y[at:] += 0.4 * torch.randn(n - at, generator=g) * torch.exp(-seg * 20)
        y[at:] += 0.15 * torch.randn(n - at, generator=g) * torch.exp(-seg * 60)
    return (y / y.abs().max() * 0.8).unsqueeze(0), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    repo = snapshot_download(ADT_STR_REPO, revision=ADT_STR_REVISION)
    sys.path.insert(0, repo)
    from model import ADTModel  # the model repo's own code, pinned by revision

    model = ADTModel.from_pretrained(str(pathlib.Path(repo) / ADT_STR_VARIANT)).float().cpu().eval()
    print(f"ADT_STR {ADT_STR_VARIANT} @ {ADT_STR_REVISION[:8]}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    clip, _ = test_clip()
    with torch.no_grad():
        ref = model.sample(src=clip, src_mask=None, tgt_mask=None, max_length=512,
                           start_token=2, end_token=3)[0].tolist()
        mel = model.compute_spectrogram(clip)
        enc, dec = Enc(model).eval(), DecFull(model).eval()
        cross = enc(mel)
        seq = [2]
        while len(seq) < 512:
            seq.append(int(dec(torch.tensor([seq]), *cross).argmax(-1)))
            if seq[-1] == 3:
                break
    print(f"parity with ADTModel.sample: {seq == ref} ({len(ref)} tokens)")
    if seq != ref:
        raise SystemExit("export wrappers do not reproduce ADTModel.sample")

    n = len(model.decoder.decoder.layers)
    names = [f"cross_{kv}{i}" for i in range(n) for kv in "kv"]
    bh = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(enc, (mel,), str(out / "adt_encoder.onnx"), input_names=["logmel"],
                          output_names=names, dynamic_axes={"logmel": bh, **{k: bh for k in names}},
                          opset_version=17, dynamo=False)
        torch.onnx.export(dec, (torch.tensor([[2, 50, 336]]), *cross), str(out / "adt_decoder.onnx"),
                          input_names=["tokens", *names], output_names=["logits"],
                          dynamic_axes={"tokens": {0: "batch", 1: "len"}, "logits": bh, **{k: bh for k in names}},
                          opset_version=17, dynamo=False)
    for f in ("adt_encoder.onnx", "adt_decoder.onnx"):
        print(f"  {f}: {(out / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
