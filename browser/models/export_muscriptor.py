"""Export MuScriptor small to ONNX for the browser engine's opt-in pitched transcriber.

  python browser/models/export_muscriptor.py --out browser/models-out/ms-small \
      [--weights model.safetensors] [--muscriptor-src DIR]

MuScriptor weights are CC-BY-NC 4.0 and gated: accept the terms at
huggingface.co/MuScriptor/muscriptor-small and `hf auth login` first (the script
downloads the weights unless --weights is given). Non-commercial use only; the
exported files carry the same licence and must not be rehosted around the gate.
The model code comes from the muscriptor package (the [muscriptor] extra), or an
unpacked muscriptor wheel given as --muscriptor-src. No onnx package is needed.

Writes, for the page to drive generation itself:
  cond.onnx   log-mel [B, 501, 512] -> prefix [B, 503, D] (mel projection plus the two
              class-conditioner rows); the log-mel runs in JS (browser/src/muscriptor.js)
  lm.onnx     one transformer step with a per-layer KV cache, float16
  emb.bin     the token embedding table, float16, looked up in JS
  meta.json   shapes, special tokens and the mel settings, including the STFT window as
              stored in the weights (it is not an exact Hann, and the near-silent top
              bands depend on its sidelobes)
Then browser/models/quantize.mjs makes lm.int8.onnx, the download format.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
import torch.nn.functional as F
from torch import nn

from torch.onnx._internal import onnx_proto_utils  # noqa: E402
onnx_proto_utils._add_onnxscript_fn = lambda proto, custom_opsets: proto

SR, CHUNK = 16000, 80000


class CondFromMel(nn.Module):
    """LMModel's condition prefix from a log-mel: projection, the last frame masked (as
    length_to_mask does for 500 frames in a 501-frame tensor), then the unconditional
    dataset and instrument rows."""

    def __init__(self, lm):
        super().__init__()
        prov = lm.condition_provider.conditioners
        self.proj = prov["self_wav"].output_proj
        with torch.no_grad():
            self.register_buffer("ds", prov["dataset_name"].embed.weight[1].clone())
            self.register_buffer("inst", prov["instrument_group"].embed.weight[1].clone())

    def forward(self, mel):
        B = mel.shape[0]
        e = self.proj(mel)
        nf = e.shape[1]
        e = e * (torch.arange(nf) < nf - 1).to(e.dtype).view(1, -1, 1)
        return torch.cat([e, self.ds.view(1, 1, -1).expand(B, 1, -1), self.inst.view(1, 1, -1).expand(B, 1, -1)], 1)


class LMStep(nn.Module):
    def __init__(self, lm, dtype):
        super().__init__()
        self.lm, self.dtype = lm, dtype
        t = lm.transformer
        self.L, self.H, self.D = len(t.layers), t.layers[0].self_attn.num_heads, lm.dim
        self.Dh, self.max_period = self.D // self.H, t.max_period

    def forward(self, x, *past):
        B, T, C = x.shape
        S = past[0].shape[2]
        pos = torch.arange(T, device=x.device).float() + S
        half = C // 2
        adim = torch.arange(half, dtype=torch.float32)
        phase = pos.view(1, -1, 1) / (self.max_period ** (adim / (half - 1))).view(1, 1, -1)
        h = (x + torch.cat([torch.cos(phase), torch.sin(phase)], -1)).to(self.dtype)
        qi, kj = torch.arange(T).view(-1, 1) + S, torch.arange(S + T).view(1, -1)
        mask = torch.where(kj <= qi, 0.0, -1e4).to(self.dtype)
        presents = []
        for li, layer in enumerate(self.lm.transformer.layers):
            att = layer.self_attn
            qkv = F.linear(layer.norm1(h), att.in_proj_weight).view(B, T, 3, self.H, self.Dh)
            q, k, v = (qkv[:, :, j].transpose(1, 2) for j in range(3))
            k = torch.cat([past[2 * li], k], dim=2)
            v = torch.cat([past[2 * li + 1], v], dim=2)
            presents += [k, v]
            w = torch.softmax((q @ k.transpose(2, 3) / math.sqrt(self.Dh) + mask).float(), dim=-1).to(self.dtype)
            h = h + att.out_proj((w @ v).transpose(1, 2).reshape(B, T, self.D))
            h = h + layer.linear2(F.gelu(layer.linear1(layer.norm2(h))))
        return (self.lm.linear(self.lm.out_norm(h[:, -1:]))[:, 0].float(), *presents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="muscriptor-small model.safetensors (default: download, gated)")
    ap.add_argument("--muscriptor-src", help="an unpacked muscriptor wheel, if the package is not installed")
    a = ap.parse_args()
    if a.muscriptor_src:
        sys.path.insert(0, a.muscriptor_src)
    try:
        from muscriptor.transcription_model import _ModelConfig, _build_model, _remap_single_codebook_keys
    except ImportError as e:
        raise SystemExit(f"needs the muscriptor package (pip install '.[muscriptor]') or --muscriptor-src: {e}")
    from safetensors.torch import load_file

    weights = a.weights
    if not weights:
        from huggingface_hub import snapshot_download
        weights = str(pathlib.Path(snapshot_download("MuScriptor/muscriptor-small")) / "model.safetensors")
    cfgd = json.loads((pathlib.Path(weights).parent / "config.json").read_text())
    cfg = _ModelConfig(cfgd["dim"], cfgd["num_heads"], cfgd["num_layers"], cfgd["card"])
    lm = _build_model(torch.device("cpu"), cfg).eval()
    lm.load_state_dict(_remap_single_codebook_keys(load_file(weights)))
    print(f"muscriptor {cfgd.get('variant', '?')}: dim {cfg.dim}, {cfg.num_layers} layers, "
          f"{sum(p.numel() for p in lm.parameters()) / 1e6:.1f}M params")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    melc = lm.condition_provider.conditioners["self_wav"]
    ms = melc.mel_spec_transform
    cond = CondFromMel(lm).eval()
    # a noise chunk through the stock mel transform, to trace the graphs with; the page's
    # numbers are checked against native muscriptor by browser/verify (onset F1)
    with torch.no_grad():
        wav = torch.randn(1, CHUNK) * 0.1
        mel = torch.log(ms(wav) + melc.eps).transpose(1, 2)            # [1, 501, M]
        lm.emb.half(); lm.transformer.half(); lm.out_norm.half(); lm.linear.half()
        step = LMStep(lm, torch.float16).eval()
        torch.onnx.export(cond, (mel,), str(out / "cond.onnx"), input_names=["mel"], output_names=["prefix"],
                          dynamic_axes={"mel": {0: "B"}, "prefix": {0: "B"}}, opset_version=17, dynamo=False)
        L, H, Dh = step.L, step.H, step.Dh
        past = [torch.zeros(1, H, 3, Dh, dtype=torch.float16) for _ in range(2 * L)]
        names_in = ["x"] + [f"past.{i}.{kv}" for i in range(L) for kv in ("key", "value")]
        names_out = ["logits"] + [f"present.{i}.{kv}" for i in range(L) for kv in ("key", "value")]
        dyn = {"x": {0: "B", 1: "T"}, "logits": {0: "B"},
               **{n: {0: "B", 2: "S"} for n in names_in[1:]}, **{n: {0: "B", 2: "S_T"} for n in names_out[1:]}}
        torch.onnx.export(step, (torch.randn(1, 4, cfg.dim), *past), str(out / "lm.onnx"), input_names=names_in,
                          output_names=names_out, dynamic_axes=dyn, opset_version=17, dynamo=False)
    (out / "emb.bin").write_bytes(lm.emb.weight.detach().half().contiguous().numpy().tobytes())
    meta = {"variant": cfgd.get("variant", "small"), "dtype": "fp16", "dim": cfg.dim, "heads": H, "layers": L,
            "headDim": Dh, "card": cfg.card, "embRows": int(lm.emb.weight.shape[0]), "bos": lm.initial_token_id,
            "eos": 1, "condInput": "mel",
            "mel": {"sampleRate": SR, "nFft": ms.n_fft, "hop": ms.hop_length, "nMels": int(ms.mel_scale.fb.shape[1]),
                    "eps": float(melc.eps), "fMin": 0.0, "fMax": SR / 2,
                    "window": [float(v) for v in ms.spectrogram.window.float()]},
            "licence": "CC-BY-NC-4.0 (MuScriptor weights; non-commercial use only)"}
    (out / "meta.json").write_text(json.dumps(meta))
    for f in sorted(out.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
