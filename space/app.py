"""Coming Undone on a Hugging Face Space, on ZeroGPU.

The public page (Coming Undone's own static page) calls the `split` endpoint through
Gradio's JavaScript client, so visitors never see this Gradio UI; it is here for the
Space's own page and for debugging.

ZeroGPU: GPU work happens only inside the @spaces.GPU function, and models go to cuda
here at module level, as the docs ask (https://huggingface.co/docs/hub/en/spaces-zerogpu).
Outside ZeroGPU the decorator does nothing, so this also runs on a CPU laptop:

    .venv/bin/python space/app.py           # after scripts/stage_space.py, or with
                                            # src/ on the path (see below)
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import uuid

HERE = pathlib.Path(__file__).resolve().parent
# In the Space the stemscribe package is copied in beside this file (stage_space.py);
# in the repo it lives in ../src.
if not (HERE / "stemscribe").is_dir():
    sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import gradio as gr  # noqa: E402
import spaces  # noqa: E402
import torch  # noqa: E402

import split  # noqa: E402
from stemscribe import separate as _separate  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _preload() -> None:
    """Fetch every model once at start-up, so a visitor's GPU minutes pay for work,
    not downloads. None of the weights ship with the Space: each comes from its own
    official source under its own licence (see README)."""
    try:
        from demucs.pretrained import get_model

        for name in split.DEMUCS_MODELS:
            model = get_model(name)
            model.to(DEVICE)
            model.eval()
            _separate.PRELOADED[name] = model
        print(f"demucs ready on {DEVICE}: {', '.join(split.DEMUCS_MODELS)}")
    except Exception as e:  # the run still loads it itself, only slower
        print(f"demucs preload failed ({type(e).__name__}: {e})")
    try:
        from huggingface_hub import snapshot_download

        from stemscribe.backends import ADT_STR_REPO, ADT_STR_REVISION

        snapshot_download(ADT_STR_REPO, revision=ADT_STR_REVISION)
        print("ADT_STR drums ready")
    except Exception as e:
        print(f"ADT_STR preload failed ({type(e).__name__}: {e})")
    try:
        # MuScriptor's weights are gated (CC-BY-NC): the Space needs an HF_TOKEN secret
        # from an account that accepted the terms. Loading once fills the cache the
        # `muscriptor` command reads from.
        from muscriptor import TranscriptionModel

        TranscriptionModel.load_model(device="cpu")
        print("MuScriptor weights ready")
    except Exception as e:
        print(f"MuScriptor preload failed ({type(e).__name__}: {e})")


if os.environ.get("COMING_UNDONE_PRELOAD", "1") != "0":
    _preload()


def _gpu_seconds(audio_path: str, opts: dict, work: str) -> int:
    return split.gpu_seconds(opts)


@spaces.GPU(duration=_gpu_seconds)
def _on_gpu(audio_path: str, opts: dict, work: str) -> dict:
    return split.run(audio_path, opts, work)


WORK_ROOT = pathlib.Path(tempfile.gettempdir()) / "coming-undone"
#: Runs' work folders and Gradio's copies of uploads and results live this long, in
#: seconds; the README promises visitors their files are gone within an hour.
KEEP_SECONDS = 1800


def split_api(audio: str | None, options: str, progress=gr.Progress()):
    """Audio file + options JSON in; result JSON, the combined MIDI, one MIDI per part,
    and the audio files (instrumental, stems, manifest) out."""
    split.sweep(WORK_ROOT, KEEP_SECONDS)   # Gradio copies the results out; old runs go
    if not audio:
        raise gr.Error("Send an audio file.")
    try:
        opts = split.parse_options(options)
    except split.OptionError as e:
        raise gr.Error(str(e)) from None
    work = WORK_ROOT / uuid.uuid4().hex[:12]
    progress(0.05, desc="waiting for a GPU")
    try:
        out = _on_gpu(audio, opts, str(work))
    except gr.Error:
        raise                              # ZeroGPU's own messages, the quota one included
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}") from None
    progress(1.0, desc="done")
    return (
        json.dumps(out["result"]),
        str(out["midi"]),
        [str(p) for p in out["parts"]],
        [str(p) for p in out["files"]],
    )


with gr.Blocks(title="Coming Undone", delete_cache=(KEEP_SECONDS // 2, KEEP_SECONDS)) as demo:
    gr.Markdown(
        "# Coming Undone.\nsimply split. Song in, stems and labeled MIDI out.\n\n"
        f"Takes up to {split.MAX_SECONDS:g} s of audio per run. Free and non-commercial: "
        "some of the models behind it are licensed for research or non-commercial use "
        "only (see this Space's README)."
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.File(label="Audio", file_types=["audio"], type="filepath")
            options_in = gr.Textbox(label="Options (JSON)", value="{}", lines=3)
            go = gr.Button("Split", variant="primary")
        with gr.Column():
            result_out = gr.Textbox(label="Result (JSON)", lines=6)
            midi_out = gr.File(label="MIDI")
            parts_out = gr.File(label="Parts", file_count="multiple")
            files_out = gr.File(label="Audio and manifest", file_count="multiple")
    go.click(split_api, [audio_in, options_in], [result_out, midi_out, parts_out, files_out],
             api_name="split")

demo.queue(default_concurrency_limit=2, max_size=32)

if __name__ == "__main__":
    demo.launch()
