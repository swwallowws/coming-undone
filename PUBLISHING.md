# Publishing checklist

Pre-publish audit of what becomes visible when this repo goes public, done
2026-09-26 on `main`. History covers all refs: 10 commits, `2451486` (first) to
`30b26f9` (before this checklist).

Status: **not blocked.** Nothing copyrighted or secret is in the tree or in
history. The open items below are the owner's call and none of them needs a
history rewrite.

## Done

- [x] **No audio, MIDI or transcriptions of real songs, in the tree or in
  history.** `git log --all --name-only` lists only `.gitignore`, `README.md`,
  `SPEC.md`, `pyproject.toml`, `src/**`, `tests/**` and the vendored design
  files. No `*.mp3`, `*.wav`, `*.flac`, `*.m4a`, `*.mid`, no `out/` or `inputs/`
  folder was ever committed.
- [x] **No model weights in the tree or in history.** No `*.pt`, `*.th`,
  `*.ckpt`, `*.safetensors`, `*.onnx` was ever committed. The checkout does hold
  an untracked `model.safetensors` at the repo root; `.gitignore` excludes it.
- [x] **No secrets in history.** `git log --all -p -G` found no API key or token
  shapes (`sk-`, `hf_`, `ghp_`, `AKIA`, `BEGIN ... PRIVATE KEY`). The keyword
  search (`api_key`, `password`, `secret`, `cookies`, `HF_TOKEN`, `bearer`) only
  hits `tests/test_web.py` (added in `2451486`), a path-traversal test that
  writes a dummy `secret.txt` and requests `/etc/passwd`. No yt-dlp cookie file
  or `.env` was ever committed.
- [x] **No personal absolute paths in history.** No `/Users/...` path, username
  or email appears in any diff.
- [x] **Large binaries:** only the two vendored fonts
  (`src/stemscribe/web/static/vendor/design/fonts/*.woff2`), SIL OFL 1.1 with
  their licence files beside them.
- [x] **`.gitignore` hardened.** It already covered `out/`, `*.mid`, `*.wav` and
  model weights. Added: `inputs/`, `*.midi`, `*.mp3`, `*.flac`, `*.m4a`, `*.aac`,
  `*.ogg`, `*.opus`, `*.webm`, `.env`, `.env.*`, `cookies*.txt`, `*.cookies`.
  Model caches live outside the repo (Hugging Face cache, torch hub), and web UI
  jobs go to the system temp dir (`web/server.py`, `JOBS_ROOT`).
- [x] **`LICENSE`** added: MIT, `Copyright (c) 2026 Bengisu Ozaydin`, covering
  stemscribe's own code. `pyproject.toml` already declares
  `license = { text = "MIT" }`; kept in that form because the bare-string form
  needs setuptools 77 and the build requires only 68.
- [x] **README licence section** states that MIT covers this repo's code only,
  that downloaded models keep their own licences, and that yt-dlp downloads are
  the user's responsibility. Table checked: demucs weights research-only,
  MuScriptor weights CC-BY-NC, basic-pitch Apache-2.0 (the commercial-safe path
  behind `STEMSCRIBE_COMMERCIAL=1`), ADT_STR CC BY-SA 4.0.
- [x] **README install** now gives the tested venv recipe (torch and torchaudio
  2.8.0, `.[web,fetch,drums,dev]`), says the default backend muscriptor is a
  separate extra, and says no audio, reference songs or weights are included.
- [x] **Tests pass:** 211 passed (`.venv/bin/pytest`).

## Open (owner's call)

- [ ] **Reference-song names in prose.** No song content is in the repo, but
  the text names one reference song and local paths:
  - `SPEC.md` lines 142 and 147: `~/Playground/rearranged/inputs/koprualti.mp3`
    and `~/Playground/rearranged/phase0/koprualti_style.mid` (in history since
    `2451486`).
  - `README.md` "Cleanup" and "Legato" sections: measurements "on `koprualti`".
  - "the reference song" and "donor" appear in `README.md`, `SPEC.md`,
    `src/stemscribe/grid.py`, `src/stemscribe/tempo.py`,
    `src/stemscribe/cli.py`, `tests/test_pipeline_grid.py`.
  Naming a song and quoting note counts is not publishing it, so this does not
  block. If you would rather not show the title or the folder layout, edit the
  current files; the old text stays in history unless you rewrite it.
- [ ] **Em dashes** remain in older text: `README.md` (23 lines), `SPEC.md` (9),
  `src/stemscribe/cli.py` (1), `src/stemscribe/core.py` (1). Style only.
- [ ] **`SPEC.md`** is an internal working spec (acceptance test on a local
  file, notes for rearranged). Keep, trim, or drop before publishing.
- [ ] **README mentions rearranged**, a private repo. Fine as prose; there is
  no link to follow.
- [ ] **GitHub settings** when flipping: description, topics, and whether to
  enable issues. Visibility change is done by the owner.

## If something bad is ever found in history

Deleting a file in a new commit does not remove it from history. The options:
rewrite with `git filter-repo --path <file> --invert-paths` in a fresh clone and
push that to a **new** repo (then delete or archive the old one), or publish a
squashed single-commit copy of the current tree as a new repo. Either way,
rotate any leaked credential first.
