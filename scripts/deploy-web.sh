#!/usr/bin/env bash
# Stage the public page. CI (.github/workflows/ci.yml) stages it on every push to
# main and publishes it to this repo's GitHub Pages, served at
# https://swwallowws.github.io/coming-undone/ (the old coming-undone-web address redirects).
#   scripts/deploy-web.sh --stage DIR       stage into DIR (no git)
#   scripts/deploy-web.sh --push CHECKOUT   stage into a checkout, commit, push (old manual route)
#
# The site is static: the studio page (web/static), whose engines are Online (the
# Hugging Face Space), This computer (localhost:8002, when it runs) and In your
# browser (models from the public Hugging Face repo, CONFIG.publicModels), plus the
# guided demo at /try/: the frozen song in try-dist/ (scripts/freeze_try.py, not
# tracked) with the current try page code on top. Every path is relative, so it
# works under the /coming-undone-web/ subpath.
set -euo pipefail

mode="${1:-}"; target="${2:-}"
if [[ "$mode" != "--stage" && "$mode" != "--push" ]] || [[ -z "$target" ]]; then
  echo "usage: $0 --stage DIR | --push CHECKOUT" >&2; exit 2
fi
root="$(cd "$(dirname "$0")/.." && pwd)"
static="$root/src/stemscribe/web/static"
if [[ ! -f "$root/try-dist/try/data.json" ]]; then
  echo "no frozen demo in try-dist/: run scripts/freeze_try.py first" >&2; exit 1
fi

mkdir -p "$target"
find "$target" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

# The studio page and everything it loads.
cp "$static/index.html" "$target/"
cp -R "$static/js" "$static/favicons" "$static/vendor" "$target/"
# The demo: the frozen song's data, then the current page code over it.
cp -R "$root/try-dist/try" "$target/"
find "$static/try" -maxdepth 1 -type f -exec cp {} "$target/try/" \;

cp "$root/deploy/README.md" "$root/LICENSE" "$target/"
{
  cat "$static/vendor/browser-engine/LICENSES.txt"
  echo
  cat "$static/vendor/design/sound/NOTICE"
  echo
  cat "$static/vendor/design/sound/spessasynth/NOTICE"
} > "$target/NOTICE"
touch "$target/.nojekyll"

if [[ "$mode" == "--push" ]]; then
  git -C "$target" add -A
  git -C "$target" commit -m "Deploy Coming Undone $(git -C "$root" rev-parse --short HEAD)"
  git -C "$target" push
fi
echo "staged in $target"
