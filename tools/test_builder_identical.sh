#!/usr/bin/env bash
# Prove that refactoring generatePDF.py did not change a single byte it emits.
#
# generatePDF.py base64s whatever --model points at, so a small dummy file
# exercises the entire PDF construction path without downloading a GGUF. The
# real models live in models/, which is gitignored, so this is the only version
# of the check that runs anywhere.
#
#   tools/test_builder_identical.sh [git-ref-to-compare-against]

set -euo pipefail
cd "$(dirname "$0")/.."

REF="${1:-HEAD}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A dummy "model": small, fixed, and not all one byte, so the base64 is real.
python3 -c "
import sys
sys.stdout.buffer.write(bytes((i * 37 + 11) % 256 for i in range(4096)))
" > "$WORK/dummy.gguf"

# A dummy llama.js: the real one is 5 MB and its content is irrelevant here.
echo "var __dummy_module__ = 1;" > "$WORK/llama.js"

build() { # build <script-path> <output>
  python3 "$1" \
    --model "$WORK/dummy.gguf" \
    --llama "$WORK/llama.js" \
    --template src/template.js \
    --output "$2" > /dev/null
}

echo "building with $REF version of scripts/generatePDF.py"
git show "$REF:scripts/generatePDF.py" > "$WORK/before.py" 2>/dev/null || {
  echo "could not read scripts/generatePDF.py from $REF" >&2; exit 1; }
build "$WORK/before.py" "$WORK/before.pdf"

echo "building with the working-tree version"
build scripts/generatePDF.py "$WORK/after.pdf"

if cmp -s "$WORK/before.pdf" "$WORK/after.pdf"; then
  echo "PASS  byte-identical ($(wc -c < "$WORK/after.pdf" | tr -d ' ') bytes)"
else
  echo "FAIL  output differs:"
  cmp "$WORK/before.pdf" "$WORK/after.pdf" || true
  exit 1
fi
