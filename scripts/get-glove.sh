#!/usr/bin/env sh
# Download free pretrained GloVe word vectors (50d, ~66MB gzipped) for the semantic
# demo. The file is public and needs no account or API key. It is downloaded once to
# vectors/ and the demo reads it directly (.gz is fine, no need to unzip).
set -e

DEST_DIR="$(dirname "$0")/../vectors"
DEST="$DEST_DIR/glove-wiki-gigaword-50.gz"
URL="https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
  echo "already present: $DEST"
else
  echo "downloading GloVe 50d (~66MB) ..."
  curl -L --fail "$URL" -o "$DEST"
  echo "saved to $DEST"
fi

echo
echo "run the semantic demo with:"
echo "  go run ./cmd/demo -vectors $DEST"
