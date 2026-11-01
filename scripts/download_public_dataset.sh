#!/usr/bin/env bash
# Downloads the public G1 Dex3 cup dataset used by the Experiment-A baseline.
# Source: https://huggingface.co/datasets/lewislf/G1_Dex3_pick_white_cup_right8_1squeeze (MIT license)
# Size on disk: ~5.1 GB. Install the CLI first: pip install -U "huggingface_hub[cli]"
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$PROJECT_ROOT/data/public/g1_cup_mit"
if [[ -f "$DEST/meta/info.json" ]]; then
  echo "Dataset already present: $DEST"
  exit 0
fi
hf download lewislf/G1_Dex3_pick_white_cup_right8_1squeeze --repo-type dataset --local-dir "$DEST"
echo "Done. Experiment A: .venv/bin/python src/train_public_baseline.py"
echo "Replay:            .venv/bin/python src/replay_public_episode.py"
