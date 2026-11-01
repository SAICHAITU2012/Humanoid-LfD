#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$PROJECT_ROOT/vendor/mujoco_menagerie"
if [[ ! -f "$VENDOR/unitree_g1/scene_with_hands.xml" ]]; then
  mkdir -p "$(dirname "$VENDOR")"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$VENDOR"
  git -C "$VENDOR" sparse-checkout set unitree_g1
fi
git -C "$VENDOR" rev-parse HEAD > "$PROJECT_ROOT/vendor/MENAGERIE_COMMIT.txt"
printf 'G1 model: %s\n' "$VENDOR/unitree_g1/scene_with_hands.xml"
printf 'Menagerie commit: %s\n' "$(cat "$PROJECT_ROOT/vendor/MENAGERIE_COMMIT.txt")"
