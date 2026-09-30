#!/usr/bin/env bash
set -euo pipefail

REPO="Simo-03/football-player-detection"
DEST="models"
MANUAL_MSG="Manual fallback: go to https://github.com/$REPO/releases/latest, download the .pt files, and place them in $DEST/"

# Create models directory
mkdir -p "$DEST"

# Check gh CLI is installed
if ! command -v gh &>/dev/null; then
  echo "Error: GitHub CLI (gh) is not installed."
  echo "Install it: https://cli.github.com/"
  echo ""
  echo "$MANUAL_MSG"
  exit 1
fi

# Check gh is authenticated
if ! gh auth status &>/dev/null; then
  echo "Error: GitHub CLI is not authenticated."
  echo "Run: gh auth login"
  echo ""
  echo "$MANUAL_MSG"
  exit 1
fi

# Download model weights from latest release
echo "Downloading model weights from $REPO ..."
if gh release download --repo "$REPO" --pattern "*.pt" --dir "$DEST"; then
  echo ""
  echo "Downloaded models:"
  ls -lh "$DEST"/*.pt
else
  echo ""
  echo "Download failed."
  echo "$MANUAL_MSG"
  exit 1
fi
