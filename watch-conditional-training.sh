#!/usr/bin/env bash
set -euo pipefail

remote="lambda-conditional-ddpm"
remote_dir="/home/ubuntu/ddim"
local_dir="$PWD/remote-runs/conditional-ddim-h100"

mkdir -p "$local_dir/checkpoints" "$local_dir/outputs"

while ssh "$remote" \
  "pgrep -f '[.]venv/bin/python -u conditional_ddim.py train' >/dev/null"
do
  echo "Training is running; syncing intermediate outputs..."

  rsync -az --partial \
    "$remote:$remote_dir/outputs/" \
    "$local_dir/outputs/"

  rsync -az --partial \
    "$remote:$remote_dir/run-conditional-ddim-100-epochs.log" \
    "$local_dir/"

  sleep 30
done

echo "Training process has stopped; performing final sync..."

rsync -az --partial \
  "$remote:$remote_dir/outputs/" \
  "$local_dir/outputs/"

rsync -az --partial \
  "$remote:$remote_dir/run-conditional-ddim-100-epochs.log" \
  "$local_dir/"

rsync -az --partial \
  "$remote:$remote_dir/checkpoints/conditional-ddim-100-epochs.pt" \
  "$local_dir/checkpoints/"

if ! ssh "$remote" \
  "test -s '$remote_dir/checkpoints/conditional-ddim-100-epochs.pt' &&
   test -s '$remote_dir/outputs/epoch_100.png'"
then
  echo "ERROR: Training did not produce the expected final artifacts."
  exit 1
fi

echo "Training completed successfully."
echo "Artifacts downloaded to: $local_dir"

# caffeinate -i ./watch-conditional-training.sh