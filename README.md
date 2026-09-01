# Minimal MNIST DDPM

This repository contains a small, educational implementation of a denoising
diffusion probabilistic model (DDPM) trained on MNIST. It is currently a
standalone Python conversion of `superminddpm.ipynb`.

## Setup

Create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it in zsh or bash:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python ddpm.py train
```

The command downloads MNIST into `data/`, writes a checkpoint to
`checkpoints/ddpm.pt`, and saves generated samples and the learning curve in
`outputs/`. By default it trains on the complete dataset. To reproduce the
notebook's quick three-batch-per-epoch experiment, run:

```bash
python ddpm.py train --max-batches 3
```

Generate images from a trained checkpoint with:

```bash
python ddpm.py sample --checkpoint checkpoints/ddpm.pt --num-samples 9
```

Both commands automatically select CUDA, then Apple MPS, then CPU. Override
that selection with, for example, `--device cpu`. Run either command with
`--help` to see options for epochs, batch size, learning rate, diffusion
timesteps, checkpoint paths, sample count, seed, and output path.

### Long-running GPU training in tmux

The following is the command used to launch the 100-epoch cloud GPU run. Run it
from the repository root after creating `.venv` and installing the dependencies:

```bash
mkdir -p checkpoints outputs .matplotlib-cache

tmux new-session -d -s ddpm-h100 "bash -lc 'cd \"$PWD\" && set -o pipefail; env MPLBACKEND=Agg MPLCONFIGDIR=\"$PWD/.matplotlib-cache\" PYTHONUNBUFFERED=1 .venv/bin/python -u ddpm.py train --device cuda --epochs 100 --batch-size 128 --timesteps 1000 --checkpoint checkpoints/ddpm-100-epochs.pt 2>&1 | while IFS= read -r line; do printf \"%s %s\\n\" \"\$(date +%s.%N)\" \"\$line\"; done | tee run-100-epochs.log'"
```

This launches a detached tmux session named `ddpm-h100`, so training continues
if the SSH connection closes. `PYTHONUNBUFFERED=1` and Python's `-u` option both
disable Python's standard output and error buffering. This is intentionally
redundant: progress is written to the log promptly even when stdout is flowing
through pipes rather than a terminal. `2>&1` captures errors in the same log,
and the line-reading loop prefixes every emitted line with a Unix timestamp.
`tee` writes the timestamped output to `run-100-epochs.log` while keeping it
visible in tmux. Finally, `set -o pipefail` makes the pipeline fail if training
fails instead of reporting only the status of `tee`. The large number at the
start of each log line is seconds since the Unix epoch; subtract timestamps on
successive `epoch=` lines to measure epoch duration.

Inspect the run without stopping it:

```bash
tail -f run-100-epochs.log
tmux attach -t ddpm-h100
```

To check that the training process is still alive and inspect GPU utilization
from another SSH shell:

```bash
ps -eo pid,etime,args | grep '[.]venv/bin/python -u ddpm.py train'
nvidia-smi
```

### Sampling while training continues

Before sampling, copy the current checkpoint to a snapshot. The trainer
rewrites its checkpoint after each epoch; sampling from the snapshot prevents a
later checkpoint update from changing the file while the sampler is reading it.

```bash
mkdir -p checkpoints/snapshots outputs .matplotlib-cache
cp checkpoints/ddpm-100-epochs.pt checkpoints/snapshots/ddpm-sampling.pt

set -o pipefail
env MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.matplotlib-cache" PYTHONUNBUFFERED=1 \
  .venv/bin/python -u ddpm.py sample \
  --device cuda \
  --checkpoint checkpoints/snapshots/ddpm-sampling.pt \
  --num-samples 9 \
  --output outputs/h100-samples-9.png 2>&1 | tee sample-9.log
```

This writes a grid of nine generated images to
`outputs/h100-samples-9.png`. It uses the same unbuffered Python and
pipeline-error settings as the training command. Sampling shares the GPU with
training, so training may slow briefly while both processes are active.

Check its log and output with:

```bash
tail -f sample-9.log
ls -lh outputs/h100-samples-9.png
```

When finished, leave the virtual environment with:

```bash
deactivate
```

The current code preserves the notebook's model and diffusion behavior.
Corrections that bring its schedule, timestep handling, model, and sampling
equations in line with the DDPM paper will be made in a subsequent step.

## Push-T data loader

Download and extract Stanford's image Push-T replay from the repository root:

```bash
mkdir -p data
curl -L https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip \
  -o data/pusht.zip
unzip data/pusht.zip -d data
```

Then create episode-disjoint loaders that return observation history and
strictly current/future action targets:

```python
from pusht_data import create_pusht_data_loaders

train_loader, val_loader, normalizer = create_pusht_data_loaders(
    "data/pusht/pusht/pusht_cchi_v7_replay.zarr",
    batch_size=64,
    n_obs_steps=2,
    prediction_horizon=16,
    val_ratio=0.02,
    device="cuda",
)

batch = next(iter(train_loader))
print(batch["obs"]["image"].shape)      # [B, 2, 3, 96, 96]
print(batch["obs"]["agent_pos"].shape)  # [B, 2, 2]
print(batch["action"].shape)             # [B, 16, 2]
```

By default, the loaders map agent positions and absolute action coordinates
from the 512x512 workspace to `[-1, 1]`. They also standardize each RGB channel
using mean and standard deviation computed from the original frames in training
episodes only. The same `normalizer` is shared by the training and validation
datasets and can invert both transformations for rollout and visualization.
Pass `normalize_coordinates=False` and/or `normalize_images=False` to inspect
unnormalized data.

For each sample, the final observation is at time `t` and the first returned
action is `a_t`. `prediction_horizon` controls the training target length.
Choose how many predicted actions to execute before replanning separately in
the rollout code.

Before launching a full training run, verify that the model can overfit one
fixed batch, timestep vector, and epsilon target:

```bash
python pusht_ddim.py overfit --device cpu --steps 100 --batch-size 2
```

This writes `pusht_outputs/fixed_batch_overfit.png`. The diagnostic intentionally
tests memorization and is not an estimate of validation or rollout performance.
