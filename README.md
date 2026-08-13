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

When finished, leave the virtual environment with:

```bash
deactivate
```

The current code preserves the notebook's model and diffusion behavior.
Corrections that bring its schedule, timestep handling, model, and sampling
equations in line with the DDPM paper will be made in a subsequent step.
