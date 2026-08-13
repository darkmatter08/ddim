"""A small, standalone DDPM implementation for MNIST."""

import argparse
import math
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 2e-4
N_CHANNELS = 1
ROWS = 28
COLS = 28
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
SAMPLE_EVERY_EPOCHS = 5

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = CHECKPOINT_DIR / "ddpm.pt"
SCHEDULE_BUFFER_NAMES = {
    "beta_t",
    "alpha_t",
    "alpha_bar_t",
    "one_over_sqrt_alpha_t",
    "sqrt_alpha_bar_t",
    "sqrt_one_minus_alpha_bar_t",
    "sqrt_beta_t",
}


def select_device(requested: str | None = None) -> torch.device:
    """Return an explicitly requested device or the best available device."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but CUDA is not available")
    if device.type == "mps" and not mps_available:
        raise ValueError("MPS was requested, but MPS is not available")
    return device


def set_seed(seed: int) -> None:
    """Seed PyTorch on the CPU and all available CUDA devices."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def schedule(
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    timesteps: int = TIMESTEPS,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Precompute values derived from the linear beta schedule."""
    beta_t = torch.zeros(timesteps + 1, device=device)
    beta_t[1:] = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "alpha_bar_t": alpha_bar_t,
        "one_over_sqrt_alpha_t": torch.rsqrt(alpha_t),
        "sqrt_alpha_bar_t": torch.sqrt(alpha_bar_t),
        "sqrt_one_minus_alpha_bar_t": torch.sqrt(1.0 - alpha_bar_t),
        "sqrt_beta_t": torch.sqrt(beta_t),
    }


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build one convolution, normalization, and activation block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class DummyEpsModel(nn.Module):
    """A small convolutional network that predicts noise in an image."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(n_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        # The notebook model accepts t but does not use it yet.
        del t
        return self.conv(x)


class DiffusionModel(nn.Module):
    """Wrap a noise predictor with DDPM training and sampling operations."""

    def __init__(
        self,
        device: str,
        image_shape: Tuple[int, int, int],
        beta_start: float,
        beta_end: float,
        eps_model: nn.Module,
        timesteps: int = TIMESTEPS,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        channels, rows, columns = image_shape
        if channels <= 0 or rows <= 0 or columns <= 0:
            raise ValueError("All image dimensions must be positive")

        self.image_shape = image_shape
        self.timesteps = timesteps
        self.eps_model = eps_model
        self.loss_fn = loss_fn or nn.MSELoss()
        for name, value in schedule(beta_start, beta_end, timesteps, device).items():
            self.register_buffer(name, value)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """Return the noise-prediction loss for a batch of clean images."""
        batch_size = x_0.shape[0]
        t = torch.randint(
            low=1,
            high=self.timesteps + 1,
            size=(batch_size,),
            device=x_0.device,
        )
        eps = torch.normal(mean=torch.zeros_like(x_0), std=1.0)

        noisy_images = (
            self.sqrt_alpha_bar_t[t, None, None, None] * x_0
            + self.sqrt_one_minus_alpha_bar_t[t, None, None, None]
            * eps
        )
        predicted_eps = self.eps_model(noisy_images, t)
        return self.loss_fn(eps, predicted_eps)

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """Generate a batch by iteratively applying the reverse process."""
        x_t = torch.normal(
            mean=torch.zeros(
                batch_size, *self.image_shape, device=self.beta_t.device
            ),
            std=1.0,
        )

        for t in range(self.timesteps, 0, -1):
            t_batch = torch.full(
                (batch_size,), t, device=x_t.device, dtype=torch.long
            )
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            x_t = (
                self.one_over_sqrt_alpha_t[t]
                * (
                    x_t
                    - self.beta_t[t]
                    / self.sqrt_one_minus_alpha_bar_t[t]
                    * self.eps_model(x_t, t_batch)
                )
                + self.sqrt_beta_t[t] * z
            )

        return x_t if batch_size != 1 else x_t.squeeze(0)


def create_data_loaders(
    device: str | torch.device, batch_size: int = BATCH_SIZE
) -> tuple[DataLoader, DataLoader]:
    """Download MNIST and return its training and test loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    train_dataset = MNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = MNIST(DATA_DIR, train=False, transform=transform, download=True)
    pin_memory = torch.device(device).type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def save_samples(
    model: DiffusionModel, path: Path, num_samples: int = 2
) -> torch.Tensor:
    """Generate samples and save them as an image grid."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size=num_samples).detach().cpu()
        if samples.ndim == 3:
            samples = samples.unsqueeze(0)
        samples = samples.clamp(-1.0, 1.0)
        samples = (samples + 1.0) / 2.0

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = math.ceil(math.sqrt(num_samples))
    rows = math.ceil(num_samples / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns, 2.5 * rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.08},
    )
    figure.patch.set_facecolor("white")

    for axis in axes.flat:
        axis.axis("off")

    for axis, image in zip(axes.flat, samples):
        if image.shape[0] == 1:
            axis.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image.permute(1, 2, 0))
        axis.axis("on")
        axis.set(xticks=[], yticks=[])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    figure.savefig(path, bbox_inches="tight", pad_inches=0.15, dpi=200)
    plt.close(figure)
    return samples


def save_learning_curve(epoch_losses: list[float], path: Path) -> None:
    """Save end-of-epoch training losses on a logarithmic scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(epoch_losses) + 1)
    plt.plot(epochs, epoch_losses, marker="o")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss (log scale)")
    plt.title("End-of-Epoch Training Loss")
    plt.grid(visible=True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_model(device: torch.device, timesteps: int) -> DiffusionModel:
    """Construct the noise predictor and diffusion wrapper."""
    eps_model = DummyEpsModel(n_channels=N_CHANNELS)
    return DiffusionModel(
        device=str(device),
        image_shape=(N_CHANNELS, ROWS, COLS),
        beta_start=BETA_START,
        beta_end=BETA_END,
        timesteps=timesteps,
        eps_model=eps_model,
    ).to(device)


def save_checkpoint(model: DiffusionModel, path: Path) -> None:
    """Save model weights together with the settings needed for sampling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "timesteps": model.timesteps,
                "image_shape": model.image_shape,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
            },
        },
        path,
    )


def train(
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    timesteps: int = TIMESTEPS,
    device_name: str | None = None,
    checkpoint: Path = CHECKPOINT_PATH,
    seed: int = 0,
    max_batches: int | None = None,
) -> DiffusionModel:
    """Train on MNIST, optionally limiting each epoch for quick experiments."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or timesteps <= 0:
        raise ValueError(
            "epochs, batch_size, learning_rate, and timesteps must be positive"
        )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    device = select_device(device_name)
    set_seed(seed)
    print(f"device={str(device)!r}")

    train_loader, _ = create_data_loaders(device, batch_size)
    model = build_model(device, timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"epoch={epoch}")
        model.train()
        epoch_loss: float | None = None
        for step, (images, _) in enumerate(train_loader):
            if max_batches is not None and step >= max_batches:
                break
            if step % 50 == 0:
                print(f"step={step}")
            images = images.to(
                device, non_blocking=device.type == "cuda"
            )

            optimizer.zero_grad()
            loss = model(images)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.detach().cpu().item()

        if epoch_loss is None:
            raise RuntimeError("training epoch processed no batches")
        epoch_losses.append(epoch_loss)

        save_checkpoint(model, checkpoint)
        save_learning_curve(epoch_losses, OUTPUT_DIR / "learning_curve.png")
        completed_epochs = epoch + 1
        if completed_epochs % SAMPLE_EVERY_EPOCHS == 0:
            print("Sampling...")
            save_samples(model, OUTPUT_DIR / f"epoch_{completed_epochs}.png")

    return model


def sample_from_checkpoint(
    checkpoint: Path,
    *,
    num_samples: int = 2,
    timesteps: int | None = None,
    device_name: str | None = None,
    seed: int = 0,
    output: Path = OUTPUT_DIR / "samples.png",
) -> torch.Tensor:
    """Load a checkpoint, generate samples, and save their image grid."""
    device = select_device(device_name)
    set_seed(seed)
    print(f"device={str(device)!r}")

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = saved["model_state_dict"]
    saved_timesteps = saved.get("config", {}).get("timesteps", TIMESTEPS)

    model_timesteps = timesteps if timesteps is not None else saved_timesteps
    model = build_model(device, model_timesteps)
    if model_timesteps != saved_timesteps:
        state_dict = {
            name: value
            for name, value in state_dict.items()
            if name not in SCHEDULE_BUFFER_NAMES
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_parameters = set(incompatible.missing_keys) - SCHEDULE_BUFFER_NAMES
    if incompatible.unexpected_keys or missing_parameters:
        raise RuntimeError(
            "checkpoint is incompatible with this model: "
            f"missing={sorted(missing_parameters)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    samples = save_samples(model, output, num_samples=num_samples)
    print(f"samples={str(output)!r}")
    return samples


def positive_int(value: str) -> int:
    """Argparse converter for integer options that must be positive."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """Argparse converter for floating-point options that must be positive."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train on MNIST")
    train_parser.add_argument("--epochs", type=positive_int, default=EPOCHS)
    train_parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    train_parser.add_argument(
        "--learning-rate", type=positive_float, default=LEARNING_RATE
    )
    train_parser.add_argument("--timesteps", type=positive_int, default=TIMESTEPS)
    train_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    train_parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument(
        "--max-batches",
        type=positive_int,
        help="maximum batches per epoch (omit to use the complete dataset)",
    )

    sample_parser = subparsers.add_parser("sample", help="sample a checkpoint")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--num-samples", type=positive_int, default=2)
    sample_parser.add_argument(
        "--timesteps",
        type=positive_int,
        help="override the checkpoint's diffusion timesteps",
    )
    sample_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "samples.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "train":
            train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                timesteps=args.timesteps,
                device_name=args.device,
                checkpoint=args.checkpoint,
                seed=args.seed,
                max_batches=args.max_batches,
            )
        else:
            sample_from_checkpoint(
                checkpoint=args.checkpoint,
                num_samples=args.num_samples,
                timesteps=args.timesteps,
                device_name=args.device,
                seed=args.seed,
                output=args.output,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        build_parser().error(str(error))


if __name__ == "__main__":
    main()
