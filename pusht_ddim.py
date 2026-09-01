"""
Conditional DDIM on Push-T.

Condition on images and agent_pos.
Diffuse the actions 

Push-T Dataset.
Images:
(BATCH_SIZE, OBS_STEPS, CHANNELS, HEIGHT, WIDTH)
[64, 2, 3, 96, 96]

agent_pos (x, y):
(BATCH_SIZE, OBS_STEPS, 2)
[64, 2, 2]

actions (dx, dy):
(BATCH_SIZE, PREDICTION_HORIZON, 2)
[64, 16, 2]
"""

from ddpm import *
from pusht_data import create_pusht_data_loaders

del DummyEpsModel, build_model

# Conditioning parameters
# Images
CHANNELS = 3
ROWS = 96
COLS = 96

# agent_pos (x, y)
AGENT_POS_DIM = 2

# action parameters - this is what we diffuse over
ACTION_DIM = 2

# prediction horizon - how many action steps ahead we predict
PREDICTION_HORIZON = 16
# number of observation steps - how many past steps are used as input (frames and agent_pos)
N_OBS_STEPS = 2

SEED = 42
OUTPUT_DIR = Path("pusht_outputs")


import sys
import traceback


def force_traceback(exctype, value, tb):
    traceback.print_exception(exctype, value, tb)
    sys.exit(1)

sys.excepthook = force_traceback

"""
Architecture for the condtional DDIM model.

R^{ACTION_DIM + PREDICTION_HORIZON * (CHANNELS*ROWS*COLS + AGENT_POS_DIM)} -> R^{ACTION_DIM}

CNN for the image conditioning -> Linear down to ACTION_DIM
FC for the AGENT_POS_DIM -> Linear down to ACTION_DIM
FC for the action -> Linear down to ACTION_DIM
Sum the outputs from all three branches together.
Then +2 FC layers.
"""

def save_checkpoint(model: DiffusionModel, path: Path) -> None:
    """Save model weights together with the settings needed for sampling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "timesteps": model.timesteps,
                "action_shape": model.action_shape,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
                "agent_pos_dim": AGENT_POS_DIM,
                "action_dim": ACTION_DIM,
                "prediction_horizon": PREDICTION_HORIZON,
                "n_obs_steps": N_OBS_STEPS,
                "channels": CHANNELS,
                "rows": ROWS,
                "cols": COLS,
            },
        },
        path,
    )

class ConditionalEpsModel(nn.Module):
    """Conditional noise predictor for DDIM."""

    def __init__(self, trunk_dim: int = 256) -> None:
        super().__init__()
        # (B, PREDICTION_HORIZON*ACTION_DIM) > (B, trunk_dim)
        self.action_branch = nn.Sequential(
            nn.Linear(PREDICTION_HORIZON*ACTION_DIM, 64),
            nn.LeakyReLU(),
            nn.Linear(64, trunk_dim),
        )

        # (B, N_OBS_STEPS*AGENT_POS_DIM) > (B, trunk_dim)
        self.agent_pos_branch = nn.Sequential(
            nn.Linear(N_OBS_STEPS*AGENT_POS_DIM, 64),
            nn.LeakyReLU(),
            nn.Linear(64, trunk_dim),
        )

        # (B, N_OBS_STEPS, CHANNELS, ROWS, COLS) > (B, trunk_dim)
        self.image_branch = nn.Sequential(
            conv_block(N_OBS_STEPS*CHANNELS, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, CHANNELS, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(CHANNELS*ROWS*COLS, trunk_dim),
        )

        # Trunk network to combine the outputs from the three branches
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 4*trunk_dim),
            nn.LeakyReLU(),
            nn.Linear(4*trunk_dim, 8*trunk_dim),
            nn.LeakyReLU(),
            nn.Linear(8*trunk_dim, 4*trunk_dim),
            nn.LeakyReLU(),
            nn.Linear(4*trunk_dim, PREDICTION_HORIZON * ACTION_DIM),
        )

    def forward(self, action: torch.Tensor, pos: torch.Tensor, image: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:
        # The notebook model accepts t but does not use it yet.
        del t

        action_flattened = action.flatten(1,2) # (B, PREDICTION_HORIZON, ACTION_DIM) > (B, PREDICTION_HORIZON * ACTION_DIM)
        x = self.action_branch(action_flattened) # (B, PREDICTION_HORIZON * ACTION_DIM) > (B, trunk_dim)
    
        pos_flattened = pos.flatten(1,2) # (B, N_OBS_STEPS, AGENT_POS_DIM) > (B, N_OBS_STEPS * AGENT_POS_DIM)
        y = self.agent_pos_branch(pos_flattened) # (B, N_OBS_STEPS * AGENT_POS_DIM) > (B, trunk_dim)

        image_flattened = image.flatten(1,2) # (B, N_OBS_STEPS, CHANNELS, ROWS, COLS) > (B, N_OBS_STEPS * CHANNELS, ROWS, COLS)
        z = self.image_branch(image_flattened) # (B, N_OBS_STEPS * CHANNELS, ROWS, COLS) > (B, trunk_dim)

        x0 = x + y + z
        pred = self.trunk(x0)
        eps_hat = pred.view(-1, PREDICTION_HORIZON, ACTION_DIM) # (B, PREDICTION_HORIZON*ACTION_DIM) > (B, PREDICTION_HORIZON, ACTION_DIM)
        assert eps_hat.shape == action.shape, f"Shape mismatch: eps_hat.shape={eps_hat.shape}, action.shape={action.shape}"
        return eps_hat


class ConditionalDiffusionModel(nn.Module):
    """Wrap a noise predictor with DDPM training and DDIM sampling operations."""

    def __init__(
        self,
        device: str,
        action_shape: Tuple[int, int],
        beta_start: float,
        beta_end: float,
        eps_model: nn.Module,
        timesteps: int = TIMESTEPS,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        prediction_horizon, action_dim = action_shape
        if prediction_horizon <= 0 or action_dim <= 0:
            raise ValueError("All action dimensions must be positive")

        self.action_shape = action_shape
        self.timesteps = timesteps
        self.eps_model = eps_model
        self.loss_fn = loss_fn or nn.MSELoss()
        for name, value in schedule(beta_start, beta_end, timesteps, device).items():
            self.register_buffer(name, value)

    def forward(self, x_0: torch.Tensor, pos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Return the noise-prediction loss for a batch of clean actions."""
        batch_size = x_0.shape[0]
        t = torch.randint(
            low=1,
            high=self.timesteps + 1,
            size=(batch_size,),
            device=x_0.device,
        )
        eps = torch.normal(mean=torch.zeros_like(x_0), std=1.0)

        noisy_actions = (
            self.sqrt_alpha_bar_t[t, None, None] * x_0
            + self.sqrt_one_minus_alpha_bar_t[t, None, None]
            * eps
        )
        predicted_eps = self.eps_model(action=noisy_actions, pos=pos, image=image, t=t)
        assert predicted_eps.shape == eps.shape, f"Shape mismatch: predicted_eps.shape={predicted_eps.shape}, eps.shape={eps.shape}"
        return self.loss_fn(eps, predicted_eps)

    def sample(self, pos: torch.Tensor, image: torch.Tensor, S: int = 5, eta: float = 1.0, batch_size: int = 1) -> torch.Tensor:
        # add in batch dimension - should already be there
        if pos.dim() == 1:
            raise ValueError
            pos = pos.unsqueeze(0)
        if image.dim() == 3:
            raise ValueError
            image = image.unsqueeze(0)

        # 1. Construct the sampling trajectory
        # tau_i = floor(i*T / S) for i = 0, 1, ..., S; S is the number of steps
        tau_i = [math.floor(i * self.timesteps / S) for i in range(S + 1)]
        # tau_0 = 0
        tau_i = [0] + tau_i[1:]  # Ensure tau_0 = 0

        # 2. Sample the initial noise
        # x_T ~ N(0, I)
        x_t = torch.normal(
            mean=torch.zeros(
                batch_size, *self.action_shape, device=self.beta_t.device
            ),
            std=1.0,
        )

        # 3. Iterate through the trajectory and apply the DDIM update rule
        # for i = S, S-1, ..., 1:
        for i in range(S, 0, -1):
            # t := tau_i
            # s := tau_{i-1}
            t = tau_i[i]
            s = tau_i[i - 1]

            # 3a. Pred noise.
            # eps_hat := eps_model(x_t, t)
            t_batch = torch.full(
                (batch_size,), t, device=x_t.device, dtype=torch.long
            )
            eps_hat = self.eps_model(action=x_t, pos=pos, image=image, t=t_batch)

            # 3b. Pred. clean sample estimate
            # x_0_hat := (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
            x_0_hat = (x_t - torch.sqrt(1 - self.alpha_bar_t[t]) * eps_hat) / torch.sqrt(self.alpha_bar_t[t])

            # 3c. Compute sigma_ts
            # sigma_ts := eta * sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_s)
            sigma_ts = eta * torch.sqrt((1 - self.alpha_bar_t[s]) / (1 - self.alpha_bar_t[t])) * torch.sqrt(1 - self.alpha_bar_t[t] / self.alpha_bar_t[s])

            # 3d. Sample noise; z ~ N(0, I) if s > 1 else z = 0
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)

            # 3e. Update x_s
            # x_s := sqrt(alpha_bar_s) * x_0_hat + sqrt(1-alpha_bar_s - sigma_ts**2) * eps_hat + sigma_ts * z
            x_t = torch.sqrt(self.alpha_bar_t[s]) * x_0_hat + torch.sqrt(1 - self.alpha_bar_t[s] - torch.pow(sigma_ts, 2)) * eps_hat + sigma_ts * z

        return x_t


def build_model(device: torch.device, timesteps: int) -> DiffusionModel:
    """Construct the noise predictor and diffusion wrapper."""
    eps_model = ConditionalEpsModel()
    return ConditionalDiffusionModel(
        device=str(device),
        action_shape=(PREDICTION_HORIZON, ACTION_DIM),
        beta_start=BETA_START,
        beta_end=BETA_END,
        timesteps=timesteps,
        eps_model=eps_model,
    ).to(device)


def train(
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    timesteps: int = TIMESTEPS,
    device_name: str | None = None,
    checkpoint: Path = CHECKPOINT_PATH,
    seed: int = SEED,
    max_batches: int | None = None,
) -> ConditionalDiffusionModel:
    """Train on Push-T, optionally limiting each epoch for quick experiments."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or timesteps <= 0:
        raise ValueError(
            "epochs, batch_size, learning_rate, and timesteps must be positive"
        )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    device = select_device(device_name)
    set_seed(seed)
    print(f"device={str(device)!r}")

    train_loader, val_loader = create_pusht_data_loaders(
        zarr_path="data/pusht/pusht/pusht_cchi_v7_replay.zarr",
        batch_size=batch_size,
        n_obs_steps=N_OBS_STEPS,
        prediction_horizon=PREDICTION_HORIZON,
        seed=seed,
        device=device,
    )
    model = build_model(device, timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"epoch={epoch}")
        model.train()
        epoch_loss: float | None = None
        for step, batch in enumerate(train_loader):
            if max_batches is not None and step >= max_batches:
                break
            if 1:
                print(f"step={step}")
            images = batch["obs"]["image"]
            agent_pos = batch["obs"]["agent_pos"]
            actions = batch["action"]
            images = images.to(
                device, non_blocking=device.type == "cuda"
            )
            agent_pos = agent_pos.to(
                device, non_blocking=device.type == "cuda"
            )
            actions = actions.to(
                device, non_blocking=device.type == "cuda"
            )

            optimizer.zero_grad()
            loss = model(image=images, pos=agent_pos, x_0=actions) # calls ConditionalDiffusionModel.forward()
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
            save_samples(model, OUTPUT_DIR / f"epoch_{completed_epochs}.png", val_loader=val_loader, device=device)

    return model


def save_samples(
    model: ConditionalDiffusionModel, path: Path, val_loader, method: str = "ddpm", device: torch.device | None = None, num_samples: int = 10,
) -> torch.Tensor:
    """Generate samples and save them as an image grid."""
    del method # ignore argument
    del num_samples # ignore argment

    print(f"{device=}")

    # Load images and pos from the val set?
    batch = next(iter(val_loader))
    images = batch["obs"]["image"].to(device, non_blocking=device.type == "cuda")
    agent_pos = batch["obs"]["agent_pos"].to(device, non_blocking=device.type == "cuda")

    num_samples: int = images.shape[0]

    model.eval()
    with torch.no_grad():
        args = {
            "pos": agent_pos,
            "image": images,
            "batch_size": images.shape[0],
        }
        samples = model.sample(**args).detach().cpu()
        if samples.ndim == 3:
            samples = samples.unsqueeze(0)
        samples = samples.clamp(-1.0, 1.0)
        samples = (samples + 1.0) / 2.0

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = math.ceil(math.sqrt(num_samples))
    rows = math.ceil(num_samples / columns)
    print(f"Saving {num_samples} samples to {path!r} in a {rows}x{columns} grid...")
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
                seed=args.seed if args.seed is not None else SEED,
                max_batches=args.max_batches,
            )
        else:
            sample_from_checkpoint(
                checkpoint=args.checkpoint,
                timesteps=args.timesteps,
                device_name=args.device,
                seed=args.seed if args.seed is not None else SEED,
                output=args.output,
                method=args.method,
                save_sample_method=save_samples,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise error
        build_parser().error(str(error))


if __name__ == "__main__":
    main()
