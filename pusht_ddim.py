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

actions (x, y):
(BATCH_SIZE, PREDICTION_HORIZON, 2)
[64, 16, 2]
"""

from torch import ne

from ddpm import *
from pusht_data import (
    PUSHT_WORKSPACE_LOWER,
    PUSHT_WORKSPACE_UPPER,
    PushTNormalizer,
    create_pusht_data_loaders,
)

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
VAL_RATIO = 0.02

BATCH_SIZE = 64
SEED = 42
OUTPUT_DIR = Path("pusht_outputs")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "pusht_ddim.pt"
PUSHT_ZARR_PATH = Path("data/pusht/pusht_cchi_v7_replay.zarr")


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

def save_checkpoint(
    model: DiffusionModel,
    path: Path,
    normalizer: PushTNormalizer,
    *,
    data_seed: int = SEED,
) -> None:
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
                "normalizer": normalizer.to_config(),
                "data_seed": data_seed,
                "val_ratio": VAL_RATIO,
            },
        },
        path,
    )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")

        half_dim = dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim)
            / (half_dim - 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


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

        self.time_branch = nn.Sequential(
            SinusoidalTimeEmbedding(trunk_dim),
            nn.Linear(trunk_dim, trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim, trunk_dim),
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
        action_flattened = action.flatten(1,2) # (B, PREDICTION_HORIZON, ACTION_DIM) > (B, PREDICTION_HORIZON * ACTION_DIM)
        x = self.action_branch(action_flattened) # (B, PREDICTION_HORIZON * ACTION_DIM) > (B, trunk_dim)
    
        pos_flattened = pos.flatten(1,2) # (B, N_OBS_STEPS, AGENT_POS_DIM) > (B, N_OBS_STEPS * AGENT_POS_DIM)
        y = self.agent_pos_branch(pos_flattened) # (B, N_OBS_STEPS * AGENT_POS_DIM) > (B, trunk_dim)

        image_flattened = image.flatten(1,2) # (B, N_OBS_STEPS, CHANNELS, ROWS, COLS) > (B, N_OBS_STEPS * CHANNELS, ROWS, COLS)
        z = self.image_branch(image_flattened) # (B, N_OBS_STEPS * CHANNELS, ROWS, COLS) > (B, trunk_dim)

        time_embedding = self.time_branch(t)
        x0 = x + y + z + time_embedding
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

        return self.noise_prediction_loss(
            x_0=x_0, pos=pos, image=image, t=t, eps=eps
        )

    def noise_prediction_loss(
        self,
        x_0: torch.Tensor,
        pos: torch.Tensor,
        image: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Return epsilon MSE for explicitly supplied timesteps and noise."""
        batch_size = x_0.shape[0]
        if t.shape != (batch_size,):
            raise ValueError(f"expected t shape [{batch_size}], got {list(t.shape)}")
        if eps.shape != x_0.shape:
            raise ValueError(
                f"expected eps shape {list(x_0.shape)}, got {list(eps.shape)}"
            )

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
    normalize_coordinates: bool = True,
    normalize_images: bool = True,
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

    train_loader, val_loader, normalizer = create_pusht_data_loaders(
        zarr_path=PUSHT_ZARR_PATH,
        batch_size=batch_size,
        n_obs_steps=N_OBS_STEPS,
        prediction_horizon=PREDICTION_HORIZON,
        val_ratio=VAL_RATIO,
        seed=seed,
        device=device,
        normalize_coordinates=normalize_coordinates,
        normalize_images=normalize_images,
    )
    model = build_model(device, timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []
    validation_losses: list[float] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"epoch={epoch}")
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for step, batch in enumerate(train_loader):
            if max_batches is not None and step >= max_batches:
                break
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
            batch_loss = loss.detach().cpu().item()
            current_batch_size = actions.shape[0]
            train_loss_total += batch_loss * current_batch_size
            train_examples += current_batch_size
            if step % 25 == 0:
                print(f"step={step}, loss={batch_loss}")

        if train_examples == 0:
            raise RuntimeError("training epoch processed no batches")
        epoch_loss = train_loss_total / train_examples
        epoch_losses.append(epoch_loss)

        model.eval()
        validation_loss_total = 0.0
        validation_examples = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["obs"]["image"].to(
                    device, non_blocking=device.type == "cuda"
                )
                agent_pos = batch["obs"]["agent_pos"].to(
                    device, non_blocking=device.type == "cuda"
                )
                actions = batch["action"].to(
                    device, non_blocking=device.type == "cuda"
                )
                loss = model(image=images, pos=agent_pos, x_0=actions)
                current_batch_size = actions.shape[0]
                validation_loss_total += loss.detach().cpu().item() * current_batch_size
                validation_examples += current_batch_size
        if validation_examples == 0:
            raise RuntimeError("validation epoch processed no batches")
        validation_loss = validation_loss_total / validation_examples
        validation_losses.append(validation_loss)
        print(
            f"epoch={epoch}, train_loss={epoch_loss:.8f}, "
            f"validation_loss={validation_loss:.8f}"
        )

        save_checkpoint(model, checkpoint, normalizer, data_seed=seed)
        save_learning_curve(
            epoch_losses,
            OUTPUT_DIR / "learning_curve.png",
            validation_losses=validation_losses,
        )
        completed_epochs = epoch + 1
        if completed_epochs % SAMPLE_EVERY_EPOCHS == 0:
            print("Sampling...")
            save_samples(
                model,
                OUTPUT_DIR / f"epoch_{completed_epochs}.png",
                val_loader=val_loader,
                normalizer=normalizer,
                device=device,
            )

    return model


def overfit_fixed_batch(
    *,
    steps: int = 100,
    batch_size: int = 2,
    learning_rate: float = LEARNING_RATE,
    timesteps: int = TIMESTEPS,
    device_name: str | None = None,
    seed: int = SEED,
    output: Path = OUTPUT_DIR / "fixed_batch_overfit.png",
) -> list[float]:
    """Overfit one deterministic batch/timestep/noise target as a wiring test."""
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0 or timesteps <= 0:
        raise ValueError("steps, batch size, learning rate, and timesteps must be positive")

    device = select_device(device_name)
    set_seed(seed)
    print(f"device={str(device)!r}")
    train_loader, _, _ = create_pusht_data_loaders(
        zarr_path=PUSHT_ZARR_PATH,
        batch_size=batch_size,
        n_obs_steps=N_OBS_STEPS,
        prediction_horizon=PREDICTION_HORIZON,
        val_ratio=VAL_RATIO,
        seed=seed,
        device=device,
    )
    fixed_batch = next(iter(train_loader))
    images = fixed_batch["obs"]["image"].to(device)
    agent_pos = fixed_batch["obs"]["agent_pos"].to(device)
    actions = fixed_batch["action"].to(device)

    model = build_model(device, timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    fixed_t = torch.randint(
        low=1,
        high=timesteps + 1,
        size=(actions.shape[0],),
        device=device,
    )
    fixed_eps = torch.randn_like(actions)
    print(f"fixed_t={fixed_t.detach().cpu().tolist()}")

    losses: list[float] = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.noise_prediction_loss(
            x_0=actions,
            pos=agent_pos,
            image=images,
            t=fixed_t,
            eps=fixed_eps,
        )
        losses.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1}, loss={losses[-1]:.8f}")

    with torch.no_grad():
        final_loss = model.noise_prediction_loss(
            x_0=actions,
            pos=agent_pos,
            image=images,
            t=fixed_t,
            eps=fixed_eps,
        ).detach().cpu().item()
    losses.append(final_loss)
    print(f"final_loss={final_loss:.8f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.plot(range(len(losses)), losses)
    plt.yscale("log")
    plt.xlabel("optimizer step")
    plt.ylabel("fixed-target epsilon MSE (log scale)")
    plt.title("Fixed-Batch Overfit Diagnostic")
    plt.grid(visible=True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"curve={str(output)!r}")
    return losses


def save_samples(
    model: ConditionalDiffusionModel,
    path: Path,
    val_loader,
    normalizer: PushTNormalizer,
    method: str = "ddpm",
    device: torch.device | None = None,
    num_samples: int = 10,
) -> torch.Tensor:
    """Generate samples and save them as an image grid."""
    del method # ignore argument
    del num_samples # ignore argment

    print(f"{device=}")

    # Load images and pos from the val set?
    batch = next(iter(val_loader))
    images = batch["obs"]["image"].to(device, non_blocking=device.type == "cuda")
    agent_pos = batch["obs"]["agent_pos"].to(device, non_blocking=device.type == "cuda")
    ground_truth_actions = batch["action"]

    num_samples: int = images.shape[0]

    model.eval()
    with torch.no_grad():
        args = {
            "pos": agent_pos,
            "image": images,
            "batch_size": images.shape[0],
        }
        samples = model.sample(**args).detach().cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    images_cpu = normalizer.unnormalize_images(images.detach().cpu())
    agent_pos_cpu = normalizer.unnormalize_coordinates(agent_pos.detach().cpu())
    action_samples_cpu = normalizer.unnormalize_coordinates(samples)
    ground_truth_actions_cpu = normalizer.unnormalize_coordinates(
        ground_truth_actions.detach().cpu()
    )
    if samples.shape != (num_samples, PREDICTION_HORIZON, ACTION_DIM):
        raise ValueError(
            "expected sampled actions with shape "
            f"[{num_samples}, {PREDICTION_HORIZON}, {ACTION_DIM}], "
            f"got {list(samples.shape)}"
        )
    if ground_truth_actions.shape != samples.shape:
        raise ValueError(
            "expected ground-truth actions to match sampled actions, "
            f"got ground truth {list(ground_truth_actions.shape)} and "
            f"samples {list(samples.shape)}"
        )

    # Push-T coordinates are expressed in the original 512x512 workspace,
    # whereas the policy observations are resized to 96x96.
    workspace_lower = torch.tensor(PUSHT_WORKSPACE_LOWER)
    workspace_size = torch.tensor(PUSHT_WORKSPACE_UPPER) - workspace_lower
    suffix = path.suffix or ".png"

    for sample_index in range(num_samples):
        figure, axes = plt.subplots(
            1,
            N_OBS_STEPS,
            figsize=(5.0 * N_OBS_STEPS, 5.0),
            squeeze=False,
            gridspec_kw={"wspace": 0.05},
        )
        axes = axes[0]

        for obs_index, axis in enumerate(axes):
            observation = images_cpu[sample_index, obs_index]
            height, width = observation.shape[-2:]
            axis.imshow(observation.permute(1, 2, 0).clamp(0.0, 1.0))
            axis.set(xticks=[], yticks=[])
            axis.set_xlim(-0.5, width - 0.5)
            axis.set_ylim(height - 0.5, -0.5)
            axis.set_title(f"Observation {obs_index + 1}")

        height, width = images_cpu[sample_index, 0].shape[-2:]
        coordinate_scale = torch.tensor(
            [width / workspace_size[0], height / workspace_size[1]]
        )
        pos1, pos2 = (
            agent_pos_cpu[sample_index] - workspace_lower
        ) * coordinate_scale

        axes[0].scatter(
            pos1[0], pos1[1], c="red", s=55, edgecolors="white", linewidths=1.0
        )
        axes[1].scatter(
            pos2[0], pos2[1], c="blue", s=55, edgecolors="white", linewidths=1.0
        )

        action_points = (
            action_samples_cpu[sample_index] - workspace_lower
        ) * coordinate_scale
        trajectory = torch.cat((pos2.unsqueeze(0), action_points), dim=0)
        axes[1].plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="blue",
            linewidth=1.5,
            alpha=0.8,
            label="prediction horizon",
        )
        axes[1].scatter(
            action_points[:, 0],
            action_points[:, 1],
            c="blue",
            s=12,
            alpha=0.8,
        )
        axes[1].scatter(
            action_points[0, 0],
            action_points[0, 1],
            c="cyan",
            marker="o",
            s=60,
            edgecolors="blue",
            linewidths=1.5,
            zorder=5,
            label="prediction: first action",
        )
        axes[1].annotate(
            "",
            xy=tuple(action_points[0].tolist()),
            xytext=tuple(pos2.tolist()),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "blue",
                "linewidth": 2.5,
                "alpha": 1.0,
                "mutation_scale": 16,
            },
            zorder=4,
        )
        axes[1].annotate(
            "P1",
            xy=tuple(action_points[0].tolist()),
            xytext=(6, -12),
            textcoords="offset points",
            color="blue",
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

        ground_truth_points = (
            ground_truth_actions_cpu[sample_index] - workspace_lower
        ) * coordinate_scale
        ground_truth_trajectory = torch.cat(
            (pos2.unsqueeze(0), ground_truth_points), dim=0
        )
        axes[1].plot(
            ground_truth_trajectory[:, 0],
            ground_truth_trajectory[:, 1],
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="ground-truth horizon",
        )
        axes[1].scatter(
            ground_truth_points[:, 0],
            ground_truth_points[:, 1],
            c="green",
            marker="x",
            s=18,
            alpha=0.8,
        )
        axes[1].scatter(
            ground_truth_points[0, 0],
            ground_truth_points[0, 1],
            c="lime",
            marker="X",
            s=70,
            edgecolors="darkgreen",
            linewidths=1.2,
            zorder=5,
            label="ground truth: first action",
        )
        axes[1].annotate(
            "",
            xy=tuple(ground_truth_points[0].tolist()),
            xytext=tuple(pos2.tolist()),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "darkgreen",
                "linestyle": "--",
                "linewidth": 2.5,
                "alpha": 1.0,
                "mutation_scale": 16,
            },
            zorder=4,
        )
        axes[1].annotate(
            "GT1",
            xy=tuple(ground_truth_points[0].tolist()),
            xytext=(6, 8),
            textcoords="offset points",
            color="darkgreen",
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )
        axes[1].legend(loc="upper right")

        output_path = path.with_name(f"{path.stem}_{sample_index:03d}{suffix}")
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.15, dpi=200)
        plt.close(figure)

    print(f"Saved {num_samples} sample visualizations beside {path!r}")
    return samples


def sample_from_checkpoint(
    checkpoint: Path,
    *,
    num_samples: int = 2,
    timesteps: int | None = None,
    device_name: str | None = None,
    seed: int = SEED,
    output: Path = OUTPUT_DIR / "samples.png",
    method: str = "ddpm",
    save_sample_method: Callable = save_samples,
) -> torch.Tensor:
    """Load a Push-T checkpoint and visualize normalized validation samples."""
    device = select_device(device_name)
    set_seed(seed)
    print(f"device={str(device)!r}")

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = saved["model_state_dict"]
    config = saved.get("config", {})
    saved_timesteps = config.get("timesteps", TIMESTEPS)
    normalizer = PushTNormalizer.from_config(config.get("normalizer"))
    data_seed = config.get("data_seed", SEED)
    val_ratio = config.get("val_ratio", VAL_RATIO)
    _, val_loader, _ = create_pusht_data_loaders(
        zarr_path=PUSHT_ZARR_PATH,
        batch_size=num_samples,
        n_obs_steps=N_OBS_STEPS,
        prediction_horizon=PREDICTION_HORIZON,
        val_ratio=val_ratio,
        seed=data_seed,
        device=device,
        normalizer=normalizer,
    )

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
    samples = save_sample_method(
        model,
        output,
        val_loader=val_loader,
        normalizer=normalizer,
        num_samples=num_samples,
        method=method,
        device=device,
    )
    print(f"samples={str(output)!r}")
    return samples


def build_parser() -> argparse.ArgumentParser:
    """Build the Push-T training and sampling command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train on Push-T")
    train_parser.add_argument("--epochs", type=positive_int, default=EPOCHS)
    train_parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    train_parser.add_argument(
        "--learning-rate", type=positive_float, default=LEARNING_RATE
    )
    train_parser.add_argument("--timesteps", type=positive_int, default=TIMESTEPS)
    train_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    train_parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.add_argument(
        "--max-batches",
        type=positive_int,
        help="maximum batches per epoch (omit to use the complete dataset)",
    )
    train_parser.add_argument(
        "--no-normalize-coordinates",
        dest="normalize_coordinates",
        action="store_false",
        help="return raw workspace coordinates from the dataset",
    )
    train_parser.add_argument(
        "--no-normalize-images",
        dest="normalize_images",
        action="store_false",
        help="return images in [0, 1] without channel standardization",
    )

    overfit_parser = subparsers.add_parser(
        "overfit", help="overfit one fixed batch/timestep/noise target"
    )
    overfit_parser.add_argument("--steps", type=positive_int, default=100)
    overfit_parser.add_argument("--batch-size", type=positive_int, default=2)
    overfit_parser.add_argument(
        "--learning-rate", type=positive_float, default=LEARNING_RATE
    )
    overfit_parser.add_argument("--timesteps", type=positive_int, default=TIMESTEPS)
    overfit_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    overfit_parser.add_argument("--seed", type=int, default=SEED)
    overfit_parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "fixed_batch_overfit.png"
    )

    sample_parser = subparsers.add_parser("sample", help="sample a Push-T checkpoint")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--num-samples", type=positive_int, default=2)
    sample_parser.add_argument(
        "--timesteps",
        type=positive_int,
        help="override the checkpoint's diffusion timesteps",
    )
    sample_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    sample_parser.add_argument("--seed", type=int, default=SEED)
    sample_parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "samples.png")
    sample_parser.add_argument("--method", default="ddim_skip")
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
                seed=args.seed if args.seed is not None else SEED,
                max_batches=args.max_batches,
                normalize_coordinates=args.normalize_coordinates,
                normalize_images=args.normalize_images,
            )
        elif args.command == "overfit":
            overfit_fixed_batch(
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                timesteps=args.timesteps,
                device_name=args.device,
                seed=args.seed,
                output=args.output,
            )
        else:
            sample_from_checkpoint(
                checkpoint=args.checkpoint,
                num_samples=args.num_samples,
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
