"""A small, standalone DDPM implementation for MNIST."""

from ddpm import *

del DummyEpsModel, build_model


class ConditionalDummyEpsModel(nn.Module):
    """A small convolutional network that predicts noise in an image."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, ROWS * COLS) # not one-hot; instead use the label as a scalar value
        self.conv = nn.Sequential(
            conv_block(n_channels + 1, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | int, y: torch.Tensor | int = 0) -> torch.Tensor:
        # The notebook model accepts t but does not use it yet.
        del t
        y = self.linear(y.float().unsqueeze(-1)) # to accomodate batch - todo - i don't understand why necessary.
        # What is the shape of x? (batch, channels, rows, cols)?
        y = y.view(-1, 1, ROWS, COLS)
        x = torch.concat([x, y], dim=1)
        # y = y.view(1, ROWS, COLS)
        # x = torch.concat([x, y], dim=0)
        return self.conv(x)


class ConditionalDiffusionModel(DiffusionModel):
    """Wrap a noise predictor with DDPM training and sampling operations."""

    def forward(self, x_0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        predicted_eps = self.eps_model(noisy_images, t, y=y)
        return self.loss_fn(eps, predicted_eps)

    def sample(self, method: str = "ddpm", **kwargs) -> torch.Tensor:
        del method
        print("Sampling with ddim_skip (method ignored)")
        return self.sample_ddim_skip(**kwargs)

    def sample_ddim_skip(self, S: int = 5, eta: float = 1.0, batch_size: int = 1, y: torch.Tensor | int = 0) -> torch.Tensor:
        # 1. Construct the sampling trajectory
        # tau_i = floor(i*T / S) for i = 0, 1, ..., S; S is the number of steps
        tau_i = [math.floor(i * self.timesteps / S) for i in range(S + 1)]
        # tau_0 = 0
        tau_i = [0] + tau_i[1:]  # Ensure tau_0 = 0

        # 2. Sample the initial noise
        # x_T ~ N(0, I)
        x_t = torch.normal(
            mean=torch.zeros(
                batch_size, *self.image_shape, device=self.beta_t.device
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
            eps_hat = self.eps_model(x_t, t_batch, y=y)

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
    eps_model = ConditionalDummyEpsModel(n_channels=N_CHANNELS)
    return ConditionalDiffusionModel(
        device=str(device),
        image_shape=(N_CHANNELS, ROWS, COLS),
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
    seed: int = 0,
    max_batches: int | None = None,
) -> ConditionalDiffusionModel:
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
        for step, (images, labels) in enumerate(train_loader):
            if max_batches is not None and step >= max_batches:
                break
            if step % 50 == 0:
                print(f"step={step}")
            images = images.to(
                device, non_blocking=device.type == "cuda"
            )
            labels = labels.to(
                device, non_blocking=device.type == "cuda"
            )
            # TODO: verify labels.shape and dtype

            optimizer.zero_grad()
            loss = model(images, y=labels) # calls ConditionalDiffusionModel.forward()
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
            save_samples(model, OUTPUT_DIR / f"epoch_{completed_epochs}.png", device=device)

    return model


def save_samples(
    model: ConditionalDiffusionModel, path: Path, method: str = "ddpm", device: torch.device | None = None,
) -> torch.Tensor:
    """Generate samples and save them as an image grid."""
    num_samples: int = 10 # sample all digits 0-9

    print(f"{device=}")

    model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size=num_samples, method=method, y=torch.arange(0, 10).to(device=device)).detach().cpu()
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


def sample_from_checkpoint(
    checkpoint: Path,
    *,
    timesteps: int | None = None,
    device_name: str | None = None,
    seed: int = 0,
    output: Path = OUTPUT_DIR / "samples.png",
    method: str = "ddpm",
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
    samples = save_samples(model, output, method=method, device=device)
    print(f"samples={str(output)!r}")
    return samples


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
                timesteps=args.timesteps,
                device_name=args.device,
                seed=args.seed,
                output=args.output,
                method=args.method,
                # condition=args.condition,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        build_parser().error(str(error))


if __name__ == "__main__":
    main()
