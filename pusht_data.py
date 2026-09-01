"""PyTorch data loading for Stanford's Push-T replay dataset.

The dataset is a Zarr group with all timesteps concatenated in ``data/*`` and
exclusive episode end offsets in ``meta/episode_ends``.  Samples are sliding
windows that never cross episode boundaries.  At the start/end of an episode,
missing timesteps are padded by repeating the nearest real timestep, matching
Stanford's Diffusion Policy sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PUSHT_WORKSPACE_LOWER = (0.0, 0.0)
PUSHT_WORKSPACE_UPPER = (512.0, 512.0)


@dataclass(frozen=True)
class PushTNormalizer:
    """Normalize Push-T observations and actions with one shared configuration."""

    coordinate_lower: tuple[float, float] | None = None
    coordinate_upper: tuple[float, float] | None = None
    image_mean: tuple[float, float, float] | None = None
    image_std: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if (self.coordinate_lower is None) != (self.coordinate_upper is None):
            raise ValueError("coordinate lower and upper bounds must be set together")
        if self.coordinate_lower is not None:
            lower = np.asarray(self.coordinate_lower, dtype=np.float64)
            upper = np.asarray(self.coordinate_upper, dtype=np.float64)
            if (
                lower.shape != (2,)
                or upper.shape != (2,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(upper <= lower)
            ):
                raise ValueError("coordinate bounds must be two increasing values")

        if (self.image_mean is None) != (self.image_std is None):
            raise ValueError("image mean and standard deviation must be set together")
        if self.image_mean is not None:
            mean = np.asarray(self.image_mean, dtype=np.float64)
            std = np.asarray(self.image_std, dtype=np.float64)
            if (
                mean.shape != (3,)
                or std.shape != (3,)
                or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(std))
                or np.any(std <= 0)
            ):
                raise ValueError("image statistics must contain three positive-channel scales")

    @staticmethod
    def _values_like(
        values: np.ndarray | torch.Tensor, constants: Sequence[float]
    ) -> np.ndarray | torch.Tensor:
        if isinstance(values, torch.Tensor):
            return torch.as_tensor(constants, dtype=values.dtype, device=values.device)
        return np.asarray(constants, dtype=values.dtype)

    def normalize_coordinates(
        self, values: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.coordinate_lower is None or self.coordinate_upper is None:
            return values
        lower = self._values_like(values, self.coordinate_lower)
        upper = self._values_like(values, self.coordinate_upper)
        return 2.0 * (values - lower) / (upper - lower) - 1.0

    def unnormalize_coordinates(
        self, values: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.coordinate_lower is None or self.coordinate_upper is None:
            return values
        lower = self._values_like(values, self.coordinate_lower)
        upper = self._values_like(values, self.coordinate_upper)
        return (values + 1.0) * (upper - lower) / 2.0 + lower

    def _image_statistics_like(
        self, images: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.image_mean is None or self.image_std is None:
            raise RuntimeError("image normalization is disabled")
        if images.ndim < 3 or images.shape[-3] != 3:
            raise ValueError("expected images with channel-first RGB dimensions")
        shape = (1,) * (images.ndim - 3) + (3, 1, 1)
        mean = self._values_like(images, self.image_mean).reshape(shape)
        std = self._values_like(images, self.image_std).reshape(shape)
        return mean, std

    def normalize_images(
        self, images: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.image_mean is None:
            return images
        mean, std = self._image_statistics_like(images)
        return (images - mean) / std

    def unnormalize_images(
        self, images: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.image_mean is None:
            return images
        mean, std = self._image_statistics_like(images)
        return images * std + mean

    def to_config(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation of this normalizer."""
        return {
            "version": 1,
            "coordinate_lower": list(self.coordinate_lower) if self.coordinate_lower is not None else None,
            "coordinate_upper": list(self.coordinate_upper) if self.coordinate_upper is not None else None,
            "image_mean": list(self.image_mean) if self.image_mean is not None else None,
            "image_std": list(self.image_std) if self.image_std is not None else None,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "PushTNormalizer":
        """Reconstruct a normalizer, treating old checkpoints as unnormalized."""
        if not config:
            return cls()
        if config.get("version") != 1:
            raise ValueError(f"unsupported Push-T normalizer version: {config.get('version')!r}")

        def optional_tuple(name: str) -> tuple[float, ...] | None:
            value = config.get(name)
            return None if value is None else tuple(float(item) for item in value)

        return cls(
            coordinate_lower=optional_tuple("coordinate_lower"),  # type: ignore[arg-type]
            coordinate_upper=optional_tuple("coordinate_upper"),  # type: ignore[arg-type]
            image_mean=optional_tuple("image_mean"),  # type: ignore[arg-type]
            image_std=optional_tuple("image_std"),  # type: ignore[arg-type]
        )


def compute_image_channel_stats(
    images: Any,
    episode_ends: Sequence[int] | np.ndarray,
    episode_indices: Sequence[int] | np.ndarray,
    *,
    chunk_size: int = 128,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute RGB moments from each selected raw episode frame exactly once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("expected images with shape [N, H, W, 3]")

    ends = np.asarray(episode_ends, dtype=np.int64)
    selected_indices = np.asarray(episode_indices, dtype=np.int64)
    if len(ends) == 0 or np.any(np.diff(ends) <= 0) or ends[-1] != images.shape[0]:
        raise ValueError("episode ends must be increasing and cover every image")
    if selected_indices.ndim != 1 or len(selected_indices) == 0:
        raise ValueError("at least one training episode is required for image statistics")
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(ends)):
        raise ValueError("episode index is out of range")
    selected = np.zeros(len(ends), dtype=bool)
    selected[np.unique(selected_indices)] = True

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    episode_start = 0
    for episode_index, episode_end_value in enumerate(ends):
        episode_end = int(episode_end_value)
        if selected[episode_index]:
            for chunk_start in range(episode_start, episode_end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, episode_end)
                chunk = np.asarray(images[chunk_start:chunk_end], dtype=np.float64) / 255.0
                channel_sum += chunk.sum(axis=(0, 1, 2))
                channel_square_sum += np.square(chunk).sum(axis=(0, 1, 2))
                pixel_count += int(np.prod(chunk.shape[:3]))
        episode_start = episode_end

    if pixel_count == 0:
        raise ValueError("training episodes contain no image pixels")
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= np.finfo(np.float32).eps):
        raise ValueError("cannot standardize an image channel with zero variance")
    return tuple(mean.tolist()), tuple(std.tolist())


def _open_zarr(path: str | Path) -> Any:
    try:
        import zarr
    except ImportError as error:
        raise ImportError(
            "Push-T loading requires zarr; run `python -m pip install -r requirements.txt`"
        ) from error

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Push-T dataset not found at {path}. See README.md for download instructions."
        )
    return zarr.open_group(str(path), mode="r")


def _array(root: Any, key: str) -> Any:
    value = root
    for part in key.split("/"):
        value = value[part]
    return value


def split_episode_indices(
    n_episodes: int, val_ratio: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes, never individual frames, into train and validation."""
    if n_episodes <= 0:
        raise ValueError("The replay must contain at least one episode")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio > 0.0 and n_episodes > 1:
        n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
        rng = np.random.default_rng(seed)
        val_mask[rng.choice(n_episodes, size=n_val, replace=False)] = True

    all_indices = np.arange(n_episodes, dtype=np.int64)
    return all_indices[~val_mask], all_indices[val_mask]


class PushTDataset(Dataset[dict[str, Any]]):
    """Return observation histories paired with current/future action targets.

    Each item has this structure::

        {
            "obs": {
                "image": float32 tensor [n_obs_steps, 3, 96, 96],
                "agent_pos": float32 tensor [n_obs_steps, 2],
            },
            "action": float32 tensor [prediction_horizon, 2],
        }

    ``state`` contains the pusher XY followed by the block pose.  Only the
    pusher XY is exposed as ``agent_pos``, following Stanford's image policy.
    When a normalizer is supplied, images are channel-standardized and the
    agent positions and actions are mapped into normalized workspace coordinates.

    If the last observation is at time ``t``, actions start at ``a_t``.  An
    internal synchronized window of length ``n_obs_steps + prediction_horizon
    - 1`` provides both sides of that alignment.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        *,
        n_obs_steps: int = 2,
        prediction_horizon: int = 16,
        episode_indices: Sequence[int] | np.ndarray | None = None,
        normalizer: PushTNormalizer | None = None,
        _root: Any | None = None,
    ) -> None:
        if n_obs_steps <= 0 or prediction_horizon <= 0:
            raise ValueError("n_obs_steps and prediction_horizon must be positive")

        self.zarr_path = Path(zarr_path).expanduser()
        self.root = _root if _root is not None else _open_zarr(self.zarr_path)
        self.n_obs_steps = n_obs_steps
        self.prediction_horizon = prediction_horizon
        self.normalizer = normalizer
        self.sequence_length = n_obs_steps + prediction_horizon - 1
        self.pad_before = n_obs_steps - 1
        self.pad_after = prediction_horizon - 1

        self.images = _array(self.root, "data/img")
        self.states = _array(self.root, "data/state")
        self.actions = _array(self.root, "data/action")
        self.episode_ends = np.asarray(
            _array(self.root, "meta/episode_ends")[:], dtype=np.int64
        )
        self._validate_replay()

        if episode_indices is None:
            episode_indices = np.arange(len(self.episode_ends), dtype=np.int64)
        selected = np.zeros(len(self.episode_ends), dtype=bool)
        selected[np.asarray(episode_indices, dtype=np.int64)] = True
        self.indices = self._build_indices(selected)

    def _validate_replay(self) -> None:
        n_steps = int(self.episode_ends[-1])
        if np.any(np.diff(self.episode_ends) <= 0):
            raise ValueError("episode_ends must be strictly increasing")
        if any(array.shape[0] != n_steps for array in (self.images, self.states, self.actions)):
            raise ValueError("img, state, and action lengths must equal the final episode end")
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("expected images with shape [N, H, W, 3]")
        if self.states.ndim != 2 or self.states.shape[1] < 2:
            raise ValueError("expected state with shape [N, >=2]")
        if self.actions.ndim != 2 or self.actions.shape[1] != 2:
            raise ValueError("expected actions with shape [N, 2]")

    def _build_indices(self, selected: np.ndarray) -> np.ndarray:
        indices: list[tuple[int, int, int, int]] = []
        episode_start = 0
        for episode_index, episode_end in enumerate(self.episode_ends):
            episode_end = int(episode_end)
            if selected[episode_index]:
                episode_length = episode_end - episode_start
                min_start = -self.pad_before
                max_start = episode_length - self.sequence_length + self.pad_after
                for relative_start in range(min_start, max_start + 1):
                    buffer_start = episode_start + max(relative_start, 0)
                    buffer_end = episode_start + min(
                        relative_start + self.sequence_length, episode_length
                    )
                    sample_start = buffer_start - (episode_start + relative_start)
                    sample_end = self.sequence_length - (
                        episode_start
                        + relative_start
                        + self.sequence_length
                        - buffer_end
                    )
                    indices.append(
                        (buffer_start, buffer_end, sample_start, sample_end)
                    )
            episode_start = episode_end
        return np.asarray(indices, dtype=np.int64).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.indices)

    def _read_and_pad(self, array: Any, index: np.ndarray) -> np.ndarray:
        buffer_start, buffer_end, sample_start, sample_end = map(int, index)
        sample = np.asarray(array[buffer_start:buffer_end])
        if sample_start == 0 and sample_end == self.sequence_length:
            return sample

        result = np.empty(
            (self.sequence_length, *array.shape[1:]), dtype=array.dtype
        )
        result[sample_start:sample_end] = sample
        if sample_start > 0:
            result[:sample_start] = sample[0]
        if sample_end < self.sequence_length:
            result[sample_end:] = sample[-1]
        return result

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        image = self._read_and_pad(self.images, index)[: self.n_obs_steps]
        image = image.astype(np.float32) / 255.0
        image = np.moveaxis(image, -1, 1)
        state = self._read_and_pad(self.states, index)[: self.n_obs_steps]
        agent_pos = state[:, :2].astype(np.float32)
        action_start = self.n_obs_steps - 1
        action_end = action_start + self.prediction_horizon
        action = self._read_and_pad(self.actions, index)[action_start:action_end]
        action = action.astype(np.float32)
        if self.normalizer is not None:
            image = self.normalizer.normalize_images(image)
            agent_pos = self.normalizer.normalize_coordinates(agent_pos)
            action = self.normalizer.normalize_coordinates(action)

        return {
            "obs": {
                "image": torch.from_numpy(image),
                "agent_pos": torch.from_numpy(agent_pos),
            },
            "action": torch.from_numpy(action),
        }


def create_pusht_data_loaders(
    zarr_path: str | Path,
    *,
    batch_size: int = 64,
    n_obs_steps: int = 2,
    prediction_horizon: int = 16,
    val_ratio: float = 0.02,
    seed: int = 42,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    normalize_coordinates: bool = True,
    normalize_images: bool = True,
    normalizer: PushTNormalizer | None = None,
) -> tuple[DataLoader, DataLoader, PushTNormalizer]:
    """Build episode-disjoint loaders sharing one training-derived normalizer.

    Supplying ``normalizer`` reuses an existing checkpoint configuration and
    takes precedence over the two normalization flags.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    root = _open_zarr(zarr_path)
    episode_ends = np.asarray(_array(root, "meta/episode_ends")[:])
    train_episodes, val_episodes = split_episode_indices(
        len(episode_ends), val_ratio, seed
    )
    if normalizer is None:
        image_mean: tuple[float, float, float] | None = None
        image_std: tuple[float, float, float] | None = None
        if normalize_images:
            image_mean, image_std = compute_image_channel_stats(
                _array(root, "data/img"), episode_ends, train_episodes
            )
        normalizer = PushTNormalizer(
            coordinate_lower=PUSHT_WORKSPACE_LOWER if normalize_coordinates else None,
            coordinate_upper=PUSHT_WORKSPACE_UPPER if normalize_coordinates else None,
            image_mean=image_mean,
            image_std=image_std,
        )
    common = {
        "n_obs_steps": n_obs_steps,
        "prediction_horizon": prediction_horizon,
        "normalizer": normalizer,
    }
    train_dataset = PushTDataset(
        zarr_path, episode_indices=train_episodes, _root=root, **common
    )
    val_dataset = PushTDataset(
        zarr_path, episode_indices=val_episodes, _root=root, **common
    )
    pin_memory = torch.device(device).type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, normalizer


if __name__ == "__main__":
    """Example usage of the Push-T data loader."""

    train_loader, val_loader, _ = create_pusht_data_loaders(
        zarr_path="data/pusht/pusht/pusht_cchi_v7_replay.zarr",
        n_obs_steps=2,
        prediction_horizon=16,
        device="cpu",
    )

    """
    Returned samples from the train_loader have this structure:
    {
        "obs": {
            "image": torch.from_numpy(image),
            "agent_pos": torch.from_numpy(state[:, :2]),
        },
        "action": torch.from_numpy(action),
    }
    """
    for step, batch in enumerate(train_loader):
        print(f"Step {step}")
        images = batch["obs"]["image"]
        agent_pos = batch["obs"]["agent_pos"]
        actions = batch["action"]

        # (BATCH_SIZE, OBS_STEPS, CHANNELS, HEIGHT, WIDTH) for images
        # [64, 2, 3, 96, 96]
        # (BATCH_SIZE, OBS_STEPS, 2) for agent_pos (x, y)
        # [64, 2, 2]
        # (BATCH_SIZE, PREDICTION_HORIZON, 2) for actions (x, y)
        # [64, 16, 2]
        print(f"{images.shape=}")
        print(f"{agent_pos.shape=}")
        print(f"{actions.shape=}")

        if step >= 2:
            break
