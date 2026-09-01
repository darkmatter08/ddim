import unittest

import numpy as np
import torch

from pusht_data import (
    PushTDataset,
    PushTNormalizer,
    compute_image_channel_stats,
    split_episode_indices,
)


class FakeRoot(dict):
    pass


def make_replay() -> FakeRoot:
    # Two episodes. Their distinct values make boundary crossing easy to catch.
    images = np.zeros((5, 2, 2, 3), dtype=np.uint8)
    images[:3] = np.arange(3, dtype=np.uint8)[:, None, None, None]
    images[3:] = 100 + np.arange(2, dtype=np.uint8)[:, None, None, None]
    states = np.column_stack(
        [np.array([0, 1, 2, 100, 101]), np.zeros(5), np.zeros((5, 3))]
    ).astype(np.float32)
    actions = np.column_stack(
        [np.array([0, 1, 2, 100, 101]), np.zeros(5)]
    ).astype(np.float32)
    return FakeRoot(
        data={"img": images, "state": states, "action": actions},
        meta={"episode_ends": np.array([3, 5], dtype=np.int64)},
    )


class PushTDatasetTest(unittest.TestCase):
    def test_windows_repeat_boundaries_without_crossing_episodes(self) -> None:
        dataset = PushTDataset(
            "unused.zarr",
            n_obs_steps=2,
            prediction_horizon=2,
            _root=make_replay(),
        )

        self.assertEqual(len(dataset), 5)
        np.testing.assert_array_equal(
            dataset[0]["obs"]["agent_pos"][:, 0].numpy(), [0, 0]
        )
        np.testing.assert_array_equal(
            dataset[0]["action"][:, 0].numpy(), [0, 1]
        )
        np.testing.assert_array_equal(
            dataset[2]["obs"]["agent_pos"][:, 0].numpy(), [1, 2]
        )
        np.testing.assert_array_equal(
            dataset[2]["action"][:, 0].numpy(), [2, 2]
        )
        np.testing.assert_array_equal(
            dataset[3]["obs"]["agent_pos"][:, 0].numpy(), [100, 100]
        )
        np.testing.assert_array_equal(
            dataset[3]["action"][:, 0].numpy(), [100, 101]
        )
        np.testing.assert_array_equal(
            dataset[4]["action"][:, 0].numpy(), [101, 101]
        )

    def test_output_shapes_and_image_scaling(self) -> None:
        dataset = PushTDataset(
            "unused.zarr",
            n_obs_steps=2,
            prediction_horizon=3,
            _root=make_replay(),
        )
        sample = dataset[1]

        self.assertEqual(sample["obs"]["image"].shape, (2, 3, 2, 2))
        self.assertEqual(sample["obs"]["agent_pos"].shape, (2, 2))
        self.assertEqual(sample["action"].shape, (3, 2))
        self.assertEqual(sample["obs"]["image"].dtype, torch.float32)
        self.assertAlmostEqual(sample["obs"]["image"][1, 0, 0, 0].item(), 1 / 255)

    def test_episode_split_is_disjoint_and_reproducible(self) -> None:
        train_a, val_a = split_episode_indices(20, 0.2, seed=7)
        train_b, val_b = split_episode_indices(20, 0.2, seed=7)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
        self.assertEqual(len(val_a), 4)
        self.assertFalse(set(train_a) & set(val_a))

    def test_normalizer_round_trips_coordinates_and_images(self) -> None:
        normalizer = PushTNormalizer(
            coordinate_lower=(0.0, 0.0),
            coordinate_upper=(512.0, 512.0),
            image_mean=(0.25, 0.5, 0.75),
            image_std=(0.1, 0.2, 0.25),
        )
        coordinates = torch.tensor([[0.0, 512.0], [256.0, 128.0]])
        images = torch.tensor(
            [[[[0.1]], [[0.4]], [[0.9]]], [[[0.2]], [[0.6]], [[0.7]]]]
        )

        normalized_coordinates = normalizer.normalize_coordinates(coordinates)
        torch.testing.assert_close(
            normalized_coordinates,
            torch.tensor([[-1.0, 1.0], [0.0, -0.5]]),
        )
        torch.testing.assert_close(
            normalizer.unnormalize_coordinates(normalized_coordinates), coordinates
        )
        torch.testing.assert_close(
            normalizer.unnormalize_images(normalizer.normalize_images(images)), images
        )
        self.assertEqual(
            PushTNormalizer.from_config(normalizer.to_config()), normalizer
        )

    def test_image_statistics_use_only_selected_raw_episode_frames(self) -> None:
        root = make_replay()
        mean, std = compute_image_channel_stats(
            root["data"]["img"], root["meta"]["episode_ends"], [0]
        )

        expected_mean = 1.0 / 255.0
        expected_std = np.std(np.array([0.0, 1.0, 2.0])) / 255.0
        np.testing.assert_allclose(mean, [expected_mean] * 3)
        np.testing.assert_allclose(std, [expected_std] * 3)

    def test_dataset_applies_one_normalizer_to_positions_and_actions(self) -> None:
        normalizer = PushTNormalizer(
            coordinate_lower=(0.0, 0.0),
            coordinate_upper=(512.0, 512.0),
        )
        dataset = PushTDataset(
            "unused.zarr",
            n_obs_steps=2,
            prediction_horizon=2,
            normalizer=normalizer,
            _root=make_replay(),
        )
        sample = dataset[2]

        expected_x = 2.0 * 2.0 / 512.0 - 1.0
        self.assertAlmostEqual(sample["obs"]["agent_pos"][-1, 0].item(), expected_x)
        self.assertAlmostEqual(sample["action"][0, 0].item(), expected_x)

    def test_no_normalizer_preserves_raw_dataset_values(self) -> None:
        dataset = PushTDataset(
            "unused.zarr",
            n_obs_steps=2,
            prediction_horizon=2,
            normalizer=None,
            _root=make_replay(),
        )
        sample = dataset[2]

        self.assertEqual(sample["obs"]["agent_pos"][-1, 0].item(), 2.0)
        self.assertEqual(sample["action"][0, 0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
