import unittest

import numpy as np
import torch

from pusht_data import PushTDataset, split_episode_indices


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


if __name__ == "__main__":
    unittest.main()
