import unittest

import numpy as np

from latent.train.build_mixed_dataset import build_mixture, episode_slices


def make_dataset(num_episodes: int, episode_length: int, offset: float) -> dict[str, np.ndarray]:
    size = num_episodes * episode_length
    observations = np.arange(size * 2, dtype=np.float32).reshape(size, 2) + offset
    terminals = np.zeros(size, dtype=bool)
    terminals[episode_length - 1 :: episode_length] = True
    return {
        'observations': observations,
        'actions': np.full((size, 1), offset, dtype=np.float32),
        'terminals': terminals,
    }


class MixedDatasetTest(unittest.TestCase):
    def test_build_mixture_is_deterministic_balanced_and_episode_safe(self):
        primary = make_dataset(num_episodes=2, episode_length=4, offset=0.0)
        secondary = make_dataset(num_episodes=8, episode_length=2, offset=1000.0)

        first, manifest = build_mixture(primary, secondary, secondary_fraction=0.5, seed=7)
        second, second_manifest = build_mixture(primary, secondary, secondary_fraction=0.5, seed=7)

        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(manifest, second_manifest)
        self.assertEqual(manifest['primary']['transitions'], 8)
        self.assertEqual(manifest['secondary']['transitions'], 8)
        self.assertEqual(manifest['secondary']['selected_episodes'], 4)
        self.assertEqual(manifest['actual_secondary_fraction'], 0.5)
        self.assertEqual(len(first['observations']), 16)
        self.assertEqual(len(episode_slices(first['terminals'])), 6)
        self.assertTrue(first['terminals'][-1])

    def test_different_seed_changes_selected_secondary_episodes(self):
        primary = make_dataset(num_episodes=1, episode_length=4, offset=0.0)
        secondary = make_dataset(num_episodes=8, episode_length=2, offset=1000.0)

        _, first = build_mixture(primary, secondary, secondary_fraction=0.5, seed=1)
        _, second = build_mixture(primary, secondary, secondary_fraction=0.5, seed=2)

        self.assertNotEqual(
            first['secondary']['selected_episode_ids'],
            second['secondary']['selected_episode_ids'],
        )

    def test_validation_rejects_invalid_fraction_and_schema(self):
        primary = make_dataset(num_episodes=1, episode_length=4, offset=0.0)
        secondary = make_dataset(num_episodes=2, episode_length=2, offset=1000.0)

        for fraction in (0.0, 1.0):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, 'secondary_fraction'):
                    build_mixture(primary, secondary, secondary_fraction=fraction, seed=0)
        with self.assertRaisesRegex(ValueError, 'Dataset keys must match'):
            build_mixture(primary, {**secondary, 'extra': np.zeros(4)}, secondary_fraction=0.5, seed=0)


if __name__ == '__main__':
    unittest.main()
