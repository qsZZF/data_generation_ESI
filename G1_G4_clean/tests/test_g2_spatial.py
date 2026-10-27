"""Spatial support and dynamics regressions without downloaded NMM recordings."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.sparse import csc_matrix

from Simulation_G1.spatial import SpatialDynamicsConfig, gaussian_weights
from Simulation_G2.g2_simulation_core import G2SimulationConfig, G2SourceSimulator
from Simulation_G2.nmm_bank import initialize_bank, projection_hash, save_bank_sample


class G2SpatialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vertices = np.array([[0., 0, 0], [10, 0, 0], [30, 0, 0],
                                  [40, 0, 0], [40, 10, 0], [50, 0, 0]])
        self.cortex = dict(Vertices=self.vertices,
            Faces=np.array([[0, 1, 2], [3, 4, 5]]),
            VertConn=csc_matrix(np.ones((6, 6)) - np.eye(6)))

    def simulator(self, projection=None):
        if projection is None:
            projection = csc_matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1.]])
        return G2SourceSimulator(np.eye(6), self.cortex, projection, self.root)

    def test_disjoint_sampling_is_reproducible_and_ignores_empty_columns(self):
        projection = csc_matrix([[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0],
                                 [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        simulator = self.simulator(projection)
        simulator.nmm_repo = SimpleNamespace(available_regions=lambda: [0, 1, 2, 3])
        first = simulator.sample_regions(2, np.random.default_rng(4))
        self.assertEqual(first, simulator.sample_regions(2, np.random.default_rng(4)))
        self.assertNotIn(2, first)
        simulator._validate_selected_regions(first)
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            simulator.sample_regions(3, np.random.default_rng(4))
        with self.assertRaises(ValueError):
            simulator.sample_regions(0, np.random.default_rng(4))

    def test_overlap_rejected_before_legacy_trace_loading(self):
        simulator = self.simulator(csc_matrix([[1, 1], [1, 0], [0, 0], [0, 1], [0, 0], [0, 0]]))
        loader = Mock()
        simulator.nmm_repo = SimpleNamespace(sample_signal=loader)
        with self.assertRaisesRegex(ValueError, "overlap"):
            simulator.generate_source(G2SimulationConfig(active_region_ids=[0, 1]))
        loader.assert_not_called()

    def test_invalid_explicit_ids_rejected(self):
        simulator = self.simulator()
        simulator.nmm_repo = SimpleNamespace(sample_signal=Mock())
        for ids in ([], [0, 0], [True], [0.0], [np.nan], [-1], [2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                simulator.generate_source(G2SimulationConfig(active_region_ids=ids))
        simulator.nmm_repo.sample_signal.assert_not_called()

    def test_stored_zeros_are_not_support_or_overlap(self):
        projection = csc_matrix((np.array([1., 0., 1.]), np.array([0, 1, 1]), np.array([0, 2, 3])), shape=(6, 2))
        simulator = self.simulator(projection)
        self.assertEqual(simulator.projection_matrix.nnz, 2)
        simulator._validate_selected_regions([0, 1])
        self.assertEqual(projection.nnz, 3)

    def test_static_projection_matches_legacy_decay_and_keeps_background(self):
        # The second column is background, so overlap with the selected column is allowed.
        projection = csc_matrix([[1, 2], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0]])
        simulator = self.simulator(projection)
        region = np.array([[1., 2, 3, 4], [5., 6, 7, 8]])
        original = region.copy()
        actual, metadata = simulator.project_regions(region, [0], srate=4,
            time_window=(0, 1), decay_mm=10, spatial=SpatialDynamicsConfig(mode="static"))
        weights = gaussian_weights(np.abs(np.array([0., 10, 30]) - 40 / 3), 10)
        expected = projection[:, [1]] @ region[[1]]
        expected[:3] += weights[:, None] * region[0]
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(region, original)
        self.assertEqual(metadata["spatial_sources"][0]["vertex_indices"], [0, 1, 2])

    def test_diffusion_contraction_and_pulse_reach_expected_endpoints(self):
        simulator = self.simulator()
        region = np.ones((2, 5))
        weights = gaussian_weights(np.abs(np.array([0., 10, 30]) - 40 / 3), 10)
        for mode in ("diffuse", "contract", "pulse"):
            actual, _ = simulator.project_regions(region, [0], srate=5, time_window=(0, 1),
                decay_mm=10, spatial=SpatialDynamicsConfig(mode=mode, start_s=0, duration_s=.8))
            with self.subTest(mode=mode):
                np.testing.assert_allclose(actual[3:], 1)
                np.testing.assert_allclose(actual[:3, 0], 1 if mode == "contract" else weights)
                np.testing.assert_allclose(actual[:3, -1], 1 if mode == "diffuse" else weights)

    def test_auto_uses_real_area_threshold(self):
        simulator = self.simulator()
        simulator.vertex_area[:] = 200
        config = SpatialDynamicsConfig(mode="auto", area_threshold_cm2=5, hop_threshold=None, duration_s=.2)
        actual, metadata = simulator.project_regions(np.ones((2, 5)), [0], srate=5,
            time_window=(0, 1), decay_mm=10, spatial=config)
        np.testing.assert_allclose(actual[:3, -1], 1)
        self.assertEqual(metadata["spatial_sources"][0]["area_cm2"], 6)

    def test_legacy_generation_uses_dynamic_projection(self):
        simulator = self.simulator()
        trace = np.ones((2, 500))
        simulator.nmm_repo = SimpleNamespace(sample_signal=lambda *args: (trace.copy(), {}))
        source, region, metadata = simulator.generate_source(G2SimulationConfig(active_region_ids=[0],
            normalize_source=False, normalize_region_signals=False, weight_decay_mm=10,
            spatial=SpatialDynamicsConfig(mode="diffuse", duration_s=.2)))
        self.assertFalse(np.allclose(source[:3, 0], 1))
        np.testing.assert_allclose(source[:3, -1], 1)
        self.assertTrue(np.all(region[1] == 0))
        self.assertTrue(metadata["spatial_supports_disjoint"])

    def test_bank_selected_dynamics_and_network_rejection(self):
        simulator = self.simulator()
        initialize_bank(self.root, atlas_id="test", n_regions=2,
            projection_sha256=projection_hash(simulator.projection_matrix))
        source = np.ones((2, 500))
        save_bank_sample(self.root, "one", source, np.arange(500) * 2.,
            dict(signal_type="alpha_jr", output_srate_hz=500, active_regions=[0],
                 n_regions_output=2, normalize=False, model={"class": "JansenRit"}), {})
        simulator = self.simulator()
        config = G2SimulationConfig(nmm_sample_id="one", nmm_source_mode="network",
            normalize_source=False, normalize_region_signals=False,
            spatial=SpatialDynamicsConfig(mode="diffuse", duration_s=.2))
        with self.assertRaisesRegex(ValueError, "network"):
            simulator.generate_source(config)
        config.nmm_source_mode = "selected"
        config.spatial = {"mode": "diffuse", "duration_s": .2}
        vertex, _, metadata = simulator.generate_source(config)
        np.testing.assert_allclose(vertex[:3, -1], 1)
        self.assertTrue(np.all(vertex[3:] == 0))
        self.assertTrue(metadata["spatial_supports_disjoint"])


if __name__ == "__main__":
    unittest.main()
