"""Atlas selection reaches both generation and projection without changing legacy defaults."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import simulate_source_sensor as pipeline
from Simulation_G2.nmm_bank import initialize_bank, projection_hash


AAL_ROOT = Path(__file__).resolve().parents[1] / "Anatomy"
AAL_ASSETS_AVAILABLE = all((AAL_ROOT / name).exists() for name in (
    "G4_AAL_cortical/atlas.npz", "G4_AAL_cortical/projection.npz",
    "G3_AAL78_TVB/connectivity_78.zip", "G3_AAL78_TVB/manifest.json"))


class AtlasCLITests(unittest.TestCase):
    def args(self, *options):
        return pipeline.build_parser().parse_args(["--engine", "g3", "--source-type", "spike_jr", *options])

    def test_unspecified_atlas_preserves_legacy_defaults(self):
        args = self.args("--anatomy-dir", "custom_anatomy")
        self.assertIsNone(args.atlas)
        config = pipeline.build_g3_config(args, [10])
        self.assertIsNone(config.connectivity_path)
        self.assertTrue(config.reshape_to_994)
        self.assertEqual(pipeline.get_projection_path(args), Path("custom_anatomy/FS_994nmm_to_20k.mat"))
        self.assertEqual(pipeline.build_g2_config(args, [10]).projection_id, "FS_994nmm_to_20k")

    @unittest.skipUnless(AAL_ASSETS_AVAILABLE, "External AAL assets not supplied")
    def test_aal_profile_selects_matching_connectivity_and_projection(self):
        args = self.args("--atlas", "aal78", "--active-regions", "2,5,12,20")
        profile = pipeline.get_atlas_profile("aal78")
        config = pipeline.build_g3_config(args, [2, 5, 12, 20])
        self.assertFalse(config.reshape_to_994)
        self.assertEqual(Path(config.connectivity_path), profile["connectivity_path"])
        self.assertEqual(config.active_regions, [2, 5, 12, 20])
        self.assertEqual(pipeline.get_projection_path(args), profile["projection_path"])
        self.assertEqual(pipeline.build_g2_config(args, [2]).projection_id, profile["projection_path"].name)
        with patch.object(pipeline.G2SourceSimulator, "from_paths") as loader:
            pipeline.load_fs20k_backend(args)
        self.assertEqual(loader.call_args.args[2], profile["projection_path"])

    @unittest.skipUnless(AAL_ASSETS_AVAILABLE, "External AAL assets not supplied")
    def test_custom_paths_remain_explicit_overrides(self):
        args = self.args("--atlas", "aal78", "--connectivity-path", "custom.zip",
                         "--projection-path", "custom.npz")
        config = pipeline.build_g3_config(args, [1])
        self.assertEqual(config.connectivity_path, "custom.zip")
        self.assertEqual(pipeline.get_projection_path(args), Path("custom.npz"))

    @unittest.skipUnless(AAL_ASSETS_AVAILABLE, "External AAL assets not supplied")
    def test_spike_preset_matches_core_defaults_and_accepts_overrides(self):
        config = pipeline.build_g3_config(self.args(), [10])
        self.assertEqual((config.jr_active_A, config.jr_mu, config.jr_noise_scale, config.coupling_strength),
                         (3.6, .087, 1.0, 1.0))
        args = self.args("--atlas", "aal78", "--jr-active-A", "3.5", "--jr-mu", ".083",
                         "--jr-noise-scale", ".2758", "--coupling-strength", ".6")
        config = pipeline.build_g3_config(args, [2])
        self.assertEqual((config.jr_active_A, config.jr_mu, config.jr_noise_scale, config.coupling_strength),
                         (3.5, .083, .2758, .6))

    def test_aal_legacy_clips_and_g1_fail_before_loading_assets(self):
        args = self.args("--engine", "g2", "--atlas", "aal78")
        with patch.object(pipeline, "load_fs20k_backend") as loader:
            with self.assertRaisesRegex(ValueError, "legacy spike library has 994"):
                pipeline.run_g2(args, Path("unused"))
            loader.assert_not_called()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            pipeline.main(["--engine", "g1", "--atlas", "aal78"])

    @unittest.skipUnless(AAL_ASSETS_AVAILABLE, "External AAL assets not supplied")
    def test_same_size_reordered_projection_rejected_before_tvb(self):
        from Simulation_G2.g2_simulation_core import load_projection_matrix

        profile = pipeline.get_atlas_profile("aal78")
        projection = load_projection_matrix(profile["projection_path"])
        order = np.arange(78)
        order[:2] = [1, 0]
        backend = Mock(n_regions=78, projection_matrix=projection[:, order])
        with patch.object(pipeline, "load_fs20k_backend", return_value=backend), \
                patch.object(pipeline, "TVBSourceGenerator") as generator:
            with self.assertRaisesRegex(ValueError, "Projection differs from the named atlas"):
                pipeline.run_g3(self.args("--atlas", "aal78"), Path("unused"))
            generator.assert_not_called()

    @unittest.skipUnless(AAL_ASSETS_AVAILABLE, "External AAL assets not supplied")
    def test_named_g2_atlas_rejects_custom_bank_identity_and_projection(self):
        profile = pipeline.get_atlas_profile("aal78")
        projection = pipeline.load_projection_matrix(profile["projection_path"])
        swapped = projection[:, np.r_[1, 0, np.arange(2, 78)]]
        cases = [("custom78", projection, "Bank atlas_id"),
                 (profile["atlas_id"], swapped, "Projection differs")]
        for atlas_id, matrix, error in cases:
            with self.subTest(atlas_id=atlas_id), tempfile.TemporaryDirectory() as directory:
                initialize_bank(directory, atlas_id=atlas_id, n_regions=78,
                                projection_sha256=projection_hash(matrix))
                backend = Mock(n_regions=78, projection_matrix=matrix)
                args = self.args("--engine", "g2", "--atlas", "aal78", "--nmm-bank", directory)
                with patch.object(pipeline, "load_fs20k_backend", return_value=backend):
                    with self.assertRaisesRegex(ValueError, error):
                        pipeline.run_g2(args, Path(directory) / "output")
                backend.generate_source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
