"""Optional G4 diagnostics describe regional PSP coordinates, never vertex truth."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_g4_noise_covariance as fixtures
from Simulation_G4 import pipeline
from Simulation_G1.noise_covariance import whitener_from_covariance


class G4ObservabilityTests(unittest.TestCase):
    setUp = fixtures.G4NoiseCovarianceTests.setUp
    simulate = staticmethod(fixtures.G4NoiseCovarianceTests.simulate)

    def test_temporal_source_filter_commutes_with_forward_without_region_car(self):
        rng = np.random.default_rng(801)
        source = rng.normal(size=(4, 256)) + np.arange(4)[:, None]
        source += np.sin(2 * np.pi * 10 * np.arange(256) / 256)[None]
        gain = rng.normal(size=(8, 4))
        centered_gain = gain - gain.mean(axis=0, keepdims=True)
        temporal = pipeline._temporal_analysis_filter(source, 256)
        expected = pipeline.analysis_filter((gain @ source)[None], 256)[0]
        np.testing.assert_allclose(centered_gain @ temporal, expected, rtol=1e-5, atol=2e-6)
        self.assertGreater(np.linalg.norm(temporal.mean(axis=0)), .1)

    def test_training_and_reconstruction_emit_regional_labels_and_invalidate_cache(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        output_root = self.directory / 'output'
        with patch('Simulation_G4.pipeline.simulate', side_effect=self.simulate) as simulator, \
                contextlib.redirect_stdout(io.StringIO()):
            report = pipeline.run_dataset('labels', [self.measured], self.atlas, self.connectivity,
                self.bank, output_root, epochs=1, batch_size=3, profile='compact', device='cpu',
                sensor_snr_db=10., record_observability=True)
            output = output_root / 'labels'
            manifest = json.loads((output / 'run_manifest.json').read_text())
            self.assertTrue(manifest['record_observability'])
            self.assertEqual(len(manifest['observability_code_sha256']), 64)
            gain = self.atlas['leadfield'] - self.atlas['leadfield'].mean(axis=0, keepdims=True)
            clean, _ = pipeline.project_bank(self.bank, gain, 256)
            clean = pipeline.analysis_filter(clean, 256)
            with np.load(output / 'training_noise_covariance.npz', allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata_json']))
                self.assertEqual(len(metadata['draws']), 10)
                for draw in metadata['draws']:
                    label = draw['observability']
                    self.assertEqual(label['coordinate_domain'], 'atlas_region_coefficients')
                    self.assertEqual(label['source_energy_metric'], 'unweighted_regional_PSP_Euclidean')
                    self.assertFalse(label['full_vertex_nullspace_evaluated'])
                    self.assertEqual(label['n_sources'], 4)
                    self.assertEqual(label['noise_status'], 'available')
                    self.assertIsNotNone(label['whitened_target_to_baseline_power_db'])
                car = np.eye(8) - np.ones((8, 8)) / 8
                covariance = car @ archive['noise_covariance'][0] @ car
                white = whitener_from_covariance(covariance, rtol=1e-10) @ clean[0]
                expected_db = 10 * np.log10(np.mean(white ** 2))
                self.assertAlmostEqual(metadata['draws'][0]['observability']['whitened_target_to_baseline_power_db'],
                                       expected_db, places=4)
            artifact = output / report['groups'][0]['file']
            with np.load(artifact, allow_pickle=False) as archive:
                reconstructed_metadata = json.loads(str(archive['metadata_json']))
                self.assertEqual(len(reconstructed_metadata['observability']), 3)
                for label in reconstructed_metadata['observability']:
                    self.assertEqual(label['noise_status'], 'zero_noise_covariance')
                    self.assertIsNone(label['whitened_target_to_baseline_power_db'])
                    self.assertEqual(label['observation_gain_applied_to_leadfield'], float(archive['observation_gain']))
            before = simulator.call_count
            pipeline.reconstruct_dataset('labels', [self.measured], self.atlas, self.connectivity,
                                         self.bank, output_root, device='cpu', record_observability=False)
            self.assertEqual(simulator.call_count, before + 6)
            with np.load(artifact, allow_pickle=False) as archive:
                self.assertNotIn('observability', json.loads(str(archive['metadata_json'])))
            pipeline.reconstruct_dataset('labels', [self.measured], self.atlas, self.connectivity,
                                         self.bank, output_root, device='cpu', record_observability=True)
            self.assertEqual(simulator.call_count, before + 12)
            pipeline.reconstruct_dataset('labels', [self.measured], self.atlas, self.connectivity,
                                         self.bank, output_root, device='cpu', record_observability=True)
            self.assertEqual(simulator.call_count, before + 12)


if __name__ == '__main__':
    unittest.main()
