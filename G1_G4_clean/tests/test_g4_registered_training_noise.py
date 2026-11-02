"""G4 measured-noise augmentation keeps donors and crops disjoint across folds."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G1.real_noise import create_noise_registry
from Simulation_G4.bank import generate_bank, project_bank
import test_g4_bank as bank_fixture
import test_g4_pipeline as pipeline_fixture


class G4RegisteredTrainingNoiseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bank = self.root / 'bank'
        with patch('Simulation_G4.bank.simulate', side_effect=bank_fixture.small_simulator), contextlib.redirect_stdout(io.StringIO()):
            generate_bank(bank_fixture.MATRIX, self.bank, **bank_fixture.BANK_OPTIONS)
        self.leadfield = np.array([[1., .2], [.3, 1.], [-.2, -.5]])
        self.channels = ['A', 'B', 'C']
        self.mapping = {}
        records = []
        for index, (fold, split) in enumerate(zip(('train', 'validation', 'calibration', 'test'), ('train', 'val', 'val', 'test'))):
            signal = np.random.default_rng(20 + index).normal(size=(3, 700)) * 1e-6
            signal -= signal.mean(axis=0, keepdims=True)
            path = self.root / f'{fold}.npz'
            np.savez(path, noise=signal)
            record_id = f'donor_{fold}'
            records.append(dict(path=str(path), record_id=record_id, recording_id=f'dataset:S{index}:R1',
                subject_id=f'dataset:S{index}', split=split, fs_hz=100., channel_names=self.channels,
                sensor_units='V', reference='average', noise_role='resting_background'))
            self.mapping[fold] = [record_id]
        self.registry = self.root / 'registry.json'
        create_noise_registry(records, self.registry)

    def project(self, **changes):
        options = dict(sensor_snr_db=10., seed=15, return_baseline=True,
                       real_noise_registry=self.registry, noise_records_by_split=self.mapping,
                       channel_names=self.channels)
        options.update(changes)
        return project_bank(self.bank, self.leadfield, 100, **options)

    def test_replay_provenance_shared_baseline_gain_and_fold_mapping(self):
        first = self.project()
        repeated = self.project()
        for left, right in zip(first[:3], repeated[:3]):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first[3]['baseline_sensor'], repeated[3]['baseline_sensor'])
        splits = json.loads((self.bank / 'splits.json').read_text())
        for index, info in enumerate(first[3]['metadata']):
            fold = next(fold for fold, ids in splits.items() if index in ids)
            provenance = info['real_noise_provenance']
            self.assertEqual(provenance['record_id'], self.mapping[fold][0])
            self.assertEqual(provenance['baseline_stop_sample'], provenance['event_start_sample'])
            self.assertEqual(info['g4_fold'], fold)
            start = provenance['start_sample']
            with np.load(self.root / f'{fold}.npz') as data:
                segment = data['noise'][:, start:start + 300]
            baseline = segment[:, :200] - segment[:, :200].mean(axis=1, keepdims=True)
            event = segment[:, 200:] - segment[:, 200:].mean(axis=1, keepdims=True)
            baseline -= baseline.mean(axis=0, keepdims=True)
            event -= event.mean(axis=0, keepdims=True)
            np.testing.assert_allclose(first[2][index], event * info['registered_input_to_output_gain'], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(first[3]['baseline_sensor'][index], baseline * info['registered_input_to_output_gain'], rtol=1e-12, atol=1e-15)
            # Same common scaling preserves the raw baseline/event power ratio.
            expected_ratio = np.mean(baseline ** 2) / np.mean(event ** 2)
            actual_ratio = np.mean(first[3]['baseline_sensor'][index] ** 2) / np.mean(first[2][index] ** 2)
            self.assertAlmostEqual(actual_ratio, expected_ratio, places=10)
            clean = first[0][index].astype(float) - first[2][index]
            self.assertAlmostEqual(10*np.log10(np.mean(clean ** 2)/np.mean(first[2][index] ** 2)), 10., places=5)

    def test_validation_calibration_shared_donor_is_rejected(self):
        mapping = dict(self.mapping, calibration=self.mapping['validation'])
        with self.assertRaisesRegex(ValueError, 'ancestry|donor'):
            self.project(noise_records_by_split=mapping)
        with self.assertRaisesRegex(ValueError, 'subject'):
            self.project(excluded_noise_subject_ids=['dataset:S0'])
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.project(sensor_snr_db=None)

    def test_upfront_invalid_split_partition_and_channel_metadata(self):
        with self.assertRaisesRegex(ValueError, 'channel'):
            self.project(channel_names=['A', 'B', 'unknown'])
        path = self.bank / 'splits.json'
        splits = json.loads(path.read_text())
        splits['test'] = [splits['train'][0]]
        path.write_text(json.dumps(splits))
        with self.assertRaisesRegex(ValueError, 'partition'):
            self.project()

    def test_iid_branch_is_identical_to_original_seed_and_scaling(self):
        actual = project_bank(self.bank, self.leadfield, 100, sensor_snr_db=10., seed=15, return_baseline=True)
        for index in range(10):
            with np.load(self.bank / f'sample_{index:06d}.npz') as data:
                clean = self.leadfield @ data['source'][:, :100]
            clean -= clean.mean(axis=0, keepdims=True)
            clean -= clean.mean(axis=1, keepdims=True)
            noise = np.random.default_rng(np.random.SeedSequence([15, index])).standard_normal(clean.shape)
            noise -= noise.mean(axis=0, keepdims=True)
            noise -= noise.mean(axis=1, keepdims=True)
            gain = np.sqrt(np.mean(clean ** 2) / (10 * np.mean(noise ** 2)))
            noise *= gain
            baseline = np.random.default_rng(np.random.SeedSequence([15, index, 1])).standard_normal((3, 200))
            baseline -= baseline.mean(axis=0, keepdims=True)
            baseline -= baseline.mean(axis=1, keepdims=True)
            baseline *= gain
            np.testing.assert_array_equal(actual[2][index], noise)
            np.testing.assert_array_equal(actual[3]['baseline_sensor'][index], baseline)
            np.testing.assert_array_equal(actual[0][index], (clean + noise).astype(np.float32))

    def test_subset_preserves_global_rng_fold_and_order(self):
        indices = [9, 0, 7]
        for measured in (False, True):
            options = dict(sensor_snr_db=10., seed=15, return_baseline=True)
            if measured:
                options.update(real_noise_registry=self.registry, noise_records_by_split=self.mapping,
                               channel_names=self.channels)
            full = project_bank(self.bank, self.leadfield, 100, **options)
            subset = project_bank(self.bank, self.leadfield, 100, draw_indices=indices, **options)
            for actual, expected in zip(subset[:3], full[:3]):
                np.testing.assert_array_equal(actual, expected[indices])
            np.testing.assert_array_equal(subset[3]['baseline_sensor'], full[3]['baseline_sensor'][indices])
            self.assertEqual(subset[3]['metadata'], [full[3]['metadata'][index] for index in indices])
            self.assertEqual([row['draw_index'] for row in subset[3]['metadata']], indices)
        for indices in ([1, 1], [-1], [10], [1.5], [True], []):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, 'draw_indices'):
                project_bank(self.bank, self.leadfield, 100, draw_indices=indices)

    def test_tampered_record_and_registry_change_are_rejected(self):
        donor = self.root / 'test.npz'
        with np.load(donor) as data:
            original = data['noise'].copy()
        np.savez(donor, noise=original * 2)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.project()
        np.savez(donor, noise=original)
        # Restore an exact archive for the registry; rewrite it freshly with its original metadata.
        records = json.loads(self.registry.read_text())['records']
        records[-1]['path'] = str(donor)
        rebuilt = self.root / 'registry_restored.json'
        create_noise_registry(records, rebuilt)
        import Simulation_G4.bank as bank_module
        original_read = bank_module.NoiseWindowReader.read
        count = 0
        def changing_read(reader, *args, **kwargs):
            nonlocal count
            result = original_read(reader, *args, **kwargs)
            count += 1
            if count == 4:
                rebuilt.write_bytes(rebuilt.read_bytes() + b'\n')
            return result
        with patch.object(bank_module.NoiseWindowReader, 'read', changing_read), self.assertRaisesRegex(ValueError, 'registry changed'):
            self.project(real_noise_registry=rebuilt)


class G4RegisteredTrainingPipelineTests(unittest.TestCase):
    setUp = pipeline_fixture.G4PipelineTests.setUp
    _write_input = pipeline_fixture.G4PipelineTests._write_input
    _fake_simulate = staticmethod(pipeline_fixture.G4PipelineTests._fake_simulate)
    _generate_bank = pipeline_fixture.G4PipelineTests._generate_bank

    def test_train_only_checkpoint_and_covariance_keep_real_noise_identity(self):
        import hashlib
        import torch
        from Simulation_G4.pipeline import run_dataset
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self._generate_bank()
        records, mapping = [], {}
        channels = [str(name).upper() for name in self.channels]
        for index, (fold, split) in enumerate(zip(('train', 'validation', 'calibration', 'test'), ('train', 'val', 'val', 'test'))):
            values = np.random.default_rng(88 + index).normal(size=(8, 600)) * 1e-6
            values -= values.mean(axis=0, keepdims=True)
            path = self.directory / f'noise_{fold}.npz'
            np.savez(path, noise=values)
            records.append(dict(path=str(path), record_id=fold, recording_id=f'synthetic_test_only:donor_{fold}:R1',
                subject_id=f'synthetic_test_only:donor_{fold}', split=split, fs_hz=100.,
                channel_names=channels, sensor_units='V', reference='average', noise_role='resting_background'))
            mapping[fold] = [fold]
        registry = self.directory / 'noise_registry.json'
        create_noise_registry(records, registry)
        with contextlib.redirect_stdout(io.StringIO()):
            run_dataset('synthetic_test_only', [self.measured], self.atlas, self.connectivity,
                self.bank, self.directory / 'output', epochs=1, batch_size=3, profile='compact', device='cpu',
                train_only=True, sensor_snr_db=10., real_noise_registry=registry, noise_records_by_split=mapping)
        output = self.directory / 'output/synthetic_test_only'
        manifest = json.loads((output / 'run_manifest.json').read_text())
        plan = manifest['training_noise']
        self.assertEqual(plan['registry_sha256'], hashlib.sha256(registry.read_bytes()).hexdigest())
        self.assertEqual(plan['noise_records_by_split'], mapping)
        self.assertAlmostEqual(manifest['training_sensor_snr_summary']['pre_analysis_filter']['median_db'], 10., places=5)
        self.assertEqual(manifest['training_sensor_snr_summary']['post_analysis_filter']['n_informative'], 10)
        checkpoint = torch.load(output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['metadata']['training_noise'], plan)
        with np.load(output / 'training_noise_covariance.npz') as data:
            metadata = json.loads(str(data['metadata_json']))
            self.assertEqual(data['baseline_sensor'].shape, (10, 8, 200))
            for info in metadata['draws']:
                self.assertIn('real_noise_provenance', info['baseline'])
                self.assertIn(info['baseline']['g4_fold'], mapping)


if __name__ == '__main__':
    unittest.main()
