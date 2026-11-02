"""Measured-noise provenance, unit alignment and split leakage guards."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Simulation_G1.real_noise import create_noise_registry, load_noise_window


class RealNoiseRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.values = np.random.default_rng(4).normal(size=(2, 120))
        self.path = self.root / 'noise.npz'
        np.savez(self.path, noise=self.values)
        self.record = dict(path=str(self.path), record_id='segment_a', recording_id='acquisition_a',
            subject_id='subject_01', split='train', fs_hz=100., channel_names=['Cz', 'Fz'],
            sensor_units='uV', reference='as_supplied', noise_role='resting_background')
        self.manifest = self.root / 'registry.json'

    def load(self, **changes):
        options = dict(expected_srate=100., expected_channels=['Fz', 'Cz'], expected_units='V',
                       expected_reference='as_supplied', baseline_samples=20, event_samples=30, seed=5, split='train')
        options.update(changes)
        return load_noise_window(self.manifest, 'segment_a', **options)

    def test_crop_alignment_units_and_repeat(self):
        create_noise_registry([self.record], self.manifest)
        actual, info = self.load()
        repeated, repeated_info = self.load()
        np.testing.assert_array_equal(actual, repeated)
        self.assertEqual(info, repeated_info)
        start = info['start_sample']
        np.testing.assert_array_equal(actual, self.values[[1, 0], start:start + 50] * 1e-6)
        self.assertEqual(info['baseline_start_sample'], start)
        self.assertEqual(info['baseline_stop_sample'], info['event_start_sample'])
        self.assertEqual(info['event_stop_sample'], start + 50)
        self.assertEqual(info['channel_names'], ['Fz', 'Cz'])
        self.assertEqual(info['sensor_units'], ['V', 'V'])
        self.assertEqual(info['subject_id'], 'subject_01')
        self.assertTrue(info['parent_data_group_ids'])
        native, native_info = self.load(expected_units=None)
        np.testing.assert_array_equal(native, self.values[[1, 0], start:start + 50])
        self.assertEqual(native_info['sensor_units'], ['uV', 'uV'])
        changed, _ = self.load(seed=123)
        self.assertFalse(np.array_equal(actual, changed))

    def test_exact_rate_reference_channels_split_and_window_guards(self):
        create_noise_registry([self.record], self.manifest)
        for changes in ({'expected_srate': 128.}, {'expected_reference': 'average'},
                        {'expected_channels': ['Cz']}, {'expected_channels': ['Cz', 'Cz']},
                        {'expected_units': 'T'}, {'split': 'test'}, {'event_samples': 101},
                        {'baseline_samples': 1}, {'seed': -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(**changes)

    def test_file_tampering_is_detected_even_with_preserved_size_and_timestamp(self):
        import os
        create_noise_registry([self.record], self.manifest)
        self.load()
        stat = self.path.stat()
        altered = self.values.copy()
        altered[0, 0] += 1.
        np.savez(self.path, noise=altered)
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'hash|checksum'):
            self.load()

    def test_acquisition_subject_and_content_aliases_cannot_cross_splits(self):
        other_path = self.root / 'different.npz'
        np.savez(other_path, noise=self.values * 2)
        for link in ('recording_id', 'subject_id', 'content'):
            other = dict(self.record, record_id='segment_b', recording_id='acquisition_b',
                         subject_id='subject_02', split='test', path=str(other_path))
            if link == 'content':
                alias = self.root / 'alias.npz'
                np.savez_compressed(alias, signal=self.values)
                other.update(path=str(alias), array_key='signal')
            else:
                other[link] = self.record[link]
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, 'split'):
                create_noise_registry([self.record, other], self.manifest)
            self.assertFalse(self.manifest.exists())

    def test_connected_group_ids_and_immutable_manifest(self):
        paths = []
        for index in (2, 3):
            path = self.root / f'noise{index}.npz'
            np.savez(path, noise=self.values * index)
            paths.append(str(path))
        second = dict(self.record, path=paths[0], record_id='segment_b', subject_id='subject_02')
        third = dict(self.record, path=paths[1], record_id='segment_c', subject_id='subject_02', recording_id='acquisition_c')
        manifest = create_noise_registry([self.record, second, third], self.manifest)
        self.assertEqual(len({row['parent_group_id'] for row in manifest['records']}), 1)
        original = self.manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            create_noise_registry([self.record], self.manifest)
        self.assertEqual(self.manifest.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            create_noise_registry([self.record, self.record], self.root / 'duplicate.json')

    def test_reordered_content_alias_and_transitive_groups_block_leakage(self):
        path = self.root / 'permuted.npz'
        np.savez(path, noise=self.values[::-1])
        alias = dict(self.record, path=str(path), record_id='alias', recording_id='other_acquisition',
                     subject_id='other_subject', split='val', channel_names=['Fz', 'Cz'])
        with self.assertRaisesRegex(ValueError, 'split'):
            create_noise_registry([self.record, alias], self.manifest)
        second_path = self.root / 'second.npz'
        third_path = self.root / 'third.npz'
        np.savez(second_path, noise=self.values * 2)
        np.savez(third_path, noise=self.values * 3)
        second = dict(self.record, path=str(second_path), record_id='second', subject_id='subject_02')
        third = dict(self.record, path=str(third_path), record_id='third', subject_id='subject_02',
                     recording_id='third_acquisition', split='test')
        with self.assertRaisesRegex(ValueError, 'split'):
            create_noise_registry([self.record, second, third], self.manifest)

    def test_metadata_edits_cannot_silently_change_reference_or_units(self):
        create_noise_registry([self.record], self.manifest)
        metadata = json.loads(self.manifest.read_text())
        metadata['records'][0]['sensor_units'] = ['V', 'V']
        self.manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'metadata checksum'):
            self.load()

    def test_optional_subject_and_mixed_meg_unit_conversion(self):
        record = dict(self.record, subject_id=None, sensor_units=['fT/cm', 'fT'])
        create_noise_registry([record], self.manifest)
        actual, metadata = self.load(expected_units=['T', 'T/m'])
        start = metadata['start_sample']
        expected = self.values[[1, 0], start:start + 50] * np.array([1e-15, 1e-13])[:, None]
        np.testing.assert_array_equal(actual, expected)
        self.assertIsNone(metadata['subject_id'])
        self.assertFalse(any('real_noise_subject_' in group for group in metadata['parent_data_group_ids']))

    def test_invalid_records_and_epoched_or_object_arrays_fail(self):
        for updates in ({'channel_names': ['Cz', 'Cz']}, {'channel_names': ['Cz']},
                        {'sensor_units': ['V']}, {'sensor_units': 'unknown'}, {'fs_hz': 0},
                        {'recording_id': ''}, {'split': 'validation'}, {'reference': 'CAR'},
                        {'noise_role': 'unknown'}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                create_noise_registry([dict(self.record, **updates)], self.manifest)
        for values in (np.zeros((2, 3, 4)), np.zeros((2, 1)), np.full((2, 10), np.nan),
                       np.array([[object(), object()], [object(), object()]], dtype=object)):
            np.savez(self.path, noise=values)
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                create_noise_registry([self.record], self.manifest)


if __name__ == '__main__':
    unittest.main()
