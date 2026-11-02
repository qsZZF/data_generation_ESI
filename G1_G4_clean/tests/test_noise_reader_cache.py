"""Cached immutable noise inputs retain strict replay and bounded memory contracts."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G1 import real_noise


class NoiseReaderCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = np.random.default_rng(4).normal(size=(2, 120)).astype(np.float32)
        self.path = self.root/'noise.npz'
        np.savez(self.path, noise=self.values)
        self.spec = dict(path=str(self.path), record_id='a', recording_id='recording', split='train',
                         fs_hz=100., channel_names=['A', 'B'], sensor_units='uV',
                         reference='as_supplied', noise_role='resting_background')
        self.registry = self.root/'registry.json'
        real_noise.create_noise_registry([self.spec], self.registry)
        self.options = dict(expected_srate=100., expected_channels=['B', 'A'], expected_units='V',
                            expected_reference='as_supplied', baseline_samples=20,
                            event_samples=30, seed=5, split='train')

    def test_cache_hit_crop_replay_and_mutable_output_cannot_poison_cache(self):
        reader = real_noise.NoiseWindowReader(max_cache_bytes=4096)
        with patch.object(real_noise, '_read_array', wraps=real_noise._read_array) as load:
            a, info = reader.read(self.registry, 'a', **self.options)
            a.fill(99.)
            b, repeated_info = reader.read(self.registry, 'a', **self.options)
            self.assertEqual(load.call_count, 1)
        expected, expected_info = real_noise.load_noise_window(self.registry, 'a', **self.options)
        np.testing.assert_array_equal(b, expected)
        self.assertEqual(info, repeated_info)
        self.assertEqual(info, expected_info)
        self.assertLessEqual(reader.cache_bytes, 4096)

    def test_stat_change_and_strict_same_stat_tamper_are_rejected(self):
        reader = real_noise.NoiseWindowReader()
        reader.read(self.registry, 'a', **self.options)
        changed = self.values.copy()
        changed[0, 0] += 1
        np.savez(self.path, noise=changed)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            reader.read(self.registry, 'a', **self.options)
        # Simulate a filesystem that reports identical identity for a same-size edit.
        np.savez(self.path, noise=self.values)
        reader.read(self.registry, 'a', **self.options)
        identity = real_noise._file_signature(self.path)
        np.savez(self.path, noise=changed)
        with patch.object(real_noise, '_file_signature', return_value=identity), \
                patch.object(real_noise, '_read_array', wraps=real_noise._read_array) as load:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                reader.read(self.registry, 'a', strict=True, **self.options)
            self.assertEqual(load.call_count, 0)

    def test_bounded_cache_and_process_reset(self):
        reader = real_noise.NoiseWindowReader(max_cache_bytes=16)
        with patch.object(real_noise, '_read_array', wraps=real_noise._read_array) as load:
            for _ in range(2):
                reader.read(self.registry, 'a', **self.options)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(reader.cache_bytes, 0)
        reader = real_noise.NoiseWindowReader(max_cache_bytes=4096)
        reader.read(self.registry, 'a', **self.options)
        with patch.object(real_noise.os, 'getpid', return_value=os.getpid()+1), \
                patch.object(real_noise, '_read_array', wraps=real_noise._read_array) as load:
            reader.read(self.registry, 'a', **self.options)
            self.assertEqual(load.call_count, 1)

    def test_npy_mmap_and_npz_produce_identical_windows(self):
        npy = self.root/'noise.npy'
        np.save(npy, self.values)
        registry = self.root/'mapped.json'
        real_noise.create_noise_registry([dict(self.spec, path=str(npy))], registry)
        reader = real_noise.NoiseWindowReader(max_cache_bytes=0)
        with patch.object(real_noise, '_read_array', wraps=real_noise._read_array) as load:
            first, info = reader.read(registry, 'a', **self.options)
            second, _ = reader.read(registry, 'a', **self.options)
            self.assertEqual(load.call_count, 1)
        expected, _ = real_noise.load_noise_window(self.registry, 'a', **self.options)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)
        self.assertEqual(reader.cache_bytes, 0)
        self.assertTrue(all(not entry['array'].flags.writeable for entry in reader._cache.values()))
        self.assertEqual(info['input_shape'], list(self.values.shape))

    def test_lru_eviction_and_pickle_do_not_duplicate_recording_arrays(self):
        import pickle
        other = self.root/'other.npz'
        np.savez(other, noise=self.values+1)
        registry = self.root/'two.json'
        real_noise.create_noise_registry([self.spec, dict(self.spec, path=str(other), record_id='b')], registry)
        reader = real_noise.NoiseWindowReader(max_cache_bytes=self.values.nbytes)
        with patch.object(real_noise, '_read_array', wraps=real_noise._read_array) as load:
            for record_id in ('a', 'b', 'a'):
                reader.read(registry, record_id, **self.options)
            self.assertEqual(load.call_count, 3)
        self.assertEqual(reader.cache_bytes, self.values.nbytes)
        restored = pickle.loads(pickle.dumps(reader))
        self.assertEqual(restored.cache_bytes, 0)
        self.assertEqual(len(restored._cache), 0)

    def test_prepare_real_noise_can_export_mmap_ready_npy_with_sidecar(self):
        from scripts.prepare_real_noise import prepare_real_noise
        source = self.root/'source.npz'
        np.savez(source, samples=np.tile(self.values, (1, 3)), names=['A', 'B'],
                 sampling_rate_hz=100., channel_types=['eeg', 'eeg'], channel_units=['V', 'V'])
        output = self.root/'prepared.npy'
        spec = prepare_real_noise(source, output, record_id='prepared', recording_id='acq',
            noise_role='prestimulus', split='train', target_fs=100., npz_key='samples', npz_channels_key='names')
        self.assertTrue(output.with_suffix('.record.json').exists())
        registry = self.root/'prepared_registry.json'
        real_noise.create_noise_registry([spec], registry)
        values, _ = real_noise.NoiseWindowReader().read(registry, 'prepared', **self.options)
        self.assertEqual(values.shape, (2, 50))


if __name__ == '__main__':
    unittest.main()
