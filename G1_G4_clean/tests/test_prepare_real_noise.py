"""Offline preparation preserves units, provenance, channels and continuous time."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prepare_real_noise import prepare_real_noise


class PrepareRealNoiseTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'MNE_LOGGING_LEVEL': 'ERROR'})
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / 'raw.npz'
        rng = np.random.default_rng(901)
        self.data = rng.normal(size=(3, 1000)) * 1e-6
        self.write_source(self.data)
        self.options = dict(record_id='fixture', recording_id='acquisition-1', subject_id='subject-1',
            noise_role='resting_background', split='train', npz_key='continuous', npz_channels_key='names')

    def write_source(self, values, **updates):
        arrays = dict(continuous=values, names=np.array(['C3.', 'C4.', 'Fz.']), sampling_rate_hz=100.,
                      channel_types=np.array(['eeg'] * 3), channel_units=np.array(['V'] * 3))
        arrays.update(updates)
        np.savez(self.source, **arrays)

    def test_crop_resample_reference_and_registry_record(self):
        target = self.directory / 'prepared.npz'
        record = prepare_real_noise(self.source, target, **self.options, target_fs=256,
            start_s=.25, duration_s=2., reference='average', channel_names=['C4', 'C3'])
        self.assertEqual(record['array_key'], 'noise')
        self.assertEqual(record['fs_hz'], 256)
        self.assertEqual(record['channel_names'], ['C4', 'C3'])
        self.assertEqual(record['sensor_units'], ['V', 'V'])
        self.assertEqual(record['noise_role'], 'resting_background')
        self.assertEqual(record['provenance']['crop_native_samples'], [25, 225])
        self.assertEqual(record['provenance']['original_channel_names'], ['C4.', 'C3.'])
        self.assertEqual(len(record['provenance']['source_file_sha256']), 64)
        with np.load(target, allow_pickle=False) as archive:
            self.assertEqual(archive['noise'].shape, (2, 512))
            np.testing.assert_allclose(archive['noise'].mean(axis=0), 0, atol=1e-12)
        self.assertEqual(json.loads(target.with_suffix('.record.json').read_text()), record)
        with self.assertRaises(FileExistsError):
            prepare_real_noise(self.source, target, **self.options)

    def test_rejects_epochs_missing_channels_aliases_and_undeclared_units(self):
        target = self.directory / 'prepared.npz'
        with self.assertRaisesRegex(ValueError, 'channel'):
            prepare_real_noise(self.source, target, **self.options, channel_names=['T3'])
        with self.assertRaisesRegex(ValueError, 'explicit'):
            prepare_real_noise(self.source, target, **{**self.options, 'npz_key': None})
        self.write_source(self.data[None])
        with self.assertRaisesRegex(ValueError, '2D'):
            prepare_real_noise(self.source, target, **self.options)
        self.write_source(self.data, channel_units=np.array(['uV'] * 3))
        with self.assertRaisesRegex(ValueError, 'SI'):
            prepare_real_noise(self.source, target, **self.options)
        self.write_source(self.data, names=np.array(['C3.', 'C3', 'Fz']))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare_real_noise(self.source, target, **self.options)
        self.assertFalse(target.exists())

    def test_polyphase_downsampling_suppresses_out_of_band_component(self):
        time = np.arange(2048) / 512
        values = np.array([np.sin(2 * np.pi * 100 * time), np.sin(2 * np.pi * 10 * time)]) * 1e-6
        self.write_source(values, names=np.array(['C3', 'C4']), sampling_rate_hz=512.,
                          channel_types=np.array(['eeg'] * 2), channel_units=np.array(['V'] * 2))
        target = self.directory / 'antialias.npz'
        prepare_real_noise(self.source, target, **self.options, target_fs=128)
        with np.load(target) as archive:
            rms = np.sqrt(np.mean(archive['noise'][:, 32:-32] ** 2, axis=1))
            self.assertLess(rms[0], rms[1] * .02)

    def test_fif_mixed_units_bad_channel_exclusion_and_average_reference_guard(self):
        import mne
        names = ['C3', 'C4', 'BAD', 'MAG001', 'GRAD001', 'STIM']
        types = ['eeg', 'eeg', 'eeg', 'mag', 'grad', 'stim']
        data = np.random.default_rng(902).normal(size=(6, 300)) * np.array([1e-6, 1e-6, 1e-6, 1e-15, 1e-13, 1])[:, None]
        raw = mne.io.RawArray(data, mne.create_info(names, 100, types), verbose='ERROR')
        raw.info['bads'] = ['BAD']
        path = self.directory / 'recording_raw.fif'
        raw.save(path, overwrite=False, verbose='ERROR')
        options = {k: v for k, v in self.options.items() if not k.startswith('npz_')}
        target = self.directory / 'mixed.npz'
        with self.assertRaisesRegex(ValueError, 'EEG'):
            prepare_real_noise(path, target, **options, reference='average')
        record = prepare_real_noise(path, target, **options)
        self.assertEqual(record['channel_names'], ['C3', 'C4', 'MAG001', 'GRAD001'])
        self.assertEqual(record['sensor_units'], ['V', 'V', 'T', 'T/m'])
        self.assertEqual(record['provenance']['selected_input_channel_indices'], [0, 1, 3, 4])
        self.assertIn('BAD', record['provenance']['excluded_bad_channels'])


if __name__ == '__main__':
    unittest.main()
