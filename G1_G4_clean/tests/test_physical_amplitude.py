"""Unit-aware amplitude and noise calibration, independent of source power."""
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np

from Simulation_G1.amplitude import (
    foreground_amplitude_gain, generate_fixed_sensor_noise,
    normalize_sensor_units, validate_amplitude_settings,
)


class ColoredBackend:
    def _make_sensor_noise(self, clean, ts, noise_type, snr_db, **kwargs):
        self.arguments = (clean.copy(), snr_db, kwargs)
        return np.cumsum(ts.rng.standard_normal(clean.shape), axis=1)


class PhysicalAmplitudeTests(unittest.TestCase):
    def test_flat_import_noise_family_remains_supported(self):
        folder = Path(__file__).resolve().parents[1] / 'Simulation_G1'
        code = f"""import sys, numpy as np
sys.path.insert(0, {str(folder)!r})
from amplitude import generate_fixed_sensor_noise
from g1_simulation_core import G1SourceSimulator
backend = object.__new__(G1SourceSimulator)
a,b,_ = generate_fixed_sensor_noise(backend,np.ones((3,64)),srate=128,noise_type='pink',seed=5,baseline_samples=256,average_reference=False,noise_std=1e-6,sensor_units='V')
assert a.shape==(3,64) and b.shape==(3,256) and np.isfinite(a).all()
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
    def physical(self, **changes):
        options = dict(amplitude_mode='physical', leadfield_source_unit='A*m',
                       sensor_units='V', foreground_peak_am=1e-9, sensor_noise_std=1e-6,
                       sensor_snr_db=None, normalize_source=False, n_channels=2)
        options.update(changes)
        return validate_amplitude_settings(**options)

    def test_explicit_physical_contract_and_relative_default(self):
        self.assertEqual(validate_amplitude_settings()['amplitude_mode'], 'relative')
        self.assertEqual(self.physical()['sensor_units'], ['V', 'V'])
        for settings in ({'leadfield_source_unit': 'mV_PSP'}, {'sensor_units': None},
                         {'foreground_peak_am': None}, {'foreground_peak_am': 0},
                         {'foreground_peak_am': np.nan}, {'sensor_noise_std': -1},
                         {'sensor_snr_db': 10}, {'normalize_source': True},
                         {'normalize_sensor': True}, {'sensor_noise_peak_ratio': 1},
                         {'sensor_units': ['V', 'T']}, {'sensor_noise_std': [1]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.physical(**settings)
        with self.assertRaises(ValueError):
            validate_amplitude_settings(foreground_peak_am=1e-9)
        with self.assertRaises(ValueError):
            validate_amplitude_settings(amplitude_mode='unknown')

    def test_leadfield_row_units_convert_to_SI(self):
        leadfield = np.ones((6, 2))
        converted, units = normalize_sensor_units(leadfield, ['V', 'uV', 'T', 'fT', 'T/m', 'fT/cm'])
        np.testing.assert_allclose(converted[:, 0], [1, 1e-6, 1, 1e-15, 1, 1e-13])
        self.assertEqual(units, ['V', 'V', 'T', 'T', 'T/m', 'T/m'])
        np.testing.assert_array_equal(leadfield, 1.)
        for declaration in ('unknown', ['V'], ['V'] * 7):
            with self.assertRaises(ValueError):
                normalize_sensor_units(leadfield, declaration)

    def test_foreground_gain_is_shared_not_independent_normalization(self):
        foreground = np.array([[0., 2., -1.], [0., -4., 1.]])
        background = foreground * .2
        gain = foreground_amplitude_gain(foreground, 2e-9)
        self.assertEqual(gain, .5e-9)
        self.assertEqual(np.max(abs(foreground * gain)), 2e-9)
        np.testing.assert_allclose(background * gain, foreground * gain * .2)
        for values in (np.zeros((2, 3)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                foreground_amplitude_gain(values, 2e-9)

    def test_iid_noise_is_independent_of_clean_power_and_has_disjoint_segments(self):
        clean = np.tile(np.sin(np.arange(256)), (2, 1)) * 1e-6
        options = dict(srate=256, noise_type='iid_white', seed=31, baseline_samples=512,
                       average_reference=False, noise_std=2e-7, sensor_units='V')
        noise, baseline, meta = generate_fixed_sensor_noise(None, clean, **options)
        other, other_baseline, _ = generate_fixed_sensor_noise(None, clean * 2, **options)
        np.testing.assert_array_equal(noise, other)
        np.testing.assert_array_equal(baseline, other_baseline)
        raw = np.random.default_rng(31).standard_normal((2, 768)) * 2e-7
        for actual, expected in ((baseline, raw[:, :512]), (noise, raw[:, 512:])):
            expected = expected - expected.mean(axis=1, keepdims=True)
            np.testing.assert_array_equal(actual, expected)
        first_snr = 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))
        second_snr = 10 * np.log10(np.mean((2 * clean) ** 2) / np.mean(other ** 2))
        self.assertAlmostEqual(second_snr - first_snr, 20 * np.log10(2))
        self.assertEqual(meta['scale_calibration'], 'theoretical_iid_standard_deviation')
        self.assertEqual(meta['baseline_interval_samples'], [0, 512])
        self.assertEqual(meta['event_interval_samples'], [512, 768])

    def test_colored_noise_uses_baseline_scale_before_reference(self):
        backend = ColoredBackend()
        clean = np.zeros((2, 256))  # Zero signal is valid with fixed noise amplitude.
        options = dict(srate=256, noise_type='pink', seed=31, baseline_samples=512,
                       noise_std=[1e-6, 2e-6], sensor_units='V')
        event, baseline, meta = generate_fixed_sensor_noise(backend, clean, average_reference=False, **options)
        np.testing.assert_allclose(baseline.std(axis=1), [1e-6, 2e-6], rtol=1e-14)
        before = backend.arguments[0]
        other, _, _ = generate_fixed_sensor_noise(backend, clean + 123., average_reference=False, **options)
        np.testing.assert_array_equal(event, other)
        np.testing.assert_array_equal(before, backend.arguments[0])
        referenced, referenced_baseline, _ = generate_fixed_sensor_noise(backend, clean, average_reference=True, **options)
        np.testing.assert_allclose(referenced, event - event.mean(axis=0, keepdims=True), atol=1e-20)
        np.testing.assert_allclose(referenced_baseline, baseline - baseline.mean(axis=0, keepdims=True), atol=1e-20)
        self.assertEqual(meta['scale_calibration'], 'baseline_standard_deviation_before_reference')

    def test_noiseless_and_mixed_sensor_unit_validation(self):
        options = dict(srate=256, noise_type='iid_white', seed=0, baseline_samples=512,
                       average_reference=False, sensor_units=['V', 'T'])
        event, baseline, _ = generate_fixed_sensor_noise(None, np.zeros((2, 256)), noise_std=None, **options)
        np.testing.assert_array_equal(event, 0.)
        np.testing.assert_array_equal(baseline, 0.)
        with self.assertRaises(ValueError):
            generate_fixed_sensor_noise(None, event, noise_std=1e-6, **options)
        with self.assertRaises(ValueError):
            generate_fixed_sensor_noise(None, event, noise_std=[1e-6, 1e-15],
                                       **dict(options, average_reference=True))
        event, _, _ = generate_fixed_sensor_noise(None, event, noise_std=[1e-6, 1e-15], **options)
        self.assertGreater(event[0].std() / event[1].std(), 1e8)

    def test_real_noise_keeps_disjoint_intervals_and_one_baseline_calibration(self):
        raw = np.random.default_rng(51).normal(size=(2, 768))
        raw[:, 512:] *= 3  # A nonstationary event must retain its larger amplitude.
        expected_std = np.array([1e-6, 2e-6])
        options = dict(srate=256, noise_type='real', seed=0, baseline_samples=512,
                       average_reference=False, noise_std=expected_std, sensor_units='V')
        event, baseline, meta = generate_fixed_sensor_noise(None, np.zeros((2, 256)),
                                                            real_sensor_noise=raw, **options)
        gain = expected_std / raw[:, :512].std(axis=1)
        for actual, segment in ((baseline, raw[:, :512]), (event, raw[:, 512:])):
            expected = segment * gain[:, None]
            expected -= expected.mean(axis=1, keepdims=True)
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(event.std(axis=1) > 2 * baseline.std(axis=1)))
        self.assertEqual(meta['real_noise_start_sample'], 0)
        with self.assertRaisesRegex(ValueError, 'no tiling'):
            generate_fixed_sensor_noise(None, np.zeros((2, 256)), real_sensor_noise=raw[:, :512], **options)

    def test_existing_colored_family_is_power_independent(self):
        from Simulation_G1.g1_simulation_core import G1SourceSimulator
        backend = object.__new__(G1SourceSimulator)
        options = dict(srate=256, noise_type='pink', seed=77, baseline_samples=512,
                       average_reference=False, noise_std=1e-6, sensor_units='V')
        first, baseline, _ = generate_fixed_sensor_noise(backend, np.zeros((3, 256)), **options)
        second, _, _ = generate_fixed_sensor_noise(backend, np.ones((3, 256)) * 1e6, **options)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(baseline.std(axis=1), 1e-6, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
