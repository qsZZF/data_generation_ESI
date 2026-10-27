"""Component SNRs remain exact through projection, reference and normalization."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from Simulation_G2.g2_simulation_core import G2SourceSimulator
from Simulation_G2.nmm_mixing import make_sensors, mix_sources


class NMMMixingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.fg = rng.normal(size=(2, 200)) + 12
        self.bg = rng.normal(size=(2, 200)) - 8
        self.projection = csc_matrix([[1, 0], [2, 0], [0, 0.5]])
        self.areas = np.array([1., 3., 9.])
        self.backend = object.__new__(G2SourceSimulator)
        self.backend.leadfield = np.array([[1., 2., 0.], [0., 1., 3.], [2., -1., 1.]])

    def sensors(self, source, **kwargs):
        if np.any(source['source_background']):
            kwargs.setdefault('baseline_source', np.random.default_rng(611).normal(size=(3, 500)))
        return make_sensors(self.backend, source, **kwargs)

    def mix(self, snr=3., normalize=True):
        return mix_sources(self.fg, self.bg, self.projection, self.areas, snr, normalize)

    def test_area_weighted_snr_and_common_normalization(self):
        raw, _ = self.mix(normalize=False)
        arrays, meta = self.mix()
        self.assertAlmostEqual(meta['realized_source_snr_db'], 3., places=12)
        pf = np.average(np.mean(arrays['source_foreground'] ** 2, axis=1), weights=self.areas)
        pb = np.average(np.mean(arrays['source_background'] ** 2, axis=1), weights=self.areas)
        self.assertAlmostEqual(10 * np.log10(pf / pb), 3., places=12)
        factor = 1 / np.max(np.abs(raw['source']))
        for name in arrays:
            np.testing.assert_allclose(arrays[name], raw[name] * factor, atol=1e-14)
        np.testing.assert_allclose(arrays['source'], self.projection @ arrays['region_source'])
        np.testing.assert_allclose(arrays['source'], arrays['source_foreground'] + arrays['source_background'])
        self.assertAlmostEqual(float(np.max(np.abs(arrays['source']))), 1.)

    def test_sensor_reference_snr_and_effective_target(self):
        source, _ = self.mix()
        for reference in (True, False):
            for snr in (-3., 0., 3.):
                arrays, meta = self.sensors(source, srate=500, sensor_snr_db=snr,
                                           sensor_seed=7, average_reference=reference)
                self.assertAlmostEqual(meta['realized_sensor_snr_db'], snr, places=12)
                np.testing.assert_allclose(arrays['sensor'], arrays['clean_sensor'] + arrays['sensor_noise'])
                np.testing.assert_allclose(arrays['clean_sensor'], arrays['foreground_sensor'] + arrays['background_sensor'])
                if reference:
                    np.testing.assert_allclose(arrays['sensor'].mean(axis=0), 0., atol=1e-14)
                effective = 10 * np.log10(np.mean(arrays['foreground_sensor'] ** 2) /
                                         np.mean((arrays['background_sensor'] + arrays['sensor_noise']) ** 2))
                self.assertAlmostEqual(meta['effective_target_snr_db'], effective)

    def test_independent_sensor_seed_and_reproducibility(self):
        source, _ = self.mix()
        a, _ = self.sensors(source, srate=500, sensor_snr_db=3, sensor_seed=8)
        b, _ = self.sensors(source, srate=500, sensor_snr_db=3, sensor_seed=8)
        c, _ = self.sensors(source, srate=500, sensor_snr_db=3, sensor_seed=9)
        np.testing.assert_array_equal(a['sensor'], b['sensor'])
        np.testing.assert_array_equal(a['clean_sensor'], c['clean_sensor'])
        self.assertFalse(np.array_equal(a['sensor_noise'], c['sensor_noise']))

    def test_no_noise_is_explicit_and_has_no_fake_snr(self):
        source, meta = mix_sources(self.fg, None, self.projection, self.areas)
        self.assertIsNone(meta['realized_source_snr_db'])
        arrays, meta = self.sensors(source, srate=500)
        self.assertFalse(np.any(arrays['sensor_noise']))
        self.assertIsNone(meta['realized_sensor_snr_db'])
        self.assertIsNone(meta['projected_source_snr_db'])
        self.assertIsNone(meta['effective_target_snr_db'])

    def test_reject_invalid_or_silent_source_components(self):
        for fg, bg, snr in ((self.fg, None, 3), (self.fg, self.bg, None),
                            (self.fg, self.bg, np.nan), (self.fg * 0, self.bg, 3),
                            (self.fg, self.bg * 0, 3), (self.fg * np.nan, self.bg, 3)):
            with self.assertRaises(ValueError):
                mix_sources(fg, bg, self.projection, self.areas, snr)
        with self.assertRaises(ValueError):
            mix_sources(self.fg, self.bg, self.projection, [1, -1, 3], 3)

    def test_zero_sensor_power_after_reference_fails(self):
        source, _ = self.mix()
        self.backend.leadfield[:] = self.backend.leadfield[0]
        with self.assertRaisesRegex(ValueError, 'zero power'):
            self.sensors(source, srate=500, sensor_snr_db=3)

    def test_real_noise_validation_and_scale(self):
        source, _ = self.mix()
        for invalid in (np.ones(200), np.zeros((3, 200)), np.ones((2, 200)),
                        np.ones((3, 100)), np.full((3, 200), np.nan)):
            with self.assertRaises(ValueError):
                self.sensors(source, srate=500, sensor_noise_type='real',
                             sensor_snr_db=3, real_sensor_noise=invalid)
        real = np.random.default_rng(9).normal(size=(3, 750))
        arrays, meta = self.sensors(source, srate=500, sensor_noise_type='real',
                                   sensor_snr_db=3, real_sensor_noise=real)
        self.assertAlmostEqual(meta['realized_sensor_snr_db'], 3., places=12)
        self.assertEqual(arrays['sensor'].shape, (3, 200))
        with self.assertRaises(ValueError):
            self.sensors(source, srate=500, sensor_snr_db=3, real_sensor_noise=real)

    def test_legacy_white_generator_unchanged(self):
        source, _ = self.mix()
        pattern = np.random.default_rng(9).normal(size=(3, 700))
        with patch.object(self.backend, '_make_broadband_sensor_noise', return_value=pattern) as old:
            self.sensors(source, srate=500, sensor_noise_type='white', sensor_snr_db=3)
            old.assert_called_once()

    def test_extreme_representable_snr_has_no_legacy_floor(self):
        for snr in (-150., 150.):
            source, meta = self.mix(snr=snr, normalize=False)
            self.assertAlmostEqual(meta['realized_source_snr_db'], snr, places=10)
            _, meta = self.sensors(source, srate=500, sensor_snr_db=snr)
            self.assertAlmostEqual(meta['realized_sensor_snr_db'], snr, places=10)
        with self.assertRaises(ValueError):
            self.mix(snr=1e10)

    def test_iid_does_not_call_legacy_filtered_white(self):
        source, _ = self.mix()
        with patch.object(self.backend, '_make_broadband_sensor_noise', side_effect=AssertionError('filtered')):
            arrays, _ = self.sensors(source, srate=500, sensor_snr_db=3)
        self.assertGreater(np.var(arrays['sensor_noise']), 0.)

    def test_common_mode_real_noise_rejected(self):
        source, _ = self.mix()
        common = np.tile(np.sin(np.arange(700)), (3, 1))
        with self.assertRaisesRegex(ValueError, 'zero power'):
            self.sensors(source, srate=500, sensor_snr_db=3,
                         sensor_noise_type='real', real_sensor_noise=common)


if __name__ == '__main__':
    unittest.main()
