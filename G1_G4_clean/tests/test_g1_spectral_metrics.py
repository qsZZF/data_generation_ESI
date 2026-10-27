"""Controlled signal checks for descriptive metrics, not realism claims."""

import json
import unittest

import numpy as np

from Simulation_G1.spectral_metrics import evoked_features, resting_features


def colored_alpha(seed=3, rate=256, duration=40, exponent=1.5):
    rng = np.random.default_rng(seed)
    n_times = int(rate * duration)
    frequency = np.fft.rfftfreq(n_times, 1 / rate)
    white = rng.normal(size=(6, n_times))
    spectrum = np.fft.rfft(white, axis=-1)
    scale = np.maximum(frequency, 1 / duration) ** (-exponent / 2)
    scale[0] = 0
    signal = np.fft.irfft(spectrum * scale, n=n_times, axis=-1)
    signal /= signal.std(axis=-1, keepdims=True)
    time = np.arange(n_times) / rate
    return signal + .9 * np.sin(2 * np.pi * 10 * time)[None, :]


def event_trials(*, erd=1., jitter=0., seed=7, n_trials=40):
    rate = 256
    time = -2 + np.arange(5 * rate) / rate
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=(n_trials, 3, 1))
    alpha = np.sin(2 * np.pi * 10 * time + phases)
    envelope = np.where(time >= .25, erd, 1.)
    beta = .25 * np.sin(2 * np.pi * 20 * time + phases / 2)
    offsets = rng.normal(0, jitter, size=(n_trials, 1, 1))
    erp = 3 * np.exp(-.5 * ((time[None, None, :] - .2 - offsets) / .04) ** 2)
    topography = np.array([1., -.5, -.5])[None, :, None]
    return alpha * envelope + beta + erp * topography + .1 * rng.normal(size=alpha.shape)


class RestingMetricsTests(unittest.TestCase):
    def test_known_colored_slope_and_alpha_peak(self):
        features = resting_features(colored_alpha(), 256)
        self.assertAlmostEqual(features["aperiodic_exponent"], 1.5, delta=.2)
        self.assertAlmostEqual(features["alpha_peak_hz"], 10, delta=.5)
        self.assertGreater(features["alpha_peak_prominence_db"], 10)
        self.assertGreater(features["alpha_relative_power"], .2)
        self.assertTrue(all(-1 <= value <= 1 for value in features["autocorrelation"]["values"]))
        self.assertLessEqual(sum(features["band_relative_power"].values()), 1.000001)
        json.dumps(features, allow_nan=False)

    def test_amplitude_scaling_and_constant_offset_invariance(self):
        signal = colored_alpha()
        original = resting_features(signal, 256)
        scaled = resting_features(signal * 1e-6 + 1e-4, 256)
        for name in ("aperiodic_exponent", "alpha_peak_hz", "alpha_peak_prominence_db",
                     "alpha_relative_power", "hjorth_mobility_hz", "hjorth_complexity"):
            self.assertAlmostEqual(original[name], scaled[name], places=9)
        np.testing.assert_allclose(original["autocorrelation"]["values"],
                                   scaled["autocorrelation"]["values"], atol=1e-12)

    def test_short_flat_nonfinite_and_invalid_rate_rejected(self):
        for signal, rate in [(np.ones((2, 1024)), 256),
                             (np.full((2, 1024), np.nan), 256),
                             (np.arange(200)[None, :], 256),
                             (np.arange(1024)[None, :], 60),
                             (np.arange(1024)[None, :], np.nan)]:
            with self.subTest(rate=rate, shape=signal.shape), self.assertRaises(ValueError):
                resting_features(signal, rate)

    def test_flat_channels_are_explicitly_excluded(self):
        signal = colored_alpha()
        with_flat = np.concatenate([signal, np.zeros((1, signal.shape[-1]))])
        features = resting_features(with_flat, 256)
        self.assertEqual(features["metadata"]["n_flat_channels_excluded"], 1)
        self.assertAlmostEqual(features["alpha_relative_power"],
                               resting_features(signal, 256)["alpha_relative_power"])


class EvokedMetricsTests(unittest.TestCase):
    def test_known_erp_latency_and_alpha_suppression(self):
        reference = evoked_features(event_trials(), 256, -2)
        suppression = evoked_features(event_trials(erd=.4), 256, -2)
        self.assertAlmostEqual(reference["erp_peak_latency_s"], .2, delta=.025)
        self.assertAlmostEqual(reference["alpha_ersp_db"], 0, delta=.6)
        self.assertLess(suppression["alpha_ersp_db"], -7)
        self.assertLess(suppression["alpha_induced_ersp_db"], -7)
        self.assertAlmostEqual(suppression["beta_ersp_db"], 0, delta=.6)
        self.assertGreater(reference["evoked_power_fraction"], 0)
        self.assertLessEqual(reference["evoked_power_fraction"], 1)
        json.dumps(suppression, allow_nan=False)

    def test_jitter_reduces_evoked_peak_and_power_fraction(self):
        locked = evoked_features(event_trials(), 256, -2)
        jittered = evoked_features(event_trials(jitter=.09), 256, -2)
        self.assertLess(jittered["erp_peak_gfp"], locked["erp_peak_gfp"] * .75)
        self.assertLess(jittered["evoked_power_fraction"], locked["evoked_power_fraction"])

    def test_amplitude_scaling_preserves_ratios(self):
        signal = event_trials(erd=.5)
        original = evoked_features(signal, 256, -2)
        scaled = evoked_features(signal * 1e-6 + 1e-4, 256, -2)
        for name in ("alpha_ersp_db", "beta_ersp_db", "alpha_induced_ersp_db",
                     "beta_induced_ersp_db", "evoked_power_fraction", "erp_peak_latency_s"):
            self.assertAlmostEqual(original[name], scaled[name], places=8)
        self.assertAlmostEqual(original["erp_peak_gfp"] * 1e-6, scaled["erp_peak_gfp"], places=13)
        self.assertAlmostEqual(original["erp_response_rms"] * 1e-6, scaled["erp_response_rms"], places=13)

    def test_missing_or_postevent_baseline_and_invalid_windows_rejected(self):
        signal = event_trials()
        options = [dict(baseline=(-3, -1)), dict(baseline=(-1, .1)),
                   dict(baseline=(-1, -1)), dict(response=(0, 4)),
                   dict(ersp_window=(-.2, 1)), dict(ersp_window=(.5, np.nan))]
        for option in options:
            with self.subTest(option=option), self.assertRaises(ValueError):
                evoked_features(signal, 256, -2, **option)
        with self.assertRaises(ValueError):
            evoked_features(signal * np.nan, 256, -2)

    def test_induced_undefined_for_single_or_identical_trials(self):
        signal = event_trials(n_trials=1)
        for trials in (signal, np.repeat(signal, 5, axis=0)):
            features = evoked_features(trials, 256, -2)
            self.assertIsNone(features["alpha_induced_ersp_db"])
            self.assertFalse(features["metadata"]["induced_estimable"])
            json.dumps(features, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
