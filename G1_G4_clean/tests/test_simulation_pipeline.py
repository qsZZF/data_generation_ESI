"""Small regressions for source identity, timing, amplitude, and forward projection."""

import tempfile
import unittest
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import h5py
import numpy as np
import scipy.io
import scipy.sparse

import simulate_source_sensor as pipeline
from Simulation_G1.g1_simulation_core import G1SimulationConfig, G1SourceSimulator, estimate_power_spectrum_summary
from Simulation_G2.g2_simulation_core import (
    G2SimulationConfig,
    G2SourceSimulator,
    NMMRegionSignalRepository,
    load_brainstorm_leadfield,
    prepare_nmm_clip,
)


class PipelineTests(unittest.TestCase):
    def run_g2(self, *options, spike_time=None):
        args = pipeline.build_parser().parse_args(["--decay-mm", "0", *options])
        projection = scipy.sparse.eye(30, format="csc")
        backend = SimpleNamespace(
            n_regions=30, n_vertices=30, n_sensors=30,
            projection_matrix=projection, leadfield=np.eye(30),
            sample_regions=lambda count, rng, **kwargs: rng.choice(30, count, replace=False).tolist(),
            project_regions=lambda region, selected, **kwargs: (projection @ region, {"spatial_sources": []}),
            _finalize_sensor_from_source=lambda source, region, meta, config:
                (source, source, np.zeros_like(source), source, meta),
        )

        def clip_loader(directory, region, clip_id):
            clip = np.full((30, 500), region / 100.0, dtype=np.float32)
            clip[region] = np.sin(2 * np.pi * 10 * np.arange(500) / 500)
            if spike_time is not None:
                clip[region] = np.exp(-((np.arange(500) / 500 - spike_time) / 0.01) ** 2)
            return clip

        with patch.object(pipeline, "load_fs20k_backend", return_value=backend), \
             patch.object(pipeline, "load_nmm_spike_clip", side_effect=clip_loader), \
             patch.object(pipeline, "save_standard_sample", side_effect=lambda *a, **kw: (a[6], a[7])):
            return pipeline.run_g2(args, Path("unused"))

    def test_background_preserves_every_active_trace(self):
        region, meta = self.run_g2("--active-regions", "10,25", "--keep-nmm-background")
        expected = np.sin(2 * np.pi * 10 * np.arange(500) / 500)
        np.testing.assert_allclose(region[10], expected, atol=1e-6)
        np.testing.assert_allclose(region[25], expected, atol=1e-6)
        np.testing.assert_allclose(region[0], 0.1)
        self.assertEqual(meta["selected_regions"], [10, 25])

    def test_g2_rejects_time_warp(self):
        with self.assertRaisesRegex(ValueError, "duration|one.second|1.second"):
            self.run_g2("--duration-s", "2")

    def test_resampling_preserves_frequency_and_default_support(self):
        for rate in (250, 500, 1000):
            region, _ = self.run_g2("--srate-hz", str(rate), "--active-regions", "10")
            self.assertEqual(region.shape[1], rate)
            freq = np.fft.rfftfreq(rate, 1 / rate)[np.argmax(abs(np.fft.rfft(region[10])))]
            self.assertEqual(freq, 10)
            self.assertEqual(np.flatnonzero(np.max(abs(region), axis=1)).tolist(), [10])

    def test_invalid_or_duplicate_regions_fail(self):
        for ids in ("-1", "30", "10,10"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_g2("--active-regions", ids)

    def test_explicit_tvb_overrides_win(self):
        args = pipeline.build_parser().parse_args([
            "--source-type", "alpha_jr", "--coupling-strength", "0.1",
            "--jr-mu", "0.2", "--jr-noise-scale", "0.2", "--jr-active-A", "3.8",
            "--jr-coupling-mode", "dense",
        ])
        config = pipeline.build_g3_config(args, [10])
        self.assertEqual(config.jr_coupling_mode, 'dense')
        self.assertEqual((config.coupling_strength, config.jr_mu, config.jr_noise_scale, config.jr_active_A),
                         (0.1, 0.2, 0.2, 3.8))

    def test_tvb_defaults_are_preserved(self):
        args = pipeline.build_parser().parse_args(["--source-type", "alpha_jr"])
        config = pipeline.build_g3_config(args, [10])
        self.assertEqual((config.coupling_strength, config.jr_mu, config.jr_active_A), (0.8, 0.1, 3.35))

    def test_tvb_warmup_is_extra_to_requested_output_duration(self):
        args = pipeline.build_parser().parse_args(['--source-type', 'alpha_jr', '--duration-s', '5'])
        config = pipeline.build_g3_config(args, [10])
        self.assertEqual(config.warmup_ms, 1000)
        self.assertEqual(config.simulation_length_ms, 5000)
        self.assertEqual(config.discard_ms, 0)

    def test_loader_preserves_supplied_forward_model(self):
        expected = np.array([[1., 2.], [3., 4.]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gain.mat"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("Gain", data=expected.T)
            np.testing.assert_array_equal(load_brainstorm_leadfield(path), expected)

    def test_repository_preserves_raw_amplitude(self):
        data = np.arange(1500, dtype=float).reshape(500, 3)
        buffer = BytesIO()
        scipy.io.savemat(buffer, {"data": data})
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(Path(directory) / "clips.zip", "w") as archive:
                archive.writestr("a1/nmm_1.mat", buffer.getvalue())
            loaded, _ = NMMRegionSignalRepository(directory).sample_signal(1, np.random.default_rng(1))
            np.testing.assert_array_equal(loaded, data.T)

    def test_core_clip_timing_validation(self):
        clip = np.zeros((3, 500))
        for rate in (0, -1, float("nan"), float("inf"), 250.5, 250.001):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                prepare_nmm_clip(clip, rate, (0, 1))
        with self.assertRaises(ValueError):
            prepare_nmm_clip(clip, 500, (0, 2))

    def test_boundary_margin_is_fifty_milliseconds_at_every_rate(self):
        for rate in (250, 500, 1000):
            _, meta = self.run_g2("--srate-hz", str(rate), spike_time=0.04)
            self.assertTrue(meta["region_info"][0]["peak_is_boundary"])

    def test_noise_preparation_does_not_modify_recordings(self):
        original = np.array([[1., -1., 2., -2.], [2., -2., 1., -1.]])
        for cls in (G1SourceSimulator, G2SourceSimulator):
            recording = original.copy()
            prepared = cls._prepare_real_sensor_noise(recording, recording.shape, np.random.default_rng(1))
            prepared -= prepared.mean(axis=0)
            np.testing.assert_array_equal(recording, original)

    def test_psd_relative_power_is_scale_invariant(self):
        signal = np.sin(2 * np.pi * 10 * np.arange(1000) / 500)
        large = estimate_power_spectrum_summary(signal, 500)
        small = estimate_power_spectrum_summary(signal * 1e-12, 500)
        self.assertAlmostEqual(large["alpha_relative_power"], small["alpha_relative_power"], places=8)

    def test_low_level_g2_normalization_and_timing(self):
        backend = object.__new__(G2SourceSimulator)
        backend.n_regions = backend.n_vertices = backend.n_sensors = 3
        backend.leadfield = np.eye(3)
        backend.projection_matrix = scipy.sparse.eye(3, format="csc")
        trace = np.tile(np.sin(2 * np.pi * 10 * np.arange(500) / 500), (3, 1)) * 8
        backend.nmm_repo = SimpleNamespace(sample_signal=lambda *args: (trace.copy(), {}))
        for normalize in (False, True):
            config = G2SimulationConfig(active_region_ids=[1], normalize_region_signals=normalize,
                                        normalize_source=False, weight_decay_mm=None, srate=250)
            source, region, meta = backend.generate_source(config)
            expected = prepare_nmm_clip(trace, 250, (0, 1))
            if normalize:
                expected /= abs(expected).max()
            np.testing.assert_allclose(region[1], expected[1], atol=1e-6)
            np.testing.assert_allclose(source, region)
            self.assertEqual(meta["normalize_region_signals"], normalize)
        config.time_window = (0, 2)
        with self.assertRaises(ValueError):
            backend.generate_source(config)

    def test_zero_power_referenced_noise_is_rejected(self):
        source = np.array([[1., -1., 2., -2.], [2., -2., 1., -1.]])
        backend = object.__new__(G2SourceSimulator)
        backend.leadfield = np.eye(2)
        # Include the two-second baseline plus the four event samples.
        common_noise = np.tile([1., -1., 1., -1.], (2, 251))
        with self.assertRaisesRegex(ValueError, "noise.*power|power.*noise"):
            backend._finalize_sensor_from_source(source, source, {}, G2SimulationConfig(), common_noise)

    def test_reference_and_snr_for_both_sensor_backends(self):
        source = np.array([[1., -1., 2., -2.], [2., -2., 1., -1.]])
        leadfield = np.array([[1., 2.], [3., 4.], [2., -1.]])
        for cls, config_cls in ((G1SourceSimulator, G1SimulationConfig), (G2SourceSimulator, G2SimulationConfig)):
            for modality, reference, scale in (("EEG", False, 1), ("EEG", True, 1), ("MEG", True, 1e-12)):
                with self.subTest(backend=cls.__name__, modality=modality, reference=reference):
                    backend = object.__new__(cls)
                    backend.leadfield = leadfield * scale
                    config = config_cls(modality=modality, average_reference=reference,
                                        sensor_snr_db=10, sensor_snr_mode="global_rms", random_seed=5)
                    if cls is G1SourceSimulator:
                        with patch.object(backend, "generate_source", return_value=(source.copy(), {})):
                            sensor, clean, noise, _, meta = backend.generate_sensor(config)
                    else:
                        sensor, clean, noise, _, meta = backend._finalize_sensor_from_source(
                            source.copy(), source.copy(), {}, config)
                    expected = backend.leadfield @ source
                    if modality == "EEG" and reference:
                        expected -= expected.mean(axis=0)
                    np.testing.assert_allclose(clean, expected, atol=scale * 1e-6)
                    np.testing.assert_allclose(sensor, clean + noise, atol=scale * 1e-6)
                    self.assertAlmostEqual(meta["realized_sensor_snr_db"], 10, places=5)

    @unittest.skipUnless(os.environ.get("RUN_DATA_SMOKE") == "1", "Optional local anatomy/NMM/TVB integration")
    def test_real_high_level_engines(self):
        def check_sample(output, sample_id, sensor, clean, noise, source, region, meta, plot, **kwargs):
            for array in (sensor, clean, noise, source):
                self.assertTrue(np.isfinite(array).all())
            np.testing.assert_allclose(sensor, clean + noise, rtol=1e-5, atol=1e-4)
            self.assertEqual(sensor.shape[0], 64)
            self.assertEqual(source.shape[0], 20484)
            self.assertAlmostEqual(meta["realized_sensor_snr_reference_db"], 10, places=5)
            if region is not None:
                self.assertEqual(region.shape, (994, sensor.shape[1]))
            if "time_ms" in kwargs:
                np.testing.assert_allclose(np.diff(kwargs["time_ms"]), 2)
                self.assertTrue(meta["normalize_region_signals"])
            return meta

        with patch.object(pipeline, "save_standard_sample", side_effect=check_sample):
            for engine, kind in (("g1", "erp"), ("g2", "nmm_spike"), ("g3", "alpha_jr")):
                options = ["--engine", engine, "--source-type", kind]
                if engine == "g3":
                    options += ["--duration-s", "0.1", "--warmup-ms", "100", "--active-regions", "7"]
                args = pipeline.build_parser().parse_args(options)
                with self.subTest(engine=engine):
                    getattr(pipeline, "run_" + engine)(args, Path("unused"))


if __name__ == "__main__":
    unittest.main()
