"""Small deterministic regressions for the TVB adapter (TVB itself is mocked)."""
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Simulation_G3.g3_tvb_core import TVBSignalConfig, TVBSourceGenerator


class G3GenerationTests(unittest.TestCase):
    def setUp(self):
        self.generator = TVBSourceGenerator(__file__)
        self.models = SimpleNamespace(
            JansenRit=lambda **kw: SimpleNamespace(**kw),
            Generic2dOscillator=lambda: SimpleNamespace(),
            WilsonCowan=lambda: SimpleNamespace(A=3.25),
        )
        # Keep orchestration tests independent of the optional TVB installation.
        self.numerics_patch = patch.dict(sys.modules, {
            'Simulation_G3.tvb_jr': SimpleNamespace(
                StableJansenRit=self.models.JansenRit,
                StableSigmoidalJansenRit=lambda **kw: kw,
                SparseSigmoidalJansenRit=lambda **kw: kw,
            ),
        })
        self.numerics_patch.start()
        self.addCleanup(self.numerics_patch.stop)

    def generate_fake(self, config, n_regions=998, startup_artifact=False, save_dir=None):
        captured = {}
        conn = SimpleNamespace(number_of_regions=n_regions, configure=lambda: None)

        def simulator_factory(**kwargs):
            captured.update(kwargs)

            def run(simulation_length):
                captured['run_numpy_errors'] = np.geterr()
                times = np.arange(config.dt_ms, simulation_length + config.dt_ms / 2, config.dt_ms)
                raw = np.zeros((len(times), 6, n_regions, 1))
                raw[:, 1, :, 0] = kwargs['model'].A
                if startup_artifact:
                    raw[times <= config.warmup_ms + config.discard_ms, 1, :, :] = 1e6
                return [(times, raw)]

            sim = SimpleNamespace(run=run)
            sim.configure = lambda: sim
            return sim

        lab = SimpleNamespace(
            connectivity=SimpleNamespace(Connectivity=SimpleNamespace(from_file=lambda **kw: conn)),
            coupling=SimpleNamespace(SigmoidalJansenRit=lambda **kw: kw, Linear=lambda **kw: kw),
            integrators=SimpleNamespace(HeunStochastic=lambda **kw: kw),
            models=self.models,
            monitors=SimpleNamespace(Raw=lambda: None),
            noise=SimpleNamespace(Additive=lambda **kw: kw),
            simulator=SimpleNamespace(Simulator=simulator_factory),
        )
        with patch.dict(sys.modules, {'tvb.simulator.lab': lab}):
            result = (self.generator.generate(config) if save_dir is None
                      else self.generator.save_sample(save_dir, config))
        return result, captured

    def test_saved_sample_excludes_startup_before_resampling_and_normalizing(self):
        for normalize in (False, True):
            config = TVBSignalConfig(warmup_ms=20, simulation_length_ms=100,
                                     discard_ms=10, normalize=normalize)
            with self.subTest(normalize=normalize), tempfile.TemporaryDirectory() as directory:
                manifest, _ = self.generate_fake(config, startup_artifact=True, save_dir=directory)
                with np.load(Path(directory) / manifest['region_source']) as saved:
                    self.assertEqual(saved['region_source'].shape, (994, 45))
                    np.testing.assert_allclose(saved['region_source'][0], 0 if normalize else 3.25)
                    self.assertLessEqual(float(abs(saved['region_source']).max()), 3.6)
                    np.testing.assert_array_equal(saved['time_ms'], np.arange(45) * 2)
                self.assertEqual(manifest['simulated_duration_ms'], 120)
                self.assertEqual(manifest['discarded_initial_ms'], 30)
                self.assertEqual(manifest['retained_duration_ms'], 90)

    def test_output_atlas_active_ids_target_surviving_raw_regions(self):
        cfg = TVBSignalConfig(signal_type='alpha_jr', active_regions=[7, 325, 921, 949], normalize=False)
        (source, _, manifest), captured = self.generate_fake(cfg)
        raw_ids = [994, 997, 996, 995]
        np.testing.assert_allclose(captured['model'].A[raw_ids], 3.35)
        np.testing.assert_allclose(source[cfg.active_regions], 3.35)
        self.assertEqual(manifest['active_regions'], cfg.active_regions)
        self.assertEqual(manifest['active_regions_raw'], raw_ids)
        self.assertEqual(captured['run_numpy_errors']['over'], 'raise')

    def test_other_models_preserve_the_callers_overflow_policy(self):
        with np.errstate(over='warn'):
            _, captured = self.generate_fake(TVBSignalConfig(signal_type='wilson_cowan'))
        self.assertEqual(captured['run_numpy_errors']['over'], 'warn')

    def test_explicit_jr_parameters_override_type_defaults(self):
        for kind in ('resting_jr', 'alpha_jr', 'spike_jr'):
            cfg = TVBSignalConfig(signal_type=kind, jr_active_A=4.1, jr_mu=.2)
            model, _, metadata = self.generator._build_model(cfg, 20, [10], self.models)
            self.assertEqual(model.A[10], 4.1)
            self.assertEqual(metadata['mu'], .2)

    def test_jr_type_defaults_are_preserved(self):
        for kind, active_a, mu in [('resting_jr', 3.25, .087), ('alpha_jr', 3.35, .1), ('spike_jr', 3.6, .087)]:
            _, _, metadata = self.generator._build_model(TVBSignalConfig(signal_type=kind), 20, [10], self.models)
            self.assertEqual(metadata['active_A'], active_a)
            self.assertEqual(metadata['mu'], mu)

    def test_noise_scale_preserves_existing_intensity_and_origin_conversion(self):
        metadata = {'class': 'JansenRit'}
        default = self.generator._noise_sigma(TVBSignalConfig(), metadata)
        np.testing.assert_array_equal(default[[0, 1, 2, 3, 5]], 0)
        self.assertAlmostEqual(default[4], .0025)
        origin_scale = np.sqrt((.1 * 3.25 * .12 * .5)**2 / 2) / .05
        origin = self.generator._noise_sigma(TVBSignalConfig(jr_noise_scale=origin_scale), metadata)
        self.assertAlmostEqual(origin[4], .000190125)

    def test_invalid_coupling_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'jr_coupling_mode'):
            self.generate_fake(TVBSignalConfig(jr_coupling_mode='invalid'))

    def test_sampling_grid_excludes_duplicate_warmup_boundary(self):
        cfg = TVBSignalConfig(warmup_ms=20, simulation_length_ms=100, discard_ms=10)
        (source, times, manifest), _ = self.generate_fake(cfg)
        self.assertEqual(source.shape, (994, 45))
        np.testing.assert_allclose(times, np.arange(45) * 2)
        self.assertEqual(manifest['discard_ms'], 10)
        self.assertEqual(manifest['retained_duration_ms'], 90)

    def test_resampling_counts_last_sample_interval(self):
        times = np.arange(200) * .5
        output_time, source = self.generator._resample(times, np.zeros((2, 200)), 500)
        self.assertEqual(source.shape, (2, 50))
        np.testing.assert_allclose(output_time, np.arange(50) * 2)

    def test_discard_removes_boundary_endpoint(self):
        times = np.arange(.5, 3.5, .5)
        kept_time, kept_source = self.generator._discard_initial(times, times[None, :], 1)
        np.testing.assert_array_equal(kept_time, [1.5, 2, 2.5, 3])
        np.testing.assert_array_equal(kept_source[0], kept_time)

    def test_resampling_preserves_physical_frequency(self):
        times = np.arange(2000) * .5
        signal = np.sin(2 * np.pi * 10 * times / 1000)[None, :]
        output_time, output = self.generator._resample(times, signal, 500)
        np.testing.assert_allclose(output[0], np.sin(2 * np.pi * 10 * output_time / 1000), atol=1e-12)

    def test_single_region_and_single_time_axes_are_preserved(self):
        self.assertEqual(self.generator._extract_source(np.zeros((1, 6, 1, 1)), 'jr_pyramidal_difference').shape, (1, 1))

    def test_invalid_time_and_region_inputs_fail_before_simulation(self):
        bad_configs = [dict(dt_ms=0), dict(output_srate_hz=float('nan')), dict(discard_ms=1000),
                       dict(warmup_ms=-1), dict(simulation_length_ms=100.1),
                       dict(active_regions=[994]), dict(active_regions=[7, 7]), dict(active_regions=[1.2])]
        for overrides in bad_configs:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.generate_fake(TVBSignalConfig(**overrides))

    def test_reshape_rejects_incompatible_atlas(self):
        with self.assertRaises(ValueError):
            self.generator._reshape_regions(np.zeros((1000, 10)), True)

    def test_oscillator_does_not_claim_unimplemented_frequency(self):
        _, _, metadata = self.generator._build_model(TVBSignalConfig(signal_type='oscillator'), 20, [], self.models)
        self.assertNotIn('nominal_frequency_hz', metadata)

    def test_export_rejects_float32_overflow(self):
        cfg = TVBSignalConfig(jr_active_A=1e40, normalize=False, simulation_length_ms=10, warmup_ms=0)
        with self.assertRaises(FloatingPointError):
            self.generate_fake(cfg)

    @unittest.skipUnless(os.environ.get('RUN_TVB_SMOKE') == '1', 'Optional installed-TVB smoke test')
    def test_real_tvb_seed_and_sampling(self):
        self.numerics_patch.stop()
        generator = TVBSourceGenerator()
        cfg = TVBSignalConfig(signal_type='alpha_jr', active_regions=[7], simulation_length_ms=100,
                              warmup_ms=100, discard_ms=10, coupling_strength=.8, random_seed=142)
        first, times, manifest = generator.generate(cfg)
        repeat, _, _ = generator.generate(cfg)
        other, _, _ = generator.generate(replace(cfg, random_seed=143))
        self.assertEqual(first.shape, (994, 45))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(times, np.arange(45) * 2)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(manifest['active_regions_raw'], [994])


if __name__ == '__main__':
    unittest.main()
