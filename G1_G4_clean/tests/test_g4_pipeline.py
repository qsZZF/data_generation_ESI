"""Synthetic fixtures only: integration behavior, never measured-data evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Simulation_G4.bank import generate_bank
from Simulation_G4.inverse import TARGET_BOUNDS, create_model
from Simulation_G4.pipeline import load_measured, reconstruct_dataset, reconstruct_groups, run_dataset, validate_bank_identity
from Simulation_G4.simulator import sample_parameters


class G4PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="g4_synthetic_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bank = self.directory / "bank"
        rng = np.random.default_rng(73)
        self.channels = np.array(["Fp1", "Fp2", "F3", "F4", "C3", "C4", "T7", "T8"])
        self.atlas = {
            "channel_names": self.channels,
            "region_names": np.array(["test_roi_1", "test_roi_2", "test_roi_3", "test_roi_4"]),
            "aal_roi_numbers": np.array([1, 2, 3, 4]),
            "leadfield": rng.normal(size=(8, 4)),
            "manifest": {"atlas_id": "synthetic_test_only",
                         "bundle_sha256": "unit_fixture_not_an_anatomical_atlas",
                         "template_limitation": "synthetic unit-test fixture; no anatomical evidence"},
        }
        self.connectivity = np.ones((4, 4)) * 0.01
        np.fill_diagonal(self.connectivity, 0)
        self.native_eeg = rng.normal(size=(8, 8, 200)) * 1e-6
        time = np.arange(200) / 200
        self.native_eeg += np.sin(2 * np.pi * 9 * time)[None, None] * np.arange(1, 9)[None, :, None] * 1e-6
        self.measured = self.directory / "synthetic_fixture_only.npz"
        self._write_input(self.measured, aliases=True)

    def _write_input(self, path, *, aliases=False):
        names = self.channels.copy()
        eeg = self.native_eeg
        if aliases:
            names = np.array(["EEG Fp1-REF", "EEG Fp2-REF", "EEG F3-REF", "EEG F4-REF",
                              "EEG C3-REF", "EEG C4-REF", "EEG T3-REF", "EEG T4-REF"])[::-1]
            eeg = eeg[:, ::-1]
        np.savez_compressed(
            path, eeg=eeg, ch_names=names, units="V", sfreq_hz=200,
            subject=np.full(8, "synthetic_subject"), condition=np.full(8, "synthetic_condition"),
            split=np.full(8, "test"), trial=np.repeat(["fixture_trial_1", "fixture_trial_2"], 4),
            start_sample=np.arange(8) * 200, stop_sample=(np.arange(8) + 1) * 200,
            metadata_json=json.dumps({"dataset": "synthetic_test_only", "evidence_kind": "synthetic_fixture",
                                      "real_human_recording": False}),
        )

    @staticmethod
    def _fake_simulate(parameters, connectivity, *, duration_s, sampling_rate_hz,
                       seed, warmup_s, **kwargs):
        """Cheap seed-sensitive stand-in, deliberately not an NMM validation."""
        rng = np.random.default_rng(seed)
        times = np.arange(round(duration_s * sampling_rate_hz)) / sampling_rate_hz
        frequency = 6 + np.asarray(parameters["C1"])
        source = np.sin(2 * np.pi * frequency[:, None] * times)
        source += rng.normal(size=source.shape) * 0.1
        return source, times, {"seed": int(seed), "evidence_kind": "mock_simulator_unit_test",
                               "warmup_s": warmup_s}

    def _generate_bank(self, sampling_rate_hz=100):
        with patch("Simulation_G4.bank.simulate", side_effect=self._fake_simulate), contextlib.redirect_stdout(io.StringIO()):
            return generate_bank(
                self.connectivity, self.bank, n_draws=10, duration_s=1,
                sampling_rate_hz=sampling_rate_hz, seed=19, atlas_id=self.atlas["manifest"]["atlas_id"],
                atlas_hash=self.atlas["manifest"]["bundle_sha256"],
            )

    def test_alias_mapping_and_native_200_to_analysis_100_hz(self):
        canonical = self.directory / "canonical_synthetic_fixture.npz"
        self._write_input(canonical)
        alias_eeg, provenance, names, leadfield, sources = load_measured(
            [self.measured], self.atlas, sampling_rate_hz=100,
        )
        canonical_eeg, _, _, _, _ = load_measured([canonical], self.atlas, sampling_rate_hz=100)
        self.assertEqual(alias_eeg.shape, (8, 8, 100))
        self.assertEqual(names, [str(name).upper() for name in self.channels])
        np.testing.assert_array_equal(leadfield, self.atlas["leadfield"])
        np.testing.assert_allclose(alias_eeg, canonical_eeg, atol=1e-12)
        np.testing.assert_allclose(alias_eeg.mean(axis=1), 0, atol=1e-12)
        self.assertEqual(len(np.unique(provenance["window_id"])), 8)
        self.assertEqual(sources[0]["native_sfreq_hz"], 200)
        self.assertEqual(sources[0]["metadata"]["evidence_kind"], "synthetic_fixture")

    def test_bank_identity_mismatches_are_rejected(self):
        config, _ = self._generate_bank()
        validate_bank_identity(config, self.atlas, self.connectivity)
        for key in ("atlas_id", "atlas_hash", "n_regions", "connectivity_sha256", "simulator_sha256"):
            invalid = copy.deepcopy(config)
            invalid[key] = "wrong_identity"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_bank_identity(invalid, self.atlas, self.connectivity)
        config["atlas_hash"] = "different_atlas"
        (self.bank / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "atlas_hash"):
            run_dataset("synthetic_test_only", [self.measured], self.atlas,
                        self.connectivity, self.bank, self.directory / "output", epochs=1, device="cpu")

    def test_native_grids_align_to_default_256_without_changing_duration_or_rhythm(self):
        with np.load(self.measured) as data:
            template = dict(data)
        for rate in (160, 200, 250, 256, 500):
            with self.subTest(native_rate=rate):
                time = np.arange(rate) / rate
                signal = np.sin(2 * np.pi * 10 * time)
                template.update(sfreq_hz=rate, eeg=np.broadcast_to(
                    signal[None, None] * np.arange(1, 9)[None, :, None], (8, 8, rate)) * 1e-6,
                    start_sample=np.arange(8) * rate, stop_sample=(np.arange(8) + 1) * rate)
                np.savez_compressed(self.measured, **template)
                eeg, provenance, _, _, sources = load_measured([self.measured], self.atlas)
                self.assertEqual(eeg.shape, (8, 8, 256))
                peak_hz = np.argmax(abs(np.fft.rfft(eeg[0, 0])))  # one-second window
                self.assertEqual(peak_hz, 10)
                np.testing.assert_array_equal(provenance['start_sample'].astype(int), np.arange(8) * rate)
                self.assertEqual(sources[0]['native_sfreq_hz'], rate)
                self.assertEqual(sources[0]['analysis_sfreq_hz'], 256)
                self.assertEqual(sources[0]['grid_upsampled'], rate < 256)
                self.assertEqual(sources[0]['epoch_sample_indices_grid'], 'native')

    def test_sampling_grid_cannot_compensate_for_insufficient_native_bandwidth(self):
        with np.load(self.measured) as data:
            template = dict(data)
        for rate in (80, float('nan'), float('inf')):
            with self.subTest(native_rate=rate):
                template['sfreq_hz'] = rate
                np.savez_compressed(self.measured, **template)
                with self.assertRaisesRegex(ValueError, 'native sampling rate'):
                    load_measured([self.measured], self.atlas)

    def test_256_hz_bank_training_checkpoint_and_reconstruction_agree(self):
        self._generate_bank(sampling_rate_hz=256)
        with patch('Simulation_G4.pipeline.simulate', side_effect=self._fake_simulate), \
                contextlib.redirect_stdout(io.StringIO()):
            run_dataset('synthetic_test_only', [self.measured], self.atlas, self.connectivity,
                        self.bank, self.directory / 'output', epochs=1, batch_size=3,
                        profile='compact', device='cpu', train_only=True)
            report = reconstruct_dataset('synthetic_test_only', [self.measured], self.atlas,
                                         self.connectivity, self.bank, self.directory / 'output', device='cpu')
        output = self.directory / 'output/synthetic_test_only'
        checkpoint = torch.load(output / 'best_checkpoint.pt', map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['model_config']['sampling_rate_hz'], 256)
        self.assertEqual(checkpoint['metadata']['window_samples'], 256)
        with np.load(output / report['groups'][0]['file']) as data:
            self.assertEqual(data['region_source_mV'].shape, (3, 4, 256))
            self.assertTrue(np.isfinite(data['clean_sensor_V']).all())

    def test_one_epoch_checkpoint_estimation_and_independent_forward_check(self):
        config, splits = self._generate_bank()
        simulation_calls = []

        def tracked_simulation(parameters, connectivity, **kwargs):
            simulation_calls.append((copy.deepcopy(parameters), kwargs["seed"]))
            return self._fake_simulate(parameters, connectivity, **kwargs)

        with patch("Simulation_G4.pipeline.simulate", side_effect=tracked_simulation), contextlib.redirect_stdout(io.StringIO()):
            report = run_dataset(
                "synthetic_test_only", [self.measured], self.atlas, self.connectivity,
                self.bank, self.directory / "output", epochs=1, batch_size=3,
                profile="compact", device="cpu", seed=4242,
            )
        output = self.directory / "output/synthetic_test_only"
        training = json.loads((output / "training_report.json").read_text())
        self.assertEqual(len(training["train_loss"]), 1)
        self.assertEqual(training["best_epoch"], 1)
        self.assertIn("heldout_test_metrics", training)
        self.assertIn("prior_mean_test_metrics", training)
        self.assertIn("synthetic_marginal_calibration", training)
        self.assertEqual(len((output / "training_progress.jsonl").read_text().splitlines()), 1)
        checkpoint = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=True)
        rebuilt = create_model(leadfield=checkpoint["leadfield"].numpy(), **checkpoint["model_config"])
        rebuilt.load_state_dict(checkpoint["state_dict"])
        self.assertFalse(checkpoint["metadata"]["human_parameter_truth_available"])
        self.assertEqual(checkpoint["metadata"]["source_files"][0]["metadata"]["evidence_kind"], "synthetic_fixture")
        self.assertFalse(list(output.glob("*.part")))
        with np.load(output / "estimated_parameters.npz", allow_pickle=False) as estimated:
            for name, (low, high) in TARGET_BOUNDS.items():
                self.assertTrue(np.all((estimated[name] >= low) & (estimated[name] <= high)))
            for branch in (1, 2):
                self.assertTrue(np.all(estimated[f"tau_i{branch}_ms"] >= estimated[f"tau_e{branch}_ms"]))
        group = report["groups"][0]
        self.assertEqual(group["status"], "forward_checked")
        self.assertEqual(group["split_kind"], "held_out_trials")
        self.assertEqual(group["fitting_windows"], 4)
        self.assertEqual(group["evaluation_windows"], 4)
        self.assertIn("prior_baseline", group)
        self.assertTrue(np.isfinite(group["log_psd_error_improvement_over_prior"]))
        with np.load(output / group["file"], allow_pickle=False) as regenerated:
            self.assertEqual(regenerated["region_source_mV"].shape, (3, 4, 100))
            self.assertEqual(regenerated["prior_sensor_V"].shape, (3, 8, 100))
            self.assertFalse(np.intersect1d(regenerated["fitting_window_ids"], regenerated["evaluation_window_ids"]).size)
            self.assertFalse(np.array_equal(regenerated["region_source_mV"][0], regenerated["region_source_mV"][1]))
            self.assertTrue(np.isfinite(regenerated["clean_sensor_V"]).all())
            metadata = json.loads(str(regenerated["metadata_json"]))
            self.assertFalse(metadata["human_source_truth_available"])
            self.assertEqual([value["seed"] for value in metadata["simulations"]], [4242, 4243, 4244])
        self.assertEqual(len(simulation_calls), 6)  # fitted and prior draws, three seeds each
        self.assertEqual([seed for _, seed in simulation_calls], [4242, 4242, 4243, 4243, 4244, 4244])
        for name in TARGET_BOUNDS:
            training_targets = []
            for index in splits["train"]:
                with np.load(self.bank / f"sample_{index:06d}.npz", allow_pickle=False) as sample:
                    training_targets.append(sample[name])
            expected_prior = np.mean(training_targets, axis=0)
            for index in (1, 3, 5):
                np.testing.assert_allclose(simulation_calls[index][0][name], expected_prior, rtol=1e-6)
        cached_seeds = []
        for sample_path in self.bank.glob("sample_*.npz"):
            with np.load(sample_path, allow_pickle=False) as sample:
                cached_seeds.append(int(sample["noise_seed"]))
        self.assertFalse(set(cached_seeds).intersection(seed for _, seed in simulation_calls))
        self.assertEqual(config["atlas_id"], "synthetic_test_only")

    def test_train_only_then_reconstruct_checkpoint_without_retraining(self):
        self._generate_bank()
        output = self.directory / "output/synthetic_test_only"
        with patch("Simulation_G4.pipeline.simulate") as simulator, contextlib.redirect_stdout(io.StringIO()):
            history = run_dataset(
                "synthetic_test_only", [self.measured], self.atlas, self.connectivity,
                self.bank, self.directory / "output", epochs=1, batch_size=3,
                profile="compact", device="cpu", seed=91, train_only=True,
            )
            simulator.assert_not_called()
        self.assertEqual(history["best_epoch"], 1)
        self.assertTrue((output / "training_prior.npz").exists())
        self.assertFalse((output / "estimated_parameters.npz").exists())
        self.assertFalse((output / "reconstruction_report.json").exists())
        checkpoint_before = (output / "best_checkpoint.pt").read_bytes()
        with patch("Simulation_G4.pipeline.train_from_arrays") as trainer, \
                patch("Simulation_G4.pipeline.simulate", side_effect=self._fake_simulate), \
                contextlib.redirect_stdout(io.StringIO()):
            report = reconstruct_dataset(
                "synthetic_test_only", [self.measured], self.atlas, self.connectivity,
                self.bank, self.directory / "output", device="cpu", seed=91,
            )
            trainer.assert_not_called()
        self.assertEqual(report["groups"][0]["status"], "forward_checked")
        self.assertEqual((output / "best_checkpoint.pt").read_bytes(), checkpoint_before)
        self.assertEqual(report["measured_window_count"], 8)

    def _reconstruction_fixture(self, groups=1):
        eeg, provenance, _, leadfield, _ = load_measured([self.measured], self.atlas, sampling_rate_hz=100)
        prior = sample_parameters(np.random.default_rng(33), 4)
        parameters = {name: np.tile(np.atleast_1d(value), (len(eeg) * groups, 1))
                      for name, value in prior.items()}
        if groups > 1:
            eeg = np.concatenate([eeg] * groups)
            provenance = {name: np.concatenate([value] * groups) for name, value in provenance.items()}
            provenance["subject"] = np.repeat([f"synthetic_subject_{index}" for index in range(groups)], 8)
            provenance["window_id"] = np.array([f"unit_window_{index}" for index in range(len(eeg))])
        return eeg, provenance, leadfield, prior, parameters

    def test_reconstruction_cache_reuses_only_identical_inputs(self):
        eeg, provenance, leadfield, prior, parameters = self._reconstruction_fixture()
        output = self.directory / "cached"
        output.mkdir()
        with patch("Simulation_G4.pipeline.predict", return_value=parameters), \
                patch("Simulation_G4.pipeline.simulate", side_effect=self._fake_simulate) as simulator, \
                contextlib.redirect_stdout(io.StringIO()):
            options = dict(sampling_rate_hz=100, prior_parameters=prior, device="cpu")
            first = reconstruct_groups(None, eeg, provenance, self.atlas, self.connectivity, leadfield, output, **options)
            self.assertEqual(simulator.call_count, 6)
            second = reconstruct_groups(None, eeg.copy(), provenance, self.atlas, self.connectivity, leadfield, output, **options)
            self.assertEqual(simulator.call_count, 6)
            self.assertEqual(first, second)
            altered = eeg.copy()
            altered[0, 0, 0] += 1e-6
            reconstruct_groups(None, altered, provenance, self.atlas, self.connectivity, leadfield, output, **options)
            self.assertEqual(simulator.call_count, 12)
            parameters["C1"][0, 0] += 0.01
            reconstruct_groups(None, altered, provenance, self.atlas, self.connectivity, leadfield, output, **options)
            self.assertEqual(simulator.call_count, 18)
            artifact = output / first[0]["file"]
            artifact.write_bytes(b"interrupted or corrupted archive")
            reconstruct_groups(None, altered, provenance, self.atlas, self.connectivity, leadfield, output, **options)
            self.assertEqual(simulator.call_count, 24)
        self.assertFalse(list(output.rglob("*.part")))
        cache = json.loads(artifact.with_suffix(".json").read_text())
        self.assertEqual(len(cache["fingerprint"]), 64)

    def test_recordings_are_held_out_when_trials_are_absent(self):
        eeg, provenance, leadfield, prior, parameters = self._reconstruction_fixture()
        provenance["trial"][:] = ""
        provenance["recording"] = np.repeat(["recording_A", "recording_B"], 4)
        output = self.directory / "recording_split"
        output.mkdir()
        with patch("Simulation_G4.pipeline.predict", return_value=parameters), \
                patch("Simulation_G4.pipeline.simulate", side_effect=self._fake_simulate), \
                contextlib.redirect_stdout(io.StringIO()):
            results = reconstruct_groups(None, eeg, provenance, self.atlas, self.connectivity, leadfield, output,
                                         sampling_rate_hz=100, prior_parameters=prior, device="cpu")
        self.assertEqual(results[0]["split_kind"], "held_out_recordings")
        with np.load(output / results[0]["file"], allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["fitting_window_ids"], provenance["window_id"][:4])
            np.testing.assert_array_equal(artifact["evaluation_window_ids"], provenance["window_id"][4:])

    def test_process_workers_match_serial_real_toy_simulator(self):
        eeg, provenance, leadfield, prior, parameters = self._reconstruction_fixture(groups=2)
        results = []
        for workers in (1, 2):
            output = self.directory / f"workers_{workers}"
            output.mkdir()
            with patch("Simulation_G4.pipeline.predict", return_value=parameters), contextlib.redirect_stdout(io.StringIO()):
                results.append(reconstruct_groups(
                    None, eeg, provenance, self.atlas, self.connectivity, leadfield, output,
                    sampling_rate_hz=100, prior_parameters=prior, device="cpu", warmup_s=0.1,
                    reconstruction_workers=workers,
                ))
        self.assertEqual(results[0], results[1])
        for result in results[0]:
            with np.load(self.directory / "workers_1" / result["file"], allow_pickle=False) as serial, \
                    np.load(self.directory / "workers_2" / result["file"], allow_pickle=False) as parallel:
                for name in ("region_source_mV", "clean_sensor_V", "prior_sensor_V", "fitting_window_ids", "evaluation_window_ids"):
                    np.testing.assert_array_equal(serial[name], parallel[name])


if __name__ == "__main__":
    unittest.main()
