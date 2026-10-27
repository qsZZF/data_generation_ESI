"""Inverse topology, parameter contract, and leak-free miniature training."""
import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from Simulation_G4.inverse import (
        TARGET_BOUNDS, TARGET_NAMES, compute_lead_field_pinv, create_model,
        pack_parameters, unpack_parameters, predict, prepare_eeg,
        train_from_arrays, weighted_mae,
    )
    from Simulation_G4.simulator import sample_parameters


@unittest.skipUnless(HAS_TORCH, "optional PyTorch is not installed")
class G4InverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, regions=5, samples=12):
        rng = np.random.default_rng(12)
        eeg = rng.normal(size=(samples, 4, 64)).astype(np.float32)
        leadfield = rng.normal(size=(4, regions)).astype(np.float32)
        parameters = np.stack([pack_parameters(sample_parameters(rng, regions)) for _ in range(samples)])
        return eeg, leadfield, parameters

    def test_packing_round_trip_and_physical_units(self):
        values = sample_parameters(np.random.default_rng(2), 7)
        packed = pack_parameters(values)
        self.assertEqual(packed.shape, (78,))
        recovered = unpack_parameters(packed)
        for key, value in values.items():
            np.testing.assert_allclose(recovered[key], np.atleast_1d(value), rtol=1e-6)
        self.assertGreater(recovered["tau_e1_ms"].min(), 9.99)
        with self.assertRaises(ValueError):
            unpack_parameters(np.zeros(13))

    def test_forward_backward_joint_bounds_and_single_region(self):
        for regions in (1, 5):
            eeg, leadfield, packed = self.fixture(regions=regions)
            model = create_model(regions, 4, 64, leadfield, seed=4)
            outputs = model(torch.from_numpy(prepare_eeg(eeg[:2])))
            self.assertEqual(set(outputs), set(TARGET_NAMES))
            for name, (low, high) in TARGET_BOUNDS.items():
                self.assertEqual(tuple(outputs[name].shape), (2, 1 if name == "delay_ms" else regions))
                self.assertTrue(torch.all(outputs[name] >= low))
                self.assertTrue(torch.all(outputs[name] <= high))
            for branch in (1, 2):
                e, i = outputs[f"tau_e{branch}_ms"], outputs[f"tau_i{branch}_ms"]
                self.assertTrue(torch.all(i >= e))
                self.assertTrue(torch.all(i <= 2.4 * e))
            target = {k: torch.from_numpy(v) for k, v in unpack_parameters(packed[:2]).items()}
            loss = weighted_mae(outputs, target)
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_compact_994_forward(self):
        eeg, leadfield, _ = self.fixture(regions=994, samples=1)
        model = create_model(994, 4, 64, leadfield)
        outputs = predict(model, eeg, device="cpu")
        self.assertEqual(outputs["C1"].shape, (1, 994))
        self.assertTrue(all(np.isfinite(v).all() for v in outputs.values()))

    def test_spectral_bins_follow_physical_frequency_and_duration(self):
        _, leadfield, _ = self.fixture()
        for samples, rate, bins, profile in ((300, 100, 121, "compact"),
                                              (1000, 100, 401, "compact"),
                                              (1280, 256, 201, "paper")):
            model = create_model(5, 4, samples, leadfield, profile=profile,
                                 sampling_rate_hz=rate)
            self.assertEqual(model.spec_branch.n_freq_bins, bins)
            rebuilt = create_model(leadfield=leadfield, **model.config)
            self.assertEqual(rebuilt.spec_branch.n_freq_bins, bins)
            self.assertEqual(model.sampling_rate_hz, rate)
            self.assertEqual(model.spectral_max_hz, 40)
        for rate, limit in ((0, 40), (np.nan, 40), (100, 51), (100, 0)):
            with self.assertRaises(ValueError):
                create_model(5, 4, 300, leadfield, sampling_rate_hz=rate,
                             spectral_max_hz=limit)

    def test_relative_ridge_handles_referenced_rank_deficient_leadfield(self):
        _, leadfield, _ = self.fixture()
        leadfield -= leadfield.mean(axis=0, keepdims=True)
        inverse = compute_lead_field_pinv(leadfield)
        self.assertEqual(inverse.shape, (5, 4))
        np.testing.assert_allclose(inverse, compute_lead_field_pinv(leadfield * 1e-10), rtol=1e-5, atol=1e-5)
        with self.assertRaises(ValueError):
            compute_lead_field_pinv(np.zeros((4, 5)))

    def test_training_is_seeded_restores_best_and_updates_weights(self):
        eeg, leadfield, parameters = self.fixture()
        runs = []
        for _ in range(2):
            model = create_model(5, 4, 64, leadfield, seed=9)
            initial = model.heads["C1"][-1].weight.detach().clone()
            progress = []
            history = train_from_arrays(
                model, eeg, parameters, train_indices=np.arange(8),
                validation_indices=np.arange(8, 12), groups=np.arange(12),
                epochs=2, batch_size=3, device="cpu", seed=3, progress=progress.append,
            )
            self.assertFalse(torch.equal(initial, model.heads["C1"][-1].weight))
            self.assertEqual(history["best_epoch"], 1 + np.argmin(history["validation_loss"]))
            self.assertEqual([record["epoch"] for record in progress], [1, 2])
            self.assertTrue(progress[history["best_epoch"] - 1]["is_best"])
            prediction = predict(model, eeg[8:], device="cpu")
            physical_target = unpack_parameters(parameters[8:])
            loss = weighted_mae(
                {k: torch.tensor(v) for k, v in prediction.items()},
                {k: torch.tensor(v) for k, v in physical_target.items()},
            )
            self.assertAlmostEqual(float(loss), min(history["validation_loss"]), places=6)
            runs.append(pack_parameters(prediction))
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_training_rejects_leakage_and_invalid_targets(self):
        eeg, leadfield, parameters = self.fixture()
        model = create_model(5, 4, 64, leadfield)
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            train_from_arrays(model, eeg, parameters, train_indices=[0, 1],
                              validation_indices=[2, 3], groups=np.zeros(12), epochs=1)
        with self.assertRaisesRegex(ValueError, "indices overlap"):
            train_from_arrays(model, eeg, parameters, train_indices=[0, 1],
                              validation_indices=[1, 3], epochs=1)
        bad = parameters.copy()
        bad[:, 5:10] = 10  # valid marginal tau_i, invalid conditional prior
        with self.assertRaisesRegex(ValueError, "joint time-constant"):
            train_from_arrays(model, eeg, bad, train_indices=[0, 1],
                              validation_indices=[2, 3], epochs=1)

    def test_preprocessing_constant_and_nonfinite_windows(self):
        values = prepare_eeg(np.ones((2, 4, 32)))
        self.assertTrue(np.isfinite(values).all())
        self.assertFalse(values.any())
        rng = np.random.default_rng(44)
        raw = rng.normal(size=(2, 4, 64)) * np.array([1, 2, 3, 4])[None, :, None]
        referenced = raw - raw.mean(axis=-1, keepdims=True)
        referenced -= referenced.mean(axis=1, keepdims=True)
        scaled = prepare_eeg(raw)
        np.testing.assert_allclose(scaled.mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(
            scaled.std(axis=-1) / scaled.std(axis=-1)[:, :1],
            referenced.std(axis=-1) / referenced.std(axis=-1)[:, :1], rtol=1e-6,
        )
        with self.assertRaises(ValueError):
            prepare_eeg(np.full((2, 4, 32), np.nan))


if __name__ == "__main__":
    unittest.main()
