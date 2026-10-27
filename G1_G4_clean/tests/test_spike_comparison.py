"""Fixed event comparison rules: no edge padding, warping, or empty-score fiction."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from Simulation_G3.compare_spike_bank import (
    PRE, POST, detect_events, event_metrics, load_references, normalized_event, reference_paths, summarize,
)


class SpikeComparisonTests(unittest.TestCase):
    def test_focus_duplicates_cross_containers_and_splits_are_not_refilled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "nmm_spikes" / "a0"
            folder.mkdir(parents=True)
            for number in range(22):
                data = np.zeros((500, 994))
                identity = 0 if number in (1, 20) else number
                data[:, 0] = identity * .01
                data[150, 0] += 10
                # Different non-focus content and metadata make containers distinct.
                data[0, 1] = number
                savemat(folder / f"nmm_{number}.mat", {"data": data, "container_id": number},
                        do_compression=True)
            with patch("Simulation_G3.compare_spike_bank.ROOT", root):
                references, provenance, rows = load_references([0])
            ref = references[0]
            self.assertEqual(len(ref["groups"]["development"]), 19)
            self.assertEqual(len(ref["groups"]["heldout"]), 1)
            self.assertEqual(ref["exclusion_details"]["development"]["duplicate"], 1)
            self.assertEqual(ref["exclusion_details"]["heldout"]["duplicate"], 1)
            self.assertEqual(ref["availability"]["selected_files"], 22)
            self.assertEqual(len(rows), 20)
            self.assertNotEqual(provenance[0]["sha256"], provenance[20]["sha256"])
            self.assertEqual(provenance[0]["waveform_sha256"], provenance[20]["waveform_sha256"])
            self.assertEqual(provenance[20]["duplicate_of"], provenance[0]["path"])
            self.assertEqual(provenance[1]["exclusion_reason"], "duplicate")
            self.assertFalse(provenance[20]["included"])

    def test_reference_availability_uses_numeric_order_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for number in range(30):
                (folder / f"nmm_{number}.mat").touch()
            paths, available = reference_paths(folder)
            self.assertEqual(available, 30)
            self.assertEqual(len(paths[:20]), 20)
            self.assertEqual(len(paths[20:]), 10)
            self.assertEqual(paths[2].name, "nmm_2.mat")
            for number in range(30, 45):
                (folder / f"nmm_{number}.mat").touch()
            paths, available = reference_paths(folder)
            self.assertEqual(available, 45)
            self.assertEqual(len(paths), 40)
            self.assertEqual(paths[-1].name, "nmm_39.mat")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for number in range(20):
                (folder / f"nmm_{number}.mat").touch()
            with self.assertRaisesRegex(ValueError, "at least 21"):
                reference_paths(folder)

    def test_strict_threshold_spacing_and_edge_exclusion(self):
        trace = np.zeros(1000)
        trace[[10, 200, 300, 500, 700, 900]] = [10, 9, 10, 8, 11, 12]
        peaks, excluded = detect_events(trace)
        np.testing.assert_array_equal(peaks, [300, 700])
        self.assertEqual(excluded, 2)

    def test_crop_aligns_without_flipping_or_stretching(self):
        trace = np.full(1000, 3.)
        trace[400] = 13
        trace[430] = -2
        crop, shape = normalized_event(trace, 400)
        self.assertEqual(len(shape), PRE + POST + 1)
        self.assertEqual(shape[PRE], 1)
        self.assertEqual(shape[PRE + 30], -.5)
        self.assertEqual(crop[PRE], 13)
        self.assertIsNone(normalized_event(trace, PRE - 1))
        self.assertIsNone(normalized_event(trace, len(trace) - POST))
        self.assertIsNone(normalized_event(np.ones(1000), 400))

    def test_similarity_is_shape_only_and_records_raw_peak(self):
        t = np.arange(1000)
        trace = 3 + 10 * np.exp(-((t - 400) / 5.) ** 2)
        crop, shape = normalized_event(trace, 400)
        scaled_crop, scaled_shape = normalized_event(trace * 7 + 20, 400)
        np.testing.assert_allclose(shape, scaled_shape, atol=1e-14)
        result = event_metrics(scaled_crop, scaled_shape, shape)
        self.assertAlmostEqual(result["correlation"], 1)
        self.assertAlmostEqual(result["normalized_psd_l1"], 0)
        self.assertEqual(result["raw_peak"], 111)
        self.assertGreater(result["half_prominence_width_ms"], 0)

    def test_zero_events_are_explicit(self):
        peaks, excluded = detect_events(np.zeros(1000))
        self.assertEqual(peaks.size, 0)
        self.assertEqual(excluded, 0)
        result = summarize([])
        self.assertEqual(result["n_events"], 0)
        self.assertIsNone(result["correlation"])


if __name__ == "__main__":
    unittest.main()
