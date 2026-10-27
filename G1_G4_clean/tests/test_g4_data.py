"""Measured-data boundaries: units, channels, subject leakage and window clocks."""
import os
import json
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import MagicMock, Mock, patch

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
import mne
import numpy as np

from Simulation_G4.data import (_hub_voltage_scale, _window_recording, load_registry,
                                _get, load_hub_recording, prepare_hub_recordings,
                                download_file, save_windows, subject_split)
from Simulation_G4 import data as data_module


class G4DataTests(unittest.TestCase):
    def test_interrupted_payload_restarts_without_replacing_existing_target(self):
        import requests
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "chunk"
            target.write_bytes(b"previous target")
            complete = b"new verified payload"
            first, second = MagicMock(), MagicMock()
            for response in (first, second):
                response.__enter__.return_value = response
                response.__exit__.return_value = False
                response.headers = {}
            def interrupted(*args):
                yield b"incomplete prefix"
                self.assertEqual(target.read_bytes(), b"previous target")
                raise requests.ConnectionError("stream read timeout")
            first.iter_content.side_effect = interrupted
            second.iter_content.return_value = iter([complete])
            with patch("Simulation_G4.data._get", side_effect=[first, second]) as get:
                with patch("time.sleep") as sleep:
                    result = download_file("https://example.org/chunk", target,
                                           sha256=hashlib.sha256(complete).hexdigest())
                    self.assertEqual(get.call_count, 2)
                    sleep.assert_called_once_with(5.)
            self.assertEqual(result.read_bytes(), complete)
            self.assertFalse(target.with_suffix(".partial").exists())

    def test_payload_attempts_bounded_and_header_exhaustion_not_retried(self):
        import requests
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "chunk"
            response = MagicMock()
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            response.headers = {}
            def interrupted(*args):
                yield b"uncommitted"
                raise requests.exceptions.ChunkedEncodingError("connection closed")
            response.iter_content.side_effect = interrupted
            with patch("Simulation_G4.data._get", return_value=response) as get:
                with patch("time.sleep") as sleep:
                    with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                        download_file("https://example.org/chunk", target)
                    self.assertEqual(get.call_count, 3)
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [5., 10.])
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix(".partial").read_bytes(), b"uncommitted")
            with patch("Simulation_G4.data._get", side_effect=requests.HTTPError("headers exhausted")) as get:
                with patch("time.sleep") as sleep:
                    with self.assertRaisesRegex(requests.HTTPError, "headers exhausted"):
                        download_file("https://example.org/chunk", target)
                    self.assertEqual(get.call_count, 1)
                    sleep.assert_not_called()

    def test_windows_extended_paths_preserve_resolved_containment(self):
        root = PureWindowsPath("E:/SourceDataGeneration/cache")
        inside = root / "recording_128/data/zarr.json"
        extended_root = PureWindowsPath("\\\\?\\" + str(root))
        extended_inside = PureWindowsPath("\\\\?\\" + str(inside))
        for destination, boundary in ((extended_inside, root), (inside, extended_root),
                                      (extended_inside, extended_root)):
            self.assertTrue(data_module._resolved_path_within_root(destination, boundary))
        for outside in (PureWindowsPath("E:/SourceDataGeneration/cache_other/file"),
                        PureWindowsPath("E:/SourceDataGeneration/cache/../outside/file"),
                        PureWindowsPath("F:/SourceDataGeneration/cache/file")):
            self.assertFalse(data_module._resolved_path_within_root(outside, extended_root))
        unc_root = PureWindowsPath("//server/share/cache")
        unc_child = PureWindowsPath("//?/UNC/server/share/cache/recording/data")
        self.assertTrue(data_module._resolved_path_within_root(unc_child, unc_root))
        self.assertFalse(data_module._resolved_path_within_root(
            PureWindowsPath("//?/UNC/other/share/cache/file"), unc_root))
        # Real filesystem resolution must still remove traversal before the comparison.
        with tempfile.TemporaryDirectory() as temp:
            boundary = (Path(temp) / "cache").resolve()
            outside = (boundary / ".." / "outside").resolve()
            self.assertFalse(data_module._resolved_path_within_root(outside, boundary))

    def test_parallel_records_commit_serially_and_drain_success_on_failure(self):
        entry = load_registry()["arithmetic_zyma2019"]
        for fail_record_zero in (False, True):
            with self.subTest(fail_record_zero=fail_record_zero), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "arithmetic_zyma2019" / entry["revision"]
                store = root / "sourcedata/braindecode/dataset.zarr"
                store.mkdir(parents=True)
                (root / "hub_metadata.json").write_text(json.dumps({"sha": entry["revision"]}))
                (store / "zarr.json").write_text(json.dumps({"attributes": {"n_datasets": 2}}))
                def item(path):
                    value = path.read_bytes()
                    return dict(type="file", path=path.relative_to(root).as_posix(), size=len(value),
                                oid=hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest())
                (root / "root-tree.json").write_text(json.dumps([item(store / "zarr.json")]))
                for recording in (0, 1):
                    directory = store / f"recording_{recording}"
                    directory.mkdir()
                    (directory / "data").write_bytes(b"verified recording")
                    (root / f"recording_{recording}-tree.json").write_text(json.dumps([item(directory / "data")]))
                barrier = threading.Barrier(2)
                def load(dataset, store, recording, **kwargs):
                    barrier.wait(timeout=5)  # Both independent records must actually execute concurrently.
                    if fail_record_zero and recording == 0:
                        raise RuntimeError("invalid recording zero")
                    return dict(eeg=np.zeros((1, 1, 2), dtype=np.float32), sfreq_hz=np.asarray(200.),
                                ch_names=np.asarray(["Cz"]), units=np.asarray("V"), subject=np.asarray(["00"]),
                                condition=np.asarray(["rest"]), split=np.asarray(["train"]),
                                metadata_json=np.asarray("{}"))
                with patch("Simulation_G4.data._get", side_effect=AssertionError("must use cache")):
                    with patch("Simulation_G4.data.load_hub_recording", side_effect=load):
                        if fail_record_zero:
                            with self.assertRaisesRegex(RuntimeError, "invalid recording zero"):
                                prepare_hub_recordings("arithmetic_zyma2019", [1, 0], temp, record_workers=2)
                        else:
                            result = prepare_hub_recordings("arithmetic_zyma2019", [1, 0], temp, record_workers=2)
                            self.assertEqual([path.name for path in result], ["recording_1.npz", "recording_0.npz"])
                manifest = json.loads((root / "download_manifest.json").read_text())
                self.assertEqual(manifest["recordings"], [1] if fail_record_zero else [0, 1])
                self.assertEqual(manifest["full_dataset"], not fail_record_zero)
                self.assertTrue((root / "recording_1.npz").exists())

    def test_retry_after_and_backoff_without_hammering_host(self):
        import requests
        limited = requests.Response()
        limited.raw = Mock()
        limited.status_code = 429
        limited.headers["Retry-After"] = "17"
        unavailable = requests.Response()
        unavailable.raw = Mock()
        unavailable.status_code = 503
        success = requests.Response()
        success.status_code = 200
        with patch("requests.get", side_effect=[limited, unavailable, success]) as get:
            with patch("time.sleep") as sleep:
                self.assertIs(_get("https://example.org/data"), success)
                self.assertEqual(get.call_count, 3)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [17., 10.])

    def test_retry_after_above_wait_cap_and_authorization_fail_immediately(self):
        import requests
        for status, headers in ((403, {}), (429, {"Retry-After": "3600"})):
            response = requests.Response()
            response.raw = Mock()
            response.status_code = status
            response.headers.update(headers)
            with patch("requests.get", return_value=response) as get, patch("time.sleep") as sleep:
                with self.assertRaises(requests.HTTPError):
                    _get("https://example.org/data")
                self.assertEqual(get.call_count, 1)
                sleep.assert_not_called()

    def test_retry_http_date_and_retry_count_are_bounded(self):
        import requests
        response = requests.Response()
        response.raw = Mock()
        response.status_code = 429
        response.headers["Retry-After"] = "Thu, 01 Jan 1970 00:00:17 GMT"
        with patch("requests.get", return_value=response) as get:
            with patch("time.time", return_value=0), patch("time.sleep") as sleep:
                with self.assertRaises(requests.HTTPError):
                    _get("https://example.org/data")
                self.assertEqual(get.call_count, 6)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [17., 17., 20., 40., 60.])

    def test_later_download_failure_preserves_first_verified_export(self):
        entry = load_registry()["arithmetic_zyma2019"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "arithmetic_zyma2019" / entry["revision"]
            store = root / "sourcedata/braindecode/dataset.zarr"
            store.mkdir(parents=True)
            (root / "hub_metadata.json").write_text(json.dumps({"sha": entry["revision"]}))
            (store / "zarr.json").write_text(json.dumps({"attributes": {"n_datasets": 2}}))
            record = store / "recording_0"
            record.mkdir()
            (record / "data").write_bytes(b"verified recording")
            def item(path):
                value = path.read_bytes()
                return dict(type="file", path=path.relative_to(root).as_posix(), size=len(value),
                            oid=hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest())
            (root / "root-tree.json").write_text(json.dumps([item(store / "zarr.json")]))
            (root / "recording_0-tree.json").write_text(json.dumps([item(record / "data")]))
            windows = dict(eeg=np.zeros((1, 1, 2), dtype=np.float32), sfreq_hz=np.asarray(200.),
                           ch_names=np.asarray(["Cz"]), units=np.asarray("V"), subject=np.asarray(["00"]),
                           condition=np.asarray(["rest"]), split=np.asarray(["train"]),
                           metadata_json=np.asarray("{}"))
            with patch("Simulation_G4.data._get", side_effect=RuntimeError("host unavailable")):
                with patch("Simulation_G4.data.load_hub_recording", return_value=windows):
                    with self.assertRaisesRegex(RuntimeError, "host unavailable"):
                        prepare_hub_recordings("arithmetic_zyma2019", [0, 1], temp)
                    # A completed recording can subsequently resume with the network unavailable.
                    resumed = prepare_hub_recordings("arithmetic_zyma2019", [0], temp)
                    self.assertEqual(len(resumed), 1)
            self.assertTrue((root / "recording_0.npz").exists())
            manifest = json.loads((root / "download_manifest.json").read_text())
            self.assertEqual(manifest["recordings"], [0])
            self.assertFalse(manifest["full_dataset"])

    def test_registry_pins_and_disjoint_subjects(self):
        registry = load_registry()
        self.assertEqual(len(registry), 12)
        for entry in registry.values():
            self.assertEqual(len(entry["revision"]), 40)
            split = entry["benchmark_split"]
            if split["val"] is not None:
                train, val, test = [set(split[key]) for key in ("train", "val", "test")]
                self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(subject_split("arithmetic_zyma2019", "00"), "train")
        self.assertEqual(subject_split("arithmetic_zyma2019", "28"), "val")
        self.assertEqual(subject_split("physionet", 90), "test")
        with self.assertRaises(ValueError):
            subject_split("arithmetic_zyma2019", 0)  # zero-padding is meaningful
        with self.assertRaises(ValueError):
            subject_split("seed_v", "01")  # session split is not subject-disjoint

    def test_export_preserves_amplitude_order_and_boundaries(self):
        data = np.arange(42, dtype=float).reshape(2, 21) * 1e-6
        raw = mne.io.RawArray(data, mne.create_info(["C4", "C3"], 10, "eeg"), verbose="ERROR")
        result = _window_recording(raw, dataset="arithmetic_zyma2019", subject="32", run=2,
                                   condition="arithmetic", duration_s=1)
        self.assertEqual(result["eeg"].shape, (2, 2, 10))
        np.testing.assert_array_equal(result["ch_names"], ["C4", "C3"])
        np.testing.assert_allclose(result["eeg"][1], data[:, 10:20], rtol=1e-7)
        np.testing.assert_array_equal(result["start_sample"], [0, 10])
        self.assertTrue(np.all(result["split"] == "test"))
        with tempfile.TemporaryDirectory() as temp:
            path = save_windows(Path(temp) / "eeg.npz", result)
            with np.load(path, allow_pickle=False) as restored:
                self.assertEqual(str(restored["units"]), "V")
                np.testing.assert_array_equal(restored["eeg"], result["eeg"])
            result["units"] = "unknown"
            with self.assertRaises(ValueError):
                save_windows(Path(temp) / "bad.npz", result)

    def test_invalid_units_and_window_lengths_rejected(self):
        raw = mne.io.RawArray(np.zeros((1, 20)), mne.create_info(["Cz"], 10, "eeg"), verbose="ERROR")
        args = dict(dataset="arithmetic_zyma2019", subject="00", run=1, condition="rest")
        with self.assertRaises(ValueError):
            _window_recording(raw, duration_s=.15, **args)
        raw.info["chs"][0]["unit"] = -1
        with self.assertRaises(ValueError):
            _window_recording(raw, duration_s=1, **args)

    def test_hub_scaling_history_overrides_stale_mne_unit_label(self):
        info = {"chs": [{"unit": {"value": 107}}]}
        attributes = {"raw_preproc_kwargs": [{"fn": "package._scale_data", "apply_on_array": True,
                                              "kwargs": {"factor": 1e6}}]}
        self.assertEqual(_hub_voltage_scale(attributes, info), 1e-6)
        attributes["raw_preproc_kwargs"][0]["fn"] = "unknown_normalization"
        with self.assertRaises(ValueError):
            _hub_voltage_scale(attributes, info)

    def test_native_zarr_provenance_and_pickle_free_export(self):
        import zarr

        with tempfile.TemporaryDirectory() as temp:
            store = Path(temp) / "dataset.zarr"
            group = zarr.open_group(str(store), mode="w")
            group.attrs.update({"raw_preproc_kwargs": json.dumps([
                {"fn": "_scale_data", "apply_on_array": True, "kwargs": {"factor": 1e6}}]),
                "window_kwargs": json.dumps([["windows", {"mapping": None}]])})
            record = group.create_group("recording_0")
            record.attrs.update({"description": {"subject": "080", "run": 0},
                                 "info": {"sfreq": 200., "ch_names": ["Cz"], "chs": [{"unit": 107}]}})
            record.create_array("data", data=np.ones((2, 1, 10), dtype=np.float32) * 10)
            (store / "recording_0/metadata.tsv").write_text(
                "\ttarget\temotion_label\tvideo_index\ti_start_in_trial\ti_stop_in_trial\n"
                "0\t3\tSadness\t11\t100\t110\n1\t2\tFear\t4\t400\t410\n", encoding="utf-8")
            result = load_hub_recording("faced", store, 0)
            np.testing.assert_allclose(result["eeg"], 10e-6)
            np.testing.assert_array_equal(result["trial"], ["11", "4"])
            np.testing.assert_array_equal(result["run"], ["0", "0"])
            np.testing.assert_array_equal(result["condition"], ["Sadness", "Fear"])
            self.assertTrue(np.all(result["split"] == "val"))
            path = save_windows(Path(temp) / "export.npz", result)
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    self.assertFalse(archive[key].dtype.hasobject)


if __name__ == "__main__":
    unittest.main()
