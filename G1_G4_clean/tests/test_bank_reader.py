"""Native-rate window decoding, integrity, ownership and bounded caching."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Simulation_G2.bank_reader import BankWindowReader
from Simulation_G2.nmm_bank import initialize_bank, load_bank_sample, save_bank_sample
from Simulation_G2.nmm_codec import decode_regions, encode_regions


class BankReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = np.random.default_rng(7).normal(size=(5, 31)).astype(np.float32)

    def bank(self, codec=None):
        path = self.root / str(codec)
        initialize_bank(path, atlas_id="test", n_regions=5, projection_sha256="a" * 64,
                        schema_version=1 if codec is None else 2)
        options = None if codec is None else {"codec": codec, "protected_regions": [2]}
        if codec == "lowrank":
            options["rank"] = 2
        for name in ("a", "b"):
            save_bank_sample(path, name, self.source, np.arange(31) * 1000 / 256,
                             {"signal_type": "spike", "output_srate_hz": 256, "active_regions": [1]},
                             {"seed": 8}, storage=options)
        return path

    def test_all_codecs_windows_match_full_and_outputs_are_owned(self):
        for codec in (None, "dense", "int16", "lowrank"):
            with self.subTest(codec=codec):
                path = self.bank(codec)
                full, _, _ = load_bank_sample(path, "a")
                reader = BankWindowReader(path)
                self.assertEqual(reader.bank_sha256, hashlib.sha256((path / "bank.json").read_bytes()).hexdigest())
                for first, last in ((0, 1), (2, 13), (13, 31), (0, 31)):
                    actual, entry = reader.read("a", first, last)
                    np.testing.assert_array_equal(actual, full[:, first:last])
                    actual[:] = 99
                    entry["active_regions"].clear()
                snapshot = reader.bank
                snapshot["samples"].clear()
                actual, entry = reader.read("a", 0, 31)
                np.testing.assert_array_equal(actual, full)
                self.assertEqual(entry["active_regions"], [1])
                self.assertEqual(reader.cache_info()["misses"], 1)

    def test_half_open_boundary_validation(self):
        reader = BankWindowReader(self.bank())
        for first, last in ((-1, 2), (0, 32), (31, 31), (3, 2), (0.0, 2), (False, 2)):
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                reader.read("a", first, last)
        for name in ("missing", "../a"):
            with self.assertRaises(ValueError):
                reader.read(name, 0, 2)

    def test_lru_bounds_oversize_and_clear(self):
        path = self.bank()
        reader = BankWindowReader(path, max_cache_bytes=self.source.nbytes)
        for name in ("a", "a", "b", "a"):
            reader.read(name, 0, 2)
        self.assertEqual(reader.cache_info(), {"bytes": self.source.nbytes, "entries": 1,
                         "hits": 1, "misses": 3, "max_cache_bytes": self.source.nbytes})
        reader.clear()
        self.assertEqual(reader.cache_info()["bytes"], 0)
        self.assertEqual(reader.cache_info()["hits"], 0)
        reader = BankWindowReader(path, max_cache_bytes=1)
        for _ in range(2):
            reader.read("a", 0, 2)
        self.assertEqual(reader.cache_info()["entries"], 0)
        self.assertEqual(reader.cache_info()["misses"], 2)
        with self.assertRaises(ValueError):
            BankWindowReader(path, max_cache_bytes=-1)

    def test_file_corruption_on_cached_entry_is_rejected(self):
        path = self.bank("lowrank")
        reader = BankWindowReader(path)
        reader.read("a", 0, 2)
        with (path / "a.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            reader.read("a", 0, 2)
        self.assertEqual(reader.cache_info()["bytes"], 0)

    def test_index_changes_reject_cached_and_uncached_reads(self):
        path = self.bank()
        reader = BankWindowReader(path)
        reader.read("a", 0, 2)
        with (path / "bank.json").open("a") as handle:
            handle.write(" ")
        for name in ("a", "b"):
            with self.assertRaisesRegex(ValueError, "index changed"):
                reader.read(name, 0, 2)

    def test_threaded_reads_share_one_cache_entry(self):
        reader = BankWindowReader(self.bank("int16"))
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda _: reader.read("a", 2, 12)[0], range(12)))
        for output in outputs:
            np.testing.assert_array_equal(output, outputs[0])
        self.assertEqual(reader.cache_info()["misses"], 1)
        self.assertEqual(reader.cache_info()["hits"], 11)

    def test_cached_lowrank_decodes_requested_columns_only(self):
        reader = BankWindowReader(self.bank("lowrank"))
        from Simulation_G2 import bank_reader
        original = bank_reader._decode_regions_window
        with patch.object(bank_reader, "_decode_regions_window", wraps=original) as decode:
            reader.read("a", 5, 9)
            reader.read("a", 10, 12)
        self.assertEqual([call.args[-2:] for call in decode.call_args_list], [(5, 9), (10, 12)])
        self.assertEqual(reader.cache_info()["misses"], 1)

    def test_window_validates_entire_payload_not_only_requested_columns(self):
        payload, metadata = encode_regions(self.source, {"codec": "lowrank", "rank": 2})
        payload["background_temporal"][0, -1] = np.nan
        with self.assertRaisesRegex(ValueError, "background_temporal"):
            decode_regions(payload, metadata, 5, 31, start=0, stop=2)

    def test_forged_recording_clock_or_protection_is_rejected(self):
        path = self.bank("dense")
        source_path = path / "a.npz"
        with np.load(source_path, allow_pickle=False) as data:
            original = {name: data[name] for name in data.files}
        for mode in ("clock", "protection", "shape"):
            payload = {key: value.copy() for key, value in original.items()}
            record = json.loads(str(payload["record_json"]))
            if mode == "clock":
                payload["time_ms"][-1] += 10
            elif mode == "protection":
                record["active_regions"] = [4]
                payload["record_json"] = np.asarray(json.dumps(record, sort_keys=True))
            else:
                payload["background_source"] = payload["background_source"][:, :-1]
            np.savez_compressed(source_path, **payload)
            index_path = path / "bank.json"
            index = json.loads(index_path.read_text())
            index["samples"]["a"] = {**record, "source_file": "a.npz",
                                     "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
            index_path.write_text(json.dumps(index))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                BankWindowReader(path).read("a", 0, 2)


if __name__ == "__main__":
    unittest.main()
