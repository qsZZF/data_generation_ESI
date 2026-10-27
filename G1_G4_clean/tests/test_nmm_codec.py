"""Storage fidelity, fixed-rank behavior and schema-compatible bank replay."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Simulation_G2.nmm_bank import initialize_bank, load_bank_sample, read_bank, save_bank_sample
from Simulation_G2.nmm_codec import decode_regions, encode_regions


class RegionalCodecTests(unittest.TestCase):
    def setUp(self):
        self.source = np.random.default_rng(14).normal(size=(8, 128)).astype(np.float32)
        self.source += np.arange(8, dtype=np.float32)[:, None]

    def roundtrip(self, codec, source=None, protected=(5, 1), rank=2):
        source = self.source if source is None else source
        options = {"codec": codec, "protected_regions": list(protected)}
        if codec == "lowrank":
            options["rank"] = rank
        arrays, metadata = encode_regions(source, options)
        decoded = decode_regions(arrays, metadata, *source.shape)
        return decoded, arrays, metadata

    def test_all_codecs_preserve_protected_rows_and_row_order_exactly(self):
        for codec in ("dense", "int16", "lowrank"):
            with self.subTest(codec=codec):
                decoded, arrays, metadata = self.roundtrip(codec)
                np.testing.assert_array_equal(decoded[[5, 1]], self.source[[5, 1]])
                self.assertEqual(decoded.dtype, np.float32)
                self.assertNotIn("region_source", arrays)
                self.assertEqual(metadata["protected_regions"], [5, 1])
                if codec == "dense":
                    np.testing.assert_array_equal(decoded, self.source)

    def test_quantization_error_bound_and_constant_rows(self):
        source = self.source.copy()
        source[2] = 7.25
        source[3] = 0
        decoded, arrays, _ = self.roundtrip("int16", source, protected=())
        error = np.abs(decoded.astype(float) - source)
        rounding = np.finfo(np.float32).eps * np.abs(source).max(axis=1)
        bound = arrays["background_scale"] / 2 + rounding
        self.assertTrue(np.all(error <= bound[:, None]))
        np.testing.assert_array_equal(decoded[[2, 3]], source[[2, 3]])

    def test_empty_foreground_all_foreground_and_constant_background(self):
        for codec in ("dense", "int16", "lowrank"):
            for protected in ((), tuple(range(8))):
                with self.subTest(codec=codec, protected=protected):
                    source = np.full((8, 128), 3.125, dtype=np.float32)
                    decoded, _, _ = self.roundtrip(codec, source, protected=protected)
                    np.testing.assert_array_equal(decoded, source)

    def test_lowrank_fixed_k_keeps_loss_and_clips_only_to_dimensions(self):
        decoded, _, metadata = self.roundtrip("lowrank", protected=(), rank=1)
        self.assertEqual(metadata["rank_stored"], 1)
        centered = self.source - self.source.mean(axis=1, keepdims=True)
        self.assertGreater(np.linalg.norm(decoded - self.source) / np.linalg.norm(centered), .5)
        decoded, _, metadata = self.roundtrip("lowrank", protected=(), rank=999)
        self.assertEqual(metadata["rank_requested"], 999)
        self.assertEqual(metadata["rank_stored"], 8)
        np.testing.assert_allclose(decoded, self.source, atol=2e-6)
        short = self.source[:, :3]
        decoded, _, metadata = self.roundtrip("lowrank", short, protected=(), rank=999)
        self.assertEqual(metadata["rank_stored"], 2)
        np.testing.assert_allclose(decoded, short, atol=2e-6)

    def test_zero_rank_is_mean_only(self):
        decoded, _, metadata = self.roundtrip("lowrank", protected=(), rank=0)
        self.assertEqual(metadata["rank_stored"], 0)
        expected = np.broadcast_to(self.source.mean(axis=1, dtype=float)[:, None], self.source.shape).astype(np.float32)
        np.testing.assert_array_equal(decoded, expected)

    def test_rejects_invalid_options_and_nonfinite_sources(self):
        for options in ({"codec": "unknown"}, {"codec": "dense", "protected_regions": [1, 1]},
                        {"codec": "dense", "protected_regions": [8]},
                        {"codec": "dense", "protected_regions": [True]},
                        {"codec": "dense", "protected_regions": [1.0]},
                        {"codec": "lowrank", "rank": -1}, {"codec": "lowrank", "rank": 1.5},
                        {"codec": "lowrank", "rank": True}, {"codec": "dense", "unexpected": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                encode_regions(self.source, options)
        for source in (self.source * np.nan, self.source[:, :1], self.source[0], self.source.astype(complex)):
            with self.assertRaises(ValueError):
                encode_regions(source, {"codec": "dense"})

    def test_rejects_malformed_arrays_and_rank_metadata(self):
        _, arrays, metadata = self.roundtrip("lowrank")
        for replacement in (arrays["background_spatial"][:-1], arrays["background_spatial"].astype(np.float64),
                            arrays["background_spatial"] * np.nan):
            with self.assertRaises(ValueError):
                decode_regions({**arrays, "background_spatial": replacement}, metadata, *self.source.shape)
        with self.assertRaises(ValueError):
            decode_regions({**arrays, "region_source": self.source}, metadata, *self.source.shape)
        with self.assertRaises(ValueError):
            decode_regions(arrays, {**metadata, "rank_stored": 5}, *self.source.shape)
        _, arrays, metadata = self.roundtrip("int16")
        arrays["background_scale"][0] = 0
        with self.assertRaises(ValueError):
            decode_regions(arrays, metadata, *self.source.shape)


class CodecBankTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.source = np.random.default_rng(12).normal(size=(4, 256)).astype(np.float32)
        self.times = np.arange(256) * 1000 / 256
        self.manifest = {"signal_type": "spike", "output_srate_hz": 256, "active_regions": [2],
                         "n_regions_output": 4, "model": {"class": "JansenRit"}}

    def initialize(self, schema=2):
        initialize_bank(self.path, atlas_id="test", n_regions=4, projection_sha256="a" * 64,
                        schema_version=schema)

    def save(self, sample="one", **kwargs):
        return save_bank_sample(self.path, sample, self.source, self.times, self.manifest, {}, **kwargs)

    def rewrite_archive(self, mutate):
        archive = self.path / "one.npz"
        with np.load(archive, allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        mutate(payload)
        np.savez_compressed(archive, **payload)
        bank = read_bank(self.path)
        bank["samples"]["one"]["source_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.path / "bank.json").write_text(json.dumps(bank))

    def test_schema1_unchanged_default_and_rejects_codec_request(self):
        self.initialize(1)
        self.save()
        source, _, entry = load_bank_sample(self.path, "one")
        np.testing.assert_array_equal(source, self.source)
        self.assertNotIn("storage", entry)
        with np.load(self.path / "one.npz") as data:
            self.assertEqual(set(data.files), {"region_source", "time_ms", "record_json"})
        with self.assertRaisesRegex(ValueError, "schema 2"):
            self.save("other", storage={"codec": "dense"})

    def test_schema2_default_dense_and_all_codecs_replay(self):
        self.initialize()
        self.save()
        decoded, _, entry = load_bank_sample(self.path, "one")
        np.testing.assert_array_equal(decoded, self.source)
        self.assertEqual(entry["storage"]["protected_regions"], [2])
        for codec in ("dense", "int16", "lowrank"):
            entry = self.save(codec, storage={"codec": codec, "rank": 1, "protected_regions": [2]})
            decoded, times, loaded = load_bank_sample(self.path, codec)
            np.testing.assert_array_equal(decoded[2], self.source[2])
            np.testing.assert_array_equal(times, self.times)
            self.assertEqual(entry, loaded)
            with np.load(self.path / f"{codec}.npz") as data:
                self.assertNotIn("region_source", data.files)

    def test_derived_views_preserve_explicit_simulation_group(self):
        self.initialize()
        group = "d" * 64
        entries = [self.save(codec, storage={"codec": codec, "rank": 1}, simulation_group_id=group)
                   for codec in ("dense", "int16", "lowrank")]
        self.assertTrue(all(entry["simulation_group_id"] == group for entry in entries))
        with self.assertRaises(ValueError):
            self.save("bad", simulation_group_id="not-a-hash")

    def test_codec_request_cannot_drop_configured_foreground(self):
        self.initialize()
        for name,options in [('missing',{'codec':'lowrank','rank':1}),
                             ('extra',{'codec':'int16','protected_regions':[0]})]:
            entry=self.save(name,storage=options)
            decoded,_,_=load_bank_sample(self.path,name)
            self.assertIn(2,entry['storage']['protected_regions'])
            np.testing.assert_array_equal(decoded[2],self.source[2])

    def test_unindexed_recording_and_wrong_schema_rejected(self):
        self.initialize()
        (self.path / "one.npz").write_bytes(b"orphan")
        with self.assertRaisesRegex(ValueError, "Unindexed"):
            self.save()
        self.assertEqual((self.path / "one.npz").read_bytes(), b"orphan")
        with self.assertRaises(ValueError):
            self.initialize(1)

    def test_checksum_metadata_and_malformed_dimensions_rejected(self):
        self.initialize()
        self.save(storage={"codec": "lowrank", "rank": 1})
        archive = self.path / "one.npz"
        original = archive.read_bytes()
        archive.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_bank_sample(self.path, "one")
        archive.write_bytes(original)
        self.rewrite_archive(lambda payload: payload.__setitem__("background_temporal", np.zeros((2, 256), dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "codec array"):
            load_bank_sample(self.path, "one")
        archive.write_bytes(original)
        bank = read_bank(self.path)
        bank["samples"]["one"]["source_sha256"] = hashlib.sha256(original).hexdigest()
        bank["samples"]["one"]["storage"]["rank_stored"] = 2
        (self.path / "bank.json").write_text(json.dumps(bank))
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_bank_sample(self.path, "one")


if __name__ == "__main__":
    unittest.main()
