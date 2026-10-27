"""Window boundaries, reproducibility and leakage checks without anatomical assets."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Simulation_G2.nmm_bank import initialize_bank, save_bank_sample
from Simulation_G2.period_index import build_period_index


class PeriodIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = self.root / "bank"
        initialize_bank(self.bank, atlas_id="test", n_regions=3, projection_sha256="a"*64,
                        schema_version=2)

    def save(self, name, *, bank=None, group=None, parents=(), seed=1, count=40,
             resting=False, normalize=False, active=None):
        bank = self.bank if bank is None else bank
        active = ([] if resting else [0]) if active is None else active
        manifest = dict(signal_type="resting_jr" if resting else "spike_jr",
            output_srate_hz=20., active_regions=active, normalize=normalize,
            random_seed=seed, parent_data_group_ids=list(parents),
            model=dict(**{"class": "JansenRit"}, background_A=3.25,
                       active_A=3.25 if resting else 3.6))
        source = np.random.default_rng(seed).normal(size=(3, count))
        return save_bank_sample(bank, name, source, np.arange(count)*50., manifest, {},
                                storage={"codec": "dense", "protected_regions": active},
                                simulation_group_id=group)

    def build(self, output="index", **kwargs):
        return build_period_index([self.bank], self.root/output,
                                  **({"split_fractions": (1., 0., 0.), "baseline_duration_s": 1.} | kwargs))

    def test_default_two_second_baseline_windows_are_disjoint(self):
        self.save('long', count=80)
        build_period_index([self.bank], self.root/'default_baseline',
            split_fractions=(1.,0.,0.), source_mode='partition', source_snr_db=10)
        for row in self.rows('default_baseline'):
            self.assertEqual(row['config']['baseline_duration_s'], 2.)
            base, event = row['baseline_background'], row['foreground']
            self.assertEqual(base['stop_sample'] - base['start_sample'], 40)
            self.assertTrue(base['stop_sample'] <= event['start_sample'] or base['start_sample'] >= event['stop_sample'])

    def rows(self, output="index"):
        return [json.loads(line) for line in (self.root/output/"periods.jsonl").read_text().splitlines()]

    def test_fixed_boundaries_offsets_and_checksum(self):
        entry = self.save("one")
        manifest = self.build(variants_per_period=2)
        rows = self.rows()
        self.assertEqual([(r["foreground"]["start_sample"], r["foreground"]["stop_sample"])
                          for r in rows], [(0, 20), (0, 20), (20, 40), (20, 40)])
        self.assertEqual(manifest["table"]["count"], 4)
        self.assertEqual(manifest["statistics"]["discarded_tail_samples"], 0)
        self.assertEqual(rows[0]["foreground"]["source_sha256"], entry["source_sha256"])
        with (self.root/"index"/"periods.jsonl").open("rb") as handle:
            for offset, expected in zip(np.load(self.root/"index"/"offsets.npy"), rows):
                handle.seek(int(offset))
                self.assertEqual(json.loads(handle.readline()), expected)
        np.testing.assert_array_equal(np.load(self.root/"index"/"train_indices.npy"), np.arange(4))

    def test_explicit_sample_selection_and_missing_id(self):
        self.save('one'); self.save('two',seed=2)
        result=self.build(sample_ids=['two'])
        self.assertEqual(result['statistics']['recordings'],1)
        self.assertTrue(all(row['foreground']['sample_id']=='two' for row in self.rows()))
        with self.assertRaises(ValueError):
            self.build('missing',sample_ids=['absent'])

    def test_sliding_tail_drop_error_and_grid(self):
        self.save("one", count=45)
        manifest = self.build(stride_s=.5)
        self.assertEqual([r["foreground"]["start_sample"] for r in self.rows()], [0, 10, 20])
        self.assertEqual(manifest["statistics"]["discarded_tail_samples"], 5)
        for kwargs in ({"stride_s": .51}, {"period_s": .01}, {"period_s": 3.},
                       {"tail_policy": "error"}, {"stride_s": 0}, {"tail_policy": "pad"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build("invalid", **kwargs)
        self.assertFalse((self.root/"invalid").exists())

    def test_connected_parents_and_codec_views_share_one_split(self):
        self.save("one", group="1"*64, parents=["shared-real-person"])
        self.save("two", group="2"*64, parents=["shared-real-person", "second-parent"], seed=2)
        self.save("third", group="3"*64, parents=["second-parent"], seed=3)
        self.save("codec_view", group="1"*64, seed=4)
        self.save("independent_a", group="a"*64, seed=5)
        self.save("independent_b", group="b"*64, seed=6)
        manifest = self.build(split_fractions=(.8, .1, .1))
        joined = {manifest["parent_splits"][key] for key in
                  ["1"*64, "2"*64, "3"*64, "shared-real-person", "second-parent"]}
        self.assertEqual(len(joined), 1)
        self.assertEqual(set(manifest["parent_splits"].values()), {"train", "val", "test"})
        parent_folds = {}
        for row in self.rows():
            for parent in row["parent_data_group_ids"]:
                parent_folds.setdefault(parent, set()).add(row["split"])
        self.assertTrue(all(len(folds) == 1 for folds in parent_folds.values()))

    def test_historical_nested_conversion_ancestry_is_recovered(self):
        self.save('one',group='1'*64)
        self.save('two',group='2'*64,seed=2)
        self.save('three',group='3'*64,seed=3)
        self.save('four',group='4'*64,seed=4)
        # Index construction reads metadata only; exercise historical wrapper layout.
        path=self.bank/'bank.json'; bank=json.loads(path.read_text())
        for name in ('one','two'):
            bank['samples'][name]['provenance']={'original_provenance':{
                'original_provenance':{'parent_data_group_ids':['same-real-subject']}}}
        path.write_text(json.dumps(bank))
        result=self.build(split_fractions=(.8,.1,.1))
        self.assertEqual(result['parent_splits']['1'*64],result['parent_splits']['2'*64])
        self.assertEqual(result['parent_splits']['1'*64],result['parent_splits']['same-real-subject'])

    def test_reproducible_variants_do_not_depend_on_destination(self):
        self.save("one")
        options = dict(source_mode="partition", source_snr_db=20., variants_per_period=7,
                       sensor_snrs=(0., 10.), sensor_noise_types=("white", "pink"),
                       spatial_profiles=("uniform", "decay", "diffuse"))
        self.build("first", **options)
        self.build("second", **options)
        self.assertEqual(self.rows("first"), self.rows("second"))
        rows = self.rows("first")
        self.assertEqual(len({r["config"]["sensor_seed"] for r in rows}), len(rows))
        self.assertTrue(all(r["config"]["selected_regions"] == [0] for r in rows))

    def test_independent_background_requires_rest_different_seed_and_same_fold(self):
        foreground = self.save("one", seed=10)
        bg = self.root/"background"
        initialize_bank(bg, atlas_id="test", n_regions=3, projection_sha256="a"*64, schema_version=2)
        self.save("rest", bank=bg, seed=11, resting=True)
        self.build(background_bank_dirs=[bg], source_snr_db=20.)
        row = self.rows()[0]
        self.assertEqual(row["background"]["sample_id"], "rest")
        self.assertIn(foreground["simulation_group_id"], row["parent_data_group_ids"])
        self.assertEqual(len(row["parent_data_group_ids"]), 2)
        self.assertLessEqual(row["background"]["stop_sample"], 40)
        self.assertGreaterEqual(row["background"]["start_sample"], 0)

    def test_background_rejects_same_seed_and_overlapping_parent(self):
        self.save("one", seed=10, parents=["person"])
        for name, seed, parents in [("same_seed", 10, []), ("same_parent", 11, ["person"])]:
            bg = self.root/name
            initialize_bank(bg, atlas_id="test", n_regions=3, projection_sha256="a"*64, schema_version=2)
            self.save("rest", bank=bg, seed=seed, parents=parents, resting=True)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "independent same-split"):
                self.build(name+"_index", background_bank_dirs=[bg], source_snr_db=20.)

    def test_background_rejects_invalid_model_normalization_or_active_region(self):
        self.save("one")
        for name, options in [("active", {"active": [1]}), ("normalized", {"normalize": True}),
                              ("not_rest", {"resting": False})]:
            bg = self.root/name
            initialize_bank(bg, atlas_id="test", n_regions=3, projection_sha256="a"*64, schema_version=2)
            self.save("rest", bank=bg, seed=33, **({"resting": True} | options))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "eligible"):
                self.build(name+"_index", background_bank_dirs=[bg], source_snr_db=20.)

    def test_validation_and_no_overwrite(self):
        self.save("one")
        for kwargs in ({"split_fractions": (.8, .1, .1)}, {"split_fractions": (1., 1., 0.)},
                       {"source_mode": "partition"}, {"source_snr_db": 20.},
                       {"spatial_profiles": ["diffuse"]}, {"sensor_noise_types": ["real"]},
                       {"sensor_snrs": [float("nan")]}, {"variants_per_period": 0},
                       {"source_mode": "selected", "selected_regions": [0, 0]},
                       {"source_mode": "partition", "source_snr_db": 20., "selected_regions": [0, 1, 2]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build("invalid", **kwargs)
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()

    def test_selected_without_background_is_valid_and_does_not_claim_source_snr(self):
        self.save("one")
        self.build(source_mode="selected", selected_regions=[0, 1, 2])
        row = self.rows()[0]
        self.assertIsNone(row["background"])
        self.assertIsNone(row["config"]["source_snr_db"])

    def test_each_positive_fold_pairs_only_its_own_independent_background(self):
        for index in range(6):
            self.save(f"foreground{index}", seed=index+1)
        bg = self.root/"background"
        initialize_bank(bg, atlas_id="test", n_regions=3, projection_sha256="a"*64, schema_version=2)
        for index in range(6):
            self.save(f"background{index}", bank=bg, seed=index+100, resting=True)
        manifest = self.build(background_bank_dirs=[bg], source_snr_db=20., split_fractions=(.8, .1, .1))
        seen = {fold: set() for fold in ("train", "val", "test")}
        for row in self.rows():
            seen[row["split"]].update(row["parent_data_group_ids"])
            self.assertEqual(manifest["parent_splits"][row["background"]["simulation_group_id"]], row["split"])
        self.assertTrue(all(seen.values()))
        self.assertFalse(seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"])

    def test_filtered_record_still_connects_shared_parent_ancestry(self):
        self.save("left", parents=["left-person"], seed=1)
        self.save("right", parents=["right-person"], seed=2)
        self.save("excluded_bridge", parents=["left-person", "right-person"], resting=True, seed=3)
        self.save("independent_a", seed=4)
        self.save("independent_b", seed=5)
        manifest = self.build(signal_types=["spike_jr"], split_fractions=(.8, .1, .1))
        self.assertEqual(manifest["parent_splits"]["left-person"], manifest["parent_splits"]["right-person"])
        self.assertEqual(manifest["statistics"]["recordings"], 4)

    def test_background_cannot_share_a_transitive_parent_component(self):
        self.save("foreground", parents=["left-person"], seed=1)
        self.save("bridge", parents=["left-person", "right-person"], seed=2, resting=True)
        bg = self.root/"background"
        initialize_bank(bg, atlas_id="test", n_regions=3, projection_sha256="a"*64, schema_version=2)
        self.save("background", bank=bg, parents=["right-person"], seed=3, resting=True)
        with self.assertRaisesRegex(ValueError, "independent same-split"):
            self.build(signal_types=["spike_jr"], background_bank_dirs=[bg], source_snr_db=20.)


if __name__ == "__main__":
    unittest.main()
