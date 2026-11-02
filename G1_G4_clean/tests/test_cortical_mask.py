"""Cortical eligibility applies to seeds and complete patch supports."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from Simulation_G1.g1_simulation_core import G1SimulationConfig, G1SourceSimulator


def simulator(mask=None):
    vertices = np.array([[20., 0, 0], [21, 0, 0], [22, 0, 0],
                         [20, 1, 0], [21, 1, 0], [22, 1, 0]])
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    rows, cols = [], []
    for face in faces:
        for i in face:
            for j in face:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    conn = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
    return G1SourceSimulator(np.ones((2, 6)), vertices, faces, conn,
                             cortex_mask=mask, cortex_mask_provenance='test_mask' if mask is not None else None)


class CorticalMaskTests(unittest.TestCase):
    def test_excluded_explicit_seed_rejected(self):
        sim = simulator([True, False, True, True, False, True])
        cfg = G1SimulationConfig(seed_vertices=[1], patch_area_mm2=.1)
        with self.assertRaisesRegex(ValueError, 'cortex mask'):
            sim.generate_source(cfg)

    def test_explicit_support_cannot_cross_mask(self):
        sim = simulator([True, False, True, True, False, True])
        cfg = G1SimulationConfig(seed_vertices=[0], patch_vertices=[[0, 1, 3]], patch_area_mm2=.1)
        with self.assertRaisesRegex(ValueError, 'cortex mask'):
            sim.generate_source(cfg)

    def test_growth_cannot_cross_excluded_bridge(self):
        sim = simulator([True, False, True, True, False, True])
        cfg = G1SimulationConfig(seed_vertices=[0], patch_area_mm2=.8)
        with self.assertRaisesRegex(ValueError, 'connected'):
            sim.generate_source(cfg)

    def test_automatic_disjoint_supports_stay_in_mask(self):
        mask = np.array([True, False, True, True, False, True])
        sim = simulator(mask)
        for seed in range(12):
            cfg = G1SimulationConfig(n_patch=2, patch_area_mm2=.45,
                                     min_seed_distance_mm=0, random_seed=seed)
            source, metadata = sim.generate_source(cfg)
            self.assertTrue(np.all(source[~mask] == 0))
            supports = [set(p['vertices']) for p in metadata['patches']]
            self.assertFalse(supports[0] & supports[1])
            self.assertEqual(metadata['cortex_mask']['eligible_vertices'], 4)
            self.assertEqual(metadata['cortex_mask']['provenance'], 'test_mask')
        self.assertTrue(mask[sim.sample_spaced_vertices(4, 0, np.random.default_rng(2))].all())

    def test_mask_validation_and_unspecified_metadata(self):
        for invalid in ([True], [0, 1, 2, 1, 1, 1], [False] * 6):
            with self.assertRaises(ValueError):
                simulator(invalid)
        sim = simulator()
        self.assertEqual(sim.cortex_mask_metadata['provenance'], 'unspecified_all_vertices')
        self.assertEqual(sim.cortex_mask_metadata['eligible_vertices'], 6)
        cortex = {'Vertices': sim.vertices, 'Faces': sim.faces, 'VertConn': sim.vert_conn,
                  'CortexMask': [True, False, True, True, False, True],
                  'CortexMaskProvenance': 'embedded'}
        clone = G1SourceSimulator.from_brainstorm_cortex(sim.leadfield, cortex)
        self.assertEqual(clone.cortex_mask_metadata['provenance'], 'embedded')

    def test_optional_embedded_atlas(self):
        import h5py
        from Simulation_G1.cortical import load_cortex_mask
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cortex.mat'
            with h5py.File(path, 'w') as file:
                cortex = file.create_group('Cortex')
                cortex.create_dataset('Vertices', data=np.zeros((3, 6)))
                cortex.create_dataset('Faces', data=np.ones((3, 1)))
            mask, provenance = load_cortex_mask(path, 6)
            self.assertIsNone(mask)
            self.assertEqual(provenance, 'unspecified_all_vertices')
            with h5py.File(path, 'a') as file:
                atlas = file['Cortex'].create_group('Atlas')
                def text(name, value):
                    return file.create_dataset(name, data=np.array([ord(c) for c in value], dtype='uint16')).ref
                atlas.create_dataset('Name', data=[text('atlas_name', 'Desikan-Killiany')], dtype=h5py.ref_dtype)
                scouts = file.create_group('scouts')
                atlas.create_dataset('Scouts', data=[scouts.ref], dtype=h5py.ref_dtype)
                scouts.create_dataset('Label', data=[text('a', 'precentral L'), text('b', 'unknown R'),
                                                     text('c', 'corpuscallosum L')], dtype=h5py.ref_dtype)
                refs = [file.create_dataset('v' + str(i), data=values).ref
                        for i, values in enumerate(([1, 2, 3], [4, 5], [6]))]
                scouts.create_dataset('Vertices', data=refs, dtype=h5py.ref_dtype)
            mask, provenance = load_cortex_mask(path, 6)
            np.testing.assert_array_equal(mask, [1, 1, 1, 0, 0, 0])
            self.assertIn('Desikan', provenance)


if __name__ == '__main__':
    unittest.main()
