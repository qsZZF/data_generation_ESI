"""Geodesic WX invariants on a small mesh, not a physiological validation."""
import unittest
import numpy as np

from Simulation_G1.geometry import compute_basis
from Simulation_G1.wx_pilot import build_weights
from Simulation_G1.wx_partitioned import build_partitioned_weights


class PartitionedWXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = np.meshgrid(np.arange(11)*10., np.arange(9)*10.)
        cls.vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        faces = []
        for j in range(8):
            for i in range(10):
                a = j*11+i
                faces.extend(((a,a+1,a+11),(a+1,a+12,a+11)))
        cls.faces = np.asarray(faces)
        cls.bundle = compute_basis(cls.vertices, cls.faces, 32)
        cls.maps = np.column_stack([np.exp(-((x.ravel()-cx)**2+(y.ravel()-40)**2)/1800)
                                   for cx in (25,50,75)])
        cls.mask = np.ones(len(x.ravel()), dtype=bool)
        cls.mask[:2] = False

    def build(self, **kwargs):
        return build_partitioned_weights(self.bundle,self.maps,self.mask,self.vertices,self.faces,
                                         length_mm=15., **kwargs)

    def test_reproducible_signed_masked_unit_columns(self):
        one, two = self.build(), self.build()
        np.testing.assert_array_equal(one['W'],two['W'])
        np.testing.assert_array_equal(one['W'][~self.mask],0.)
        np.testing.assert_allclose(np.sum(one['mass'][:,None]*one['W']**2,axis=0),1)
        for group in range(3):
            columns = np.flatnonzero(one['groups']==group)
            self.assertTrue(np.all(one['W'][:,columns[1:]]*one['W'][:,columns[0],None]>=0))
            self.assertEqual(len(set(one['metadata']['centers_zero_based'][group])),2)

    def test_less_redundant_local_columns_on_grid(self):
        old = build_weights(self.bundle,self.maps,self.mask,length_mm=15.)
        new = self.build()
        def overlap(w):
            return np.mean([abs(np.dot(w['mass']*w['W'][:,3*g+1],w['W'][:,3*g+2])) for g in range(3)])
        self.assertLess(overlap(new),overlap(old))
        np.testing.assert_array_equal(old['W'][:,::3],new['W'][:,::3])

    def test_disconnected_mesh_kernels_do_not_cross(self):
        vertices = np.vstack((self.vertices,self.vertices+np.array([200.,0,0])))
        faces = np.vstack((self.faces,self.faces+len(self.vertices)))
        bundle = compute_basis(vertices,faces,32)
        maps = np.vstack((np.ones((len(self.vertices),1)),np.ones((len(self.vertices),1))))
        result = build_partitioned_weights(bundle,maps,np.ones(len(vertices),dtype=bool),vertices,faces,
                                            n_networks=1,length_mm=15.)
        centers = result['metadata']['centers_zero_based'][0]
        self.assertNotEqual(centers[0]//len(self.vertices),centers[1]//len(self.vertices))
        for column,center in enumerate(centers,1):
            other = np.arange(len(vertices))//len(self.vertices) != center//len(self.vertices)
            np.testing.assert_array_equal(result['W'][other,column],0)

    def test_invalid_width_and_geometry(self):
        for width in (0,-1,np.nan):
            with self.assertRaises(ValueError):
                self.build(local_width_mm=width)
        with self.assertRaisesRegex(ValueError,'match'):
            build_partitioned_weights(self.bundle,self.maps,self.mask,self.vertices+1,self.faces)


if __name__=='__main__':
    unittest.main()
