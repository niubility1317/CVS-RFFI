import unittest
import numpy as np
from experiments.pamf_u_history.response_transfer_geometry import cycle_statistics,conservative_projection


class ResponseGeometryTest(unittest.TestCase):
    def test_exact_potential_table_has_zero_cycle_and_is_unchanged(self):
        p=np.arange(5)[:,None]*np.linspace(.001,.004,8)[None,:]*(1+.3j);v=np.exp(p[None,:,:]-p[:,None,:])
        g=cycle_statistics(v);self.assertLess(g['triangles']['max_complex_error'],1e-14)
        projected,_=conservative_projection(v);np.testing.assert_allclose(projected,v,rtol=1e-14,atol=1e-14)

    def test_inconsistent_edges_are_detected_and_projection_restores_cycles(self):
        v=np.ones((5,5,16),complex);v[0,1]=np.exp(.05+.02j)
        before=cycle_statistics(v);self.assertGreater(before['reciprocal']['max_complex_error'],.01);self.assertGreater(before['nonconservative_phase_energy_fraction'],0)
        projected,_=conservative_projection(v);after=cycle_statistics(projected)
        self.assertLess(after['triangles']['max_complex_error'],1e-14);self.assertLess(after['reciprocal']['max_complex_error'],1e-14)
        self.assertLessEqual(after['per_edge_max_abs_gain_db'],.5+1e-12);self.assertLessEqual(after['per_edge_max_abs_phase_rad'],.025+1e-12)

    def test_receiver_permutation_and_invalid_table(self):
        rng=np.random.default_rng(7);v=np.exp(rng.uniform(-.04,.04,(5,5,11))+1j*rng.uniform(-.02,.02,(5,5,11)))
        for r in range(5):v[r,r]=1
        p=np.array([3,0,4,1,2]);a,_=conservative_projection(v);b,_=conservative_projection(v[p][:,p]);np.testing.assert_allclose(b,a[p][:,p],rtol=1e-13,atol=1e-13)
        with self.assertRaises(ValueError):cycle_statistics(np.zeros((5,5,11),complex))
        with self.assertRaises(ValueError):conservative_projection(v,max_gain_db=0)


if __name__=='__main__':unittest.main()
