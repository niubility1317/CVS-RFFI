import unittest
import numpy as np
from experiments.pamf_u_history.bounded_response import fit,Calibration,Correction,RXS
from experiments.cvs_rff_physics.known_excitation import template


class BoundedResponseTests(unittest.TestCase):
    def test_RX_balancing_and_training_role(self):
        coeff=[];rx=[]
        for i,r in enumerate(RXS):
            mean=np.array([i,i*i,0.,0.]);n=2*(i+1)
            coeff.extend([mean]*n);rx.extend([r]*n)
        cal,meta=fit(np.asarray(coeff),np.asarray(rx),role='U_s')
        np.testing.assert_allclose(cal.center,np.mean([[i,i*i,0,0] for i in range(5)],0))
        np.testing.assert_allclose(cal.basis.T@cal.basis,np.diag([1,1,0,0]),atol=1e-12)
        self.assertEqual(meta['TX_labels_consumed'],0)
        with self.assertRaises(ValueError):fit(np.asarray(coeff),np.asarray(rx),role='V')
        with self.assertRaises(ValueError):cal.center[0]=0

    def test_zero_control_and_hard_gain_phase_bounds(self):
        z=template(np.arange(256),0.)
        iq=np.stack((z.real,z.imag))[None].astype(np.float32)
        iq=np.concatenate((iq,iq*2,np.zeros_like(iq)),0)
        correction=Correction(Calibration(np.array([1e6,-1e6,1e6,-1e6]),np.eye(4)[:2]))
        same,stats=correction(iq,strength=0)
        np.testing.assert_array_equal(same,iq);self.assertTrue(stats['identity_control'])
        out,stats=correction(iq)
        self.assertEqual(out.dtype,iq.dtype);self.assertTrue(np.isfinite(out).all())
        self.assertLessEqual(stats['max_abs_gain_db'],.5+1e-12);self.assertLessEqual(stats['max_abs_phase_rad'],.025+1e-12)
        p=np.square(out[:2].astype(np.float64)).sum((1,2))/np.square(iq[:2].astype(np.float64)).sum((1,2))
        self.assertTrue((np.abs(10*np.log10(p))<=.50001).all())
        np.testing.assert_allclose(out[1],2*out[0],rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(out[2],np.zeros_like(out[2]))

    def test_each_packet_is_independent(self):
        rng=np.random.default_rng(7);iq=rng.normal(size=(3,2,256)).astype(np.float32)
        correction=Correction(Calibration(np.zeros(4),np.eye(4)[:2]))
        together,_=correction(iq);single=np.concatenate([correction(iq[i:i+1])[0] for i in range(3)])
        np.testing.assert_allclose(together,single,rtol=1e-6,atol=1e-7)
        with self.assertRaises(ValueError):correction(iq,strength=.2)
        with self.assertRaises(ValueError):correction(iq*np.nan)


if __name__=='__main__':unittest.main()
