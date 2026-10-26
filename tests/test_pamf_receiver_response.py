import unittest
import numpy as np
from experiments.pamf_u_history.receiver_response import Response,decompose,RXS,DAYS
from experiments.cvs_rff_physics.known_excitation import template
from experiments.pamf_u_history.receiver_response_probe import validate_context


class ReceiverResponseTests(unittest.TestCase):
    def test_ideal_scalar_gain_and_zero_inputs(self):
        r=Response();z=template(np.arange(256),3.5)*np.exp(1j*.001*np.arange(256))
        x=np.stack((z.real,z.imag))[None];a,q=r(x);b,_=r(np.stack(((z*2j).real,(z*2j).imag))[None])
        np.testing.assert_allclose(a,b,atol=1e-9);self.assertLess(np.linalg.norm(a),1e-8);self.assertGreater(q[0,0],.99)
        a,q=r(np.zeros((2,2,256)));self.assertTrue(np.isfinite(a).all() and np.isfinite(q).all())
        with self.assertRaises(ValueError):r(np.zeros((2,2,255)))

    def test_matches_existing_PAMF_reference_branch(self):
        import torch
        from experiments.cvs_reference_identity.model import ExcitationResponse
        rng=np.random.default_rng(27);z=template(np.arange(256),2.125)[None]+.03*(rng.normal(size=(4,256))+1j*rng.normal(size=(4,256)))
        x=np.stack((z.real,z.imag),1);r=Response();actual,_=r(x)
        with torch.no_grad():value=ExcitationResponse('reference_response').double().components(torch.tensor(x.tolist(),dtype=torch.float64))['features']
        values=np.asarray(value.tolist());bounded=values[:,:12]+1j*values[:,12:24];low=bounded@r.basis
        np.testing.assert_allclose(actual,np.concatenate((low.real,low.imag),1),atol=1e-11)

    def test_crossed_generalization_and_balanced_variance(self):
        rng=np.random.default_rng(15);a=rng.normal(size=(6,4));b=rng.normal(size=(5,4));c=rng.normal(size=(3,4))
        coeff=[];tx=[];rx=[];day=[]
        for t in range(6):
            for ri,r in enumerate(RXS):
                for di,d in enumerate(DAYS):
                    for _ in range(4):coeff.append(a[t]+b[ri]+c[di]+rng.normal(0,.001,4));tx.append(t);rx.append(r);day.append(d)
        result,_=decompose(np.array(coeff),np.array(tx),np.array(rx),np.array(day))
        self.assertTrue(result['common_component_supported']);self.assertEqual(len(result['crossed_folds']),18)
        self.assertLess(abs(sum(result['variance'].values())-result['total_variance']),1e-9)
        # Same in-sample fit can be misleading when each TX/day has an unrelated RX pattern.
        interactions=rng.normal(size=(6,5,3,4));z=np.array([interactions[t,RXS.index(r),DAYS.index(d)] for t,r,d in zip(tx,rx,day)])
        negative,_=decompose(z,np.array(tx),np.array(rx),np.array(day));self.assertFalse(negative['common_component_supported'])
        with self.assertRaises(ValueError):decompose(z[:-1],np.array(tx[:-1]),np.array(rx[:-1]),np.array(day[:-1]))

    def test_context_rejects_model_and_target_handles(self):
        c=dict(project='/p',python='/python',frozen_sources={},source_contract='/contract',source_cache='/cache',U_channel_cache='/p/runs/20261026-diagnostic-PAMF-URealizations-cache-m3-r01')
        validate_context(c)
        for key in ('G','source_G','target_inputs','truth','checkpoint'):
            with self.assertRaises(ValueError):validate_context(dict(c,**{key:'/forbidden'}))
        with self.assertRaises(ValueError):validate_context(dict(c,frozen_sources={'F':'/F'}))
