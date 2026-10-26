import unittest
import numpy as np
from experiments.pamf_channel_pl.sig1_repl_rx_geometry import estimate,compare_directions,analyze_views
from experiments.pamf_channel_pl.sig1_repl_rx_source import validate_runtime,RUN,RELEASE


class ReceiverGeometryTests(unittest.TestCase):
    def runtime(self):
        release='/p/releases/'+RELEASE
        return dict(run_id=RUN,project='/p',python='python',release_root=release,run_root='/p/runs/'+RUN,
            context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='a'*40,frozen_sources={},
            allowed_gpus=list(range(8)),launch_owner='codex-pamf-sig1-repl-rx-geometry-20261026')

    def test_runtime_rejects_target_or_foreign_roots(self):
        rt=self.runtime();validate_runtime(rt)
        for bad in (dict(rt,truth='query'),dict(rt,run_root='/foreign'),dict(rt,context='/target'),dict(rt,launch_owner='other')):
            with self.assertRaises(ValueError):validate_runtime(bad)

    def data(self):
        x=[];y=[];r=[]
        for cls in range(3):
            for rx in range(2):
                for _ in range(1+cls+rx):
                    v=np.zeros(6);v[cls]=3;v[3]=(-1 if rx==0 else 1)
                    x.append(v);y.append(cls);r.append(rx)
        return np.array(x),np.array(y),np.array(r)

    def test_class_balanced_rx_offset(self):
        x,y,r=self.data();model=estimate(x,y,r,range(3),range(2))
        np.testing.assert_allclose(model['bias'][:,:3],0,atol=1e-12)
        np.testing.assert_allclose(model['bias'][:,3],[-1,1],atol=1e-12)
        self.assertAlmostEqual(model['shared_RX_fraction'],1)

    def test_identity_projection_preserves_orthogonal_rx(self):
        x,y,r=self.data();model=estimate(x,y,r,range(3),range(2))
        np.testing.assert_allclose(model['protected_bias'],model['bias'],atol=1e-12)
        np.testing.assert_allclose(model['protected_bias']@model['class_basis'],0,atol=1e-12)
        self.assertEqual(model['class_rank'],2)

    def test_identity_leak_is_removed_without_rescale(self):
        x,y,r=self.data();x[:,0]+=r*2
        model=estimate(x,y,r,range(3),range(2))
        self.assertGreater(model['identity_energy_fraction'],0)
        self.assertTrue(np.linalg.norm(model['protected_bias'])<=np.linalg.norm(model['bias']))

    def test_missing_cell_and_nonfinite_rejected(self):
        x,y,r=self.data();mask=~((y==0)&(r==0))
        with self.assertRaises(ValueError):estimate(x[mask],y[mask],r[mask],range(3),range(2))
        x[0,0]=np.nan
        with self.assertRaises(ValueError):estimate(x,y,r,range(3),range(2))

    def test_loco_and_inductive_validation(self):
        x,y,r=self.data();model=estimate(x,y,r,range(3),range(2));before=model['bias'].copy()
        result=compare_directions(model,estimate(x,y,r,range(3),range(2)))
        self.assertAlmostEqual(result['mean_cosine'],1)
        np.testing.assert_array_equal(model['bias'],before)
        evidence=analyze_views({'raw':(x,x)},y,r,y,r,classes=range(3),receivers=range(2))
        self.assertTrue(evidence['geometry_pass'])

    def test_class_rx_interaction_fails_shared_hypothesis(self):
        x,y,r=self.data();x[:,3]=np.array([(-1 if rx==0 else 1)*(-1 if cls==0 else .5) for cls,rx in zip(y,r)])
        model=estimate(x,y,r,range(3),range(2))
        self.assertAlmostEqual(model['shared_RX_fraction'],0)


if __name__=='__main__':unittest.main()
