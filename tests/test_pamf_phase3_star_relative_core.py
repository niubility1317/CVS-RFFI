from copy import deepcopy
import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pamf_phase3_open_world import source_open_readout as geometry_api


class StarRelativeCoreTests(unittest.TestCase):
    def module(self):
        name='experiments.pamf_phase3_open_world.star_relative_core'
        self.assertTrue((Path(__file__).resolve().parents[1]/(name.replace('.','/')+'.py')).is_file())
        return importlib.import_module(name)

    def fixture(self):
        module=self.module();rng=np.random.default_rng(20261030);y=np.repeat(np.arange(6),8).astype(np.int64)
        branches={}
        for name,d in module.reg.DIMENSIONS.items():
            x=np.eye(6,d)[y]+rng.normal(0.,.025,(len(y),d));x[:,8]+=.2
            if name=='reference29':x[:,27]=0.
            branches[name]=x.astype(np.float32)
        geometry=geometry_api.fit_geometry(branches,y)
        logits=np.zeros((len(y),6),np.float32);logits[np.arange(len(y)),y]=8.
        return branches,y,geometry,logits

    def test_actual_class_means_covariances_and_manual_raw_MD_RMD(self):
        module=self.module();b,y,g,l=self.fixture();before=deepcopy((b,g))
        metric=module.fit_metric(b,y,g)
        self.assertEqual(set(metric),{'schema','branches','score_mean','score_scale'})
        self.assertEqual(metric['score_mean'].shape,(2,5));self.assertEqual(metric['score_scale'].shape,(2,5))
        raw=[]
        for name in module.BRANCHES:
            a=g['branches'][name];u=geometry_api._unit((b[name].astype(np.float64)-a['mean'])/a['scale'])
            means=np.asarray([u[y==c].mean(axis=0) for c in range(6)])
            background=u.mean(axis=0);within=u-means[y];residual=u-background
            def whitener(rows,denom):
                cov=rows.T@rows/denom;v=max(float(np.trace(cov)/len(cov)),1e-8)
                return np.linalg.solve(np.linalg.cholesky(.8*cov+.2*v*np.eye(len(cov))),np.eye(len(cov)))
            ww=whitener(within,len(y)-6);w0=whitener(residual,len(y)-1)
            p=metric['branches'][name]
            np.testing.assert_allclose(p['class_means'],means,atol=1e-13)
            self.assertTrue(np.any(np.linalg.norm(p['class_means'],axis=1)<.999))
            np.testing.assert_allclose(p['background_mean'],background,atol=1e-13)
            np.testing.assert_allclose(p['within_whitener'],ww,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(p['background_whitener'],w0,rtol=1e-12,atol=1e-12)
            minmd=np.min(np.column_stack([np.sum(((u-center)@ww.T)**2,axis=1) for center in means]),axis=1)
            md0=np.sum(((u-background)@w0.T)**2,axis=1)
            expected=np.column_stack((-minmd,md0-minmd));raw.append(expected)
            np.testing.assert_allclose(module._branch_scores(u,p),expected,rtol=1e-11,atol=1e-10)
        raw=np.stack(raw,axis=2)
        np.testing.assert_allclose(metric['score_mean'],raw.mean(axis=0),rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(metric['score_scale'],np.maximum(raw.std(axis=0),1e-6),rtol=1e-12,atol=1e-10)
        expected=((raw-metric['score_mean'][None])/metric['score_scale'][None]).mean(axis=2)
        np.testing.assert_allclose(module.metric_scores(b,g,metric),expected,rtol=1e-11,atol=1e-10)
        for name in b:
            self.assertTrue(np.array_equal(b[name],before[0][name]))
            for key in g['branches'][name]:self.assertTrue(np.array_equal(g['branches'][name][key],before[1]['branches'][name][key]))

    def test_known_points_score_higher_and_all_scores_use_exact_seven_method_order(self):
        module=self.module();b,y,g,l=self.fixture();metric=module.fit_metric(b,y,g)
        unknown={}
        for name,d in module.reg.DIMENSIONS.items():
            a=g['branches'][name];u=np.zeros((2,d));u[:,10]=1.
            unknown[name]=(a['mean']+a['scale']*u).astype(np.float32)
            if name=='reference29':unknown[name][:,27]=0.
        known=module.metric_scores(b,g,metric);outsider=module.metric_scores(unknown,g,metric)
        self.assertTrue(np.all(known.mean(axis=0)>outsider.mean(axis=0)))
        result=module.all_scores(b,l,g,metric)
        from experiments.pamf_phase3_open_world.scores import knownness_scores
        np.testing.assert_array_equal(result[:,:5],knownness_scores(b['fused160'],l,g['branches']['fused160']['prototypes']))
        np.testing.assert_array_equal(result[:,5:],known)
        self.assertEqual(result.dtype,np.float64);self.assertEqual(result.shape,(48,7))
        self.assertEqual(module.METHODS,('msp','maxlogit','margin','energy','cosine','mahalanobis','relative_mahalanobis'))
        self.assertEqual(module.PRIMARY,'relative_mahalanobis');self.assertEqual(module.PRIMARY_INDEX,6)

    def test_class_permutation_invariance_and_source_score_centering(self):
        module=self.module();b,y,g,l=self.fixture();metric=module.fit_metric(b,y,g)
        perm=np.asarray([3,0,5,1,4,2]);other=module.fit_metric(b,perm[y],g)
        np.testing.assert_allclose(module.metric_scores(b,g,metric),module.metric_scores(b,g,other),rtol=1e-11,atol=1e-10)
        np.testing.assert_allclose(module.metric_scores(b,g,metric).mean(axis=0),np.zeros(2),atol=1e-12)
        self.assertEqual(set(metric['branches']),set(module.BRANCHES))
        for name in module.BRANCHES:
            d=module.reg.DIMENSIONS[name]
            self.assertEqual(metric['branches'][name]['within_whitener'].shape,(d,d))
            self.assertEqual(metric['branches'][name]['background_whitener'].shape,(d,d))

    def test_gaussian_kernel_coherent_translation_and_aggregate_freeze(self):
        module=self.module();b,y,g,l=self.fixture();metric=module.fit_metric(b,y,g);before=deepcopy(metric)
        name='fused160';a=g['branches'][name];u=geometry_api._unit((b[name]-a['mean'])/a['scale'])
        branch=metric['branches'][name];shift=np.linspace(-.3,.3,160)
        translated=deepcopy(branch);translated['class_means']+=shift;translated['background_mean']+=shift
        np.testing.assert_allclose(module._branch_scores(u+shift,translated),module._branch_scores(u,branch),rtol=1e-11,atol=1e-10)
        # This is translation of the already standardized/unit-coordinate
        # Gaussian kernel, not a claim that arbitrary pre-L2 raw shifts vanish.
        with patch.object(module,'fit_metric',side_effect=AssertionError('Evaluation never refits')):
            module.all_scores(b,l,g,metric)
        for key in ('score_mean','score_scale'):self.assertTrue(np.array_equal(metric[key],before[key]))
        for name in module.BRANCHES:
            for key in branch:self.assertTrue(np.array_equal(metric['branches'][name][key],before['branches'][name][key]))

    def test_constant_features_keep_full_directions_and_scale_floor(self):
        module=self.module();y=np.repeat(np.arange(6),2).astype(np.int64)
        b={name:np.zeros((12,d),np.float32) for name,d in module.reg.DIMENSIONS.items()}
        g=geometry_api.fit_geometry(b,y);metric=module.fit_metric(b,y,g)
        np.testing.assert_array_equal(metric['score_scale'],np.full((2,5),1e-6))
        np.testing.assert_array_equal(module.metric_scores(b,g,metric),np.zeros((12,2)))
        for name in module.BRANCHES:
            self.assertTrue(np.all(np.diag(metric['branches'][name]['within_whitener'])>0))
            self.assertTrue(np.isfinite(metric['branches'][name]['background_whitener']).all())

    def test_bad_labels_shapes_nonfinite_and_invalid_frozen_metric_rejected(self):
        module=self.module();b,y,g,l=self.fixture();metric=module.fit_metric(b,y,g)
        for labels in (y.astype(np.float64),np.full_like(y,-1),y%5,y[:-1]):
            with self.subTest(labels_shape=labels.shape,dtype=labels.dtype),self.assertRaises(ValueError):module.fit_metric(b,labels,g)
        bad=deepcopy(b);bad['fused160'][0,0]=np.nan
        with self.assertRaises(ValueError):module.fit_metric(bad,y,g)
        bad=deepcopy(b);bad['time160']=bad['time160'][:,:159]
        with self.assertRaises(ValueError):module.metric_scores(bad,g,metric)
        for key,value in (('score_mean',np.zeros((5,2))),('score_scale',np.zeros((2,5))),('score_mean',np.full((2,5),np.nan))):
            with self.subTest(key=key),self.assertRaises(ValueError):module.metric_scores(b,g,dict(metric,**{key:value}))
        with self.assertRaises(ValueError):module.all_scores(b,l[:,:5],g,metric)

    def test_single_F0_bias_preserves_existing_core_parameters_without_dual_model_claim(self):
        module=self.module();b,y,g,l=self.fixture();given=dict(applied=True,parameters=dict(module.reg.CORE_PARAMETERS))
        zeros={name:np.zeros(d) for name,d in module.reg.DIMENSIONS.items()}
        with patch.object(module.reg,'fit_bias',return_value=(zeros,given)) as fit:
            bias,diag=module.fit_bias_single(b,l,g)
        self.assertIs(fit.call_args.args[1],l);self.assertIs(fit.call_args.args[2],l)
        self.assertTrue(diag['single_F0_core']);self.assertFalse(diag['independent_dual_F_agreement'])
        self.assertEqual(diag['parameters'],module.reg.CORE_PARAMETERS);self.assertNotIn('single_F0_core',given)
        self.assertIs(bias,zeros)


if __name__=='__main__':unittest.main()
