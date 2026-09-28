"""Synthetic physical-objective, isolation and immutable-state MVRidge tests."""
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi import stage2_d92_multiview_ridge as m


def fixture(c=3,k=3,seed=781):
    rng=np.random.default_rng(seed)
    return dict(support_identity_views=rng.normal(size=(c*k,4,160)),support_fft=rng.normal(size=(c*k,96)),
        support_labels=np.repeat(np.arange(c),k),support_ids=tuple(f'id-{i:04d}' for i in range(c*k)),
        classes=tuple(f'class-{i:02d}' for i in range(c)),old_classes=('class-00',))


def objective(x,y,W,b):
    targets=np.eye(W.shape[1])[y]-1./W.shape[1]
    residual=x@W+b-targets[:,None,:]
    return float(np.sum(residual**2)/8.+np.sum(W**2)/2.)


class MVRidgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.limits=threadpool_limits(limits=2)
    @classmethod
    def tearDownClass(cls): cls.limits.restore_original_limits()

    def test_closed_form_equals_independent_augmented_least_squares(self):
        for k in (1,2,5):
            a=fixture(k=k);x=m._features(a['support_identity_views'],a['support_fft']);n=len(x);c=3
            W,b,details=m._fit_train(x,a['support_labels'],a['support_ids'],a['classes'],scope='final',fold=None)
            stacked=x.reshape(-1,256);design=np.concatenate((stacked,np.ones((4*n,1))),axis=1)/2.
            ridge=np.concatenate((np.eye(256),np.zeros((256,1))),axis=1)
            design=np.concatenate((design,ridge),axis=0)
            targets=np.eye(c)[a['support_labels']]-1/c
            rhs=np.concatenate((np.repeat(targets,4,axis=0)/2.,np.zeros((256,c))),axis=0)
            solution=np.linalg.lstsq(design,rhs,rcond=None)[0]
            np.testing.assert_allclose(W,solution[:256],atol=3e-14,rtol=3e-12)
            np.testing.assert_allclose(b,solution[256],atol=3e-14,rtol=3e-12)
            self.assertAlmostEqual(objective(x,a['support_labels'],W,b),details['loss_total'],places=12)
            self.assertLess(details['gradient_norm'],2e-12)
            self.assertEqual(details['physical_loss_mass'],n)
            self.assertAlmostEqual(details['target_norm_squared'],n*(1-1/c))

    def test_mean_variance_objective_and_gradient_decomposition(self):
        a=fixture(k=2);x=m._features(a['support_identity_views'],a['support_fft']);y=a['support_labels'];n=len(x)
        rng=np.random.default_rng(40);W=rng.normal(scale=.2,size=(256,3));b=rng.normal(size=3)
        target=np.eye(3)[y]-1/3;mean=x.mean(axis=1);diff=x-mean[:,None,:]
        R=x@W+b-target[:,None,:];r=mean@W+b-target
        V=diff.reshape(-1,256).T@diff.reshape(-1,256)/4
        split=.5*np.sum(r*r)+.5*np.trace(W.T@V@W)+.5*np.sum(W*W)
        self.assertAlmostEqual(split,objective(x,y,W,b),places=12)
        grad=x.reshape(-1,256).T@R.reshape(-1,3)/4+W
        np.testing.assert_allclose(grad,mean.T@r+V@W+W,atol=1e-13)
        gradb=R.sum(axis=(0,1))/4
        for j,l in ((2,0),(103,1),(240,2)):
            delta=np.zeros_like(W);delta[j,l]=1e-6
            numerical=(objective(x,y,W+delta,b)-objective(x,y,W-delta,b))/2e-6
            self.assertAlmostEqual(numerical,grad[j,l],delta=2e-8)
        delta=np.array([1e-6,0,0]);numerical=(objective(x,y,W,b+delta)-objective(x,y,W,b-delta))/2e-6
        self.assertAlmostEqual(numerical,gradb[0],delta=2e-8)
        self.assertGreater(.5*np.trace(W.T@V@W),0)

    def test_k1_non_nearest_center_and_no_validation(self):
        z=np.zeros((3,4,160));z[0,:,0]=1;z[1,:,1]=1;z[2,:,0]=-1
        fft=np.zeros((3,96));fft[:,0]=1
        with patch.object(m,'_diagnostics',side_effect=AssertionError('K1 has no independent holdout')):
            state=m.fit_multiview_ridge(support_identity_views=z,support_fft=fft,support_labels=[0,1,2],support_ids=['a','b','c'],classes=['a','b','c'])
        query=np.zeros((1,4,160));query[0,:,0]=.7;query[0,:,1]=np.sqrt(.51)
        qfft=fft[:1];x=m._features(z,fft)[:,0];qx=m._features(query,qfft)[:,0]
        nearest=np.argmin(np.sum((x-qx)**2,axis=1))
        self.assertEqual(nearest,1);self.assertEqual(state.predict(query,qfft)[0],'a')
        audit=state.audit_dict();self.assertIsNone(audit['oof']);self.assertEqual(audit['fold_count'],0)
        self.assertEqual(audit['optimizer_steps'],0);self.assertEqual(audit['steps'],[])
        self.assertEqual(audit['final_fit']['loss_view'],0)

    def test_physical_folds_rebuild_every_statistic(self):
        a=fixture(k=5);records=[];original=m._fit_train
        def capture(x,y,ids,classes,**kw):
            means=x.mean(axis=1);center=means.mean(axis=0);diff=(x-means[:,None,:]).reshape(-1,256)
            scatter=diff.T@diff/4;targets=np.eye(len(classes))[y]-1/len(classes)
            gram=np.eye(256)+(means-center).T@(means-center)+scatter;cross=(means-center).T@targets
            value=original(x,y,ids,classes,**kw)
            records.append((center,scatter,gram,cross,value));return value
        with patch.object(m,'_fit_train',side_effect=capture): state=m.fit_multiview_ridge(**a)
        audit=state.audit_dict();self.assertEqual([f['train_k'] for f in audit['folds']],[3,3,4])
        baseline=records[0];held=set(audit['folds'][0]['heldout_ids']);records.clear()
        changed=dict(a);z=a['support_identity_views'].copy();f=a['support_fft'].copy()
        mask=np.array([i in held for i in a['support_ids']]);z[mask]*=-2;f[mask]+=88
        changed.update(support_identity_views=z,support_fft=f)
        with patch.object(m,'_fit_train',side_effect=capture): m.fit_multiview_ridge(**changed)
        for old,new in zip(baseline[:4],records[0][:4]): np.testing.assert_array_equal(old,new)
        np.testing.assert_array_equal(baseline[4][0],records[0][4][0]);np.testing.assert_array_equal(baseline[4][1],records[0][4][1])
        self.assertEqual(len(audit['oof']['physical_records']),15)
        self.assertEqual(len({r['physical_id'] for r in audit['oof']['physical_records']}),15)
        for fold in audit['folds']:
            self.assertFalse(set(fold['train_ids']) & set(fold['heldout_ids']))
            self.assertEqual(fold['training']['physical_loss_mass'],3*fold['train_k'])

    def test_registry_support_phase_permutation_and_old_roles(self):
        a=fixture();state=m.fit_multiview_ridge(**a);change=dict(a);order=np.array([2,0,1]);idx=np.arange(9)[::-1]
        change['classes']=tuple(a['classes'][i] for i in order);change['support_labels']=np.argsort(order)[a['support_labels']][idx]
        for key in ('support_identity_views','support_fft'): change[key]=a[key][idx]
        change['support_ids']=tuple(a['support_ids'][i] for i in idx);change['old_classes']=()
        other=m.fit_multiview_ridge(**change)
        np.testing.assert_array_equal(other.W,state.W[:,order]);np.testing.assert_array_equal(other.b,state.b[order])
        change=dict(a);change['support_identity_views']=a['support_identity_views'][:,[2,0,3,1]]
        other=m.fit_multiview_ridge(**change)
        np.testing.assert_allclose(other.W,state.W,atol=2e-14,rtol=1e-12)
        np.testing.assert_allclose(other.b,state.b,atol=2e-14,rtol=1e-12)

    def test_query_independence_and_immutable_storage(self):
        a=fixture();state=m.fit_multiview_ridge(**a);z=a['support_identity_views'];f=a['support_fft']
        snapshot=(state.W.tobytes(),state.b.tobytes(),json.dumps(state.audit_dict(),sort_keys=True))
        scores=state.score(z,f)
        np.testing.assert_array_equal(np.concatenate([state.score(z[:2],f[:2]),state.score(z[2:],f[2:])]),scores)
        np.testing.assert_array_equal(state.score(z[::-1],f[::-1]),scores[::-1])
        np.testing.assert_allclose(state.score(z[:,[1,3,0,2]],f),scores,atol=2e-14,rtol=1e-12)
        self.assertEqual(state.score(z[:0],f[:0]).shape,(0,3))
        self.assertEqual(snapshot,(state.W.tobytes(),state.b.tobytes(),json.dumps(state.audit_dict(),sort_keys=True)))
        with self.assertRaises(ValueError): state.W.setflags(write=True)
        with self.assertRaises(TypeError): state.audit['x']=0
        self.assertEqual(set(vars(state)),{'W','b','classes','audit'})

    def test_spd_zero_repeated_and_storage_c26(self):
        a=fixture(c=26,k=1);state=m.fit_multiview_ridge(**a)
        self.assertEqual(state.W.nbytes+state.b.nbytes,53456)
        self.assertEqual(state.audit_dict()['persistent_state_bytes'],53456)
        x=m._features(a['support_identity_views'],a['support_fft']);center=x.mean(axis=(0,1));r=(x-center).reshape(-1,256)
        eig=np.linalg.eigvalsh(np.eye(256)+r.T@r/4)
        self.assertGreaterEqual(eig.min(),1-2e-13);self.assertLessEqual(eig.max()/eig.min(),27+2e-13)
        json.dumps(state.audit_dict(),allow_nan=False)
        a=fixture(k=2);a['support_identity_views'][:]=0;a['support_fft'][:]=0;state=m.fit_multiview_ridge(**a)
        np.testing.assert_array_equal(state.W,0);np.testing.assert_array_equal(state.b,0)
        self.assertTrue(np.all(state.predict(a['support_identity_views'],a['support_fft'])=='class-00'))
        a=fixture(k=2);a['support_identity_views'][:]=a['support_identity_views'][:,:1,:]
        state=m.fit_multiview_ridge(**a);self.assertEqual(state.audit_dict()['final_fit']['loss_view'],0.)
        one=fixture(c=1,k=1);np.testing.assert_array_equal(m.fit_multiview_ridge(**one).W,0)

    def test_validation_config_and_preprocessing(self):
        a=fixture(k=1);x=m._features(a['support_identity_views'],a['support_fft'])
        np.testing.assert_allclose(np.sum(x[:,:,:160]**2,axis=2),1/17,atol=1e-15)
        np.testing.assert_allclose(np.sum(x[:,:,160:]**2,axis=2),16/17,atol=1e-15)
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/d92_multiview_ridge_frozen_20260929.json').read_text(encoding='utf-8'))
        self.assertEqual(config['algorithm'],m.FROZEN_CONFIG)
        for key,value in [('support_labels',[0.,1.,2.]),('support_labels',[0,0,2]),('support_ids',['a','a','c']),
                          ('classes',['a','a','b']),('old_classes',['absent']),('support_fft',np.zeros((3,95))),
                          ('support_identity_views',np.full((3,4,160),np.inf)),('support_identity_views',np.zeros((3,4,160),dtype=bool))]:
            bad=dict(a);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): m.fit_multiview_ridge(**bad)
        self.assertFalse(any('query' in p or 'source' in p or 'ground' in p for p in inspect.signature(m.fit_multiview_ridge).parameters))
        with self.assertRaises(ValueError): m.MultiviewRidgeState(np.zeros((255,3)),np.zeros(3),a['classes'],{})
        a=fixture(k=2)
        for key in ('support_identity_views','support_fft','support_labels'): a[key]=a[key][:-1]
        a['support_ids']=a['support_ids'][:-1]
        with self.assertRaises(ValueError): m.fit_multiview_ridge(**a)

    def test_zero_identity_view_keeps_exact_joint_normalization(self):
        a=fixture(k=1);a['support_identity_views'][0,0]=0
        x=m._features(a['support_identity_views'],a['support_fft'])
        self.assertAlmostEqual(float(np.sum(x[0,0,160:]**2)),1.)
        self.assertAlmostEqual(float(np.sum(x[0,1,160:]**2)),16/17)
        scatter=x[0]-x[0].mean(axis=0)
        self.assertGreater(float(np.sum(scatter[:,160:]**2)),0.)
        state=m.fit_multiview_ridge(**a)
        self.assertTrue(np.isfinite(state.W).all())
        self.assertLess(state.audit_dict()['final_fit']['gradient_norm'],2e-12)


if __name__=='__main__': unittest.main()
