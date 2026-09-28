"""Synthetic-only checks for OSC; no data, checkpoint or result access."""
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi import stage2_d92_orbit_shared as m


def fixture(c=3,k=3,seed=192):
    rng=np.random.default_rng(seed)
    return dict(support_identity_views=rng.normal(size=(c*k,4,160)),support_fft=rng.normal(size=(c*k,96)),
        support_labels=np.repeat(np.arange(c),k),support_ids=tuple(f'id-{i:04d}' for i in range(c*k)),
        classes=tuple(f'class-{i:02d}' for i in range(c)),old_classes=('class-00',))


class OSCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.limit=threadpool_limits(limits=2)
    @classmethod
    def tearDownClass(cls): cls.limit.restore_original_limits()

    def test_views_fft_and_config(self):
        from cvsrffi import stage2_d92_bnna as b
        iq=np.random.default_rng(2).normal(size=(2,2,256))
        for scale in (0,1e-11,1):
            np.testing.assert_array_equal(m.make_received_views(iq*scale),b.make_received_views(iq*scale))
            np.testing.assert_array_equal(m.received_fft96(iq*scale),b.received_fft96(iq*scale))
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/d92_orbit_shared_frozen_20260929.json').read_text(encoding='utf-8'))
        self.assertEqual(config['algorithm'],m.FROZEN_CONFIG)

    def test_compiled_matches_quadratic_up_to_query_common_term(self):
        a=fixture();x=m._features(a['support_identity_views'],a['support_fft']);y=a['support_labels']
        aligned,means,_,_=m._align(x,y,a['support_ids'],a['classes'])
        Q,_=m._covariance(aligned,y,means,3);precision=256*np.linalg.inv(Q)
        W,b,_=m._fit_train(x,y,a['support_ids'],a['classes'],scope='final',fold=None)
        query=x[:2];compiled=m._score(query,W,b);direct=[]
        for row in query:
            perclass=[]
            for center in means:
                components=[]
                for shift in range(4):
                    residual=row-np.roll(center,-shift,axis=0)
                    components.append(-np.sum((residual@precision)*residual)/8.)
                maximum=max(components)
                perclass.append(maximum+np.log(np.exp(np.asarray(components)-maximum).sum())-np.log(4.))
            direct.append(perclass)
        common=np.sum((query@precision)*query,axis=(1,2))/8.
        np.testing.assert_allclose(compiled,np.asarray(direct)+common[:,None],atol=1e-12,rtol=1e-12)

    def test_medoid_and_alignment_reference(self):
        a=fixture(c=2,k=4);x=m._features(a['support_identity_views'],a['support_fft'])
        aligned,means,medoids,shifts=m._align(x,a['support_labels'],a['support_ids'],a['classes'])
        for cls in range(2):
            idx=np.flatnonzero(a['support_labels']==cls)
            dist=np.array([[[sum(np.sum((x[i,v]-x[j,(v+s)%4])**2) for v in range(4))/4 for s in range(4)] for j in idx] for i in idx])
            chosen=int(np.argmin(dist.min(axis=2).sum(axis=1)));self.assertEqual(medoids[cls]['physical_id'],a['support_ids'][idx[chosen]])
            expected=np.argmin(dist[chosen],axis=1);expected[chosen]=0
            np.testing.assert_array_equal(shifts[idx],expected)
            np.testing.assert_allclose(means[cls],aligned[idx].mean(axis=0))

    def test_k1_fixed_and_zero_state_bytes(self):
        a=fixture(c=26,k=1)
        with patch.object(m,'_diagnostics',side_effect=AssertionError('K1 must not validate')):
            state=m.fit_orbit_shared(**a)
        audit=state.audit_dict();self.assertIsNone(audit['oof']);self.assertEqual(audit['fold_count'],0)
        self.assertEqual(audit['final_fit']['physical_df'],0);self.assertEqual(audit['final_fit']['shrinkage'],1.)
        self.assertEqual(audit['optimizer_steps'],0);self.assertEqual(audit['persistent_state_bytes'],213200)
        self.assertEqual(state.W.nbytes+state.b.nbytes,213200)
        self.assertEqual(set(vars(state)),{'W','b','classes','audit'})
        json.dumps(audit,allow_nan=False)
        a['support_identity_views'][:]=0;a['support_fft'][:]=0;zero=m.fit_orbit_shared(**a)
        np.testing.assert_array_equal(zero.W,0);np.testing.assert_array_equal(zero.b,0)
        self.assertTrue(np.all(zero.predict(a['support_identity_views'],a['support_fft'])=='class-00'))

    def test_covariance_physical_df_positive_definite_and_nearzero(self):
        a=fixture(k=5);x=m._features(a['support_identity_views'],a['support_fft']);y=a['support_labels']
        aligned,means,_,_=m._align(x,y,a['support_ids'],a['classes']);Q,audit=m._covariance(aligned,y,means,5)
        eig=np.linalg.eigvalsh(Q);self.assertGreater(eig.min(),0)
        self.assertLessEqual(eig.max()/eig.min(),audit['condition_bound']+1e-10)
        self.assertEqual(audit['physical_df'],12);self.assertEqual(audit['shrinkage'],256/268)
        repeated=np.repeat(x[::5],5,axis=0);means=np.stack([repeated[y==c].mean(axis=0) for c in range(3)])
        Q,deg=m._covariance(repeated,y,means,5)
        np.testing.assert_array_equal(Q,np.eye(256));self.assertEqual(deg['status'],'ZERO_PHYSICAL_RESIDUAL')
        repeated+=np.random.default_rng(3).normal(scale=1e-16,size=repeated.shape)
        means=np.stack([repeated[y==c].mean(axis=0) for c in range(3)])
        np.testing.assert_array_equal(m._covariance(repeated,y,means,5)[0],np.eye(256))

    def test_fold_all_state_isolation_and_actual_n(self):
        args=fixture(k=5);records=[];original=m._fit_train
        def capture(*a,**kw):
            value=original(*a,**kw);records.append((kw['scope'],kw['fold'],value));return value
        with patch.object(m,'_fit_train',side_effect=capture): state=m.fit_orbit_shared(**args)
        audit=state.audit_dict();self.assertEqual([v['train_k'] for v in audit['folds']],[3,3,4])
        self.assertEqual(len(audit['oof']['physical_records']),15)
        self.assertEqual(len({v['physical_id'] for v in audit['oof']['physical_records']}),15)
        original_fold=records[0][2];held=set(audit['folds'][0]['heldout_ids'])
        changed=dict(args);z=args['support_identity_views'].copy();f=args['support_fft'].copy()
        mask=np.array([id_ in held for id_ in args['support_ids']]);z[mask]*=-7;f[mask]+=91
        changed.update(support_identity_views=z,support_fft=f);records.clear()
        with patch.object(m,'_fit_train',side_effect=capture): m.fit_orbit_shared(**changed)
        new_fold=records[0][2]
        np.testing.assert_array_equal(original_fold[0],new_fold[0]);np.testing.assert_array_equal(original_fold[1],new_fold[1])
        for key in ('medoid_physical_ids','alignment_shifts','covariance_trace','feature_energy','physical_df','shrinkage'):
            self.assertEqual(original_fold[2][key],new_fold[2][key])
        for fold in audit['folds']:
            self.assertEqual(fold['training']['physical_df'],3*(fold['train_k']-1))
            self.assertFalse(set(fold['train_ids']) & set(fold['heldout_ids']))

    def test_registry_support_permutation_and_old_membership_no_fit_effect(self):
        a=fixture();original=m.fit_orbit_shared(**a);order=np.array([2,0,1]);inverse=np.argsort(order)
        change=dict(a);change['classes']=tuple(a['classes'][i] for i in order);change['support_labels']=inverse[a['support_labels']]
        idx=np.random.default_rng(12).permutation(9)
        for key in ('support_identity_views','support_fft','support_labels'): change[key]=change[key][idx]
        change['support_ids']=tuple(a['support_ids'][i] for i in idx);change['old_classes']=()
        state=m.fit_orbit_shared(**change)
        self.assertEqual(state.classes,change['classes'])
        np.testing.assert_array_equal(original.W[order],state.W);np.testing.assert_array_equal(original.b[order],state.b)

    def test_query_cyclic_batch_and_immutable(self):
        a=fixture();state=m.fit_orbit_shared(**a);z=a['support_identity_views'];f=a['support_fft']
        snapshot=(state.W.tobytes(),state.b.tobytes(),json.dumps(state.audit_dict(),sort_keys=True))
        full=state.score(z,f)
        for shift in range(4): np.testing.assert_allclose(state.score(np.roll(z,shift,axis=1),f),full,atol=2e-13,rtol=1e-13)
        np.testing.assert_array_equal(np.concatenate([state.score(z[:2],f[:2]),state.score(z[2:],f[2:])]),full)
        np.testing.assert_array_equal(state.score(z[::-1],f[::-1]),full[::-1])
        self.assertEqual(state.score(z[:0],f[:0]).shape,(0,3))
        self.assertEqual(snapshot,(state.W.tobytes(),state.b.tobytes(),json.dumps(state.audit_dict(),sort_keys=True)))
        with self.assertRaises(ValueError): state.W.setflags(write=True)
        with self.assertRaises(TypeError): state.audit['x']=1

    def test_orbit_structure_is_used_and_identical_view_degeneracy(self):
        z=np.zeros((2,4,160));z[0,np.arange(4),np.arange(4)]=1;z[1]=z[0,[0,2,1,3]];f=np.zeros((2,96))
        np.testing.assert_array_equal(z[0].mean(axis=0),z[1].mean(axis=0))
        state=m.fit_orbit_shared(support_identity_views=z,support_fft=f,support_labels=[0,1],support_ids=['a','b'],classes=['a','b'])
        scores=state.score(z,f);self.assertGreater(scores[0,0],scores[0,1]);self.assertGreater(scores[1,1],scores[1,0])
        a=fixture(k=1);a['support_identity_views'][:]=a['support_identity_views'][:,:1,:]
        state=m.fit_orbit_shared(**a);x=m._features(a['support_identity_views'],a['support_fft'])
        expected=x[:,0]@state.W[:,0].T+state.b
        np.testing.assert_allclose(state.score(a['support_identity_views'],a['support_fft']),expected,atol=1e-12)

    def test_input_validation_and_no_query_fit_api(self):
        a=fixture(k=1)
        for key,value in [('support_labels',[0.,1.,2.]),('support_labels',[0,1,1]),('support_ids',['a','a','c']),
                          ('classes',['a','a','b']),('old_classes',['absent']),('support_fft',np.zeros((3,95))),
                          ('support_identity_views',np.full((3,4,160),np.nan)),('support_identity_views',np.zeros((3,4,160),dtype=bool))]:
            bad=dict(a);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): m.fit_orbit_shared(**bad)
        self.assertFalse(any('query' in p or 'source' in p or 'ground' in p for p in inspect.signature(m.fit_orbit_shared).parameters))
        with self.assertRaises(ValueError): m.OrbitSharedState(np.zeros((3,4,159)),np.zeros(3),a['classes'],{})
        unequal=fixture(k=2)
        for key in ('support_identity_views','support_fft','support_labels'): unequal[key]=unequal[key][:-1]
        unequal['support_ids']=unequal['support_ids'][:-1]
        with self.assertRaises(ValueError): m.fit_orbit_shared(**unequal)


if __name__=='__main__': unittest.main()
