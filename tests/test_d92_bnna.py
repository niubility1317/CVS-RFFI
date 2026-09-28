"""Synthetic-only analytical and physical-isolation checks for frozen BNNA."""
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi import stage2_d92_bnna as m


def reference_leave_one_ce(examples,labels,count):
    """Original scalar-entity implementation retained only as a test oracle."""
    total=len(examples)
    sums=np.stack([examples[labels==c].sum(axis=0) for c in range(count)])
    gradient=np.zeros_like(examples);sum_gradient=np.zeros_like(sums);self_gradient=np.zeros_like(examples)
    loss=0.;near_zero=0
    for i,(example,label) in enumerate(zip(examples,labels)):
        raw=sums.copy();raw[label]-=example
        prototype=m._unit(raw)
        logits=10.*np.sum(prototype*example[None,:],axis=1)
        maximum=float(logits.max());exp=np.exp(logits-maximum);prob=exp/exp.sum()
        loss+=(maximum+np.log(exp.sum())-logits[label])/total
        delta=prob;delta[label]-=1.;delta/=total
        gradient[i]=10.*np.sum(delta[:,None]*prototype,axis=0)
        proto_gradient=10.*delta[:,None]*example[None,:]
        raw_gradient=m._unit_backward(raw,proto_gradient)
        sum_gradient+=raw_gradient;self_gradient[i]=raw_gradient[label]
        near_zero+=int(np.count_nonzero(np.linalg.norm(raw,axis=1)<=m.EPS))
    gradient+=sum_gradient[labels]-self_gradient
    return float(loss),gradient,near_zero


def fixture(c=3,k=2,seed=901):
    rng=np.random.default_rng(seed)
    labels=np.repeat(np.arange(c),k)
    center=rng.normal(size=(c,160))
    z=center[labels,None,:]+rng.normal(scale=.7,size=(c*k,4,160))
    fft=m._unit(rng.normal(size=(c*k,96)))
    classes=tuple(f'class-{i:02d}' for i in range(c))
    return dict(support_identity_views=z,support_fft=fft,support_labels=labels,
        support_ids=tuple(f'physical-{i:04d}' for i in range(c*k)),classes=classes,old_classes=classes[:max(1,c//2)])


class BNNATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls): cls.threads.restore_original_limits()

    def test_views_and_fft_match_historical_formula(self):
        from cvsrffi.stage2_diag_cosine_exploration import spectral_logmag_sketch
        iq=np.random.default_rng(7).normal(size=(2,2,256));views=m.make_received_views(iq)
        self.assertEqual(views.shape,(2,4,2,256))
        np.testing.assert_array_equal(views[:,0],iq)
        np.testing.assert_array_equal(views[:,2],-iq)
        np.testing.assert_array_equal(views[:,1,0],-iq[:,1])
        np.testing.assert_array_equal(views[:,3,1],-iq[:,0])
        for scale in (1,1e-11,0):
            np.testing.assert_array_equal(m.received_fft96(iq*scale),spectral_logmag_sketch(iq*scale))
        self.assertFalse(views.flags.writeable)
        self.assertEqual(m.make_received_views(np.empty((0,2,256))).shape,(0,4,2,256))

    def test_analytical_gradient_matches_central_difference(self):
        for k in (1,2,3):
            args=fixture(k=k);z=m._unit(args['support_identity_views']);fft=args['support_fft'];y=args['support_labels']
            basis,_=m._nuisance_basis(z);gate=np.linspace(.07,.31,len(basis))
            loss,gradient,metrics=m._objective_gradient(gate,z,fft,y,basis,k)
            numerical=np.empty_like(gate)
            for j in range(len(gate)):
                delta=np.zeros_like(gate);delta[j]=1e-6
                numerical[j]=(m._objective_gradient(gate+delta,z,fft,y,basis,k)[0]
                    -m._objective_gradient(gate-delta,z,fft,y,basis,k)[0])/2e-6
            np.testing.assert_allclose(gradient,numerical,atol=3e-8,rtol=3e-6)
            self.assertEqual(metrics['loss_gate_weight'],1/(k+1))
            self.assertAlmostEqual(loss,metrics['loss_cls']+(metrics['loss_view']+metrics['loss_gate'])/(k+1))

    def test_vectorized_leave_one_matches_original_loss_and_gradient(self):
        rng=np.random.default_rng(452)
        fixtures=[]
        # K1 has four training views per class; K>=2 uses physical representatives.
        for c,entities_per_class in ((1,4),(3,4),(4,2),(26,20)):
            fixtures.append((m._unit(rng.normal(size=(c*entities_per_class,256))),np.repeat(np.arange(c),entities_per_class),c))
        fixtures.append((rng.normal(scale=1e-15,size=(12,256)),np.repeat(np.arange(3),4),3))
        fixtures.append((np.zeros((12,256)),np.repeat(np.arange(3),4),3))
        # Class sums are exactly zero while each positive leave-one sum is not.
        pair=m._unit(rng.normal(size=(3,256)))
        fixtures.append((np.stack([v for row in pair for v in (row,-row)]),np.repeat(np.arange(3),2),3))
        for x,labels,c in fixtures:
            expected=reference_leave_one_ce(x,labels,c);actual=m._leave_one_ce(x,labels,c)
            self.assertAlmostEqual(actual[0],expected[0],places=12)
            np.testing.assert_allclose(actual[1],expected[1],rtol=3e-13,atol=3e-13)
            self.assertEqual(actual[2],expected[2])

    def test_vectorized_objective_matches_original_for_both_training_branches(self):
        for k in (1,2,5):
            args=fixture(k=k);z=m._unit(args['support_identity_views']);basis,_=m._nuisance_basis(z)
            gate=np.linspace(.06,.29,len(basis))
            actual=m._objective_gradient(gate,z,args['support_fft'],args['support_labels'],basis,k)
            with patch.object(m,'_leave_one_ce',side_effect=reference_leave_one_ce):
                expected=m._objective_gradient(gate,z,args['support_fft'],args['support_labels'],basis,k)
            self.assertAlmostEqual(actual[0],expected[0],places=13)
            np.testing.assert_allclose(actual[1],expected[1],atol=2e-13,rtol=2e-12)

    def test_repeated_eigen_basis_is_canonical_and_orthogonal(self):
        z=np.zeros((2,4,160));z[:,0,0]=1;z[:,1,0]=-1;z[:,2,1]=1;z[:,3,1]=-1
        b,a=m._nuisance_basis(z)
        self.assertEqual(a['active_rank'],2)
        np.testing.assert_allclose(b,np.eye(160)[:2],atol=1e-13)
        np.testing.assert_array_equal(b,m._nuisance_basis(z[::-1])[0])
        np.testing.assert_allclose(b@b.T,np.eye(2),atol=1e-13)

    def test_fixed_k1_budget_and_actual_scalar_training_logs(self):
        args=fixture(k=1)
        with patch.object(m,'_risk',side_effect=AssertionError('K1 cannot validate')):
            state=m.fit_bnna(**args)
        audit=state.audit_dict()
        self.assertEqual(audit['selection'],'k1_fixed_trained');self.assertEqual(audit['fold_count'],0)
        self.assertEqual(audit['candidate_count'],0);self.assertEqual(audit['final_optimizer_steps'],64)
        self.assertEqual([s['step'] for s in audit['steps']],list(range(1,65)))
        keys=set(audit['steps'][0])
        for step in audit['steps']:
            self.assertEqual(set(step),keys);self.assertEqual(step['scope'],'final');self.assertIsNone(step['fold'])
            self.assertEqual(step['learning_rate'],.03);self.assertEqual(step['loss_view_weight'],.5)
            self.assertTrue(0<=step['gate_after_min']<=step['gate_after_max']<=.5)
            self.assertGreaterEqual(step['gradient_norm'],0)
        final=audit['final_fit']
        self.assertLessEqual(final['measured_max_residual_norm'],final['residual_norm_upper_bound']+1e-10)
        self.assertLessEqual(final['measured_max_angle_degrees'],final['angular_upper_bound_degrees']+1e-5)
        self.assertEqual(audit['persistent_state_bytes'],state.basis.nbytes+state.gate.nbytes+state.prototypes.nbytes)
        json.dumps(audit,allow_nan=False)

    def test_rank_zero_is_exact_identity_with_zero_optimizer_steps(self):
        args=fixture(k=1)
        args['support_identity_views']=np.repeat(args['support_identity_views'][:,:1,:],4,axis=1)
        state=m.fit_bnna(**args);a=state.audit_dict()
        self.assertEqual(a['active_rank'],0);self.assertEqual(a['steps'],[])
        self.assertEqual(a['final_fit']['status'],'NO_VIEW_VARIATION')
        self.assertEqual(a['final_fit']['steps'],0)
        z=m._unit(args['support_identity_views'])
        _,physical,_=m._adapt(z,args['support_fft'],np.empty((0,160)),np.empty(0))
        np.testing.assert_array_equal(state.prototypes,m._class_prototypes(physical,args['support_labels'],3))
        zero=dict(args,support_identity_views=np.zeros_like(args['support_identity_views']),support_fft=np.zeros_like(args['support_fft']))
        empty=m.fit_bnna(**zero)
        np.testing.assert_array_equal(empty.score(zero['support_identity_views'],zero['support_fft']),np.zeros((3,3)))
        self.assertTrue(np.all(empty.predict(zero['support_identity_views'],zero['support_fft'])==min(args['classes'])))

    def test_a_zero_matches_identity_and_bounded_nonlinearity(self):
        args=fixture();z=m._unit(args['support_identity_views']);basis,_=m._nuisance_basis(z)
        base=m._adapt(z,args['support_fft'],np.empty((0,160)),np.empty(0))[0]
        zero=m._adapt(z,args['support_fft'],basis,np.zeros(len(basis)))[0]
        np.testing.assert_array_equal(base,zero)
        gate=np.full(len(basis),.5)
        _,_,cache=m._adapt(z,args['support_fft'],basis,gate)
        self.assertLessEqual(np.linalg.norm(cache[-1],axis=-1).max(),.5*np.sqrt(len(basis)/160)+1e-12)
        projected=z@basis.T
        derivative=1-gate*(1-np.tanh(m.TAU*projected)**2)
        self.assertTrue(np.all(derivative>=.5))
        # Concrete nonlinear coordinate-map midpoint, not a universal claim about finite datasets.
        coord=np.array([.01,.10,.19])
        transformed=coord-.5*np.tanh(m.TAU*coord)/m.TAU
        self.assertGreater(abs(transformed[1]-(transformed[0]+transformed[2])/2),1e-3)

    def test_outer_physical_fold_state_is_fully_isolated(self):
        args=fixture(c=2,k=2);original=m._fit_train;captured=[]
        def spy(z,fft,labels,ids,count,**kwargs):
            output=original(z,fft,labels,ids,count,**kwargs)
            if kwargs['scope']=='fold' and kwargs['fold']==0:
                captured.append((tuple(ids),tuple(v.copy() for v in output[:3])))
            return output
        with patch.object(m,'_fit_train',side_effect=spy): first=m.fit_bnna(**args)
        assignment=first.audit_dict()['physical_fold_assignment']
        held={v['physical_id'] for v in assignment if v['fold']==0}
        changed=dict(args,support_identity_views=args['support_identity_views'].copy(),support_fft=args['support_fft'].copy())
        for i,id_ in enumerate(args['support_ids']):
            if id_ in held:
                changed['support_identity_views'][i]*=-3
                changed['support_fft'][i]=np.roll(changed['support_fft'][i],7)
        with patch.object(m,'_fit_train',side_effect=spy): m.fit_bnna(**changed)
        self.assertEqual(captured[0][0],captured[1][0]);self.assertFalse(set(captured[0][0])&held)
        for one,two in zip(captured[0][1],captured[1][1]):np.testing.assert_array_equal(one,two)
        audit=first.audit_dict();self.assertEqual(audit['candidate_count'],2)
        for fold in audit['folds']:
            self.assertEqual(fold['train_k'],1);self.assertEqual(fold['heldout_k'],1)
            self.assertEqual(set(fold['training']['training_physical_ids']),set(fold['train_ids']))
            self.assertFalse(set(fold['train_ids'])&set(fold['heldout_ids']))
            for step in audit['steps']:
                if step['scope']=='fold' and step['fold']==fold['fold']:
                    self.assertEqual(step['train_k'],1);self.assertEqual(step['loss_view_weight'],.5)

    def test_k5_fold_counts_and_candidate_selection(self):
        state=m.fit_bnna(**fixture(c=2,k=5));a=state.audit_dict()
        self.assertEqual(a['fold_count'],3)
        self.assertEqual([f['train_k'] for f in a['folds']],[3,3,4])
        all_held=[id_ for f in a['folds'] for id_ in f['heldout_ids']]
        self.assertEqual(len(all_held),len(set(all_held)))
        self.assertEqual(len(all_held),10)
        chosen=min(a['candidates'],key=lambda v:(round(v['objective'],10),round(v['macro_nll'],10),v['candidate']!='identity'))
        self.assertEqual(a['selected'],chosen['candidate'])
        if a['selected']=='identity':
            self.assertEqual(a['final_optimizer_steps'],0);self.assertEqual(state.basis.shape,(0,160))
        else:self.assertEqual(a['final_optimizer_steps'],64)

    def test_permutations_and_query_immutable_chunk_independence(self):
        args=fixture(k=2);state=m.fit_bnna(**args)
        rng=np.random.default_rng(4);row=rng.permutation(6);cp=np.array([2,0,1]);inv=np.argsort(cp)
        perm=dict(args,support_identity_views=args['support_identity_views'][row],support_fft=args['support_fft'][row],
            support_labels=inv[args['support_labels'][row]],support_ids=tuple(args['support_ids'][i] for i in row),
            classes=tuple(args['classes'][i] for i in cp))
        other=m.fit_bnna(**perm)
        self.assertEqual(state.audit_dict()['selected'],other.audit_dict()['selected'])
        q=rng.normal(size=(5,4,160));fft=m._unit(rng.normal(size=(5,96)))
        before=tuple(v.tobytes() for v in (state.basis,state.gate,state.prototypes));score=state.score(q,fft)
        np.testing.assert_allclose(other.score(q,fft)[:,inv],score,atol=1e-10)
        np.testing.assert_array_equal(score,np.concatenate([state.score(q[:2],fft[:2]),state.score(q[2:],fft[2:])]))
        np.testing.assert_array_equal(score[::-1],state.score(q[::-1],fft[::-1]))
        self.assertEqual(before,tuple(v.tobytes() for v in (state.basis,state.gate,state.prototypes)))
        for value in (state.basis,state.gate,state.prototypes):
            with self.assertRaises(ValueError):value.setflags(write=True)
        with self.assertRaises(TypeError):state.audit['k']=10
        self.assertEqual(state.score(np.empty((0,4,160)),np.empty((0,96))).shape,(0,3))

    def test_k1_26_classes_and_near_zero_features(self):
        args=fixture(c=26,k=1);state=m.fit_bnna(**args)
        self.assertEqual(state.prototypes.shape,(26,256));self.assertLessEqual(len(state.gate),8)
        self.assertTrue(np.isfinite(state.score(args['support_identity_views'],args['support_fft'])).all())
        tiny=fixture(k=1);tiny['support_identity_views']*=1e-20
        small=m.fit_bnna(**tiny)
        self.assertGreater(small.audit_dict()['final_fit']['subunit_identity_count'],0)
        self.assertTrue(np.isfinite(small.score(tiny['support_identity_views'],tiny['support_fft'])).all())

    def test_bad_inputs_and_no_query_training_interface(self):
        args=fixture();changes=[dict(support_labels=np.ones(6)),dict(support_labels=np.ones(6,dtype=bool)),
            dict(support_labels=np.array([0,0,0,1,2,2])),dict(support_labels=np.full(6,99)),
            dict(support_ids=['same']*6),dict(classes=['same']*3),dict(old_classes=['missing']),
            dict(support_identity_views=np.zeros((6,160))),dict(support_identity_views=np.zeros((6,4,160),dtype=bool)),
            dict(support_identity_views=np.full((6,4,160),np.nan)),dict(support_fft=np.full((6,96),np.inf)),
            dict(support_fft=np.zeros((5,96)))]
        for change in changes:
            with self.subTest(keys=list(change)):
                with self.assertRaises(ValueError):m.fit_bnna(**dict(args,**change))
        with self.assertRaises(ValueError):m.make_received_views(np.zeros((2,256)))
        self.assertEqual(set(inspect.signature(m.fit_bnna).parameters),
            {'support_identity_views','support_fft','support_labels','support_ids','classes','old_classes'})

    def test_frozen_config_matches_file(self):
        path=Path(__file__).resolve().parents[1]/'configs/d92_bnna_frozen_20260929.json'
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'algorithm':m.FROZEN_CONFIG})


if __name__=='__main__':unittest.main()
