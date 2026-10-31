"""Uncentered source-only geometry with a fixed eighty-dimensional residual."""
from copy import deepcopy
import importlib
import numpy as np
import pytest


def api(): return importlib.import_module('experiments.pamf_phase3_open_world.ground_geometry_core')


def cloud(seed=9,n=180):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,160));x=(x/(2*np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1))).astype(np.float32)
    logits=np.tile(np.asarray([2.,1.,0.,-1.,-2.,-3.],np.float32),(n,1))
    return x,logits,(np.arange(n)%15).astype(np.int64)


def shifted(rank=1):
    rng=np.random.default_rng(23);n=100
    base=np.zeros((n,160),np.float32);base[:,rank:]=(rng.normal(size=(n,160-rank))*.02).astype(np.float32)
    shifts=np.zeros((15,160),np.float32)
    if rank==1:shifts[:,0]=np.linspace(-.2,.2,15)
    else:shifts[1:rank+1,:rank]=np.eye(rank,dtype=np.float32)*.1
    x=np.concatenate([base+row for row in shifts]);domains=np.repeat(np.arange(15,dtype=np.int64),n)
    logits=np.tile(np.asarray([2.,1.,0.,-1.,-2.,-3.],np.float32),(len(x),1))
    return x,logits,domains


def test_standard_matches_existing_uncentered_vim_and_has_no_row_bank():
    m=api();from experiments.pamf_phase3_open_world import vim_readout_core as old
    x,l,d=cloud();state,diag=m.fit(x,l,d,'standard');expected,_=old.fit(x,l)
    assert set(state)=={'schema','mode','origin','residual_basis','nuisance_basis','alpha'}
    assert state['schema'].item()==m.SCHEMA and state['mode'].item()=='standard'
    assert state['nuisance_basis'].shape==(160,0) and not any(v.flags.writeable for v in state.values())
    np.testing.assert_allclose(state['residual_basis']@state['residual_basis'].T,expected['residual_basis']@expected['residual_basis'].T,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(m.raw_scores(state,x,l),old.raw_scores(expected,x,l))
    assert float(state['alpha'])==float(expected['alpha']) and m.classifier_coordinates is old.classifier_coordinates
    assert diag['residual_dimension']==80 and diag['principal_dimension']==80 and diag['nuisance_rank']==0
    assert diag['domain_counts']==[12]*15 and diag['deployed_numeric_bytes']==103688
    assert diag['source_TX_labels_used'] is False and diag['target_U_reads']==diag['target_Q_reads']==diag['neural_updates']==0
    assert m.validate_state(state) is state


def test_domain_mean_direction_removed_without_normalizing_or_centering_scores():
    m=api();x,l,d=shifted();state,diag=m.fit(x,l,d,'domain_orthogonal')
    R,N=state['residual_basis'],state['nuisance_basis']
    assert N.shape==(160,1) and R.shape==(160,80) and diag['principal_dimension']==79
    np.testing.assert_allclose(N@N.T,np.diag([1.]+[0.]*159),atol=1e-12,rtol=0)
    np.testing.assert_allclose(N.T@R,0,atol=1e-12,rtol=0)
    means=np.stack([x[d==j].astype(np.float64).mean(0) for j in range(15)])
    np.testing.assert_allclose((means-means[0])@R,0,atol=1e-12,rtol=0)
    q=x[:3].copy();q2=q.copy();q2[:,0]+=.1
    np.testing.assert_allclose(m.raw_scores(state,q,l[:3]),m.raw_scores(state,q2,l[:3]),atol=1e-12,rtol=0)
    scores=m.raw_scores(state,q,l[:3]);changed=m.raw_scores(state,q,l[:3]+1.)
    np.testing.assert_allclose(changed[:,1]-scores[:,1],1.,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(changed[:,0],scores[:,0])
    assert diag['logit_path_domain_invariance_guaranteed'] is False
    residual=np.linalg.norm(x.astype(np.float64)@R,axis=1)
    assert float(state['alpha'])==pytest.approx(l.max(1).astype(np.float64).mean()/residual.mean(),rel=1e-14)


def test_fourteen_nuisance_directions_keep_eighty_residual_and_sixtysix_principal():
    m=api();x,l,d=shifted(14);state,diag=m.fit(x,l,d,'domain_orthogonal')
    assert diag['nuisance_rank']==14 and diag['effective_dimension']==146
    assert diag['residual_dimension']==80 and diag['principal_dimension']==66
    N,R=state['nuisance_basis'],state['residual_basis']
    np.testing.assert_allclose(N.T@N,np.eye(14),atol=1e-12,rtol=0)
    np.testing.assert_allclose(R.T@R,np.eye(80),atol=1e-12,rtol=0)
    np.testing.assert_allclose(N.T@R,0,atol=1e-12,rtol=0)
    assert len(diag['eigenvalues_ascending'])==146 and diag['state_numeric_bytes']==103688+160*14*8
    assert diag['nuisance_basis_audit_only'] is True


def test_no_domain_difference_is_standard_equivalent_and_query_rows_are_independent():
    m=api();base,l,_=cloud(n=100);x=np.tile(base,(15,1));logits=np.tile(l,(15,1));d=np.repeat(np.arange(15,dtype=np.int64),len(base))
    state,diag=m.fit(x,logits,d,'domain_orthogonal');standard,_=m.fit(x,logits,d,'standard')
    assert diag['nuisance_rank']==0 and state['nuisance_basis'].shape==(160,0)
    np.testing.assert_array_equal(m.raw_scores(state,x[:5],logits[:5]),m.raw_scores(standard,x[:5],logits[:5]))
    before=deepcopy(state);q=np.vstack((x[:5],np.zeros((1,160),np.float32)));ql=np.vstack((logits[:5],np.zeros((1,6),np.float32)))
    score=m.raw_scores(state,q,ql);p=np.asarray([5,3,0,2,1,4])
    np.testing.assert_allclose(m.raw_scores(state,q[p],ql[p]),score[p],atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,q[i:i+1],ql[i:i+1]) for i in range(len(q))]),score,atol=1e-12,rtol=0)
    rotation=np.linalg.qr(np.random.default_rng(7).normal(size=(80,80)))[0]
    changed=dict(state,residual_basis=state['residual_basis']@rotation)
    np.testing.assert_allclose(m.raw_scores(changed,q,ql),score,atol=1e-12,rtol=0)
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    assert score[-1,0]==0 and score[-1,1]==pytest.approx(np.log(6.))


def test_fit_rejects_missing_domain_labels_nonfinite_and_degenerate_inputs(subtests):
    m=api();x,l,d=cloud()
    for defect in ('short','zero','constant','negative_alpha','nan','dtype','norm','logit_dtype','domain_dtype','domain_nan','missing','illegal','mode','extra_y'):
        with subtests.test(defect=defect),pytest.raises((ValueError,TypeError)):
            if defect=='short':m.fit(x[:80],l[:80],d[:80],'standard')
            elif defect=='zero':m.fit(np.zeros_like(x),l,d,'domain_orthogonal')
            elif defect=='constant':m.fit(np.tile(x[0],(len(x),1)),l,d,'domain_orthogonal')
            elif defect=='negative_alpha':m.fit(x,np.full_like(l,-1),d,'standard')
            elif defect=='nan':a=x.copy();a[0,0]=np.nan;m.fit(a,l,d,'standard')
            elif defect=='dtype':m.fit(x.astype(np.float64),l,d,'standard')
            elif defect=='norm':m.fit(x*4,l,d,'standard')
            elif defect=='logit_dtype':m.fit(x,l.astype(np.float64),d,'standard')
            elif defect=='domain_dtype':m.fit(x,l,d.astype(np.int32),'standard')
            elif defect=='domain_nan':a=d.astype(np.float64);a[0]=np.nan;m.fit(x,l,a,'standard')
            elif defect=='missing':m.fit(x[d!=14],l[d!=14],d[d!=14],'standard')
            elif defect=='illegal':a=d.copy();a[0]=15;m.fit(x,l,a,'domain_orthogonal')
            elif defect=='mode':m.fit(x,l,d,'learned')
            else:m.fit(x,l,d,'standard',y=np.zeros(len(x),np.int64))


def test_exact_state_schema_rejects_nonorthogonal_crossover_and_private_extra(subtests):
    m=api();x,l,d=cloud();state,_=m.fit(x,l,d,'domain_orthogonal')
    for defect in ('extra','schema','mode','origin','dtype','rank15','nuisance','residual','cross','alpha','nan'):
        with subtests.test(defect=defect),pytest.raises(ValueError):
            bad={k:v.copy() for k,v in state.items()}
            if defect=='extra':bad['true_TX']=np.zeros(len(x))
            elif defect=='schema':bad['schema']=np.asarray('other')
            elif defect=='mode':bad['mode']=np.asarray('standard')
            elif defect=='origin':bad['origin'][0]=.1
            elif defect=='dtype':bad['residual_basis']=bad['residual_basis'].astype(np.float32)
            elif defect=='rank15':bad['nuisance_basis']=np.eye(160,15)
            elif defect=='nuisance':bad['nuisance_basis'][:,0]*=2
            elif defect=='residual':bad['residual_basis'][:,0]*=2
            elif defect=='cross':bad['nuisance_basis']=bad['residual_basis'][:,:1].copy()
            elif defect=='alpha':bad['alpha']=np.asarray(0.)
            else:bad['residual_basis'][0,0]=np.nan
            m.validate_state(bad)
