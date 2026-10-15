"""Closed-form geometry examples and source-role rejection cases."""
import copy
import json
import numpy as np
from .core import basis,dictionary,project,source,decision
from .design import config,validate


def fixture():
    p=np.zeros((18,5,8));p[:,:,0]=np.arange(-2,3)[None,:]
    obs=dict(labels=dict(TX=list(range(6)),day=[1,2,3],RX=[1,3,4,6,8],cell_count=70),
        folds=[dict(TX_index=t,day_index=d) for t in range(6) for d in range(3)],
        predicted_contrasts=p.tolist(),observed_contrasts=p.tolist(),relative_MSE=dict(prediction=0.))
    ids=['L'+str(i) for i in range(6300)]
    return dict(row={'completion.json':dict(status='SOURCE_DIAGNOSTIC_COMPLETE',identity_updates=0,checkpoint_loaded=False,target_access=False,L_count=6300),
        'spectral_crossfit.json':dict(U_used=False,V_used=False,target_access=False,primary='rectangular/DCT1to8',observations={'rectangular/DCT1to8':obs}),
        'provenance.json':dict(L_ids=ids,source_contract=dict(role_ids=dict(L_s=ids,U_s=['U0'],V=['V0'])))})


def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Illegal source case accepted')


def run():
    b=basis();assert np.allclose(b.T@b,np.eye(8),atol=1e-14) and np.allclose(b.mean(0),0,atol=1e-15)
    q,parameters,r=dictionary(.06,5,128)
    assert q.shape==(513,8) and parameters.shape==(513,2) and np.all(q[0]==0) and np.isfinite(q).all() and r>0
    assert np.abs(parameters[:,0]+1j*parameters[:,1]).max()<=.06+1e-15
    # A strength-limited x-axis dictionary cannot recover a y-axis direction.
    p=np.array([[2.,0.],[0.,2.],[1.,1.]])
    f=project(p,np.array([[0.,0.],[1.,0.],[-1.,0.]]))
    assert np.allclose(f['direction_energy'],[4,0,1])
    assert np.allclose(((f['bounded']-p)**2).sum(1),[1,4,1])
    assert np.allclose(f['relaxed'],[[2,0],[0,0],[1,0]])
    assert np.allclose(project(np.zeros((2,2)),np.zeros((1,2)))['relaxed'],0)
    valid=fixture();predicted,actual,ratio=source(valid);assert predicted.shape==(18,5,8) and ratio==0 and np.array_equal(actual,predicted)
    mutations=[lambda v:v['row']['spectral_crossfit.json'].__setitem__('target_access',True),
        lambda v:v['row']['spectral_crossfit.json'].__setitem__('U_used',True),
        lambda v:v['row']['spectral_crossfit.json'].__setitem__('V_used',True),
        lambda v:v['row']['completion.json'].__setitem__('checkpoint_loaded',True),
        lambda v:v['row']['completion.json'].__setitem__('identity_updates',1),
        lambda v:v['row']['completion.json'].__setitem__('target_access',True),
        lambda v:v['row']['provenance.json']['L_ids'].__setitem__(0,'U0'),
        lambda v:v['row']['spectral_crossfit.json'].__setitem__('primary','hann/DCT1to8'),
        lambda v:v['row']['spectral_crossfit.json']['observations']['rectangular/DCT1to8']['folds'][0].__setitem__('TX_index',5)]
    for mutation in mutations:
        v=copy.deepcopy(valid);mutation(v);rejects(lambda:source(v))
    c=config();validate(c);bad=copy.deepcopy(c);bad['radius']=.08;rejects(lambda:validate(bad))
    for a,z,status in ((.2,.205,'IDEAL_MAGNITUDE_DIRECTION_GAP'),(.9,.9,'SUBSTANTIAL_IDEAL_DIRECTION_COVERAGE'),(.6,.6,'MIXED_IDEAL_DIRECTION_COVERAGE'),(.3,.35,'INCONCLUSIVE_GRID_RESOLUTION')):
        assert decision(dict(weighted_direction_coverage=a),dict(weighted_direction_coverage=z),c)==status
    print(json.dumps(dict(status='PASS',closed_form_cases=5,source_negative_cases=len(mutations),configuration_negative_cases=1,
        decision_cases=4,raw_IQ_reads=0,checkpoint_reads=0,model_forwards=0,optimizer_updates=0)))


if __name__=='__main__':run()
