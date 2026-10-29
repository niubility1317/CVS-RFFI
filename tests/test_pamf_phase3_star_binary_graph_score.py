"""Independent scoring behavior on synthetic graph and registry predictions."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.pamf_phase3_open_world import star_binary_graph_score as m

TARGETS=Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'
STATES=m.STATES


def fixture(root,ks=(2,2)):
    uy=np.tile(np.repeat(np.arange(8,dtype=np.int64),8),6)
    y=np.tile(np.repeat(np.arange(9,dtype=np.int64),16),6)
    ui=np.asarray(['U'+str(i) for i in range(384)]);qi=np.asarray(['Q'+str(i) for i in range(864)])
    qs=np.repeat(m.SCENES,144);us=np.repeat(m.SCENES,64)
    closed=np.broadcast_to(np.where(y<6,y,0),(2,864)).copy()
    p=dict(ids=qi,U_ids=ui,state_names=np.asarray(STATES),closed_prediction=closed,
        baseline_closed_prediction=closed[0].copy(),prediction=np.zeros((4,864),np.int64),knownness=np.zeros((4,864),np.float64))
    diag,states={},{}
    for di,da in enumerate(('DA0','DA1')):
        k=ks[di];ass=np.where((uy>=6)&(uy<6+min(k,2)),uy-6,-1).astype(np.int64)
        diag[da]=dict(ids=ui,assignments=ass,candidate_mask=uy>=6,component_labels=uy.copy(),
            component_sizes=np.bincount(uy),registered_component_indices=np.arange(6,6+k,dtype=np.int64),
            components=[dict(component_index=c) for c in range(8)])
        states[da]=dict(centers=np.zeros((k,160)),radii=np.ones(k))
    for si,name in enumerate(STATES):
        k=ks[si%2];reg=si>=2;a=np.full((864,6+(k if reg else 0)),-2.,np.float64)
        a[np.arange(864),closed[si%2]]=np.where(y<6,1.,-1.)
        if si%2==0:
            for j in range(6):a[j*144,:]=-1.
        if reg:
            for c in range(min(k,2)):a[y==6+c,6+c]=1.
        p['joint_margins_'+name]=a;refresh(p,name)
    done=dict(run_id=m.RUN_ID,code_commit='a'*40,Q_day='2021-03-15',costs=dict(neural_updates=0))
    ut=root/'U.npz';qt=root/'Q.npz'
    np.savez(ut,ids=ui[::-1],y=uy[::-1],receiver=np.full(384,26,np.int64),scenario=us[::-1])
    np.savez(qt,ids=qi[::-1],y=y[::-1],receiver=np.full(864,26,np.int64),scenario=qs[::-1])
    return (done,p,diag,states),ut,qt,uy,y,qs


def refresh(p,name):
    si=STATES.index(name);a=p['joint_margins_'+name]
    p['knownness'][si]=a.max(1);p['prediction'][si]=np.where(a.max(1)>=0,a.argmax(1),-1)


def evaluate(root,v,name='score'):
    public,ut,qt,*_=v
    with patch.object(m,'_prediction_gate',return_value=public):return m.score(root,ut,qt,TARGETS,root/name)


def test_both_U_maps_freeze_before_Q_and_all_states_have_correct_denominators(tmp_path):
    v=fixture(tmp_path);events=[];truth=m._truth;matching=m.prior._matching
    def read(p,ids):events.append('U' if Path(p)==v[1] else 'Q');return truth(p,ids)
    def match(*a):events.append('map');return matching(*a)
    with patch.object(m,'_truth',side_effect=read),patch.object(m.prior,'_matching',side_effect=match):r=evaluate(tmp_path,v)
    assert events==['U','map','map','Q'] and r['primary_pass']
    assert r['primary_state']=='DA1_REG1' and r['states']['DA0_REG0']['overall']['A_N'] is None
    assert r['primary_summary']['A_N']==1. and r['primary_summary']['R_U_labels']==[8]
    assert r['states']['DA1_REG0']['overall']['R_U_labels']==[6,7,8]
    assert r['full_goal_status']=='ACTIVE' and not r['final_phase3_complete']
    assert r['deltas']['DA1_REG1_minus_DA1_REG0']['primary_R_U_and_H_comparable'] is False
    assert json.loads((tmp_path/'score/complete.json').read_text())['code_commit']=='a'*40
    with pytest.raises(FileExistsError):evaluate(tmp_path,v)


def test_public_gate_precedes_both_truths(tmp_path):
    with patch.object(m,'_prediction_gate',side_effect=ValueError('incomplete')),patch.object(m,'_truth') as truth:
        with pytest.raises(ValueError,match='incomplete'):m.score(tmp_path,'U','Q',TARGETS,tmp_path/'out')
        truth.assert_not_called()


def test_missing_discovered_class_is_zero_in_AN_not_removed(tmp_path):
    v=fixture(tmp_path,(0,1));r=evaluate(tmp_path,v)
    assert r['states']['DA0_REG1']['overall']['A_N']==0.
    assert r['primary_summary']['A_N']==.5 and r['primary_summary']['new_macro_denominator']==2
    assert not r['primary_pass']


def test_unmatched_slots_are_accepted_wrong(tmp_path):
    v=fixture(tmp_path);d,p,diag,states=v[0];y=v[4]
    diag['DA1']['assignments']=np.where(v[3]==0,0,np.where(v[3]==1,1,-1)).astype(np.int64)
    a=p['joint_margins_DA1_REG1'];a[y>=6,6:]=-2.;a[y>=6,6]=1.;refresh(p,'DA1_REG1')
    r=evaluate(tmp_path,v)
    assert r['primary_summary']['unmatched_anonymous_accepted']==288
    assert r['primary_summary']['A_N']==0. and r['primary_summary']['R_U']==0.


@pytest.mark.parametrize('failure',['RU','no_gain','missing_scene','AN'])
def test_final_absolute_constraints_cannot_be_replaced_by_partial_success(tmp_path,failure):
    v=fixture(tmp_path);p=v[0][1];y=v[4];scene=v[5]
    if failure=='RU':
        p['joint_margins_DA1_REG0'][np.flatnonzero(y>=6)[:69],0]=1.;refresh(p,'DA1_REG0')
    elif failure=='no_gain':
        for name in ('DA0_REG0','DA0_REG1'):
            for i in range(6):p['joint_margins_'+name][i*144,0]=1.
            refresh(p,name)
    elif failure=='missing_scene':
        with np.load(v[2],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        data['scenario'][data['scenario']==m.SCENES[0]]=m.SCENES[1];np.savez(v[2],**data)
    else:
        p['joint_margins_DA1_REG1'][y==7,6:]=-2.;refresh(p,'DA1_REG1')
    assert not evaluate(tmp_path,v)['primary_pass']


def test_failed_DA0_absolute_metrics_are_diagnostic_not_extra_entry_gate(tmp_path):
    v=fixture(tmp_path);p=v[0][1];y=v[4];scene=v[5]
    for name in ('DA0_REG0','DA0_REG1'):
        a=p['joint_margins_'+name]
        for s in m.SCENES:
            for c in range(6):a[np.flatnonzero((y==c)&(scene==s))[:2],:6]=-1.
        if name=='DA0_REG0':a[np.flatnonzero(y>=6)[:69],0]=1.
        refresh(p,name)
    r=evaluate(tmp_path,v)
    assert not r['frozen_rejection_stage_passed'] and r['primary_pass']
