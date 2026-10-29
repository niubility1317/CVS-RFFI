"""Synthetic reciprocal scorer contracts; no real data, model or target access."""
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')
ARMS=('self_pl','reciprocal')
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
TARGETS=Path(__file__).resolve().parents[1]/'configs/pamf_phase3_targets_20261029.json'


def module():return importlib.import_module('experiments.pamf_phase3_open_world.star_reciprocal_score')


def fixture(root, counts=None):
    counts=counts or {arm:{'DA0':2,'DA1':2} for arm in ARMS}
    uy=np.tile(np.repeat(np.arange(8,dtype=np.int64),8),6);y=np.tile(np.repeat(np.arange(9,dtype=np.int64),16),6)
    ui=np.asarray(['U'+str(i) for i in range(384)]);qi=np.asarray(['Q'+str(i) for i in range(864)])
    qs=np.repeat(SCENES,144);us=np.repeat(SCENES,64);closed=np.broadcast_to(np.where(y<6,y,0),(2,2,864)).copy()
    p=dict(ids=qi,U_ids=ui,arms=np.asarray(ARMS),state_names=np.asarray(STATES),closed_prediction=closed,
        baseline_closed_prediction=closed[0,0].copy(),prediction=np.zeros((2,4,864),np.int64),knownness=np.zeros((2,4,864),np.float64))
    diag={};states={};pseudo={}
    for ai,arm in enumerate(ARMS):
        diag[arm]={};states[arm]={};pseudo[arm]=dict(ids=ui.copy(),labels=np.where(uy<6,uy,-1).astype(np.int64),
            selected=uy<6,trusted=uy<6,expanded=np.zeros(384,bool))
        for da in ('DA0','DA1'):
            k=counts[arm][da];ass=np.where((uy>=6)&(uy<6+min(k,2)),uy-6,-1).astype(np.int64)
            diag[arm][da]=dict(ids=ui.copy(),assignments=ass,candidate_mask=uy>=6,component_labels=uy.copy(),
                component_sizes=np.bincount(uy),registered_component_indices=np.arange(6,6+k,dtype=np.int64),
                components=[dict(component_index=c) for c in range(8)],neighbors=np.zeros((384,1),np.int64),reciprocal=np.ones(384,bool),registered_count=k)
            states[arm][da]=dict(schema='synthetic-public-state',centers=np.zeros((k,160),np.float64),radii=np.ones(k,np.float64))
        for si,name in enumerate(STATES):
            da='DA'+str(si%2);reg=si>=2;k=counts[arm][da];joint=np.full((864,6+(k if reg else 0)),-2.,np.float64)
            joint[np.arange(864),closed[ai,si%2]]=np.where(y<6,1.,-1.)
            if si%2==0:
                for s in range(6):joint[s*144,:]=-1.
            if reg:
                for c in range(min(k,2)):joint[y==6+c,6+c]=1.
            p['joint_'+arm+'_'+name]=joint;p['knownness'][ai,si]=joint.max(1)
            p['prediction'][ai,si]=np.where(joint.max(1)>=0,joint.argmax(1),-1)
    done=dict(run_id='20261030-phase3-star-reciprocal-adapter-s2026092701-r01',code_commit='a'*40,Q_day='2021-03-15',
        costs=dict(adapter_updates_per_arm=200),optimizer_updates=400)
    ut=root/'U.npz';qt=root/'Q.npz'
    np.savez(ut,ids=ui[::-1],y=uy[::-1],receiver=np.full(384,26,np.int64),scenario=us[::-1])
    np.savez(qt,ids=qi[::-1],y=y[::-1],receiver=np.full(864,26,np.int64),scenario=qs[::-1])
    return (done,p,diag,states,pseudo),ut,qt,uy,y,qs


def refresh(p,arm,name):
    ai=ARMS.index(arm);si=STATES.index(name);a=p['joint_'+arm+'_'+name]
    p['knownness'][ai,si]=a.max(1);p['prediction'][ai,si]=np.where(a.max(1)>=0,a.argmax(1),-1)


def evaluate(m,root,v,name='score'):
    public,ut,qt,*_=v
    with patch.object(m,'_prediction_gate',return_value=public),patch.object(m.base,'_gate',side_effect=AssertionError('No legacy neighborhood gate')):
        return m.score(root,ut,qt,TARGETS,root/name)


def test_all_four_U_maps_freeze_before_Q_truth_and_full_primary(tmp_path):
    m=module();v=fixture(tmp_path);events=[];truth=m._truth;match=m.prior._matching
    def read(path,ids):events.append('U' if Path(path)==v[1] else 'Q');return truth(path,ids)
    def matching(*a):events.append('map');return match(*a)
    with patch.object(m,'_truth',side_effect=read),patch.object(m.prior,'_matching',side_effect=matching):r=evaluate(m,tmp_path,v)
    assert events==['U','map','map','map','map','Q']
    assert r['primary_pass'] and r['primary_arm']=='reciprocal' and r['primary_state']=='DA1_REG1'
    assert r['next_step']=='READY_FOR_SEPARATE_CONFIRMATION'
    assert r['arms']['reciprocal']['mappings']['DA1']['mapping']=={'0':6,'1':7}
    assert r['arms']['reciprocal']['adaptation_goals']['old_macro_open_accuracy_gain']>0
    assert r['arms']['reciprocal']['states']['DA1_REG1']['overall']['A_N']==1.
    assert r['arms']['reciprocal']['states']['DA0_REG0']['overall']['A_N'] is None
    assert r['costs']==v[0][0]['costs'] and not r['final_phase3_complete'] and r['full_goal_status']=='ACTIVE'
    assert json.loads((tmp_path/'score/complete.json').read_text())['code_commit']=='a'*40
    with pytest.raises(FileExistsError):evaluate(m,tmp_path,v)


@pytest.mark.parametrize('reason',['incomplete predictions','registered overwrote old accepted','foreign model changed'])
def test_public_gate_blocks_both_truths(tmp_path,reason):
    m=module()
    with patch.object(m,'_prediction_gate',side_effect=ValueError(reason)),patch.object(m,'_truth') as truth:
        with pytest.raises(ValueError,match=reason):m.score(tmp_path,'U','Q',TARGETS,tmp_path/'score')
        truth.assert_not_called()


def test_zero_and_different_K_keep_all_arrivals_in_denominator(tmp_path):
    m=module();v=fixture(tmp_path,{'self_pl':{'DA0':0,'DA1':1},'reciprocal':{'DA0':1,'DA1':2}})
    r=evaluate(m,tmp_path,v)
    assert r['arms']['self_pl']['states']['DA0_REG1']['overall']['A_N']==0.
    assert r['arms']['self_pl']['states']['DA1_REG1']['overall']['A_N']==.5
    assert r['arms']['self_pl']['states']['DA1_REG1']['overall']['new_macro_denominator']==2
    assert r['arms']['reciprocal']['states']['DA1_REG1']['overall']['A_N']==1.


def test_unmatched_anonymous_is_accepted_wrong_and_current_winner_drives_curves(tmp_path):
    m=module();v=fixture(tmp_path);d,p,diag,states,_=v[0];y=v[4]
    diag['reciprocal']['DA1']['assignments']=np.where(v[3]==0,0,np.where(v[3]==1,1,-1)).astype(np.int64)
    a=p['joint_reciprocal_DA1_REG1'];a[y>=6,6:]=-2.;a[y>=6,6]=1.;refresh(p,'reciprocal','DA1_REG1')
    r=evaluate(m,tmp_path,v);row=r['primary_summary']
    assert row['unmatched_anonymous_accepted']==288 and row['R_U']==0. and row['A_N']==0.
    assert not r['primary_pass']
    a=p['joint_reciprocal_DA1_REG0'];wrong=(y+1)%6;a[:,:6]=-2.;a[np.arange(864),wrong]=np.where(y<6,1.,-1.)
    p['closed_prediction'][1,1]=wrong;refresh(p,'reciprocal','DA1_REG0')
    r=evaluate(m,tmp_path,v,'wrong');row=r['arms']['reciprocal']['states']['DA1_REG0']['overall']
    assert row['original_F0_macro_closed_accuracy']==1. and row['oscr_auc_macro']==0.


@pytest.mark.parametrize('mode',['main_failed','DA1_RU76','no_gain','later_scene'])
def test_absolute_v3_and_fixed_main_cannot_be_replaced(tmp_path,mode):
    m=module();v=fixture(tmp_path);p=v[0][1];y=v[4];sc=v[5]
    if mode=='main_failed':
        a=p['joint_reciprocal_DA1_REG1'];a[y==7,6:]=-2.;refresh(p,'reciprocal','DA1_REG1')
    elif mode=='DA1_RU76':
        a=p['joint_reciprocal_DA1_REG0'];a[np.flatnonzero(y>=6)[:69],0]=1.;refresh(p,'reciprocal','DA1_REG0')
    elif mode=='no_gain':
        for name in ('DA0_REG0','DA0_REG1'):
            a=p['joint_reciprocal_'+name]
            for s in range(6):a[s*144,0]=1.
            refresh(p,'reciprocal',name)
    else:
        a=p['joint_reciprocal_DA1_REG1'];a[(y==8)&(sc==SCENES[0]),6]=1.;refresh(p,'reciprocal','DA1_REG1')
    r=evaluate(m,tmp_path,v)
    assert r['arms']['self_pl']['joint_pass'] and not r['primary_pass'] and r['next_step']=='NO_PROMOTION'


def test_DA0_diagnostic_failure_does_not_block_valid_DA1_with_gain(tmp_path):
    m=module();v=fixture(tmp_path);p=v[0][1];y=v[4];sc=v[5]
    for name in ('DA0_REG0','DA0_REG1'):
        a=p['joint_reciprocal_'+name]
        # The fixed baseline misses CK90 and old-identity protection; DA1
        # independently restores them without changing its final requirements.
        for s in SCENES:
            for c in range(6):a[np.flatnonzero((y==c)&(sc==s))[:2],:6]=-1.
        if name=='DA0_REG0':a[np.flatnonzero(y>=6)[:69],0]=1.
        refresh(p,'reciprocal',name)
    r=evaluate(m,tmp_path,v);arm=r['arms']['reciprocal']
    assert not arm['states']['DA0_REG0']['goals']['passed']
    assert not arm['frozen_rejection_stage_passed'] and not r['frozen_rejection_stage_passed']
    assert arm['states']['DA1_REG0']['goals']['passed'] and arm['states']['DA1_REG1']['goals']['passed']
    assert arm['adaptation_goals']['old_macro_open_accuracy_gain']>0 and arm['adaptation_goals']['passed']
    assert 'DA0_absolute_v3_constraints' not in arm['adaptation_goals']['checks']
    assert r['primary_pass'] and r['next_step']=='READY_FOR_SEPARATE_CONFIRMATION'
    assert r['full_goal_status']=='ACTIVE' and not r['final_phase3_complete']


def test_missing_scene_coverage_is_NA_and_U_purity_is_truth_only(tmp_path):
    m=module();v=fixture(tmp_path);d,p,diag,states,pseudo=v[0];y=v[4];sc=v[5].copy()
    sc[(y==7)&(sc==SCENES[0])]=SCENES[1];np.savez(v[2],ids=p['ids'],y=y,receiver=np.full(864,26,np.int64),scenario=sc)
    pseudo['reciprocal']['selected'][:]=True;pseudo['reciprocal']['labels'][v[3]>=6]=0
    r=evaluate(m,tmp_path,v);row=r['arms']['reciprocal']['states']['DA1_REG1']['by_scenario'][SCENES[0]]
    assert row['A_N'] is None and row['H_score'] is None and not row['coverage']['complete'] and not r['primary_pass']
    pd=r['arms']['reciprocal']['pseudo_label_diagnostics']['selected']
    assert pd['unknown_count']==96 and pd['old_label_accuracy']==1. and pd['unknown_fraction']==.25


def test_target_limits_and_CLI_are_not_silently_changed(tmp_path):
    m=module();v=fixture(tmp_path);bad=json.loads(TARGETS.read_text());bad['known_macro_acceptance']['minimum']=.89
    path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    with patch.object(m,'_prediction_gate',return_value=v[0]),patch.object(m,'_truth') as truth:
        with pytest.raises(ValueError):m.score(tmp_path,v[1],v[2],path,tmp_path/'score')
        truth.assert_not_called()
    args=['--predictions-root','p','--U-truth','u','--Q-truth','q','--targets','t','--output','o']
    with patch.object(m,'score',return_value={'status':m.STATUS,'primary_pass':False}) as call:assert m.main(args)==0
    call.assert_called_once_with('p','u','q','t','o')
