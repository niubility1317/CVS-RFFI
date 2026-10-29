import importlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.pamf_phase3_open_world.star_binary_graph_run')
    except ModuleNotFoundError:pytest.fail('binary graph runner is not implemented')


def config(root):
    r=module()
    return dict(run_id=r.RUN_ID,code_commit='a'*40,source_root=str(root/r.PARENT/'source'),
        U_public_root=str(root/'discovery/public'),U_capsule_id='p3dev-star-g10-6e1292b62ab09286ebf1a0df',U_split_id='83c5cc1fe2f2a1512dbc1287',
        Q_public_root=str(root/'evaluation/public'),Q_capsule_id='p3dev-star-g10-71a5a0dec23ac6ccc3b479c2',Q_split_id='05de65cd4818f3e5e33ced5c',
        output_root=str(root/'predict'),batch_size=128,dura_checkpoint=str(root/'G.pt'),dura_release=str(root/'G-release'))


def pipeline(tmp_path,skip=False,bad_parent=False,registered=False):
    r=module();c=config(tmp_path);phases=[]
    prior={'marker':{'run_id':r.PARENT,'code_commit':'b'*40 if bad_parent else r.PARENT_COMMIT}}
    backend={'reader_identity':{'model':'original_F0_E50'},'dura_identity':{'deployment_id':'dura_v1'}}
    def smoke(*args):phases.append('smoke');return dict(status='PASS',source_or_target_real_rows_read=0,neural_updates=0)
    def opened(cfg,role):
        phases.append(role)
        if role=='Q':
            frozen=json.loads((Path(c['output_root'])/'adaptation/complete.json').read_text())
            assert frozen['status']==r.U_STATUS and frozen['Q_rows_read']==0
            assert (Path(c['output_root'])/'adaptation/graph.npz').is_file()
        n=384 if role=='U' else 864
        x=np.zeros((n,2,256),np.float32);x[:,0,0]=np.arange(n)%2;x[:,0,1]=np.arange(n)
        return {},np.asarray([role+str(i) for i in range(n)]),x
    def encode(b,values,device):
        y=values[:,0,0].astype(int);raw=np.eye(2,160,dtype=np.float32)[y]
        if registered:
            angle=(values[:,0,1]//2)*(2*np.pi/192)
            raw[y==1,1]=np.sqrt(.99);raw[y==1,2]=.1*np.cos(angle[y==1]);raw[y==1,3]=.1*np.sin(angle[y==1])
        l=np.zeros((len(y),6),np.float32);l[y==0,0]=10
        return {'fused160':raw},l,{'fused160':raw.copy()},l.copy()
    with patch.object(r.previous,'load_source',return_value=prior),patch.object(r.previous,'_target_backend',return_value=backend),\
        patch.object(r.previous,'_snapshot',return_value=[]),patch.object(r.previous,'_check_models'),\
        patch.object(r,'_smoke',side_effect=smoke),patch.object(r.public,'_open',side_effect=opened),\
        patch.object(r,'_encode',side_effect=encode),patch.object(r,'_margin',side_effect=lambda b1,l1,b0,l0,p: np.ones(len(l1),np.float64) if skip else np.where(l1[:,0]>0,1.,-1.)):
        if bad_parent:
            with pytest.raises(ValueError):r.run(c,'cpu')
            assert not phases
            return
        done=r.run(c,'cpu')
    assert phases==['smoke','U','Q']
    return r,c,done


def test_complete_pipeline_freezes_u_memory_then_preserves_identity_and_old_gate(tmp_path):
    r,c,done=pipeline(tmp_path)
    m,p,diag,states=r._predictions(c['output_root'])
    assert done['status']==r.STATUS and p['prediction'].shape==(4,864)
    assert p['closed_prediction'].shape==(2,864)
    np.testing.assert_array_equal(p['closed_prediction'][0],p['closed_prediction'][1])
    np.testing.assert_array_equal(p['baseline_closed_prediction'],p['closed_prediction'][0])
    assert all(x['centers'].shape==(0,160) for x in states.values())
    assert done['U_memory_deployed'] and not done['source_bank_deployed']
    assert done['costs']['F0_real_full_forward_rows']==2496 and done['costs']['G_real_forward_rows']==1248
    assert done['source_parameters_updated']==done['new_threshold_calibrations']==0
    with pytest.raises(FileExistsError):r.run(c,'cpu')


def test_empty_seed_exact_four_state_fallback_and_corrupt_joint_rejected(tmp_path):
    r,c,done=pipeline(tmp_path,skip=True);root=Path(c['output_root'])
    _,p,_,_=r._predictions(root)
    assert done['graph_diagnostics']['status']=='SKIPPED_EMPTY_SEED_SIDE'
    np.testing.assert_array_equal(p['knownness'][0],p['knownness'][1])
    np.testing.assert_array_equal(p['prediction'][0],p['prediction'][1])
    p['prediction'][2,0]=6;np.savez(root/'predictions.npz',**p)
    with pytest.raises(ValueError):r._predictions(root)


def test_independent_config_rejects_truth_k_oldrun_bad_handles_and_parent(tmp_path):
    r=module();c=config(tmp_path);r.validate_config(c)
    for key,value in [('truth_path','private'),('true_K',2),('run_id',r.PARENT),('Q_capsule_id','formal')]:
        wrong=dict(c);wrong[key]=value
        with pytest.raises(ValueError):r.validate_config(wrong)
    pipeline(tmp_path,bad_parent=True)


def test_nonzero_u_registry_protects_old_acceptance_with_graph_scores(tmp_path):
    r,c,done=pipeline(tmp_path,registered=True)
    _,p,diag,states=r._predictions(c['output_root'])
    assert done['prototype_counts']==dict(DA0=1,DA1=1)
    assert all(len(s['centers'])==1 for s in states.values())
    for da in range(2):
        accepted=p['knownness'][da]>=0
        np.testing.assert_array_equal(p['prediction'][da+2,accepted],p['prediction'][da,accepted])
        assert np.any(p['prediction'][da+2,~accepted]==6)
        assert not np.any((p['prediction'][da+2,~accepted]>=0)&(p['prediction'][da+2,~accepted]<6))
