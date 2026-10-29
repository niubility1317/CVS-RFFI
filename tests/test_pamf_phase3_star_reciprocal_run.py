import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_reciprocal_run as r


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(6, 160), requires_grad=False)
        self.eval()
    def classify_features(self, x):
        return 30*F.normalize(x, dim=1)@self.weight.T


def config(root):
    return dict(run_id=r.RUN_ID,code_commit='a'*40,source_root=str(root/r.PARENT/'source'),
        U_public_root=str(root/'discovery/public'),U_capsule_id='p3dev-star-g10-6e1292b62ab09286ebf1a0df',U_split_id='83c5cc1fe2f2a1512dbc1287',
        Q_public_root=str(root/'evaluation/public'),Q_capsule_id='p3dev-star-g10-71a5a0dec23ac6ccc3b479c2',Q_split_id='05de65cd4818f3e5e33ced5c',
        output_root=str(root/'predict'),batch_size=128,dura_checkpoint=str(root/'G.pt'),dura_release=str(root/'G-release'))


def fixture_pipeline(tmp_path, empty=False):
    c=config(tmp_path);reader=Reader();g=Reader()
    prior={'marker':{'code_commit':r.PARENT_COMMIT}}
    backend=dict(reader=reader,dura=(g,None),reader_identity={'model':'original_F0_E50'},dura_identity={'deployment_id':'dura_v1'})
    phases=[]
    def opened(cfg,role):
        phases.append(role)
        n=384 if role=='U' else 864
        if role=='Q':
            frozen=json.loads((Path(c['output_root'])/'adaptation/complete.json').read_text())
            assert frozen['Q_rows_read']==0 and set(frozen['adapter_summaries'])==set(r.ARMS)
        return {},np.asarray([role+'-'+str(i) for i in range(n)]),np.zeros((n,2,256),np.float32)
    def encode(b,values,device):
        labels=np.arange(len(values))%6
        raw=np.zeros((len(values),160),np.float32);raw[np.arange(len(values)),labels]=2.
        logits=np.zeros((len(values),6),np.float32);logits[np.arange(len(values)),labels]=30.
        return {'fused160':raw},logits,{'fused160':raw.copy()},logits.copy()
    def fit(raw,labels,classify,device):
        model=r.adapter.Adapter().eval().requires_grad_(False)
        steps=200 if np.any(labels>=0) else 0
        summary=dict(status='COMPLETE' if steps else 'SKIPPED_NO_PSEUDOLABELS',steps=steps,U_count=len(raw),
            selected_count=int((labels>=0).sum()),parameters=10432,model_seed=2026092701,F0_updates=0,G_updates=0,finite=True)
        trace=[dict(step=i+1,loss_ce=0.,loss_gram=0.,loss_total=0.,grad_norm=0.,max_pre_normalization_residual=0.) for i in range(steps)]
        return model,trace,summary
    with patch.object(r.previous,'load_source',return_value=prior),patch.object(r.previous,'_target_backend',return_value=backend),\
         patch.object(r.public,'_open',side_effect=opened),patch.object(r,'_encode',side_effect=encode),\
         patch.object(r,'_margin',side_effect=lambda b1,l1,b0,l0,p:np.full(len(l1),-1. if empty else 1.,np.float64)),patch.object(r.adapter,'fit',side_effect=fit):
        done=r.run(c,'cpu')
    assert phases==['U','Q']
    return c,done


def test_real_flow_freezes_all_u_state_before_q_and_preserves_original_models(tmp_path):
    c,done=fixture_pipeline(tmp_path)
    assert done['truth_read'] is False and done['query_fit'] is False
    assert done['costs']['F0_real_full_forward_rows']==2496 and done['costs']['G_real_forward_rows']==1248
    m,p,diag,states,pseudo=r._predictions(c['output_root'])
    assert m['status']==r.STATUS and p['prediction'].shape==(2,4,864)
    assert np.array_equal(p['prediction'][0,0],p['prediction'][1,0])
    assert all(len(states[a][d]['centers'])==0 for a in r.ARMS for d in r.DAS)
    assert all(np.array_equal(p['prediction'][i,0],p['prediction'][i,2]) for i in range(2))
    with pytest.raises(FileExistsError):r.run(c,'cpu')


def test_truth_fields_formal_handles_and_source_replacement_are_rejected(tmp_path):
    c=config(tmp_path);r.validate_config(c)
    for key,value in [('truth_path','secret.npz'),('true_K',2),('Q_capsule_id','formal')]:
        wrong=dict(c);wrong[key]=value
        with pytest.raises(ValueError):r.validate_config(wrong)
    wrong=dict(c,output_root=c['source_root'])
    with pytest.raises(ValueError):r.validate_config(wrong)


def test_incomplete_state_and_corrupt_joint_gate_stay_closed(tmp_path):
    c,done=fixture_pipeline(tmp_path);root=Path(c['output_root'])
    with np.load(root/'predictions.npz',allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
    p['prediction'][0,2,0]=6
    np.savez(root/'predictions.npz',**p)
    with pytest.raises(ValueError,match='joint decisions'):r._predictions(root)
    (root/'complete.json').unlink()
    with pytest.raises(FileNotFoundError):r._predictions(root)


def test_empty_pseudo_mask_is_exact_da_identity_with_no_extra_head_or_readout(tmp_path):
    c,done=fixture_pipeline(tmp_path,empty=True)
    _,p,_,_,_=r._predictions(c['output_root'])
    assert all(x['steps']==0 for x in done['costs']['adapter_training'].values())
    assert done['costs']['additional_adapter_inference_rows_per_arm']==dict(self_pl=0,reciprocal=0)
    assert done['costs']['prior_evidence_real_row_calls']==1248
    assert np.array_equal(p['prediction'][:,0],p['prediction'][:,1])
    assert np.array_equal(p['knownness'][:,0],p['knownness'][:,1])
