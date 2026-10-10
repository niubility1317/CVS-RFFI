"""Focused recipe-difference and per-row scoring-barrier checks."""
from unittest.mock import patch
from copy import deepcopy
from experiments.cvs_legacy_no_sat import design as d,evaluate as e

def checks():
    rows=[d.config(r) for r in d.rows()]
    assert len(rows)==3 and len(e.VIEWS)==7 and not any(v.startswith('leo_') for v in e.VIEWS)
    for c in rows:
        a=d.make_args(c,'cpu')
        assert a.use_unlabeled and a.use_ema_teacher and a.pseudo_temporal_gate and a.pseudo_strong_agreement
        assert a.pseudo_temporal_mode=='batch_neighbor' and not a.pseudo_domain_gate
        assert a.label_epochs==130 and a.pseudo_epochs==70 and a.label_smoothing==.01
        assert a.phase1_lr_schedule=='cosine' and a.phase1_lr_min==1e-6 and a.lambda_u==.16 and a.lambda_ent==.01
        assert not a.use_concat_sat_channel_aug and not a.use_sat_consistency and a.sat_training_mode=='disabled'
        assert a.lambda_sat_cls==0 and a.lambda_sat_cons==0 and a.from_scratch and not a.baseline_ckpt and not a.teacher_ckpt
        bad=deepcopy(c);bad['features'].append('leo')
        try:d.validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed configuration accepted')
    old=e.EVAL_ROW;e.EVAL_ROW=rows[0]['row_id']
    value=dict(status='SOURCE_FROZEN',config=rows[0],target_access=False)
    with patch.object(e,'read',return_value=value) as read:
        assert e.snapshot(d)==[rows[0]] and read.call_count==1
    with patch.object(e,'snapshot',return_value=[rows[0]]),patch.object(e,'prediction_preflight',side_effect=ValueError('incomplete')),patch.object(e,'read') as read:
        try:e.score()
        except ValueError:pass
        else:raise AssertionError('Truth opened before complete prediction')
        read.assert_not_called()
    e.EVAL_ROW=old
    from experiments.cvs_legacy_no_sat.publish import REMOTE
    compile(REMOTE.replace('CONFIG',repr({})),'remote','exec')
    assert 'cvs_legacy_no_sat.dispatch' in REMOTE
    print('PASS: three seeds, old pseudo+EMA+cos preserved, no satellite path, immutable config, independent row freeze, truth barrier, release syntax')

def smoke(output,device='cpu'):
    from experiments.cvs_phase1_repair import smoke as s
    for k in ['ARMS','rows','config','make_args','SEEDS']:
        setattr(s,k,getattr(d,k))
    s.smoke(output,device)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke');p.add_argument('--device',default='cpu');a=p.parse_args()
    checks()
    if a.smoke:smoke(a.smoke,a.device)
