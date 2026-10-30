"""Independent density-score boundaries and true-class macro reporting."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_pamf_phase3_star_density_run import fixture, write


def completed(root,monkeypatch):
    return fixture(root,monkeypatch)


def truth_file(root,m,roles):
    n=len(roles['novel']['ids']);rows=[(tx,r,s) for tx in m.common.UNKNOWN_TX for r in m.common.RECEIVERS for s in m.common.SCENES]
    assert n==len(rows)
    d=dict(ids=roles['novel']['ids'],rx_i=roles['novel']['metadata']['rx_i'],day_i=roles['novel']['metadata']['day_i'],
        tx=np.asarray([x[0] for x in rows]),scenario=np.asarray([x[2] for x in rows]))
    p=root/'private_synthetic_truth.npz';np.savez(p,**{k:v[::-1] for k,v in d.items()});return p


def test_complete_density_formula_margin_and_source_only_primary_metrics(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_density_score')
    with completed(tmp_path,monkeypatch) as (m,c,roles,b,counts,events,r):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        p=truth_file(tmp_path,s,roles)
        monkeypatch.setattr(m,'_fit_known',lambda *a:pytest.fail('Scorer cannot refit source models'))
        monkeypatch.setattr(m,'_fit_background',lambda *a:pytest.fail('Scorer cannot fit background'))
        result=s.score(c['output_root'],p,tmp_path/'score')
        assert result['primary_method']=='density_ratio' and result['source_gate_passed']
        assert result['primary_summary']['A_K']==result['primary_summary']['C_K']==result['primary_summary']['R_U']==1.
        assert len(result['methods']['density_ratio']['groups'])==42
        assert result['A_N'] is result['H_OW'] is None and result['full_goal_status']=='ACTIVE'
        assert result['target_reads']==0 and not result['final_phase3_complete']
        assert result['paired_primary_vs_control']['counts']['unknown_rejected_net']==0
        with pytest.raises(FileExistsError):s.score(c['output_root'],p,tmp_path/'score')


def test_all_predictions_models_and_calibration_validated_before_truth(tmp_path,monkeypatch,subtests):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_density_score')
    for defect in ('marker','state','raw','margin','argmax','calibration','overlap','missing_novel','private'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with completed(root,mp) as (m,c,roles,b,counts,e,r):
                m.run(c,'cpu');mp.setattr(s,'COUNTS',counts);out=Path(c['output_root'])
                if defect=='marker':
                    d=json.loads((out/'complete.json').read_text());d['real_F_forward_rows']=1;write(out/'complete.json',d)
                elif defect=='state':
                    p=out/'models/density_ratio.npz'
                    with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                    d['source_row_bank']=np.ones((2,160));np.savez(p,**d)
                elif defect=='calibration':
                    p=out/'calibration/complete.json';d=json.loads(p.read_text());d['methods']['density_ratio']['groups'].pop('overall');write(p,d)
                elif defect=='missing_novel':(out/'novel_predictions.npz').unlink()
                else:
                    p=out/'novel_predictions.npz'
                    with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='raw':d['raw_scores'][0,1]+=.1
                    elif defect=='margin':d['scores'][0,1]+=.1
                    elif defect=='argmax':d['closed'][0]=1
                    elif defect=='overlap':d['ids'][0]='V0'
                    else:d['tx']=np.asarray(['private']*len(d['ids']))
                    np.savez(p,**d)
                mp.setattr(s,'_truth',lambda *a:pytest.fail('Private truth opened before validation'))
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,root/'not_opened',root/'score')


def test_control_cannot_replace_primary_and_acceptance_is_not_identity_accuracy(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_density_score')
    with completed(tmp_path,monkeypatch) as (m,c,roles,b,counts,e,r):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        marker,cal,tau,states,known,novel=s._prediction(c['output_root'])
        known=deepcopy(known);novel=deepcopy(novel)
        known['closed'][0]=1  # Scores can accept an identity error; it is not correct-accept.
        known['baseline_closed'][0]=1
        novel['scores'][:,1]=1.  # Control rejects, fixed primary does not.
        monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,tau,states,known,novel))
        result=s.score(c['output_root'],truth_file(tmp_path,s,roles),tmp_path/'score')
        assert result['methods']['known_only']['goals']['passed']
        assert not result['source_gate_passed'] and not result['eligible_primary']
        assert result['next_step']=='NO_PROMOTION'
        assert result['primary_summary']['C_K']>result['primary_summary']['A_K']
        assert result['paired_primary_vs_control']['counts']['unknown_rejected_net']==-90


def test_missing_scene_true_TX_coverage_is_NA_and_not_certified(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_density_score')
    with completed(tmp_path,monkeypatch) as (m,c,roles,b,counts,e,r):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        path=truth_file(tmp_path,s,roles)
        with np.load(path) as z:d={k:z[k].copy() for k in z.files}
        d['scenario'][d['tx']==s.common.UNKNOWN_TX[0]]=s.common.SCENES[0]
        np.savez(path,**d)
        result=s.score(c['output_root'],path,tmp_path/'score')
        row=result['methods']['density_ratio']['by_scene'][s.common.SCENES[1]]
        assert row['R_U'] is None and not row['coverage']['unknown_complete']


def test_paired_counts_report_gains_losses_and_net_not_only_rates():
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_density_score')
    known=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.asarray([[1,-1],[-1,1],[-1,1],[1,1]],np.float64))
    novel=dict(scores=np.asarray([[-1,1],[1,-1],[-1,-1]],np.float64))
    got=s._paired_counts(known,novel,np.ones(4,bool),np.ones(3,bool))
    assert got['known_accepted_gain']==2 and got['known_accepted_loss']==1 and got['known_accepted_net']==1
    assert got['known_correct_accepted_gain']==1 and got['known_correct_accepted_loss']==1 and got['known_correct_accepted_net']==0
    assert got['unknown_rejected_gain']==got['unknown_rejected_loss']==1 and got['unknown_rejected_net']==0
