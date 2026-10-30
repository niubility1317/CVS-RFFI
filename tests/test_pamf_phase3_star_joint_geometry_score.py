"""Synthetic union/novel reference geometry with strictly delayed labels."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_geometry_score')


def write_npz(path,data):
    with path.open('wb') as f:np.savez(f,**data)


def toy():
    y=np.repeat(np.arange(9),2);rx=np.tile(np.asarray([1,3],np.int64),9);day=np.tile(np.asarray([1,2],np.int64),9)
    return y,rx,day


@contextmanager
def fixture(root,monkeypatch):
    from tests.test_pamf_phase3_star_joint_geometry_export import fixture as export_fixture
    with export_fixture(root,monkeypatch) as (export,c,L,novel,cp,td,prediction):
        s=api();monkeypatch.setattr(s,'COUNTS',dict(export.COUNTS))
        export.export(c,'cpu');out=Path(c['output_root']);Lroot=Path(cp['feature_manifest']['config']['L_root']);Lroot.mkdir(parents=True,exist_ok=True)
        (Lroot/'manifest.json').write_text(json.dumps(cp['feature_manifest']['source_manifests']['L']),encoding='utf-8')
        np.save(Lroot/'ids.npy',L['ids']);np.savez(Lroot/'metadata.npz',**L['metadata'])
        for forbidden in ('received.npy','restored.npy','teacher_logits.npy'):(Lroot/forbidden).write_bytes(b'not-an-array-MUST-NOT-READ')
        nv=len(novel['ids']);ny=np.repeat(np.arange(6,9),nv//3);y=np.r_[L['metadata']['y'],ny]
        with np.load(out/'union.npz') as z:data={k:z[k].copy() for k in z.files}
        data['original_fused160']=np.eye(160,dtype=np.float32)[y];data['protected160']=data['original_fused160'].copy()
        data['protected160'][y==6]=np.eye(160,dtype=np.float32)[0]  # Deliberate known intrusion; novel-only remains perfect.
        write_npz(out/'union.npz',data)
        truth=dict(ids=novel['ids'],tx=np.asarray([s.common.UNKNOWN_TX[v-6] for v in ny]),rx_i=novel['rx_i'],day_i=novel['day_i'],
                   scenario=np.full(nv,s.common.SCENES[-1]))
        truth_path=root/'synthetic-truth.npz';np.savez(truth_path,**truth)
        yield s,c,Lroot,truth_path,y,data,truth


def test_hand_confusion_macro_roles_and_paired_counts_keep_full_denominators():
    s=api();y,rx,day=toy();indices=np.arange(18)^1;indices[12:14]=[0,1]
    row=s.condition_report(indices,y,rx,day)
    assert row['confusion_counts'][6][0]==2 and sum(map(sum,row['confusion_counts']))==18
    assert row['macro9']==pytest.approx(8/9) and row['macro_known6']==1. and row['macro_novel3']==pytest.approx(2/3)
    assert row['novel_queries']['to_known_count']==2 and row['novel_queries']['pooled_to_known_rate']==pytest.approx(1/3)
    assert row['novel_queries']['same_novel_TX_count']==4 and row['known_queries']['to_novel_count']==0
    comparison=s.paired_correct(indices,np.arange(18)^1,y,y>=6)
    assert comparison['gained']==0 and comparison['lost']==2 and comparison['net']==-2
    indices[0]=12
    row=s.condition_report(indices,y,rx,day)
    assert row['known_queries']['to_novel_count']==1 and row['known_queries']['pooled_to_novel_rate']==pytest.approx(1/12)


def test_all_16_neighbors_before_L_and_only_novel_truth_then_matched_comparison(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (s,c,Lroot,truth_path,y,data,truth):
        calls=[];nearest=s.geometry._nearest_indices;load_L=s._load_L_labels;private=s.common._truth
        def choose(*args):
            assert not any(x in ('L_labels','novel_truth') for x in calls)
            value=nearest(*args);assert len(value)==4;calls.append('neighbors');return value
        def visible(*args):
            assert calls==['neighbors']*4;calls.append('L_labels');return load_L(*args)
        def hidden(*args):
            assert calls==['neighbors']*4+['L_labels'];calls.append('novel_truth');return private(*args)
        monkeypatch.setattr(s.geometry,'_nearest_indices',choose);monkeypatch.setattr(s,'_load_L_labels',visible);monkeypatch.setattr(s.common,'_truth',hidden)
        result=s.score(c['output_root'],Lroot,truth_path,tmp_path/'score')
        candidate=result['representations']['protected160'];row=candidate['union_reference']['cross_RX_and_day']
        assert row['macro9']==pytest.approx(8/9) and row['macro_novel3']==pytest.approx(2/3)
        assert candidate['novel_reference']['cross_RX_and_day']['overall']['macro_hit_rate']==1.
        change=candidate['novel_reference_vs_union']['cross_RX_and_day']
        assert change['paired_correct']['lost']==15 and change['unknown_to_known_invasion_count']==15
        assert result['paired_representations']['union_reference']['cross_RX_and_day']['novel_queries']['lost']==15
        assert result['all_16_neighbors_fixed_before_truth'] and not result['neighbor_selection_TX_or_K_used']
        assert result['diagnostic_only'] and not result['eligible_primary'] and result['target_reads']==0
        assert result['A_N'] is None and result['H_OW'] is None and not result['final_phase3_complete']
        with pytest.raises(FileExistsError):s.score(c['output_root'],Lroot,truth_path,tmp_path/'score')


def test_bad_prior_or_union_public_fields_fail_before_labels(tmp_path,monkeypatch):
    for defect in ('parent','target','pool','duplicate','nan','labels'):
        with monkeypatch.context() as patch,fixture(tmp_path/defect,patch) as (s,c,Lroot,truth_path,y,data,truth):
            out=Path(c['output_root'])
            if defect in ('parent','target'):
                marker=json.loads((out/'complete.json').read_text())
                if defect=='parent':marker['parent_model_identity']['code_commit']='d'*40
                else:marker['target_reads']=1
                (out/'complete.json').write_text(json.dumps(marker))
            else:
                if defect=='pool':data['pool'][0]=1
                elif defect=='duplicate':data['ids'][1]=data['ids'][0]
                elif defect=='nan':data['protected160'][0,0]=np.nan
                else:data['y']=y
                write_npz(out/'union.npz',data)
            patch.setattr(s,'_load_L_labels',lambda *a:pytest.fail('L labels opened on invalid features'))
            patch.setattr(s.common,'_truth',lambda *a:pytest.fail('Novel truth opened on invalid features'))
            with pytest.raises(ValueError):s.score(out,Lroot,truth_path,tmp_path/defect/'score')


def test_L_public_alignment_and_legal_labels_checked_only_after_neighbor_freeze(tmp_path,monkeypatch):
    for defect in ('ids','rx','y','manifest'):
        with monkeypatch.context() as patch,fixture(tmp_path/defect,patch) as (s,c,Lroot,truth_path,y,data,truth):
            if defect=='ids':np.save(Lroot/'ids.npy',np.load(Lroot/'ids.npy')[::-1])
            elif defect=='manifest':
                m=json.loads((Lroot/'manifest.json').read_text());m['parent_F0_identity']['model']='wrong';(Lroot/'manifest.json').write_text(json.dumps(m))
            else:
                with np.load(Lroot/'metadata.npz') as z:meta={k:z[k].copy() for k in z.files}
                meta['rx_i' if defect=='rx' else 'y'][0]=3 if defect=='rx' else 9
                write_npz(Lroot/'metadata.npz',meta)
            calls=[];nearest=s.geometry._nearest_indices
            def choose(*args):calls.append(1);return nearest(*args)
            patch.setattr(s.geometry,'_nearest_indices',choose)
            patch.setattr(s.common,'_truth',lambda *a:pytest.fail('Private truth must wait for legal L metadata'))
            with pytest.raises(ValueError):s.score(c['output_root'],Lroot,truth_path,tmp_path/defect/'score')
            assert len(calls)==4


def test_missing_candidates_or_required_TX_are_NA_not_zero_or_dropped():
    s=api();y,rx,day=toy();missing=np.full(len(y),-1,np.int64)
    row=s.condition_report(missing,y,rx,day)
    assert row['macro9'] is None and row['macro_novel3'] is None
    assert row['novel_queries']['pooled_to_known_rate'] is None and row['retrieval']['overall']['missing_candidate_count']==18
    absent=y.copy();absent[absent==8]=7
    row=s.condition_report(np.arange(18)^1,absent,rx,day)
    assert row['macro9'] is None and row['macro_novel3'] is None
    assert row['retrieval']['overall']['per_TX']['8']['count']==0
    assert s.paired_correct(missing,np.arange(18)^1,y,y>=6)['net'] is None
