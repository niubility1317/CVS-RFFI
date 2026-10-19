import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.pamf_restoration.feature_diagnostic import capped_pair_margin
from experiments.pamf_restoration import runner as old
from experiments.pamf_formal_restore import reference as ref
from experiments.pamf_formal_restore import runner as formal


def test_capped_margin_matches_full_formula_when_pool_is_complete_and_label_symmetric():
    rng=np.random.default_rng(1);a=rng.normal(size=(7,5)).astype(np.float32);b=rng.normal(size=(7,5)).astype(np.float32)
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    labels=np.array([0,0,1,1,2,2,2]);scores=a@b.T;paired=scores.diagonal().copy()
    scores[labels[:,None]==labels[None,:]]=-np.inf;expected=paired-scores.max(1)
    result=capped_pair_margin(a,b,labels,7)
    assert result['mean']==pytest.approx(float(expected.mean()),abs=1e-7)
    assert result['own_pair_higher_percent']==pytest.approx(float((expected>0).mean()*100))
    relabeled=capped_pair_margin(a,b,np.array([12,12,-1,-1,89,89,89]),7)
    assert relabeled==result and result['reference_pool_count']==7 and not result['inference_or_selection_use']


def test_reference_cap_and_no_competitor_are_explicit_without_false_success():
    a=np.eye(6,dtype=np.float32);labels=np.zeros(6,dtype=np.int64)
    result=capped_pair_margin(a,a,labels,2)
    assert result['reference_pool_count']==2 and result['valid_comparison_count']==0
    assert result['mean'] is None and result['own_pair_higher_percent'] is None
    with pytest.raises(ValueError):capped_pair_margin(a,a,labels,0)
    with pytest.raises(ValueError):old.score_row(Path('missing'),Path('missing'),target_truth=Path('forbidden'),feature_reference_cap=-1)


def test_formal_reference_keeps_all_original_bytes_and_never_reads_index_labels(tmp_path,monkeypatch):
    source=tmp_path/'asset';source.mkdir();count=4
    ids=np.array(['p0','p1','p2','p3']);np.savez(source/'index.npz',ids=ids,labels=np.arange(count))
    paths={scene:source/(scene+'.npy') for scene in ('clean',*ref.SCENES)}
    for index,path in enumerate(paths.values()):np.save(path,np.full((count,2,256),index,dtype=np.float32),allow_pickle=False)
    manifest=dict(status='VALIDATED_ONCE',count=count,scenes=list(ref.SCENES),classes=ref.CLASSES,
        channel='residual/post_sync/noeq',truth_read=False,rows_share_observations=True,source_target_disjoint=True,
        clean_ref=str(paths['clean']))
    (source/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    real_load=np.load;keys=[]
    class IDsOnly:
        def __init__(self,archive):self.archive=archive
        def __enter__(self):return self
        def __exit__(self,*args):self.archive.close()
        def __getitem__(self,key):
            assert key=='ids';keys.append(key);return self.archive[key]
    def guarded(path,*args,**kwargs):
        value=real_load(path,*args,**kwargs)
        return IDsOnly(value) if Path(path).name=='index.npz' else value
    links=[]
    def link(destination,target,*args,**kwargs):
        links.append((destination.name,str(target)));shutil.copyfile(target,destination)
    monkeypatch.setattr(np,'load',guarded);monkeypatch.setattr(Path,'symlink_to',link)
    output=tmp_path/'reference';ref.prepare_reference(source,output,expected_count=count)
    assert keys==['ids'] and len(links)==7
    assert np.array_equal(real_load(output/'physical_ids.npy'),ids)
    for scene,path in paths.items():assert (output/(scene+'.npy')).read_bytes()==path.read_bytes()
    assert set(real_load(output/'rx.npy').tolist())=={'unavailable'}
    with pytest.raises(FileExistsError):ref.prepare_reference(source,output,expected_count=count)
    manifest['channel']='full/post_sync';(source/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):ref.prepare_reference(source,tmp_path/'wrong',expected_count=count)


def test_score_cli_forwards_explicit_pool_without_opening_truth(tmp_path,monkeypatch):
    captured=[]
    monkeypatch.setattr(old,'score_row',lambda *args:captured.append(args))
    monkeypatch.setattr(sys,'argv',['runner','score','--reference',str(tmp_path),'--row-output',str(tmp_path/'row'),
                                  '--target-truth',str(tmp_path/'truth'),'--feature-reference-cap','512'])
    old.main()
    assert len(captured)==1 and captured[0][-1]==512


def test_formal_candidate_is_source_selected_and_qualified_before_target(tmp_path):
    parent=tmp_path/formal.PARENT;parent.mkdir()
    (parent/'launch.json').write_text(json.dumps(dict(commit=formal.PARENT_COMMIT)),encoding='utf-8')
    decision=dict(status='SOURCE_SELECTION_FROZEN',waveform_selected=formal.CANDIDATE,
        waveform_selection_TX_F_access=False,training_TX_labels_read=0,no_target_feedback=True,
        identity_diagnostic=[dict(candidate=formal.CANDIDATE,qualified=True)])
    p=parent/'source_decision.json';p.write_text(json.dumps(decision),encoding='utf-8')
    assert formal.source_parent(parent)['source_waveform_selected']
    decision['identity_diagnostic'][0]['qualified']=False;p.write_text(json.dumps(decision),encoding='utf-8')
    with pytest.raises(ValueError):formal.source_parent(parent)
