import importlib
import json
from types import SimpleNamespace
import numpy as np
import pytest

def module():return importlib.import_module('tools.analyze_pamf_source_cross_record')

def panel():
    groups=[(y,rx,day,k) for y in range(6) for rx in range(3) for day in range(2) for k in range(2)]
    labels=np.array([g[0] for g in groups]);rx=np.array([str(g[1]) for g in groups]);day=np.array([str(g[2]) for g in groups])
    ids=np.array(['id'+str(i).zfill(3) for i in range(len(groups))]);clean=np.eye(6)[labels]
    return clean,ids,labels,rx,day

def test_cross_RX_and_day_perfect_TX_signal_and_wrong_TX():
    m=module();clean,ids,y,rx,day=panel();ref=m.reference_pool(clean,ids,y,rx,day)
    good=m.contrasts(clean,ref,y,rx,day);wrong=m.contrasts(np.roll(clean,1,axis=1),ref,y,rx,day)
    assert np.allclose(good['margin'],1.) and np.allclose(wrong['margin'],-.2)
    assert np.all(good['valid_cells']==2) and np.all(good['valid'])
    for i in range(len(y)):
        used=ref['indices'][m.eligible_reference_mask(ref,rx[i],day[i])]
        assert i not in used and np.all(rx[used]!=rx[i]) and np.all(day[used]!=day[i])

def test_pool_fixed_opaque_ID_cap_and_input_order_independence():
    m=module();clean,ids,y,rx,day=panel();a=m.reference_pool(clean,ids,y,rx,day);order=np.random.default_rng(25).permutation(len(y))
    b=m.reference_pool(clean[order],ids[order],y[order],rx[order],day[order])
    assert list(ids[a['indices']])==list(ids[order][b['indices']]) and len(a['indices'])==72
    aa=m.contrasts(clean,a,y,rx,day);bb=m.contrasts(clean[order],b,y[order],rx[order],day[order])
    assert np.allclose(aa['margin'][order],bb['margin'])

def test_zero_query_is_undefined_and_common_valid_delta_is_explicit():
    m=module();clean,ids,y,rx,day=panel();ref=m.reference_pool(clean,ids,y,rx,day)
    bad=clean.copy();bad[0]=0;before=m.contrasts(bad,ref,y,rx,day);after=m.contrasts(clean,ref,y,rx,day)
    assert not before['valid'][0] and before['zero_query'][0]
    stats=m.summarize(before,after,after,np.ones(len(y),bool))
    assert stats['baseline']['valid_count']==71 and stats['baseline']['zero_query_count']==1
    assert stats['paired_delta']['count']==71 and stats['paired_delta']['mean_margin_change']==0.

def test_missing_TX_cell_not_imputed_or_silently_treated_as_valid():
    m=module();clean,ids,y,rx,day=panel();keep=~((y==0)&(rx=='1')&(day=='1'))
    ref=m.reference_pool(clean[keep],ids[keep],y[keep],rx[keep],day[keep]);values=m.contrasts(clean,ref,y,rx,day)
    chosen=(rx=='0')&(day=='0');assert np.all(values['valid_cells'][chosen]==1) and np.all(values['missing_cells'][chosen]==1)
    assert ref['empty_TX_cells']==1

@pytest.mark.parametrize('bad',['duplicate','nonfinite','labels','cap'])
def test_invalid_features_ID_truth_or_cap_rejected(bad):
    m=module();clean,ids,y,rx,day=panel();cap=2
    if bad=='duplicate':ids[0]=ids[1]
    elif bad=='nonfinite':clean[0,0]=np.nan
    elif bad=='labels':y[0]=6
    else:cap=3
    with pytest.raises(ValueError):m.reference_pool(clean,ids,y,rx,day,cap=cap)

@pytest.mark.parametrize('mutation',['target','F','unscored','training','head'])
def test_bad_role_or_provenance_rejected_before_source_truth(tmp_path,monkeypatch,mutation):
    m=module();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');name,run,row,commit=m.ROWS[0]
    root=tmp_path/run;folder=root/row/'native_map_prior';folder.mkdir(parents=True)
    (root/'launch.json').write_text(json.dumps(dict(commit=commit)))
    summary=dict(status='ANALYZED',kind='native_map_prior',role='source',count=1800,F_run=m.F_RUN,F_seed=2026092701,original_default_head=True,
        all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,head_fit_updates=0,target_access=False)
    summary['views']={view:{} for view in m.VIEWS};m.require_summary(summary,'native_map_prior')
    if mutation=='target':summary['role']='target'
    elif mutation=='F':summary['F_seed']=2026092702
    elif mutation=='unscored':summary['status']='PREDICTIONS_COMPLETE'
    elif mutation=='training':summary['training_F_access']=True
    else:summary['head_fit_updates']=1
    (folder/'summary.json').write_text(json.dumps(summary))
    def forbidden(*a,**k):raise AssertionError('TX labels must stay closed')
    with pytest.raises(ValueError):m.analyze(SimpleNamespace(runs_root=tmp_path),label_loader=forbidden)

@pytest.mark.parametrize('mutation',['missing_view','ids','role','nonfinite'])
def test_all_feature_views_and_IDs_validate_before_truth(tmp_path,mutation):
    m=module();folder=tmp_path/'native_map_prior';pred=folder/'prediction';pred.mkdir(parents=True)
    ids=np.arange(1800);np.save(pred/'physical_ids.npy',ids)
    value=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='source',count=1800,views=list(m.VIEWS),readouts=['restored'],
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=m.F_RUN,F_seed=2026092701)
    (pred/'complete.json').write_text(json.dumps(value))
    for view in m.VIEWS:np.save(pred/(view+'.restored.features.npy'),np.ones((1800,160),np.float32))
    assert tuple(m.load_features(folder,ids,'native_map_prior','restored'))==m.VIEWS
    if mutation=='missing_view':value['views']=list(m.VIEWS[:-1])
    elif mutation=='ids':np.save(pred/'physical_ids.npy',ids[::-1])
    elif mutation=='role':value['role']='target'
    else:
        a=np.ones((1800,160),np.float32);a[0,0]=np.nan;np.save(pred/(m.VIEWS[-1]+'.restored.features.npy'),a)
    (pred/'complete.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):m.load_features(folder,ids,'native_map_prior','restored')
