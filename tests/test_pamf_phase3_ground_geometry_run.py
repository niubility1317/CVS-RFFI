"""Actual Us geometry/known calibration/four-query synthetic producer handoff."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F


def build_prediction_fixture(tmp,mp,produce=True):
    """The new NN boundary is a frozen synthetic cosine reader, not a real CP.

    Historical teacher/energy artifacts were produced by the old fixture's
    different fake reader. Replay differences are intentional, measured, and
    retained; this fixture does not claim actual-checkpoint numerical parity.
    All data export, source fits, calibrations and predictions run for real.
    """
    from test_pamf_phase3_ground_open_run import build_prediction_fixture as inherited
    from experiments.pamf_phase3_open_world import ground_open_score as historical
    from experiments.pamf_phase3_open_world import ground_open_core as risk
    original_load=np.load;original_calibrate=risk.calibrate_known
    old=inherited(tmp,mp)
    historical.score(old['root'],*[old['truths'][k] for k in ('source_known','source_novel','clean','star')],output=old['root'].parent/'score')
    assert len(old['truth_reads'])==4;old['truth_reads'].clear()
    m=importlib.import_module('experiments.pamf_phase3_open_world.ground_geometry_run')
    mp.setattr(m,'REFERENCE_CODE',old['config']['code_commit'])
    for key,value in dict(US_COUNT=90,CAL_COUNT=90,SOURCE_Q_COUNT=135,TARGET_Q_COUNT=18).items():mp.setattr(m,key,value)
    c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=old['config'],output_root=str(tmp/'geometry_prediction'),batch_size=128)
    root=Path(c['output_root']);events=[];truth_reads=[];calibration_calls=[];fit_calls=[];counter=dict(F=0)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();generator=torch.Generator().manual_seed(2026110105)
            self.weight=torch.nn.Parameter(torch.randn(6,160,generator=generator),requires_grad=False);self.scale=30.
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.core=torch.nn.Module();self.core.id_backbone=torch.nn.Module();self.core.id_backbone.cls_head=Head()
            self.padding=torch.nn.Parameter(torch.zeros(232033),requires_grad=False);self.eval()
        def features(self,x):
            counter['F']+=len(x);return x.flatten(1)[:,:160]
        def classify_features(self,raw):
            head=self.core.id_backbone.cls_head
            return head.scale*F.linear(F.normalize(raw,dim=1,eps=1e-4),F.normalize(head.weight,dim=1,eps=1e-4))
    reader=Reader().eval().requires_grad_(False);backend=dict(old['backend'],reader=reader)
    mp.setattr(m,'_backend',lambda *a:backend)
    mp.setattr(m.previous._train(),'load_trained_model',lambda *a:pytest.fail('Geometry must not load or run an SSL student'))
    before={name:value.detach().clone() for name,value in reader.state_dict().items()}
    phase=dict(allow_Us_values=True)
    def source_fit_done():
        p=root/'models/complete.json';assert p.is_file()
        done=json.loads(p.read_text());assert done['source_TX_labels_used'] is False and done['source_count']==90
        for mode in m.MODES:assert (root/'models'/mode/'state.npz').is_file()
    def frozen():
        source_fit_done();p=root/'calibration/complete.json';assert p.is_file() and (root/'thresholds.npz').is_file()
        report=json.loads(p.read_text());assert len(report['methods'])==4 and report['test_inputs_used'] is False
    def role(path,name):
        assert name in ('Us','CAL'),'Only source Us fit and held-out CAL may be opened'
        assert (root/'smoke.json').is_file()
        if name=='Us':assert not (root/'models/complete.json').exists()
        else:source_fit_done()
        events.append(name);return m.data._load(path,name,False)
    def query(path,name):
        frozen();events.append(name);return m.data._load(path,name,True)
    public=m._public;fit=m.core.fit
    def target(c,name):frozen();events.append(name);return public(c,name)
    def fit_metric(x,logits,domains,mode):
        assert x.shape==(90,160) and logits.shape==(90,6) and domains.shape==(90,)
        assert x.dtype==logits.dtype==np.float32 and domains.dtype==np.int64
        fit_calls.append(dict(mode=mode,source_count=len(x),domain_count=len(set(domains.tolist()))))
        return fit(x,logits,domains,mode)
    def calibrate(ids,y,closed,baseline,raw,rx,day):
        source_fit_done();calibration_calls.append(dict(ids=ids.copy(),closed=closed.copy(),baseline=baseline.copy()))
        return original_calibrate(ids,y,closed,baseline,raw,rx,day)
    mp.setattr(m.data,'load_role',role);mp.setattr(m.data,'load_query',query)
    mp.setattr(m,'_public',target);mp.setattr(m.core,'fit',fit_metric);mp.setattr(m.risk,'calibrate_known',calibrate)
    truth_paths={Path(p).resolve() for p in old['truths'].values()}
    def guarded(path,*args,**kwargs):
        p=Path(path) if isinstance(path,(str,Path)) else None
        if p is not None:
            if p.resolve() in truth_paths:
                assert (root/'complete.json').is_file(),'Current four truths must wait for all sixteen prediction points'
                truth_reads.append(str(p.resolve()))
            if any(part in ('L','OE') for part in p.parts) or 'U_s_y' in p.name:
                pytest.fail('Source labels/training L/OE IQ must not enter geometry')
            if not phase['allow_Us_values'] and 'Us' in p.parts and p.name=='iq.npy':
                pytest.fail('CPU scoring must not reload source Us IQ or re-fit geometry')
        return original_load(path,*args,**kwargs)
    mp.setattr(np,'load',guarded)
    result=dict(module=m,config=c,root=root,truths=old['truths'],oldfixture=old,events=events,truth_reads=truth_reads,
        fit_calls=fit_calls,calibration_calls=calibration_calls,backend=backend,reader=reader,reader_before=before,
        counter=counter,phase=phase,counters_before=dict(old['counters']),data_root=old['data_root'],data_done=old['data_done'],
        test_only_synthetic_backend=True,replay_parity_with_actual_checkpoint_not_tested=True)
    if produce:
        result['done']=m.evaluate(c,'cpu');phase['allow_Us_values']=False
    return result


def test_actual_source_fit_before_CAL_before_all_sixteen_and_exact_legacy_controls(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch);m,c,done=f['module'],f['config'],f['done']
    assert done['all_16_points_complete'] and done['source_controls_reused_exactly'] and done['source_subspaces_frozen_before_CAL']
    assert done['all_thresholds_frozen_before_Q'] and f['events']==['Us','CAL',*m.CONDITIONS]
    assert f['truth_reads']==f['oldfixture']['truth_reads']==[]
    assert f['fit_calls']==[dict(mode=mode,source_count=90,domain_count=15) for mode in m.MODES]
    assert len(f['calibration_calls'])==2 and all(set(a['baseline'])=={5} for a in f['calibration_calls'])
    assert done['source_Us_reads']==90 and done['source_U_GT_reads']==done['source_L_reads']==done['source_OE_reads']==0
    assert done['source_subspace_fit_count']==2 and done['optimizer_updates']==done['real_SSL_forward_rows']==0
    assert done['real_F_forward_rows']==90+90+2*135+2*18==486 and f['counter']['F']==488
    assert done['G_forward_rows']==36 and f['oldfixture']['counters']['G']-f['counters_before']['G']==38
    assert not done['classification_unchanged_from_cached_F0']
    assert any(v['closed_mismatch_count']>0 and v['maximum_absolute_logit_difference']>0 for v in done['replay_diagnostics'].values())
    with np.load(f['root']/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
    with np.load(f['oldfixture']['root']/'thresholds.npz',allow_pickle=False) as z:old_tau=z['conditional'].copy()
    np.testing.assert_array_equal(tau[:2],old_tau[[0,2]])
    states=m._states(f['root'])
    for condition in m.CONDITIONS:
        with subtests.test(condition=condition):
            a=m._rows(f['root'],condition);legacy=m.previous._rows(f['oldfixture']['root'],condition)
            for key in ('default_logits','closed','raw_scores','scores'):
                np.testing.assert_array_equal(a[key][:,:2],legacy[key][:,[0,2]])
            np.testing.assert_array_equal(a['baseline_logits'],legacy['default_logits'][:,0])
            np.testing.assert_array_equal(a['fresh_F0_logits'],a['default_logits'][:,2])
            for j,mode in enumerate(m.MODES,2):
                raw=m.core.raw_scores(states[mode],a['classifier_coordinates'],a['fresh_F0_logits'])
                np.testing.assert_array_equal(a['raw_scores'][:,j],raw[:,1])
                np.testing.assert_array_equal(a['residual_norms'][:,j-2],-raw[:,0])
                np.testing.assert_array_equal(a['scores'][:,j],raw[:,1]-tau[j,a['closed'][:,j]])
            replay=m._replay(a['fresh_F0_logits'],a['baseline_logits'])
            assert replay==done['replay_diagnostics'][condition]
            assert 'y' not in a and a['classifier_coordinates'].dtype==np.float32
    for first,second in zip(m.CONDITIONS[::2],m.CONDITIONS[1::2]):
        np.testing.assert_array_equal(m._rows(f['root'],first)['ids'],m._rows(f['root'],second)['ids'])
    assert all(torch.equal(v,f['reader'].state_dict()[k]) for k,v in f['reader_before'].items())
    with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_private_options_bad_shapes_and_partial_last_query_leave_truth_closed(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch,produce=False);m,c=f['module'],f['config']
    for change in (dict(labels='private'),dict(rank=14),dict(batch_size=1),dict(output_root=c['reference_config']['data_root']),
                   dict(reference_config=dict(c['reference_config'],code_commit='0'*40))):
        with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    with pytest.raises(ValueError):m._coordinates(f['reader'],np.zeros((3,256,2),np.float32),'cpu')
    public=m._public
    def fail(c,name):
        if name=='cross_star_DURA':raise ValueError('synthetic final query unavailable')
        return public(c,name)
    monkeypatch.setattr(m,'_public',fail)
    with pytest.raises(ValueError,match='final query'):m.evaluate(c,'cpu')
    assert not (f['root']/'complete.json').exists() and f['truth_reads']==[]
    assert (f['root']/'queries/cross_clean_DURA/predictions.npz').is_file()
