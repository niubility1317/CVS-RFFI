"""Frozen actual synthetic signal checkpoint and truth-blind C12 orchestration."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_macro12_run')
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_signal_open_eval import fixture as source_fixture
    with source_fixture(root,mp) as (e,ec,counts,events,_,roles):
        m=api();mp.setattr(m,'COUNTS',dict(counts));mp.setattr(m,'_smoke',e._smoke)
        tc=ec['train_config'];tc['code_commit']=m.REFERENCE_CODE;tc['native_config']['code_commit']=m.REFERENCE_CODE
        ec['code_commit']=m.REFERENCE_CODE;cp=Path(tc['output_root'])/'final.pt';saved=torch.load(cp,weights_only=False)
        saved['config']=deepcopy(tc);saved['identity']['code_commit']=m.REFERENCE_CODE;torch.save(saved,cp)
        old=e._train()._read(cp.parent/'complete.json');old['config']=deepcopy(tc);old['identity']=saved['identity'];old['code_commit']=m.REFERENCE_CODE;write(cp.parent/'complete.json',old)
        out=root/'signal_parent';out.mkdir();ec['output_root']=str(out);control=Path(tc['control_root'])
        with np.load(control/'models/learned_known_support.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        (out/'models').mkdir()
        for name in e.METHODS[:2]:np.savez(out/'models'/(name+'.npz'),**state)
        write(out/'models/complete.json',dict(status=e.DENSITY_STATUS,run_id=e.RUN_ID,code_commit=m.REFERENCE_CODE,arm='E_SIGNAL',method_names=list(e.METHODS),
            controls_reused=True,signal_known_statistics_shared=True,source_V_fit=False,novel_fit=False))
        with np.load(control/'thresholds.npz') as z:tau=z['conditional'][[1,1,2,2]];glob=z['global'][[1,1,2,2]]
        np.savez(out/'thresholds.npz',conditional=tau,**{'global':glob})
        previous=e._read(control/'calibration/complete.json');reports={e.METHODS[j]:deepcopy(previous['methods'][e.parent.METHODS[1 if j<2 else 2]]) for j in range(4)}
        write(out/'calibration/complete.json',dict(schema=e.SCHEMA,status=e.CALIBRATION_STATUS,run_id=e.RUN_ID,code_commit=m.REFERENCE_CODE,arm='E_SIGNAL',
            method_names=list(e.METHODS),primary_method=e.PRIMARY_METHOD,known_count=counts['V'],group_count=42,known_only=True,unknown_inputs_used=False,
            source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),global_thresholds=glob.tolist(),methods=reports))
        arrays={}
        for role in ('known','novel'):
            with np.load(control/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            a['signal_features']=a['learned_features'].copy();a['raw_scores']=a['raw_scores'][:,[1,1,2,2]];a['scores']=a['raw_scores']-tau[:,a['closed']].T
            np.savez(out/(role+'_predictions.npz'),**a);arrays[role]=a
        pm=dict(schema=e.SCHEMA,status=e.STATUS,run_id=e.RUN_ID,code_commit=m.REFERENCE_CODE,arm='E_SIGNAL',primary_arm='E_SIGNAL',config=ec,
            method_names=list(e.METHODS),primary_method=e.PRIMARY_METHOD,classification_path='P1',model_identity=saved['identity'],
            parent_F0_identity=saved['identity']['parent_F0_identity'],dura_identity=saved['identity']['dura_identity'],known_count=counts['V'],novel_count=counts['novel'],
            source_L_count=counts['L'],source_OE_count=counts['OE'],source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,
            source_novel_fit_reads=0,identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
            source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_V_model_fit=False,source_V_threshold_fit=True,
            feature_manifest=saved['cache_manifest'],source_feature_manifest=saved['cache_manifest']['source_feature_manifest'],
            IQ_manifests={r:a['manifest'] for r,a in roles.items()})
        write(out/'complete.json',pm)
        c=dict(run_id=m.RUN_ID,code_commit='d'*40,reference_prediction_root=str(out),train_config=deepcopy(tc),output_root=str(root/'macro12'))
        yield m,c,pm,arrays


def test_strict_checkpoint_smoke_models_and_all_thresholds_before_novel_exact_reuse(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,arrays):
        events=[];backend=m._backend;reference=m._reference;rows=m._rows
        def loaded(*a):events.append('checkpoint');return backend(*a)
        def ref(*a):
            assert events==['checkpoint'];events.append('models');return reference(*a)
        def read(root,role):
            output=Path(c['output_root']);assert (output/'models/complete.json').exists()
            if role=='novel':assert all((output/p).exists() for p in ('calibration/complete.json','thresholds.npz','known_predictions.npz'))
            events.append(role);return rows(root,role)
        monkeypatch.setattr(m,'_backend',loaded);monkeypatch.setattr(m,'_reference',ref);monkeypatch.setattr(m,'_rows',read)
        monkeypatch.setattr(m._core(),'fit_known',lambda *a:pytest.fail('No refit'))
        monkeypatch.setattr(m._core(),'fit_background',lambda *a:pytest.fail('No refit'))
        monkeypatch.setattr(m.source,'_load_role',lambda *a:pytest.fail('No real IQ/source role values may open'))
        done=m.run(c,'cpu');out=Path(c['output_root']);assert events==['checkpoint','models','known','novel']
        assert done['synthetic_signal_forward_rows']==2 and done['real_signal_forward_rows']==done['neural_updates']==done['target_reads']==0
        assert done['models_frozen_before_V'] and done['all_thresholds_frozen_before_novel'] and done['controls_reused_exactly']
        with np.load(out/'thresholds.npz') as z:assert set(z.files)=={'conditional'};tau=z['conditional'].copy()
        assert tau.shape==(4,6) and np.all(tau[[1,3]]>=tau[[0,2]])
        assert {p.name for p in (out/'models').glob('*.npz')}=={'frozen_known.npz','signal_known.npz'}
        for role,parent in arrays.items():
            with np.load(out/(role+'_predictions.npz')) as z:a={k:z[k] for k in z.files}
            for key,value in parent.items():
                if key in ('scores','raw_scores'):np.testing.assert_array_equal(a[key][:,[0,2]],value[:,:2])
                else:np.testing.assert_array_equal(a[key],value)
            np.testing.assert_array_equal(a['raw_scores'],parent['raw_scores'][:,[0,0,1,1]])
            assert np.all(~(a['scores'][:,[1,3]]>=0)|(a['scores'][:,[0,2]]>=0))
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_forbidden_config_or_smoke_failure_never_opens_cached_predictions(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,arrays):
        for delta in (dict(truth='private'),dict(run_id='wrong'),dict(output_root=c['reference_prediction_root']),dict(train_config=dict(c['train_config'],code_commit='a'*40))):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        monkeypatch.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
        monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('Cache opened before checkpoint smoke'))
        with pytest.raises(ValueError,match='smoke'):m.run(c,'cpu')
        assert not Path(c['output_root']).exists()


def test_parent_identity_or_cached_margin_mismatch_cannot_complete_or_read_novel(tmp_path,monkeypatch,subtests):
    for defect in ('identity','old_margin','private_novel'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,pm,a):
                parent=Path(c['reference_prediction_root']);novel_reads=[];read=m._rows
                if defect=='identity':pm['model_identity']['native_updates']=1;write(parent/'complete.json',pm)
                elif defect=='old_margin':a['known']['scores'][0,0]+=.1;np.savez(parent/'known_predictions.npz',**a['known'])
                else:a['novel']['y']=np.full(len(a['novel']['ids']),6,np.int64);np.savez(parent/'novel_predictions.npz',**a['novel'])
                def rows(path,role):
                    if role=='novel':novel_reads.append(role)
                    return read(path,role)
                mp.setattr(m,'_rows',rows)
                with pytest.raises(ValueError):m.run(c,'cpu')
                if defect!='private_novel':assert novel_reads==[]
                assert not Path(c['output_root'],'complete.json').exists()
