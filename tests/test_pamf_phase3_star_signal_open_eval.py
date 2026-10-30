"""Signal IQ evaluation contracts, actual synthetic checkpoint and freeze ordering."""
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch
from contextlib import contextmanager
from copy import deepcopy
import json


def test_fixed_four_readout_contract_and_no_truth_config():
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_eval')
    assert m.METHODS==('frozen_known_support','signal_known_support','frozen_density_ratio','signal_density_ratio')
    assert m.PRIMARY_METHOD=='signal_known_support' and m.ARMS==('E_SIGNAL',)
    assert m.FIELDS=={'run_id','code_commit','arm','train_config','IQ_root','feature_root','reference_prediction_root','output_root','batch_size'}
    with pytest.raises(ValueError):m.validate_config(dict(novel_truth='forbidden'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_signal_open_train import fixture as train_fixture,frozen_checkpoint
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_eval')
    with train_fixture(root,mp) as (t,tc,fm):
        frozen_checkpoint(tc,fm);L=m.parent._train().load_role(tc['feature_root'],'L');OE=m.parent._train().load_role(tc['feature_root'],'OE')
        counts=dict(L=len(L['ids']),OE=len(OE['ids']),V=180,novel=90);mp.setattr(m,'COUNTS',counts)
        core=m._core();mp.setattr(core,'KNOWN_CELL_SIZE',counts['L']//90);mp.setattr(core,'OE_COUNT',counts['OE']);mp.setattr(core,'CAPACITY',4);mp.setattr(core,'LLOYD_ITERATIONS',2)
        known,_=core.fit_known(L['u'],L['metadata']['y'],L['metadata']['rx_i'],L['metadata']['day_i']);ratio,_=core.fit_background(known,OE['u'],OE['metadata']['rx_i'],OE['metadata']['day_i'])
        parent=Path(tc['control_root']);(parent/'models').mkdir();np.savez(parent/'models/learned_known_support.npz',**known);np.savez(parent/'models/learned_density_ratio.npz',**ratio)
        oldtc=dict(run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,arm='S_SEP_TOPO',cache_root=tc['feature_root'],output_root=str(root/'control/train'),steps=2220,lr=.0001,L_seed=2026103017,OE_seed=2026103018)
        pi=m.parent._train()._identity(oldtc,fm)
        marker=dict(schema=m.parent.SCHEMA,status=m.parent.STATUS,run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,arm='S_SEP_TOPO',method_names=list(m.parent.METHODS),primary_method=m.parent.PRIMARY_METHOD,
            config=dict(train_config=oldtc,run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,arm='S_SEP_TOPO'),model_identity=pi,parent_F0_identity=fm['parent_F0_identity'],dura_identity=fm['dura_identity'],feature_manifest=fm,source_feature_manifest=fm['source_feature_manifest'],
            classification_path='P1',source_only=True,private_truth_reads=0,target_reads=0,identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,learned_known_statistics_shared=True)
        write(parent/'complete.json',marker)
        scenes=m.parent.reference.SCENES;rxs=m.parent.reference.RECEIVERS;grid=[(c,r,s) for c in range(6) for r in rxs for s in scenes]
        y=np.asarray([v[0] for v in grid],np.int64);rx=np.asarray([v[1] for v in grid],np.int64);day=np.arange(180,dtype=np.int64)%3+1
        ix=np.asarray([np.flatnonzero(L['metadata']['y']==c)[i%np.sum(L['metadata']['y']==c)] for i,c in enumerate(y)])
        v=dict(ids=np.asarray(['V'+str(i) for i in range(180)]),y=y,rx_i=rx,day_i=day,scenario=np.asarray([v[2] for v in grid]),unit_features=L['u'][ix].copy(),contribution_features=L['c'][ix].copy(),learned_features=L['u'][ix].copy(),
            default_logits=np.eye(6,dtype=np.float32)[y]*3,closed=y.copy(),baseline_closed=y.copy())
        grid=[(c,r,s) for c in range(3) for r in rxs for s in scenes];ix=np.arange(90)%len(OE['ids'])
        n=dict(ids=np.asarray(['NV'+str(i) for i in range(90)]),rx_i=np.asarray([v[1] for v in grid],np.int64),day_i=np.arange(90,dtype=np.int64)%3+1,unit_features=OE['u'][ix].copy(),contribution_features=OE['c'][ix].copy(),learned_features=OE['u'][ix].copy(),
            default_logits=np.tile(np.asarray([3,0,0,0,0,0],np.float32),(90,1)),closed=np.zeros(90,np.int64))
        thresholds=[np.full(6,-123.,np.float64)];globals=[-123.];methods={};controls={}
        for j,state in enumerate((known,ratio),1):
            raw=core.score(state,v['learned_features'],'known_only' if j==1 else 'density_ratio');s,r=m.risk.calibrate_known(v['ids'],y,y,y,raw,rx,v['scenario'])
            thresholds.append(s['conditional_thresholds']);globals.append(s['global_threshold']);methods[m.parent.METHODS[j]]=r['methods']['predicted_class'];controls[m.parent.METHODS[j]]=r['methods']['global_control']
        tau=np.asarray(thresholds,np.float64);np.savez(parent/'thresholds.npz',conditional=tau,**{'global':np.asarray(globals,np.float64)})
        write(parent/'calibration/complete.json',dict(run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,status=m.parent.CALIBRATION_STATUS,known_only=True,unknown_inputs_used=False,methods=methods,global_controls=controls))
        for role,a in (('known',v),('novel',n)):
            a['raw_scores']=np.column_stack((np.full(len(a['ids']),-222.),core.score(known,a['learned_features'],'known_only'),core.score(ratio,a['learned_features'],'density_ratio')))
            a['scores']=a['raw_scores']-tau[:,a['closed']].T;np.savez(parent/(role+'_predictions.npz'),**a)
        rng=np.random.default_rng(19);roles={}
        for role,a in (('L',L),('OE',OE),('V',v),('novel',n)):
            meta=a['metadata'] if role in ('L','OE') else {k:a[k] for k in m.IQ.METADATA_FIELDS[role]-{'domain'}}|{'domain':np.searchsorted(rxs,a['rx_i'])*3+a['day_i']-1}
            manifest=deepcopy(fm['source_manifests']['L']);manifest.update(role=role,count=counts[role],iq_shape=[counts[role],2,256],files=sorted(m.IQ.role_files(role)),metadata_fields=sorted(m.IQ.METADATA_FIELDS[role]),teacher_input_path='P1' if role=='L' else None,baseline_input_paths=['P0','P1'] if role=='V' else None)
            if role in ('L','OE'):manifest=fm['source_manifests'][role]
            roles[role]=dict(ids=a['ids'],metadata=meta,manifest=manifest,restored=rng.normal(size=(counts[role],2,256)).astype(np.float32))
            if role=='V':roles[role]['baseline_closed']=np.column_stack((y,y))
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],arm='E_SIGNAL',train_config=tc,IQ_root=tc['IQ_root'],feature_root=tc['feature_root'],reference_prediction_root=tc['control_root'],output_root=str(root/'evaluation'),batch_size=128)
        events=[];project=m._project;smoke=m._smoke;saved=m._rows
        def probe(model,device):events.append('smoke');return smoke(model,device)
        def load(path,role):
            assert events[0]=='smoke'
            if role=='V':assert (Path(c['output_root'])/'models/complete.json').exists()
            if role=='novel':assert all((Path(c['output_root'])/p).is_file() for p in ('thresholds.npz','calibration/complete.json','known_predictions.npz'))
            events.append(role);return roles[role]
        def saved_rows(path,role):
            if role=='novel':assert (Path(c['output_root'])/'calibration/complete.json').exists()
            events.append('parent_'+role);return saved(path,role)
        mp.setattr(m,'_smoke',probe);mp.setattr(m,'_load_role',load);mp.setattr(m,'_close_role',lambda a:None);mp.setattr(m,'_rows',saved_rows)
        yield m,c,counts,events,n,roles


def test_actual_headless_checkpoint_IQ_to_four_readouts_and_Torch_Numpy_safe_bridge(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,roles):
        original=torch.as_tensor
        def reject_numpy(value,*a,**k):
            if isinstance(value,np.ndarray):raise RuntimeError('No ndarray ABI bridge')
            return original(value,*a,**k)
        monkeypatch.setattr(torch,'as_tensor',reject_numpy);monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy ABI bridge forbidden'))
        done=m.evaluate(c,'cpu');out=Path(c['output_root'])
        assert events==['smoke','L','OE','parent_known','V','parent_novel','novel']
        assert done['real_signal_forward_rows']==sum(counts.values()) and done['synthetic_signal_forward_rows']==2
        assert done['real_F_forward_rows']==done['G_forward_rows']==done['received_P0_values_read']==0
        assert done['model_parameters_buffers_unchanged'] and done['signal_known_statistics_shared']
        with np.load(out/'thresholds.npz') as z:assert z['conditional'].shape==(4,6)
        for role in ('known','novel'):
            with np.load(out/(role+'_predictions.npz')) as z:a={k:z[k] for k in z.files}
            with np.load(Path(c['reference_prediction_root'])/(role+'_predictions.npz')) as z:old={k:z[k] for k in z.files}
            assert a['signal_features'].shape==(len(a['ids']),160)
            for key,v in old.items():np.testing.assert_array_equal(a[key][:,[0,2]] if key in ('scores','raw_scores') else a[key],v[:,[1,2]] if key in ('scores','raw_scores') else v)
        with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_invalid_source_or_checkpoint_fails_before_novel_and_never_completes(tmp_path,monkeypatch,subtests):
    for defect in ('truth','overlap','identity','smoke','L_ID','OE_y','V_baseline','mutation'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,counts,events,n,roles):
                output=Path(c['output_root'])/'complete.json'
                if defect=='truth':c['truth']='forbidden'
                elif defect=='overlap':c['output_root']=c['IQ_root']
                elif defect=='identity':
                    original=m._backend
                    def bad(*a):
                        model,i=original(*a);return model,dict(i,native_updates=2220)
                    mp.setattr(m,'_backend',bad)
                elif defect=='smoke':mp.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
                elif defect=='L_ID':roles['L']['ids']=roles['L']['ids'][::-1]
                elif defect=='OE_y':roles['OE']['metadata']=dict(roles['OE']['metadata'],y=np.zeros(counts['OE'],np.int64))
                elif defect=='V_baseline':roles['V']['baseline_closed'][0,1]=5
                else:
                    original=m._project
                    def changed(model,*a):
                        result=original(model,*a)
                        if len(result)>2:
                            with torch.no_grad():next(model.parameters()).add_(.01)
                        return result
                    mp.setattr(m,'_project',changed)
                with pytest.raises((ValueError,KeyError)):m.evaluate(c,'cpu')
                assert not output.exists() and 'novel' not in events

