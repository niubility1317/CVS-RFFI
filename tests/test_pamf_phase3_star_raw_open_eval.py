"""P0-only signal evaluation with immutable original-P1 identity evidence."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch
import json


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    """Actual synthetic P0 checkpoint and independently frozen P1 parents."""
    from test_pamf_phase3_star_raw_open_train import fixture as train_fixture,frozen_checkpoint
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_eval')
    with train_fixture(root,mp) as (t,tc,pm,values,_,_):
        source=m.previous;macro=m.macro;old=tc['control_train_config']
        L=source.parent._train().load_role(old['feature_root'],'L');OE=source.parent._train().load_role(old['feature_root'],'OE')
        counts=dict(L=len(L['ids']),OE=len(OE['ids']),V=180,novel=90)
        for module in (m,source,macro):mp.setattr(module,'COUNTS',counts)
        core=m._core();mp.setattr(core,'KNOWN_CELL_SIZE',counts['L']//90);mp.setattr(core,'OE_COUNT',counts['OE']);mp.setattr(core,'CAPACITY',4);mp.setattr(core,'LLOYD_ITERATIONS',2)
        known,_=core.fit_known(L['u'],L['metadata']['y'],L['metadata']['rx_i'],L['metadata']['day_i']);ratio,_=core.fit_background(known,OE['u'],OE['metadata']['rx_i'],OE['metadata']['day_i'])
        rxs=source.parent.reference.RECEIVERS;scenes=source.parent.reference.SCENES
        grid=[(cl,r,s) for cl in range(6) for r in rxs for s in scenes];y=np.asarray([v[0] for v in grid],np.int64)
        ix=np.asarray([np.flatnonzero(L['metadata']['y']==cl)[i%np.sum(L['metadata']['y']==cl)] for i,cl in enumerate(y)])
        V=dict(ids=np.asarray(['V'+str(i) for i in range(180)]),y=y,rx_i=np.asarray([v[1] for v in grid],np.int64),day_i=np.arange(180,dtype=np.int64)%3+1,
            scenario=np.asarray([v[2] for v in grid]),unit_features=L['u'][ix].copy(),contribution_features=L['c'][ix].copy(),learned_features=L['u'][ix].copy(),signal_features=L['u'][ix].copy(),
            default_logits=np.eye(6,dtype=np.float32)[y]*3,closed=y.copy(),baseline_closed=y.copy())
        ng=[(cl,r,s) for cl in range(3) for r in rxs for s in scenes];ix=np.arange(90)%len(OE['ids'])
        N=dict(ids=np.asarray(['NV'+str(i) for i in range(90)]),rx_i=np.asarray([v[1] for v in ng],np.int64),day_i=np.arange(90,dtype=np.int64)%3+1,
            unit_features=OE['u'][ix].copy(),contribution_features=OE['c'][ix].copy(),learned_features=OE['u'][ix].copy(),signal_features=OE['u'][ix].copy(),
            default_logits=np.tile(np.asarray([3,0,0,0,0,0],np.float32),(90,1)),closed=np.zeros(90,np.int64))
        rng=np.random.default_rng(31);manifests=deepcopy(pm['source_IQ_manifests'])
        for role,a in (('V',V),('novel',N)):
            meta={k:a[k] for k in t.IQ.METADATA_FIELDS[role]-{'domain'}}|dict(domain=np.searchsorted(rxs,a['rx_i'])*3+a['day_i']-1)
            manifest=deepcopy(manifests['L']);manifest.update(role=role,count=counts[role],iq_shape=[counts[role],2,256],files=sorted(t.IQ.role_files(role)),metadata_fields=sorted(t.IQ.METADATA_FIELDS[role]),teacher_input_path=None,baseline_input_paths=['P0','P1'] if role=='V' else None)
            folder=Path(tc['IQ_root'])/role;folder.mkdir();write(folder/'manifest.json',manifest);np.save(folder/'ids.npy',a['ids']);np.savez(folder/'metadata.npz',**meta)
            values[role]=rng.normal(size=(counts[role],2,256)).astype(np.float32);np.save(folder/'received.npy',values[role]);manifests[role]=manifest
        signalroot=root/'signal_parent';(signalroot/'models').mkdir(parents=True)
        for name,state in zip(source.METHODS,(known,known,ratio,ratio)):np.savez(signalroot/'models'/(name+'.npz'),**state)
        sigc=dict(run_id=source.RUN_ID,code_commit=m.SIGNAL_REFERENCE_CODE,arm='E_SIGNAL',train_config=old,IQ_root=old['IQ_root'],feature_root=old['feature_root'],reference_prediction_root=old['control_root'],output_root=str(signalroot),batch_size=128)
        si=pm['parent_model_identity'];sm=dict(schema=source.SCHEMA,status=source.STATUS,run_id=source.RUN_ID,code_commit=m.SIGNAL_REFERENCE_CODE,arm='E_SIGNAL',primary_arm='E_SIGNAL',
            method_names=list(source.METHODS),primary_method=source.PRIMARY_METHOD,config=sigc,model_identity=si,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
            IQ_manifests=manifests,classification_path='P1',known_count=180,novel_count=90,source_L_count=counts['L'],source_OE_count=counts['OE'],
            source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,identity_predictions_unchanged=True,models_frozen_before_V=True,
            all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_V_model_fit=False,source_V_threshold_fit=True)
        methods={};tau=[];glob=[]
        for j,name in enumerate(source.METHODS):
            raw=core.score(known if j<2 else ratio,V['signal_features'],'known_only' if j<2 else 'density_ratio')
            state,report=source.risk.calibrate_known(V['ids'],y,y,y,raw,V['rx_i'],V['scenario']);tau.append(state['conditional_thresholds']);glob.append(state['global_threshold']);methods[name]=report['methods']['predicted_class']
        tau=np.asarray(tau,np.float64)
        write(signalroot/'complete.json',sm);write(signalroot/'models/complete.json',dict(status=source.DENSITY_STATUS,run_id=source.RUN_ID,code_commit=m.SIGNAL_REFERENCE_CODE,arm='E_SIGNAL',method_names=list(source.METHODS),controls_reused=True,signal_known_statistics_shared=True,source_V_fit=False,novel_fit=False))
        np.savez(signalroot/'thresholds.npz',conditional=tau,**{'global':np.asarray(glob,np.float64)})
        write(signalroot/'calibration/complete.json',dict(schema=source.SCHEMA,status=source.CALIBRATION_STATUS,run_id=source.RUN_ID,code_commit=m.SIGNAL_REFERENCE_CODE,arm='E_SIGNAL',method_names=list(source.METHODS),primary_method=source.PRIMARY_METHOD,
            known_count=180,group_count=42,known_only=True,unknown_inputs_used=False,source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),methods=methods))
        raw=core.score(known,V['signal_features'],'known_only');state,report=m.risk.calibrate_known(V['ids'],y,y,y,raw,V['rx_i'],V['scenario']);mtau=np.tile(state['conditional_thresholds'],(4,1))
        parentroot=Path(tc['control_root']);(parentroot/'models').mkdir();np.savez(parentroot/'models/signal_known.npz',**known);np.savez(parentroot/'thresholds.npz',conditional=mtau)
        pm.update(method_names=list(macro.METHODS),primary_method=macro.PRIMARY_METHOD,classification_path='P1',model_identity=si,reference_prediction_identity=sm,source_IQ_manifests=manifests,identity_predictions_unchanged=True,source_states_unchanged=True)
        write(parentroot/'complete.json',pm);write(parentroot/'calibration/complete.json',dict(schema=macro.SCHEMA,status=macro.CALIBRATION_STATUS,run_id=m.REFERENCE_RUN,code_commit=m.REFERENCE_CODE,known_only=True,unknown_inputs_used=False,methods=dict(signal_known12=report)))
        for role,a in (('known',V),('novel',N)):
            kr=core.score(known,a['signal_features'],'known_only');rr=core.score(ratio,a['signal_features'],'density_ratio')
            oldrows=dict(a,raw_scores=np.column_stack((kr,kr,rr,rr)));oldrows['scores']=oldrows['raw_scores']-tau[:,a['closed']].T;np.savez(signalroot/(role+'_predictions.npz'),**oldrows)
            oldrows=dict(a,raw_scores=np.column_stack((kr,kr,kr,kr)));oldrows['scores']=oldrows['raw_scores']-mtau[:,a['closed']].T;np.savez(parentroot/(role+'_predictions.npz'),**oldrows)
        frozen_checkpoint(tc,pm)
        c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],arm='E_RAW',train_config=tc,reference_prediction_root=str(parentroot),signal_reference_root=str(signalroot),output_root=str(root/'evaluation'),batch_size=128)
        events=[];smoke=m._smoke;load=m._load_role;rows=m._reference_rows
        def probe(model,device):events.append('smoke');return smoke(model,device)
        def role(path,name):
            assert events[0]=='smoke'
            if name=='V':assert (Path(c['output_root'])/'models/complete.json').is_file()
            if name=='novel':assert all((Path(c['output_root'])/p).is_file() for p in ('thresholds.npz','calibration/complete.json','known_predictions.npz'))
            events.append(name);return load(path,name)
        def parents(config,role):
            if role=='novel':assert (Path(c['output_root'])/'calibration/complete.json').is_file()
            events.append('parent_'+role);return rows(config,role)
        mp.setattr(m,'_smoke',probe);mp.setattr(m,'_load_role',role);mp.setattr(m,'_reference_rows',parents)
        yield m,c,counts,events,N,values


def test_actual_P0_checkpoint_reads_only_received_and_preserves_both_P1_parents(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,values):
        load=np.load;as_tensor=torch.as_tensor
        def safe(path,*a,**k):
            assert Path(path).name not in ('restored.npy','teacher_logits.npy','baseline_closed.npy')
            return load(path,*a,**k)
        def bridge(value,*a,**k):
            if isinstance(value,np.ndarray):raise RuntimeError('No NumPy ABI')
            return as_tensor(value,*a,**k)
        monkeypatch.setattr(np,'load',safe);monkeypatch.setattr(torch,'as_tensor',bridge);monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No ndarray Torch bridge'))
        result=m.evaluate(c,'cpu');assert events==['smoke','L','OE','parent_known','V','parent_novel','novel']
        assert result['real_signal_forward_rows']==sum(counts.values()) and result['synthetic_signal_forward_rows']==2
        assert result['real_F_forward_rows']==result['G_forward_rows']==result['restored_P1_values_read']==0
        assert result['teacher_input_path']==result['rejection_input_path']=='P0' and result['identity_baseline_path']==result['classification_path']=='P1'
        for role in ('known','novel'):
            with load(Path(c['output_root'])/(role+'_predictions.npz')) as z:a={k:z[k] for k in z.files}
            with load(Path(c['reference_prediction_root'])/(role+'_predictions.npz')) as z:old={k:z[k] for k in z.files}
            for key,v in old.items():
                if key not in ('scores','raw_scores'):np.testing.assert_array_equal(a[key],v)
            np.testing.assert_array_equal(a['scores'][:,0],old['scores'][:,3]);assert a['raw_signal_features'].shape==(len(a['ids']),160)
        with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_invalid_source_smoke_and_parent_fail_without_opening_source_or_completing(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,values):
        for defect in ('truth','overlap','identity','smoke'):
            with subtests.test(defect=defect),monkeypatch.context() as mp:
                cc=deepcopy(c);cc['output_root']=str(tmp_path/('bad_'+defect));mp.setattr(m,'_load_role',lambda *a:pytest.fail('Source before legal checkpoint smoke'))
                if defect=='truth':cc['target_truth']='forbidden'
                elif defect=='overlap':cc['output_root']=cc['train_config']['IQ_root']
                elif defect=='identity':
                    backend=m._backend
                    def bad(*a):
                        model,i=backend(*a);return model,dict(i,teacher_input_path='P1')
                    mp.setattr(m,'_backend',bad)
                else:mp.setattr(m,'_smoke',lambda *a:dict(status='FAIL'))
                with pytest.raises(ValueError):m.evaluate(cc,'cpu')
                assert not Path(cc['output_root'],'complete.json').is_file()


def test_fixed_P0_primary_with_two_explicit_P1_reference_roots():
    m=importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_eval')
    assert m.METHODS==('P1_known12','P1_ratio12','P0_known12','P0_ratio12') and m.PRIMARY_METHOD=='P0_known12'
    assert m.FIELDS=={'run_id','code_commit','arm','train_config','reference_prediction_root','signal_reference_root','output_root','batch_size'}
    with pytest.raises(ValueError):m.validate_config(dict(truth='forbidden'))
