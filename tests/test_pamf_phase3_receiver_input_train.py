"""Only two bounded input adapters learn; source completes before U opens."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.receiver_input_train')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_pcd_cluster_registry_run import fixture as parent
    with parent(root,mp) as (reg,rc,pm,b,pcd,events):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'L_COUNT',reg.COUNTS['L'])
        c=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=rc['reference_config']['train_config'],reference_prediction_root=rc['reference_config']['output_root'],output_root=str(root/'receiver/train'),steps=2,lr=.01)
        # The older producer fixture has a thin P1 role. Materialize this
        # candidate's genuine synthetic same-P1 cached F0 teacher contract.
        t=c['reference_config']['reference_config']['reference_config']['train_config'];folder=Path(t['IQ_root'])/'L'
        role=m.reg.fresh._iq_role(rc['reference_config'],'L',pm['source_control_prediction_identity']);iq=np.asarray(role['restored']).copy()
        np.save(folder/'restored.npy',iq)
        with torch.no_grad():teacher=m._logits(b['reader'],m._tensor(iq,'cpu'))
        np.save(folder/'received.npy',iq);np.save(folder/'teacher_logits.npy',np.asarray(teacher.cpu().tolist(),np.float32))
        mp.setattr(m.cache,'COUNTS',dict(m.cache.COUNTS,L=len(iq)))
        manifest=dict(schema=m.cache.ROLE_SCHEMA,status='STAR_REPRESENTATION_ROLE_COMPLETE',run_id=m.cache.RUN_ID,role='L',count=len(iq),iq_shape=[len(iq),2,256],precision='FP32',paths=['P0','P1'],
            received_input_path='P0',restored_input_path='P1',teacher_input_path='P1',baseline_input_paths=None,files=sorted(m.cache.role_files('L')),metadata_fields=sorted(m.cache.METADATA_FIELDS['L']),
            source_only=True,all_new_IQ_satellite=True,code_commit='a'*40,parent_F0_identity=b['reader_identity'],dura_identity=b['dura_identity'])
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        def public(config):
            assert Path(c['output_root'],'SOURCE_INPUT/complete.json').is_file();events.append('U_public')
            header=m._read(Path(c['reference_config']['output_root'])/'complete.json')
            with np.load(Path(c['reference_config']['output_root'])/'U_features.npz') as z:ids=z['ids'].copy()
            return header['U_manifest'],ids,np.zeros((m.U_COUNT,2,256),np.float32)
        mp.setattr(m,'_public_U',public)
        events.clear()
        yield m,c,pm,b,pcd,events


def test_actual_input_grad_training_source_before_U_frozenNN_and_strict_reload(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,pcd,events):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI bridge'));monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No NumPy ABI bridge'))
        before=[(model,m.reg.fresh.best._snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0],pcd)]
        done=m.train(c,'cpu');out=Path(c['output_root'])
        assert done['source_completed_before_U_open'] and done['source_sampling_matched'] and done['all_adapters_changed']
        assert done['NN_optimizer_updates']==done['target_Q_reads']==done['private_truth_reads']==done['U_truth_reads']==0
        assert done['real_F_forward_rows']==2*128*2+64*2+576 and done['real_PCD_forward_rows']==64*2+576
        assert done['real_r03_forward_rows']==done['G_forward_rows']==0 and done['target_U_reads']==576
        assert events==['checkpoint_PCD','U_public']
        traces={a:[json.loads(x) for x in (out/a/'steps.jsonl').read_text().splitlines()] for a in m.ARMS}
        assert all(a['L_indices']==z['L_indices'] for a,z in zip(traces['SOURCE_INPUT'],traces['TARGET_INPUT']))
        assert all(not r['U_indices'] for r in traces['SOURCE_INPUT']) and all(len(r['U_indices'])==64 for r in traces['TARGET_INPUT'])
        assert all(rows[0]['gradient_norm_before_clip']>0 for rows in traces.values()) and all(r['NN_gradients_none'] for rows in traces.values() for r in rows)
        for model,snapshot in before:m.reg.fresh.best._check_model(model,snapshot);assert all(p.grad is None for p in model.parameters())
        # r01 training terminal was verified at release; the recovery launcher
        # now runs prediction/scoring only and must not relabel these weights.
        # Same strict FP32 theta, but CUDA's derived tanh report can differ
        # from CPU by one FP32 ULP. Only this numerical report is perturbed.
        for arm in m.ARMS:
            path=out/arm/'complete.json';row=m._read(path);fir=np.asarray(row['coefficients']['fir'],np.float32)
            fir[0,0]=np.nextafter(fir[0,0],np.float32(np.inf));row['coefficients']['fir']=fir.tolist()
            path.unlink();m._write(path,row);done['rows'][arm]=row
        (out/'complete.json').unlink();m._write(out/'complete.json',done)
        monkeypatch.setattr(m,'_U',lambda *a:pytest.fail('Loader cannot read U'));monkeypatch.setattr(m,'_L',lambda *a:pytest.fail('Loader cannot read L'));monkeypatch.setattr(m,'_models',lambda *a:pytest.fail('Loader cannot load NN'))
        adapters,loaded=m.load_adapters(c,'cpu');assert loaded==done and all(not p.requires_grad for a in adapters.values() for p in a.parameters())
        assert all(set(a.state_dict())=={'theta'} and a.theta.shape==(16,) for a in adapters.values())
        assert loaded['identities']['SOURCE_INPUT']['adapter_source_only'] and not loaded['identities']['TARGET_INPUT']['adapter_source_only']
        cp=out/'TARGET_INPUT/adapter.pt';saved=cp.read_bytes()
        for defect in ('shape','dtype','nan','theta_value','wrong_arm','identity'):
            with subtests.test(defect=defect):
                payload=torch.load(cp,weights_only=False)
                if defect=='shape':payload['state']['theta']=torch.zeros(15)
                elif defect=='dtype':payload['state']['theta']=payload['state']['theta'].double()
                elif defect=='nan':payload['state']['theta'][0]=torch.nan
                elif defect=='theta_value':payload['state']['theta'][0]=torch.nextafter(payload['state']['theta'][0],torch.tensor(torch.inf))
                elif defect=='wrong_arm':payload['arm']='SOURCE_INPUT'
                else:payload['identity']['parent_PCD_identity']={'wrong':True}
                torch.save(payload,cp)
                with pytest.raises(ValueError):m.load_adapters(c,'cpu')
                cp.write_bytes(saved)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_private_config_failed_smoke_or_partial_source_never_opens_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,pcd,events):
        for delta in (dict(truth='private'),dict(Q_root='forbidden'),dict(steps=1),dict(reference_prediction_root=c['reference_config']['output_root']),dict(output_root=c['reference_prediction_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        gradient=m._gradient_smoke;monkeypatch.setattr(m,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('probe failure')))
        with pytest.raises(ValueError,match='probe failure'):m.train(c,'cpu')
        assert events==['checkpoint_PCD'] and not Path(c['output_root']).exists()
        monkeypatch.setattr(m,'_gradient_smoke',gradient);monkeypatch.setattr(m,'_loop',lambda *a:(_ for _ in ()).throw(ValueError('source interruption')))
        monkeypatch.setattr(m,'_public_U',lambda *a:pytest.fail('Partial source cannot open U'))
        with pytest.raises(ValueError,match='source interruption'):m.train(c,'cpu')
        assert not Path(c['output_root'],'SOURCE_INPUT/complete.json').exists() and not Path(c['output_root'],'complete.json').exists()
        with pytest.raises(ValueError):m.load_adapters(c,'cpu')


def test_U_teacher_mask_is_fixed_from_unadapted_F0_and_PCD():
    m=api();logits=np.zeros((4,6),np.float32);logits[:,0]=[20.,20.,0.,20.];margin=np.asarray([1.,-1.,1.,0.])
    mask,pred,confidence=m.fixed_mask(logits,margin)
    np.testing.assert_array_equal(mask,[True,False,False,True]);np.testing.assert_array_equal(pred,np.zeros(4,np.int64));assert confidence[2]<.95


def test_derived_coefficient_reports_allow_only_two_FP32_ULP(subtests):
    m=api();model=m.core.BoundedIQ()
    with torch.no_grad():model.theta.copy_(torch.linspace(-.7,.8,16))
    expected=m._coefficients(model)
    for steps in (0,1,2):
        with subtests.test(steps=steps):
            report={k:np.asarray(v,np.float32).copy() for k,v in expected.items()}
            for _ in range(steps):report={k:np.nextafter(v,np.float32(np.inf)) for k,v in report.items()}
            assert m._coefficient_reports_equal({k:v.tolist() for k,v in report.items()},expected)
    zero=dict(fir=np.zeros((7,2),np.float32),gamma=np.zeros(2,np.float32));tiny=np.nextafter(np.float32(0),np.float32(1))
    assert m._coefficient_reports_equal(dict(fir=zero['fir']+2*tiny,gamma=zero['gamma']-2*tiny),zero)
    for defect in ('three_ULP','large','shape','nan','strings','extra','near_zero_three_ULP'):
        with subtests.test(defect=defect):
            report={k:np.asarray(v,np.float32).copy() for k,v in expected.items()}
            if defect=='three_ULP':
                for _ in range(3):report['fir'][0,0]=np.nextafter(report['fir'][0,0],np.float32(np.inf))
            elif defect=='large':report['fir'][0,0]+=.001
            elif defect=='shape':report['fir']=report['fir'][:-1]
            elif defect=='nan':report['gamma'][0]=np.nan
            elif defect=='strings':report['fir']=report['fir'].astype(str)
            elif defect=='extra':report['other']=np.zeros(1)
            else:
                report={k:v.copy() for k,v in zero.items()};report['gamma'][0]=3*tiny
            assert not m._coefficient_reports_equal(report,zero if defect=='near_zero_three_ULP' else expected)
