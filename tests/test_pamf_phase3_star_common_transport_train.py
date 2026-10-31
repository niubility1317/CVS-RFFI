"""One matched CCD student, strict synthetic parent, no old query values."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_common_transport_train')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_body_prototype_train import fixture as previous
    with previous(root,mp) as (body,tc,pm,b,U,events,_):
        m=api();tc['code_commit']=m.reference.TRAIN_CODE;old=body.train(tc,'cpu');events.clear()
        mp.setattr(m,'STEPS',2);mp.setattr(m,'WARMUP_STEPS',1);mp.setattr(m,'U_COUNT',16);mp.setattr(m,'U_BATCH',4);mp.setattr(m,'COUNTS',dict(body.COUNTS))
        ref=dict(run_id=m.reference.RUN_ID,code_commit=m.REFERENCE_CODE,train_config=tc,reference_config=tc['reference_config'],output_root=str(root/'body_r02_predict'),batch_size=128)
        Path(ref['output_root']).mkdir();body._write(Path(ref['output_root'])/'complete.json',dict(schema=m.reference.SCHEMA,status=m.reference.STATUS,run_id=ref['run_id'],code_commit=ref['code_commit'],config=ref,
            reference_prediction_identity=pm,model_identities=dict(E0=pm['model_identity'],**old['identities']),parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
            training_run_id=tc['run_id'],training_code_commit=tc['code_commit'],training_reused=True,this_run_optimizer_updates=0,private_truth_reads=0,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,all_three_DURA_conditions_complete=True))
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=ref,U_public_root=tc['U_public_root'],U_capsule_id=tc['U_capsule_id'],U_split_id=tc['U_split_id'],output_root=str(root/'common_train'),steps=2,lr=.0001)
        def public(config):
            assert Path(c['output_root'],'smoke.json').exists();events.append('U');return U
        mp.setattr(m.body.ssl,'_public',public)
        yield m,c,pm,b,U,events,old


def test_single_actual_student_matches_old_source_and_U_with_CCD_and_strict_reload(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,U,events,old):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI'));monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No NumPy ABI'))
        before=m.body._snapshot(b['model']);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','OE','U'] and done['source_sampling_matched'] and done['U_sampling_matched']
        assert done['total_optimizer_updates']==2 and not (out/'SOURCE').exists()
        assert done['real_parent_signal_forward_rows']==sum(m.COUNTS.values())+16 and done['real_student_signal_forward_rows']==(256+16+8)*2
        rows=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
        controls=[json.loads(x) for x in (Path(c['reference_config']['train_config']['output_root'])/'TARGET_PCD/steps.jsonl').read_text().splitlines()]
        for a,z in zip(rows,controls):
            assert all(a[k]==z[k] for k in ('L_indices','OE_indices','U_indices')) and len(a['CCD_beta'])==6
        assert rows[0]['CCD_active'] is False and rows[1]['CCD_active'] is True and rows[1]['CCD_transport']['iterations']==400
        assert done['CCD_first_active_step']==2 and done['CCD_only_gradient_smoke']['gradient_norms']['body']>0
        loaded,i=m.load_trained_model(c,'cpu');assert i==done['identity'] and i['arm']=='TARGET_CCD' and all(not p.requires_grad for p in loaded.parameters())
        assert done['all_student_Sinc_unchanged'] and done['all_student_buffers_unchanged'] and done['all_student_bodies_changed'] and done['all_student_projections_changed']
        assert all(torch.equal(v,b['model'].state_dict()[k].detach().cpu()) for k,v in before.items())
        with np.load(out/'U_features.npz') as z:assert set(z.files)=={'ids','views','parent_z','neighbors'}
        cp=out/'final.pt';payload=torch.load(cp,weights_only=False);key=next(k for k in payload['network'] if k.startswith(m.body.signalcore.SINC_PREFIX));payload['network'][key]+=1;torch.save(payload,cp)
        with pytest.raises(ValueError,match='Sinc'):m.load_trained_model(c,'cpu')


def test_private_config_and_sampling_mismatch_never_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,U,events,old):
        for delta in (dict(truth='private'),dict(Q_public_root='forbidden'),dict(U_capsule_id='clean'),dict(steps=1)):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        path=Path(c['reference_config']['train_config']['output_root'])/'TARGET_PCD/steps.jsonl';lines=path.read_text().splitlines();row=json.loads(lines[0]);row['U_indices'][0]=-1;lines[0]=json.dumps(row);path.write_text('\n'.join(lines)+'\n')
        with pytest.raises(ValueError,match='sampling'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists() and not Path(c['output_root'],'final.pt').exists()

