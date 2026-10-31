"""Actual small native reader training/reload; no real IQ/checkpoints."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def api(): return importlib.import_module('experiments.pamf_phase3_open_world.ground_tail_train')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_ground_open_train import fixture as inherited
    from experiments.pamf_phase3_open_world import ground_open_data as data
    actual_load=data.load_role
    with inherited(root,mp) as (old,oldc,pm,b,parent,events,batches):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(old.COUNTS))
        oldc['code_commit']=m.REFERENCE_CODE
        p=Path(oldc['data_root'])/'complete.json';d=json.loads(p.read_text());d['code_commit']=m.REFERENCE_CODE
        for role in ('L','OE','Us'):
            path=Path(oldc['data_root'])/'roles'/role/'manifest.json';a=json.loads(path.read_text());a['code_commit']=m.REFERENCE_CODE;write(path,a);d['roles'][role]=a
        write(p,d)
        reference=dict(run_id=m.previous.RUN_ID,code_commit=m.REFERENCE_CODE,reference_config=oldc['reference_config'],
            data_root=oldc['data_root'],train_config=oldc,output_root=str(root/'old_ground_prediction'),batch_size=128)
        refdone=dict(schema=m.previous.SCHEMA,status=m.previous.STATUS,run_id=reference['run_id'],code_commit=m.REFERENCE_CODE,config=reference,
            all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,all_thresholds_frozen_before_Q=True,
            private_truth_reads=0,target_U_reads=0,final_phase3_complete=False)
        write(Path(reference['output_root'])/'complete.json',refdone)
        olddone=dict(schema=old.SCHEMA,status=old.STATUS,run_id=old.RUN_ID,code_commit=m.REFERENCE_CODE,config=oldc,steps_per_arm=old.STEPS,
            updates_per_arm=old.STEPS,total_optimizer_updates=2*old.STEPS,source_L_OE_sampling_matched=True,original_models_unchanged=True,
            all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,
            source_U_GT_reads=0,final_phase3_complete=False,test_only_nominal_old_completion=True)
        write(Path(oldc['output_root'])/'complete.json',olddone)
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=reference,output_root=str(root/'ground_tail'),steps=2,lr=.0001)
        def load(path,role):
            assert role in ('L','OE'),'New two-arm run cannot open Us/CAL/sourceQ'
            assert Path(c['output_root'],'smoke.json').is_file()
            events.append(role);return actual_load(path,role)
        mp.setattr(data,'load_role',load)
        mp.setattr(data,'load_query',lambda *a:pytest.fail('Training must not open a query'))
        yield m,c,pm,b,parent,events,batches


def test_actual_matched_native_training_newloss_smoke_and_strict_checkpoint(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        initial=m.old._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['L','OE'] and batches==[2,4,6,4,4,256,256,256,256]
        assert done['steps_per_arm']==done['updates_per_arm']==2 and done['total_optimizer_updates']==4
        assert done['real_student_forward_rows']==1024 and done['synthetic_student_forward_rows']==20
        assert done['real_F_forward_rows']==done['real_parent_signal_forward_rows']==done['real_PCD_forward_rows']==done['G_forward_rows']==0
        assert done['source_Us_reads']==done['CAL_reads']==done['target_U_reads']==done['private_truth_reads']==0
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in initial.items())
        traces={arm:[json.loads(v) for v in (out/'rows'/arm/'steps.jsonl').read_text().splitlines()] for arm in m.ARMS}
        for left,right in zip(traces['DMEAN'],traces['DTAIL']):
            assert left['L_indices']==right['L_indices'] and left['OE_indices']==right['OE_indices']
            assert left['domain_pair_counts']==right['domain_pair_counts'] and left['domain_count']==right['domain_count']==15
            assert left['selected_pair_count']==left['pair_count'] and right['selected_pair_count']<right['pair_count']
            for row in (left,right):
                assert abs(row['total']-row['CE']-row['KD']-row['OE'])<2e-6
                assert len(row['domain_mean_losses'])==len(row['domain_tail_losses'])==15
                assert row['gradient_norms']['Sinc']==0 and all(row['gradient_norms'][k]>0 for k in ('head','body','shallow','middle'))
        monkeypatch.setattr(m.old._data(),'load_role',lambda *a:pytest.fail('Strict reload cannot open source role arrays'))
        for arm in m.ARMS:
            model,i=m.load_trained_model(c,arm,'cpu')
            assert i['run_id']==m.RUN_ID and i['code_commit']==c['code_commit'] and i['identity_head_updates']==2
            assert i['ground_data_identity']['code_commit']==m.REFERENCE_CODE and i['source_ground_reference_identity']['code_commit']==m.REFERENCE_CODE
            assert i['source_handles']['L']['code_commit']==m.REFERENCE_CODE and i['reader_parameters']==232993
            assert not model.training and not any(p.requires_grad for p in model.parameters())
        path=out/'rows/DTAIL/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for defect in ('Sinc','head','buffer','old_code','ground_identity'):
            with subtests.test(defect=defect):
                bad=deepcopy(payload)
                if defect=='old_code':bad['source_ground_reference_identity']['code_commit']='0'*40
                elif defect=='ground_identity':bad['ground_data_identity']['code_commit']=c['code_commit']
                else:
                    key=m.native.HEAD_WEIGHT if defect=='head' else next(k for k in bad['network'] if k.startswith(m.native.SINC_PREFIX)) if defect=='Sinc' else next(k for k in bad['network'] if k.endswith('running_mean'))
                    if defect=='head':del bad['network'][key]
                    else:bad['network'][key]=bad['network'][key]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'DTAIL','cpu')
        torch.save(payload,path)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_wrong_source_ancestor_forbidden_inputs_and_failed_smoke_never_open_IQ(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches):
        for change in (dict(Us_root='hidden'),dict(private_truth='hidden'),dict(steps=1),dict(output_root=c['reference_config']['data_root']),
                       dict(reference_config=dict(c['reference_config'],code_commit='0'*40))):
            with subtests.test(config=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
        p=Path(c['reference_config']['output_root'])/'complete.json';original=p.read_bytes();wrong=json.loads(original);wrong['all_12_points_complete']=False;write(p,wrong)
        with pytest.raises(ValueError):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists();p.write_bytes(original)
        monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('newloss checkpoint smoke failed')))
        with pytest.raises(ValueError,match='newloss checkpoint smoke'):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists()

