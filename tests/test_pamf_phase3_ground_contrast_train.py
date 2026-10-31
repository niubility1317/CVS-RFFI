"""Two actual native source loops, fixed CPAIR indices and strict checkpoints."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_contrast_train')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_ground_centered_train import fixture as inherited
    from experiments.pamf_phase3_open_world import ground_open_data as data
    actual_load=data.load_role
    with inherited(root,mp) as (centered,cc,pm,b,parent,events,batches,trace):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(centered.COUNTS));cc['code_commit']=m.REFERENCE_CODE
        _,ground,handles,tail_reference=centered._reference(cc)
        identity=centered._identity(cc,'CPAIR',pm,parent._phase3_parent_identity,ground,handles,tail_reference)
        row=dict(schema=centered.SCHEMA,status=centered.ROW_STATUS,run_id=centered.RUN_ID,code_commit=m.REFERENCE_CODE,config=cc,
            arm='CPAIR',identity=identity,updates=2,source_indices_match_DMEAN=True,final_phase3_complete=False)
        write(Path(cc['output_root'])/'rows/CPAIR/complete.json',row)
        trained=dict(schema=centered.SCHEMA,status=centered.STATUS,run_id=centered.RUN_ID,code_commit=m.REFERENCE_CODE,config=cc,
            arms=['CPAIR'],rows={'CPAIR':row},identities={'CPAIR':identity},steps_per_arm=2,updates_per_arm=2,total_optimizer_updates=2,
            real_student_forward_rows=512,source_L_OE_sampling_matched=True,source_sampling_matched=True,same_input_teacher=True,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            source_indices_match_DMEAN=True,source_Us_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,
            source_tail_reference_identity=tail_reference,source_ground_reference_identity=tail_reference['source_ground_reference_identity'],
            ground_data_identity=ground,source_handles=handles,final_phase3_complete=False,test_only_nominal_old_completion=True)
        write(Path(cc['output_root'])/'complete.json',trained)
        p=Path(cc['output_root'])/'rows/CPAIR/steps.jsonl';p.write_text(''.join(json.dumps(a)+'\n' for a in trace),encoding='utf-8')
        ref=dict(run_id=m.previous.RUN_ID,code_commit=m.REFERENCE_CODE,reference_config=cc['reference_config'],train_config=cc,
            output_root=str(root/'old_centered_prediction'),batch_size=128)
        marker=dict(schema=m.previous.SCHEMA,status=m.previous.STATUS,run_id=ref['run_id'],code_commit=m.REFERENCE_CODE,config=ref,
            all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,
            source_controls_reused_exactly=True,all_thresholds_frozen_before_Q=True,private_truth_reads=0,target_U_reads=0,
            source_tail_reference_identity=tail_reference,source_ground_reference_identity=tail_reference['source_ground_reference_identity'],
            final_phase3_complete=False)
        write(Path(ref['output_root'])/'complete.json',marker)
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=ref,output_root=str(root/'contrast'),steps=2,lr=.0001)
        def role(path,name):
            assert name in ('L','OE') and Path(c['output_root'],'smoke.json').is_file()
            events.append(name);return actual_load(path,name)
        mp.setattr(data,'load_role',role)
        yield m,c,pm,b,parent,events,batches,trace


def test_actual_two_native_loops_smoke_matched_sampling_and_strict_reload(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches,reference):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        original=m.old._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['L','OE'] and batches==[2,6,6,256,256,256,256]
        assert done['steps_per_arm']==done['updates_per_arm']==2 and done['total_optimizer_updates']==4
        assert done['real_student_forward_rows']==1024 and done['synthetic_student_forward_rows']==14
        assert done['synthetic_F_forward_rows']==done['G_synthetic_forward_rows']==2
        assert done['synthetic_r03_forward_rows']==done['synthetic_PCD_forward_rows']==done['real_parent_signal_forward_rows']==0
        assert done['source_Us_reads']==done['target_U_reads']==done['target_Q_reads']==done['private_truth_reads']==0
        assert done['source_L_OE_sampling_matched'] and done['source_sampling_matched'] and done['same_input_teacher']
        assert set(done['rows'])==set(done['identities'])=={'KCON','OECON'}
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in original.items())
        traces={arm:[json.loads(a) for a in (out/'rows'/arm/'steps.jsonl').read_text().splitlines()] for arm in m.ARMS}
        for k,o,ref in zip(traces['KCON'],traces['OECON'],reference):
            for a in (k,o):
                assert a['L_indices']==ref['L_indices'] and a['OE_indices']==ref['OE_indices'] and a['source_indices_match_CPAIR']
                assert a['domain_count']==15 and len(a['domain_mean_losses'])==15 and len(a['domain_raw_mean_losses'])==15
                assert abs(a['total']-a['CE']-a['KD']-a['OE']-.1*a['contrast'])<3e-6
                assert a['with_OE_contrast']>=a['known_only_contrast']-1e-6 and a['positive_pair_count']>0
                assert a['gradient_norms']['Sinc']==0 and all(a['gradient_norms'][v]>0 for v in ('head','body','shallow','middle'))
            assert k['selected_OE_negative_pair_count']==0 and o['selected_OE_negative_pair_count']==192*64
        monkeypatch.setattr(m.old._data(),'load_role',lambda *a:pytest.fail('No role values at strict load'))
        for arm in m.ARMS:
            model,i=m.load_trained_model(c,arm,'cpu')
            assert i['reader_parameters']==232993 and i['identity_head_updates']==2
            assert i['source_centered_reference_identity']['code_commit']==m.REFERENCE_CODE
            assert i['source_tail_reference_identity']['code_commit']==m.current.REFERENCE_CODE
            assert i['ground_data_identity']['code_commit']==m.current.tail.REFERENCE_CODE
            assert not model.training and not any(p.requires_grad for p in model.parameters())
        path=out/'rows/OECON/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for defect in ('head','Sinc','buffer','centered_ancestor','ground_data'):
            with subtests.test(defect=defect):
                bad=deepcopy(payload)
                if defect=='centered_ancestor':bad['source_centered_reference_identity']['code_commit']='0'*40
                elif defect=='ground_data':bad['ground_data_identity']['code_commit']=c['code_commit']
                else:
                    key=m.native.HEAD_WEIGHT if defect=='head' else next(k for k in bad['network'] if k.startswith(m.native.SINC_PREFIX)) if defect=='Sinc' else next(k for k in bad['network'] if k.endswith('running_mean'))
                    if defect=='head':del bad['network'][key]
                    else:bad['network'][key]=bad['network'][key]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'OECON','cpu')
        torch.save(payload,path)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_illegal_config_failed_smoke_and_damaged_CPAIR_indices_do_not_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches,trace):
        for update in (dict(U_public_root='forbidden'),dict(Us_labels='forbidden'),dict(steps=1),
                       dict(reference_config=dict(c['reference_config'],code_commit='0'*40)),dict(output_root=c['reference_config']['output_root'])):
            with subtests.test(config=next(iter(update))),pytest.raises(ValueError):m.validate_config(dict(c,**update))
        smoke=m._smoke;monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('contrast smoke failed')))
        with pytest.raises(ValueError,match='contrast smoke failed'):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists();monkeypatch.setattr(m,'_smoke',smoke)
        trace[0]['L_indices'][0]=(trace[0]['L_indices'][0]+1)%270
        p=Path(c['reference_config']['train_config']['output_root'])/'rows/CPAIR/steps.jsonl'
        p.write_text(''.join(json.dumps(a)+'\n' for a in trace),encoding='utf-8')
        with pytest.raises(ValueError,match='CPAIR'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists() and not Path(c['output_root'],'rows/KCON/final.pt').exists()
