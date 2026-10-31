"""Single centered-pair native training, fixed DMEAN indices and strict reload."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_centered_train')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_ground_tail_train import fixture as inherited
    from experiments.pamf_phase3_open_world import ground_open_data as data
    actual_load=data.load_role
    with inherited(root,mp) as (tail,tailc,pm,b,parent,events,batches):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(tail.COUNTS));tailc['code_commit']=m.REFERENCE_CODE
        _,ground,handles,ground_reference=tail._reference(tailc)
        ref=dict(run_id=m.previous.RUN_ID,code_commit=m.REFERENCE_CODE,reference_config=tailc['reference_config'],
            train_config=tailc,output_root=str(root/'old_tail_prediction'),batch_size=128)
        marker=dict(schema=m.previous.SCHEMA,status=m.previous.STATUS,run_id=ref['run_id'],code_commit=m.REFERENCE_CODE,config=ref,
            all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_controls_reused_exactly=True,
            all_thresholds_frozen_before_Q=True,private_truth_reads=0,target_U_reads=0,source_ground_reference_identity=ground_reference,
            final_phase3_complete=False)
        write(Path(ref['output_root'])/'complete.json',marker)
        done=dict(schema=tail.SCHEMA,status=tail.STATUS,run_id=tail.RUN_ID,code_commit=m.REFERENCE_CODE,config=tailc,
            steps_per_arm=2,updates_per_arm=2,total_optimizer_updates=4,source_L_OE_sampling_matched=True,source_sampling_matched=True,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,source_Us_reads=0,
            target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_ground_reference_identity=ground_reference,
            ground_data_identity=ground,source_handles=handles,final_phase3_complete=False,test_only_nominal_old_training_completion=True)
        write(Path(tailc['output_root'])/'complete.json',done)
        metadata={}
        for role in ('L','OE'):
            with np.load(Path(ref['reference_config']['data_root'])/'roles'/role/'metadata.npz',allow_pickle=False) as z:
                a={k:z[k].copy() for k in z.files}
            metadata[role]=tail.old._metadata(a,a['ids'],role=='L')
        sampler=tail.native.SourceSampler(metadata['L'],metadata['OE']);trace=[]
        for step in (1,2):
            sample=sampler.sample(step);trace.append(dict(step=step,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist()))
        path=Path(tailc['output_root'])/'rows/DMEAN/steps.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in trace),encoding='utf-8')
        c=dict(run_id=m.RUN_ID,code_commit='f'*40,reference_config=ref,output_root=str(root/'centered'),steps=2,lr=.0001)
        def role(path,name):
            assert name in ('L','OE') and Path(c['output_root'],'smoke.json').is_file()
            events.append(name);return actual_load(path,name)
        mp.setattr(data,'load_role',role);mp.setattr(data,'load_query',lambda *a:pytest.fail('No source/target query in training'))
        mp.setattr(tail,'_load_PCD',lambda *a:pytest.fail('PCD is not part of this training or smoke'))
        mp.setattr(b['model'],'forward',lambda *a:pytest.fail('r03 is never forwarded by centered native training'))
        yield m,c,pm,b,parent,events,batches,trace


def test_actual_single_native_checkpoint_smoke_matching_source_and_strict_loader(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches,reference):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        original=m.old._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['L','OE'] and batches==[2,4,256,256]
        assert done['total_optimizer_updates']==done['updates_per_arm']==2 and done['real_student_forward_rows']==512
        assert done['synthetic_student_forward_rows']==6 and done['synthetic_F_forward_rows']==done['G_synthetic_forward_rows']==2
        assert done['synthetic_r03_forward_rows']==done['synthetic_PCD_forward_rows']==done['real_parent_signal_forward_rows']==0
        assert done['source_Us_reads']==done['target_U_reads']==done['target_Q_reads']==done['private_truth_reads']==0
        assert done['source_L_OE_sampling_matched'] and done['source_sampling_matched'] and done['same_input_teacher']
        assert set(done['rows'])==set(done['identities'])=={'CPAIR'}
        assert all(torch.equal(value,parent.state_dict()[key]) for key,value in original.items())
        rows=[json.loads(x) for x in (out/'rows/CPAIR/steps.jsonl').read_text().splitlines()]
        for row,ref in zip(rows,reference):
            assert row['L_indices']==ref['L_indices'] and row['OE_indices']==ref['OE_indices'] and row['domain_count']==15
            assert len(row['domain_mean_losses'])==len(row['domain_raw_mean_losses'])==15
            assert abs(row['total']-row['CE']-row['KD']-row['OE'])<2e-6
            assert row['gradient_norms']['Sinc']==0 and all(row['gradient_norms'][k]>0 for k in ('head','body','shallow','middle'))
        monkeypatch.setattr(m.old._data(),'load_role',lambda *a:pytest.fail('No role values during strict CP loading'))
        model,identity=m.load_trained_model(c,'CPAIR','cpu')
        assert identity['reader_parameters']==232993 and identity['identity_head_updates']==2 and identity['source_Us_reads']==0
        assert identity['source_tail_reference_identity']['code_commit']==m.REFERENCE_CODE
        assert identity['ground_data_identity']['code_commit']==m.tail.REFERENCE_CODE
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        path=out/'rows/CPAIR/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for defect in ('head','Sinc','buffer','tail_ancestor','ground_data'):
            with subtests.test(defect=defect):
                bad=deepcopy(payload)
                if defect=='tail_ancestor':bad['source_tail_reference_identity']['code_commit']='0'*40
                elif defect=='ground_data':bad['ground_data_identity']['code_commit']=c['code_commit']
                else:
                    key=m.native.HEAD_WEIGHT if defect=='head' else next(k for k in bad['network'] if k.startswith(m.native.SINC_PREFIX)) if defect=='Sinc' else next(k for k in bad['network'] if k.endswith('running_mean'))
                    if defect=='head':del bad['network'][key]
                    else:bad['network'][key]=bad['network'][key]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'CPAIR','cpu')
        torch.save(payload,path)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_invalid_input_failed_smoke_and_mismatched_DMEAN_indices_preserve_state(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches,trace):
        for update in (dict(U_public_root='forbidden'),dict(Us_labels='forbidden'),dict(steps=1),
                       dict(reference_config=dict(c['reference_config'],code_commit='0'*40)),dict(output_root=c['reference_config']['output_root'])):
            with subtests.test(config=next(iter(update))),pytest.raises(ValueError):m.validate_config(dict(c,**update))
        smoke=m._smoke;monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('centered smoke failed')))
        with pytest.raises(ValueError,match='centered smoke failed'):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists();monkeypatch.setattr(m,'_smoke',smoke)
        trace[0]['L_indices'][0]=(trace[0]['L_indices'][0]+1)%270
        path=Path(c['reference_config']['train_config']['output_root'])/'rows/DMEAN/steps.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in trace),encoding='utf-8')
        with pytest.raises(ValueError,match='DMEAN'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists() and not Path(c['output_root'],'rows/CPAIR/final.pt').exists()
