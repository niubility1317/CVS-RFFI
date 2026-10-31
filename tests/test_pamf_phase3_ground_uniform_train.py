"""Actual synthetic UNIFORM native training and strict source-only reload."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_uniform_train')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_ground_contrast_train import fixture as inherited
    from experiments.pamf_phase3_open_world import ground_open_data as data
    actual_load=data.load_role
    with inherited(root,mp) as (previous,oldc,pm,b,parent,events,batches,trace):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(previous.COUNTS))
        c=dict(oldc,run_id=m.RUN_ID,code_commit='b'*40,output_root=str(root/'uniform'))
        def role(path,name):
            assert name in ('L','OE') and Path(c['output_root'],'smoke.json').is_file()
            events.append(name);return actual_load(path,name)
        mp.setattr(data,'load_role',role)
        yield m,c,pm,b,parent,events,batches,trace


def test_actual_single_native_uniform_smoke_matching_CPAIR_and_strict_reload(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches,reference):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('NumPy2 bridge prohibited'))
        before=m._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['L','OE'] and batches==[2,4,256,256]
        assert done['total_optimizer_updates']==done['steps_per_arm']==done['updates_per_arm']==2
        assert done['real_student_forward_rows']==512 and done['synthetic_student_forward_rows']==6
        assert done['synthetic_F_forward_rows']==done['G_synthetic_forward_rows']==2
        assert done['real_F_forward_rows']==done['real_parent_signal_forward_rows']==done['G_forward_rows']==0
        assert done['synthetic_r03_forward_rows']==done['synthetic_PCD_forward_rows']==done['real_PCD_forward_rows']==0
        assert done['source_Us_reads']==done['target_U_reads']==done['target_Q_reads']==done['private_truth_reads']==0
        assert done['source_sampling_matched'] and done['source_L_OE_sampling_matched'] and done['same_input_teacher']
        assert set(done['rows'])==set(done['identities'])=={'UNIFORM'}
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in before.items())
        trace=[json.loads(a) for a in (out/'rows/UNIFORM/steps.jsonl').read_text().splitlines()]
        for row,old in zip(trace,reference):
            assert row['L_indices']==old['L_indices'] and row['OE_indices']==old['OE_indices'] and row['source_indices_match_CPAIR']
            assert row['domain_count']==15 and row['domain_uniform_count']==15 and row['rank_is_diagnostic']
            assert len(row['domain_uniform_losses'])==len(row['domain_mean_losses'])==len(row['domain_raw_mean_losses'])==15
            assert all(v>0 for v in row['domain_uniform_counts']) and row['uniform']==row['OE']==row['domain_uniform_mean']
            assert abs(row['total']-row['CE']-row['KD']-row['uniform'])<3e-6
            assert row['gradient_norms']['Sinc']==0 and all(row['gradient_norms'][k]>0 for k in ('head','body','shallow','middle'))
        monkeypatch.setattr(m.old._data(),'load_role',lambda *a:pytest.fail('Strict reload may not read source roles'))
        model,i=m.load_trained_model(c,'UNIFORM','cpu')
        assert i['reader_parameters']==232993 and i['trainable_parameters']==232945 and i['identity_head_updates']==2
        assert i['source_centered_reference_identity']['code_commit']==m.REFERENCE_CODE
        assert i['source_tail_reference_identity']['code_commit']==m.ancestry.current.REFERENCE_CODE
        assert i['ground_data_identity']['code_commit']==m.ancestry.current.tail.REFERENCE_CODE
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        path=out/'rows/UNIFORM/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for defect in ('Sinc','head','buffer','source_ancestor','data_ancestor','nonfinite'):
            with subtests.test(defect=defect):
                bad=deepcopy(payload)
                if defect=='source_ancestor':bad['source_centered_reference_identity']['code_commit']='0'*40
                elif defect=='data_ancestor':bad['ground_data_identity']['code_commit']=c['code_commit']
                else:
                    key=m.native.HEAD_WEIGHT if defect in ('head','nonfinite') else next(k for k in bad['network'] if k.startswith(m.native.SINC_PREFIX)) if defect=='Sinc' else next(k for k in bad['network'] if k.endswith('running_mean'))
                    if defect=='head':del bad['network'][key]
                    elif defect=='nonfinite':bad['network'][key][0,0]=float('nan')
                    else:bad['network'][key]=bad['network'][key]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'UNIFORM','cpu')
        torch.save(payload,path)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_invalid_config_smoke_before_IQ_and_CPAIR_mismatch_do_not_complete(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches,trace):
        for update in (dict(Us_public_root='forbidden'),dict(U_truth='forbidden'),dict(steps=1),
                       dict(reference_config=dict(c['reference_config'],code_commit='0'*40)),dict(output_root=c['reference_config']['output_root'])):
            with subtests.test(config=next(iter(update))),pytest.raises(ValueError):m.validate_config(dict(c,**update))
        smoke=m._smoke;monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('uniform checkpoint smoke failed')))
        with pytest.raises(ValueError,match='uniform checkpoint smoke failed'):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists();monkeypatch.setattr(m,'_smoke',smoke)
        trace[0]['OE_indices'][0]=(trace[0]['OE_indices'][0]+1)%90
        p=Path(c['reference_config']['train_config']['output_root'])/'rows/CPAIR/steps.jsonl'
        p.write_text(''.join(json.dumps(a)+'\n' for a in trace),encoding='utf-8')
        with pytest.raises(ValueError,match='CPAIR'):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists() and not Path(c['output_root'],'rows/UNIFORM/final.pt').exists()
