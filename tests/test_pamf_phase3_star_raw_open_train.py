"""P0-only source training with a real synthetic native head/return_aux graph."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_train')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_signal_open_train import fixture as parent_fixture,frozen_checkpoint as parent_checkpoint
    with parent_fixture(root,mp) as (old,tc,handle):
        t=api();mp.setattr(t,'COUNTS',dict(L=360,OE=90));mp.setattr(t,'ROLE_COUNTS',dict(L=360,OE=90,V=180,novel=90));mp.setattr(t.IQ,'COUNTS',dict(t.ROLE_COUNTS))
        tc['code_commit']=t.SIGNAL_CODE;tc['native_config']['code_commit']=t.SIGNAL_CODE
        parent_model,identity=parent_checkpoint(tc,handle)
        L=old.parent.load_role(tc['feature_root'],'L');OE=old.parent.load_role(tc['feature_root'],'OE')
        sampler=old.Sampler(L,OE)
        with Path(tc['output_root'],'steps.jsonl').open('w') as f:
            for i in range(1,3):
                ix=sampler.sample(i);f.write(json.dumps(dict(step=i,L_indices=ix['L'].tolist(),OE_indices=ix['OE'].tolist()))+'\n')
        control=root/'macro12/predict';manifests={}
        for role,a in (('L',L),('OE',OE)):
            manifests[role]=dict(handle['source_manifests'][role],schema=t.IQ.ROLE_SCHEMA,status='STAR_REPRESENTATION_ROLE_COMPLETE',run_id=t.IQ.RUN_ID,
                role=role,count=len(a['ids']),iq_shape=[len(a['ids']),2,256],precision='FP32',paths=['P0','P1'],received_input_path='P0',restored_input_path='P1',
                source_only=True,all_new_IQ_satellite=True,metadata_fields=sorted(t.IQ.METADATA_FIELDS[role]))
        marker=dict(schema='pamf_phase3_star_macro12_prediction_v1',status='STAR_MACRO12_PREDICTIONS_COMPLETE',run_id=t.CONTROL_RUN,code_commit=t.CONTROL_CODE,
            config=dict(train_config=tc),parent_model_identity=identity,parent_F0_identity=handle['parent_F0_identity'],dura_identity=handle['dura_identity'],
            source_IQ_manifests=manifests,source_feature_handle=handle,source_only=True,private_truth_reads=0,target_reads=0,
            models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,controls_reused_exactly=True,model_parameters_buffers_unchanged=True)
        write(control/'complete.json',marker)
        rng=np.random.default_rng(20261031);values={}
        for role,a in (('L',L),('OE',OE)):
            folder=Path(tc['IQ_root'])/role;folder.mkdir(parents=True);write(folder/'manifest.json',manifests[role])
            np.save(folder/'ids.npy',a['ids']);np.savez(folder/'metadata.npz',**a['metadata'])
            values[role]=(1.5+.7*rng.normal(size=(len(a['ids']),2,256))).astype(np.float32);np.save(folder/'received.npy',values[role])
            # P1 values deliberately do not exist; the P0 loader needs no access.
        out=root/'raw_train';n=deepcopy(tc['native_config']);n.update(run_id=t.RUN_ID,code_commit='e'*40,row_id='SOURCE_E_RAW-s2026092701',output_root=str(out))
        c=dict(run_id=t.RUN_ID,code_commit='e'*40,arm='E_RAW',native_config=n,IQ_root=tc['IQ_root'],control_root=str(control),
            control_train_config=deepcopy(tc),output_root=str(out),steps=t.STEPS,lr=.0001,L_seed=2026103017,OE_seed=2026103018)
        events=[];built=[];construct=t.native.make_trainable_model
        def native(cfg,device):
            assert cfg['run_id']==t.RUN_ID and cfg['row_id']=='SOURCE_E_RAW-s2026092701'
            result=construct(cfg,device);built.append(result);events.append('construct');return result
        def smoke(model,device):
            assert all(p.requires_grad for p in model.parameters());before=t._snapshot(model)
            x=torch.stack((torch.sin(torch.arange(512).reshape(2,256)*.07),torch.cos(torch.arange(512).reshape(2,256)*.11))).float().to(device)
            with torch.no_grad():assert model(x).shape==(2,6)
            assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items());events.append('checkpoint_smoke')
            return dict(status='PASS',actual_F0_checkpoint=True,F0_forward_rows=2,G_forward_rows=0)
        mp.setattr(t.native,'make_trainable_model',native);mp.setattr(t,'_checkpoint_smoke',smoke)
        yield t,c,marker,values,events,built


def frozen_checkpoint(c,m):
    t=api();model,teacher=t._construct(c,'cpu');identity=t._identity(c,m,model);out=Path(c['output_root']);out.mkdir(parents=True)
    zero={k:0 for k in t.ZEROS};provenance=t._teacher_provenance(c,m)
    torch.save(dict(schema=t.SCHEMA,config=c,identity=identity,network=model.state_dict(),control_identity=m,teacher_provenance=provenance,
        original_parent_identity=model.original_parent_identity,source_data_handle=model.source_data_handle,source_only=True,**zero),out/'final.pt')
    write(out/'complete.json',dict(schema=t.SCHEMA,status=t.STATUS,run_id=t.RUN_ID,code_commit=c['code_commit'],arm='E_RAW',config=c,
        identity=identity,updates=t.STEPS,open_signal_updates=t.STEPS,control_identity=m,teacher_provenance=provenance,source_only=True,
        model_buffers_unchanged=True,teacher_state_unchanged=True,final_phase3_complete=False,test_only_nominal_completion=True,**zero))
    return model,identity


def test_two_steps_use_received_original_teacher_once_and_strict_headless_checkpoint(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,m,values,events,built):
        monkeypatch.setattr(t,'STEPS',2);c=dict(c,steps=2);load=np.load
        def safe(path,*args,**kwargs):
            assert Path(path).name not in ('restored.npy','teacher_logits.npy','known_predictions.npz','novel_predictions.npz')
            return load(path,*args,**kwargs)
        monkeypatch.setattr(np,'load',safe);done=t.train(c,'cpu');out=Path(c['output_root'])
        assert len(built)==1 and events[:2]==['construct','checkpoint_smoke']
        teacher=built[0];assert all(not p.requires_grad for p in teacher.parameters()) and not teacher.training
        with torch.no_grad():expected=t.core.unit(teacher.id_backbone.encoder.features(torch.tensor(values['OE'].tolist())))
        with np.load(out/'teacher_P0.npz') as z:
            assert set(z.files)=={'ids','u','rx_i','day_i','domain'}
            np.testing.assert_array_equal(z['u'],np.asarray(expected.tolist(),np.float32))
        assert done['real_F_forward_rows']==90 and done['synthetic_F_forward_rows']==2 and done['real_signal_forward_rows']==512
        assert done['synthetic_signal_forward_rows']==8 and done['teacher_state_unchanged'] and done['restored_cache_values_read']==0
        for row in (done['first_step'],done['second_step']):
            assert row['input_path']=='P0' and row['teacher_input_path']=='P0' and row['control_indices_equal']
            assert row['gradient_norms']['sinc']>0 and row['crossset_embedding_gradients']['OE']['minimum_row_norm']>0
            for k in ('middle','time_frequency','PA','reference','signature','last'):assert row['gradient_norms'][k]>0
        assert done['first_step']['gradient_norms']['hidden']==0 and done['second_step']['gradient_norms']['hidden']>0
        assert done['optimizer']['sinc_steps']==done['optimizer']['physical_clip_calls']==2
        value=torch.load(out/'final.pt',weights_only=False)
        assert not any('cls_head.weight' in k for k in value['network']) and 'teacher_bank' not in value and 'u' not in value
        (out/'teacher_P0.npz').unlink();model,identity=t.load_trained_model(c,out/'final.pt','cpu')
        assert identity==done['identity'] and identity['input_path']=='cached_P0_IQ_only' and identity['teacher_input_path']=='P0'
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        value['network'].pop('projection.network.2.bias');torch.save(value,out/'final.pt')
        with pytest.raises(ValueError,match='Strict'):t.load_trained(c,'cpu')


def test_smoke_and_gradient_probe_precede_all_role_values_and_bad_inputs_fail(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,m,v,events,built):
        for delta in (dict(truth='private'),dict(arm='E_SIGNAL'),dict(steps=222),dict(IQ_root=str(tmp_path/'wrong'))):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
        monkeypatch.setattr(t,'load_role',lambda *a:pytest.fail('Real source values opened before smoke'))
        monkeypatch.setattr(t,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('gradient smoke fail')))
        with pytest.raises(ValueError,match='gradient smoke'):t.train(c,'cpu')
        assert not Path(c['output_root']).exists()


def test_public_P0_loader_never_reads_P1_and_training_rejects_heldout_roles(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,m,v,events,built):
        for role in ('L','OE'):
            a=t.load_received(Path(c['IQ_root'])/role,role)
            assert set(a)=={'ids','metadata','manifest','received'} and isinstance(a['received'],np.memmap) and not a['received'].flags.writeable
            np.testing.assert_array_equal(a['received'],v[role]);t.close_role(a)
        for role in ('V','novel'):
            with pytest.raises(ValueError):t.load_role(c['IQ_root'],role)
        with pytest.raises(ValueError):t.load_received(Path(c['IQ_root'])/'OE','Q')
        metadata=Path(c['IQ_root'])/'OE/metadata.npz'
        with np.load(metadata) as z:meta={k:z[k] for k in z.files}
        np.savez(metadata,**meta,y=np.zeros(90,np.int64))
        with pytest.raises(ValueError):t.load_received(metadata.parent,'OE')


def test_control_indices_and_private_teacher_provenance_are_required_not_P1_checkpoint_weights(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,m,v,events,built):
        model,identity=frozen_checkpoint(c,m)
        assert not model.projection.network[-1].weight.any() and identity['signal_feature_cumulative_updates']==13320
        loaded,_=t.load_trained(c,'cpu');assert sum(p.numel() for p in loaded.parameters())==334721
        cp=Path(c['output_root'])/'final.pt';value=torch.load(cp,weights_only=False);value['teacher_provenance']['input_path']='P1';torch.save(value,cp)
        with pytest.raises(ValueError):t.load_trained(c,'cpu')
        monkeypatch.setattr(t,'STEPS',2);cc=dict(c,steps=2,output_root=str(tmp_path/'mismatch'),native_config=dict(c['native_config'],output_root=str(tmp_path/'mismatch')))
        trace=Path(c['control_train_config']['output_root'])/'steps.jsonl';rows=trace.read_text().splitlines();row=json.loads(rows[0]);row['OE_indices'][0]=-1;rows[0]=json.dumps(row);trace.write_text('\n'.join(rows)+'\n')
        with pytest.raises(ValueError,match='indices'):t.train(cc,'cpu')
        assert not Path(cc['output_root'],'complete.json').exists()
