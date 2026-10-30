"""Synthetic actual headless checkpoint, source boundaries and matched steps."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_pamf_phase3_star_crossset_metric_train import fixture as parent_fixture
from test_pamf_phase3_star_signal_open_model import source


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_train')


@contextmanager
def fixture(root,mp):
    torch.set_num_threads(1)
    with parent_fixture(root,mp) as (parent,pc,configs,m):
        t=api();mp.setattr(t,'COUNTS',dict(L=360,OE=90));m=deepcopy(m)
        m['code_commit']=t.CONTROL_CODE;m['config']['code_commit']=t.CONTROL_CODE;m['config']['source_config']['code_commit']=t.CONTROL_CODE
        Path(pc['output_root'],'complete.json').write_text(json.dumps(m),encoding='utf-8')
        n=deepcopy(pc['source_config']);out=root/'signal_train'
        n.update(run_id=t.RUN_ID,code_commit='c'*40,row_id='E_SIGNAL-s2026092701',output_root=str(out))
        c=dict(run_id=t.RUN_ID,code_commit='c'*40,arm='E_SIGNAL',native_config=n,IQ_root=str(root/'P1_IQ'),feature_root=pc['output_root'],
            control_root=str(root/'control/predict'),output_root=str(out),steps=t.STEPS,lr=.0001,L_seed=2026103017,OE_seed=2026103018)
        cp=Path(c['control_root']);cp.mkdir(parents=True);oldtrain=root/'control/train';oldtrain.mkdir()
        parent._write(cp/'complete.json',dict(schema='pamf_phase3_star_crossset_metric_prediction_v1',status='STAR_CROSSSET_METRIC_PREDICTIONS_COMPLETE',
            run_id=parent.RUN_ID,code_commit=t.CONTROL_CODE,arm='S_SEP_TOPO',source_only=True,private_truth_reads=0,target_reads=0,source_states_unchanged=True,
            model_parameters_buffers_unchanged=True,feature_manifest=m,source_feature_manifest=m['source_feature_manifest'],
            config=dict(train_config=dict(output_root=str(oldtrain),steps=2220,L_seed=2026103017,OE_seed=2026103018))))
        L=parent.load_role(pc['output_root'],'L');OE=parent.load_role(pc['output_root'],'OE');sampler=parent.Sampler(L,OE)
        with (oldtrain/'steps.jsonl').open('w') as f:
            for i in range(1,3):
                ix=sampler.sample(i);f.write(json.dumps(dict(step=i,L_indices=ix['L'].tolist(),OE_indices=ix['OE'].tolist()))+'\n')
        rng=np.random.default_rng(41);roles={}
        for role,a in (('L',L),('OE',OE)):
            roles[role]=dict(ids=a['ids'],metadata=a['metadata'],manifest=m['source_manifests'][role],restored=rng.normal(size=(len(a['ids']),2,256)).astype(np.float32))
        def construct(config,device):
            model,_=source();model._phase3_parent_identity=dict(parent_updates=11100,selection='original_fixed_final_E50')
            model._phase3_source_handle=dict(classes=t.native.CLASSES,source_rxs=t.native.RX,source_days=[1,2,3],split_seed=392005,
                counts=dict(L_s=6300,U_s=56700,V=27000),source_contract=n['source_contract'])
            return model.to(device)
        mp.setattr(t.native,'make_trainable_model',construct)
        mp.setattr(t,'_checkpoint_smoke',lambda model,device:dict(status='PASS',actual_F0_checkpoint=True,F0_forward_rows=2,G_forward_rows=0))
        def load(root,role):
            assert role in ('L','OE');return roles[role]
        mp.setattr(t,'load_role',load);mp.setattr(t,'close_role',lambda a:None)
        yield t,c,m


def frozen_checkpoint(c,m):
    """Real synthetic network state with explicit nominal-complete test metadata."""
    t=api();model=t._model(c,'cpu');identity=t._identity(c,m,model);out=Path(c['output_root']);out.mkdir(parents=True)
    z={k:0 for k in t.ZEROS};torch.save(dict(schema=t.SCHEMA,config=c,identity=identity,network=model.state_dict(),cache_manifest=m,
        original_parent_identity=model.original_parent_identity,source_data_handle=model.source_data_handle,source_only=True,**z),out/'final.pt')
    t._write(out/'complete.json',dict(schema=t.SCHEMA,status=t.STATUS,run_id=t.RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,
        identity=identity,updates=t.STEPS,open_signal_updates=t.STEPS,cache_manifest=m,source_only=True,model_buffers_unchanged=True,
        final_phase3_complete=False,test_only_nominal_completion=True,**z));return model,identity


def test_config_and_failed_smoke_never_open_IQ_or_teacher(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,m):
        for delta in (dict(truth='private'),dict(arm='S_SEP_TOPO'),dict(steps=222),dict(L_seed=1),dict(IQ_root=c['output_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
        monkeypatch.setattr(t,'_checkpoint_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke fail')))
        monkeypatch.setattr(t,'load_role',lambda *a:pytest.fail('IQ opened before checkpoint smoke'))
        monkeypatch.setattr(t.parent,'load_role',lambda *a:pytest.fail('Teacher opened before checkpoint smoke'))
        with pytest.raises(ValueError,match='smoke fail'):t.train(c,'cpu')
        assert not Path(c['output_root']).exists()


def test_two_steps_actual_signal_grads_physical_optimizer_matched_sampler_and_strict_reload(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,m):
        monkeypatch.setattr(t,'STEPS',2);c=dict(c,steps=2);load=t.load_role;events=[]
        def opened(root,role):
            assert (Path(c['output_root'])/'smoke.json').is_file();events.append(role);return load(root,role)
        monkeypatch.setattr(t,'load_role',opened);teacher_load=t.parent.load_role;teacher_arrays=[]
        def teacher(root,role):
            a=teacher_load(root,role);teacher_arrays.append((a['u'],a['u'].copy()));return a
        monkeypatch.setattr(t.parent,'load_role',teacher)
        done=t.train(c,'cpu');assert events==['L','OE'] and done['updates']==done['open_signal_updates']==2
        for a,b in teacher_arrays:np.testing.assert_array_equal(a,b);assert not a.flags.writeable
        assert done['native_updates']==0 and done['signal_parameters']==334721 and done['optimizer']['physical_clip_calls']==2
        for row in (done['first_step'],done['second_step']):
            assert len(row['L_indices'])==192 and len(row['OE_indices'])==64 and row['control_indices_equal']
            assert row['sinc_lr']==pytest.approx(.1) and row['gradient_norms']['sinc']>0
            for k in ('time_frequency','PA','reference','signature','last'):assert row['gradient_norms'][k]>0
            for k in ('L','OE'):assert row['crossset_embedding_gradients'][k]['minimum_row_norm']>0
        assert done['first_step']['gradient_norms']['hidden']==0 and done['second_step']['gradient_norms']['hidden']>0
        model,identity=t.load_trained_model(c,Path(c['output_root'])/'final.pt','cpu')
        assert identity==done['identity'] and not model.training and all(not p.requires_grad for p in model.parameters())
        assert not any('cls_head.weight' in k for k in model.state_dict())
        with pytest.raises(FileExistsError):t.train(c,'cpu')
        cp=Path(c['output_root'])/'final.pt';value=torch.load(cp,weights_only=False);value['network'].pop('projection.network.2.bias');torch.save(value,cp)
        with pytest.raises(ValueError,match='Strict'):t.load_trained(c,'cpu')


def test_private_OE_labels_or_misaligned_IDs_fail_before_optimizer(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,m):
        load=t.load_role
        for alteration in ('labels','ids'):
            with subtests.test(alteration=alteration):
                def bad(root,role):
                    a=deepcopy(load(root,role))
                    if role=='OE':
                        if alteration=='labels':a['metadata']['y']=np.zeros(len(a['ids']),np.int64)
                        else:a['ids']=a['ids'][::-1]
                    return a
                monkeypatch.setattr(t,'load_role',bad)
                cc=dict(c,output_root=str(tmp_path/alteration),native_config=dict(c['native_config'],output_root=str(tmp_path/alteration)))
                with pytest.raises(ValueError):t.train(cc,'cpu')
                assert not Path(cc['output_root'],'complete.json').exists()


def test_control_sampling_mismatch_and_checkpoint_provenance_rejected(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,m):
        model,identity=frozen_checkpoint(c,m);assert identity['signal_feature_cumulative_updates']==13320
        loaded,ident=t.load_trained(c,'cpu');assert ident==identity
        done=t._read(Path(c['output_root'])/'complete.json');done['identity']['native_updates']=1
        Path(c['output_root'],'complete.json').write_text(json.dumps(done),encoding='utf-8')
        with pytest.raises(ValueError):t.load_trained(c,'cpu')
        monkeypatch.setattr(t,'STEPS',2);c=dict(c,steps=2,output_root=str(tmp_path/'mismatch'),native_config=dict(c['native_config'],output_root=str(tmp_path/'mismatch')))
        marker=t._read(Path(c['control_root'])/'complete.json');steps=Path(marker['config']['train_config']['output_root'])/'steps.jsonl'
        rows=steps.read_text().splitlines();r=json.loads(rows[0]);r['L_indices'][0]=-1;rows[0]=json.dumps(r);steps.write_text('\n'.join(rows)+'\n')
        with pytest.raises(ValueError,match='indices'):t.train(c,'cpu')
