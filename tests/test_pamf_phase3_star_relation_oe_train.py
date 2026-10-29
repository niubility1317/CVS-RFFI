"""Synthetic train provenance, paired sampling, native size and strict reload."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from tests import test_pamf_phase3_star_representation_train as fixtures

torch.set_num_threads(1)


class FakeModel(fixtures.FakeModel):
    def __init__(self):super().__init__();self._phase3_source_handle={'source_contract':'contract.json','split_seed':392005}


class RelationOETrainTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_relation_oe_train')
    def config(self,root,arm='relation_oe'):
        old=fixtures.StarRepresentationTrainTests().config(root,'oe');run='20261030-phase3-star-relation-oe-s2026092701-r01';out=str(root/arm)
        return dict(run_id=run,code_commit='a'*40,arm=arm,native_config=dict(old['native_config'],run_id=run,row_id=arm+'-s2026092701',output_root=out),
            L_root=old['L_root'],OE_root=old['OE_root'],teacher_root=str(root/'teacher'),output_root=out,steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
    def roles(self):return {r:fixtures.StarRepresentationTrainTests().role(r) for r in ('L','OE')}
    def teacher(self,root,roles,config):
        m=self.module();path=root/'teacher';path.mkdir();units=np.eye(160,dtype=np.float32)[np.arange(30)%6]
        np.savez(path/'OE.npz',ids=roles['OE']['ids'],unit_features=units)
        oldnative=dict(config['native_config'],run_id=m.TEACHER_RUN,code_commit=m.TEACHER_COMMIT,row_id='teacher-s2026092701',output_root=str(path))
        marker=dict(schema=m.TEACHER_SCHEMA,status=m.TEACHER_STATUS,run_id=m.TEACHER_RUN,code_commit=m.TEACHER_COMMIT,
            config=dict(run_id=m.TEACHER_RUN,code_commit=m.TEACHER_COMMIT,native_config=oldnative,L_root=config['L_root'],OE_root=config['OE_root'],output_root=str(path),batch_size=128),
            source_only=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,OE_truth_reads=0,
            encoder_updates=0,identity_head_updates=0,G_updates=0,G_forward_rows=0,model_buffers_unchanged=True,model_parameters=232993,
            source_L_count=30,source_OE_count=30,parent_F0_identity=roles['L']['manifest']['parent_F0_identity'],dura_identity=roles['L']['manifest']['dura_identity'],
            input_path='cached_P1_only',feature='unit_fused160',cache_handles={r:b['manifest'] for r,b in roles.items()})
        (path/'complete.json').write_text(json.dumps(marker));return path

    def test_fixed_permission_schema_and_no_output_overwrite(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));self.assertIs(m.validate_config(c),c);self.assertEqual(m._native_config(c)['arm'],'supervised_only')
            for key in ('V_root','novel_root','target_root','truth','checkpoint','aux_lr','view_seed'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'forbidden'}))
            for key in ('target_root','increment_checkpoint'):
                d=copy.deepcopy(c);d['native_config'][key]='forbidden'
                with self.subTest(native=key),self.assertRaises(ValueError):m.validate_config(d)
            for defect in ('steps','arm','output'):
                d=copy.deepcopy(c)
                if defect=='steps':d['steps']=222
                elif defect=='arm':d['arm']='joint'
                else:d['output_root']=d['teacher_root'];d['native_config']['output_root']=d['output_root']
                with self.subTest(defect=defect),self.assertRaises(ValueError):m.validate_config(d)

    def test_actual_fixed_teacher_identity_only_OE_units_and_source_alignment(self):
        m=self.module();roles=self.roles()
        with tempfile.TemporaryDirectory() as td,patch.object(m,'COUNTS',dict(L=30,OE=30)):
            root=Path(td);c=self.config(root);p=self.teacher(root,roles,c)
            data=m.load_teacher(p,roles['L'],roles['OE'],c);self.assertEqual(set(data),{'manifest','OE'})
            self.assertFalse((p/'L.npz').exists())
            for defect in ('code','role','ids'):
                with self.subTest(defect=defect):
                    original=json.loads((p/'complete.json').read_text());bad=copy.deepcopy(original)
                    if defect=='code':bad['code_commit']='b'*40
                    elif defect=='role':bad['cache_handles']['OE']['role']='V'
                    if defect=='ids':
                        altered=copy.deepcopy(roles['OE']);altered['ids'][0]='wrong'
                    else:(p/'complete.json').write_text(json.dumps(bad));altered=roles['OE']
                    with self.assertRaises(ValueError):m.load_teacher(p,roles['L'],altered,c)
                    (p/'complete.json').write_text(json.dumps(original))

    def test_paired_sampling_smoke_first_single_optimizer_frozen_buffers_and_reload(self):
        m=self.module();roles=self.roles();torch.manual_seed(81);initial=FakeModel().state_dict();traces=[]
        for arm in m.ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
                root=Path(td);c=self.config(root,arm);c['steps']=2;self.teacher(root,roles,c);model=FakeModel();model.load_state_dict(initial);before={k:v.clone() for k,v in model.named_buffers()};events=[]
                helper=fixtures.StarRepresentationTrainTests();write=m._write
                with patch.object(m,'STEPS',2),patch.object(m,'COUNTS',dict(L=30,OE=30)),patch.object(m.previous,'COUNTS',dict(L=30,OE=30)),\
                     patch.object(m.native,'validate_config',side_effect=lambda c:c),patch.object(m.native,'make_trainable_model',return_value=model),\
                     patch.object(m,'MODEL_PARAMETERS',sum(p.numel() for p in model.id_backbone.encoder.parameters())),patch.object(m,'_smoke',side_effect=lambda *a:events.append('smoke') or {'status':'PASS'}),\
                     patch.object(m,'load_L',side_effect=lambda *a:events.append('L') or roles['L']),patch.object(m,'load_OE',side_effect=lambda *a:events.append('OE') or roles['OE']),patch.object(m,'close_role'),\
                     patch.object(m,'_optimizer',side_effect=helper.build_optimizer),patch.object(m,'_clip',side_effect=helper.clip),patch.object(m,'_write',side_effect=lambda p,v:events.append(Path(p).name) or write(p,v)):
                    result=m.train(c,'cpu')
                    with self.assertRaises(FileExistsError):m.train(c,'cpu')
                self.assertEqual(events[:3],['smoke','smoke.json','L']);self.assertEqual(events[-1],'complete.json')
                self.assertEqual(result['optimizer_step_calls'],2);self.assertEqual(result['aux_parameters'],0);self.assertEqual(result['encoder_updates'],2)
                self.assertEqual(result['L_forward_samples'],256);self.assertEqual(result['OE_forward_samples'],256)
                rows=[json.loads(line) for line in (Path(c['output_root'])/'steps.jsonl').read_text().splitlines()];traces.append(rows)
                self.assertTrue(all(r['sinc_coordinate_gradient_norm']>0 and r['head_gradient_norm']>0 for r in rows))
                for k,v in before.items():torch.testing.assert_close(v,dict(model.named_buffers())[k],rtol=0,atol=0)
                saved=torch.load(Path(c['output_root'])/'final.pt',map_location='cpu',weights_only=False)
                self.assertNotIn('auxiliary',saved);self.assertNotIn('aux_optimizer',saved)
                with patch.object(m,'STEPS',2),patch.object(m.native,'validate_config',side_effect=lambda c:c),patch.object(m.native,'make_trainable_model',return_value=FakeModel()),\
                     patch.object(m,'MODEL_PARAMETERS',sum(p.numel() for p in model.id_backbone.encoder.parameters())):
                    loaded,config,payload=m.load_candidate(c['output_root'],'cpu')
                self.assertEqual(config,c);self.assertEqual(payload['identity']['checkpoint'],str(Path(c['output_root'])/'final.pt'))
                self.assertFalse(loaded.training);self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                for k,v in loaded.state_dict().items():torch.testing.assert_close(v,saved['model'][k],rtol=0,atol=0)
        for left,right in zip(*traces):self.assertEqual(left['L_indices'],right['L_indices']);self.assertEqual(left['OE_indices'],right['OE_indices'])

    def test_bad_teacher_preserves_partial_without_optimizer_or_forbidden_roles(self):
        m=self.module();roles=self.roles();helper=fixtures.StarRepresentationTrainTests()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=self.config(root);c['steps']=2;p=self.teacher(root,roles,c);d=json.loads((p/'complete.json').read_text());d['code_commit']='b'*40;(p/'complete.json').write_text(json.dumps(d));model=FakeModel()
            with patch.object(m,'STEPS',2),patch.object(m,'COUNTS',dict(L=30,OE=30)),patch.object(m.previous,'COUNTS',dict(L=30,OE=30)),\
                 patch.object(m.native,'validate_config',side_effect=lambda c:c),patch.object(m.native,'make_trainable_model',return_value=model),patch.object(m,'MODEL_PARAMETERS',sum(p.numel() for p in model.parameters())),\
                 patch.object(m,'_smoke',return_value={'status':'PASS'}),patch.object(m,'load_L',return_value=roles['L']),patch.object(m,'load_OE',return_value=roles['OE']),patch.object(m,'close_role'),\
                 patch.object(m,'_optimizer',side_effect=AssertionError('Teacher provenance before optimizer')):
                with self.assertRaises(ValueError):m.train(c,'cpu')
            self.assertTrue((Path(c['output_root'])/'smoke.json').exists());self.assertFalse((Path(c['output_root'])/'complete.json').exists())

    def test_train_cli_requires_exact_action_and_forwards_device(self):
        m=self.module();config={'source':'synthetic'}
        with patch.object(sys,'argv',['trainer','train','--config','config.json','--device','cpu']),patch.object(m.runner,'_read',return_value=config),patch.object(m,'train') as call:
            m.main()
        call.assert_called_once_with(config,'cpu')
        with patch.object(sys,'argv',['trainer','--config','config.json']),self.assertRaises(SystemExit):m.main()

    def test_native_encoder_count_and_old_candidate_rejection(self):
        m=self.module();root=Path(__file__).resolve().parents[1];c=self.config(root);cfg=m._native_config(c);cfg['base_release']=str(root)
        frozen=root/'local_artifacts/cric_dura_comparison_20261021/frozen_sources';cfg['frozen_sources']={k:str(frozen/k) for k in ('F','AF','BASE','FOLD','G')};m.runner._activate(cfg)
        parent=dict(run_id='synthetic-parent',row_id='synthetic-parent',model_seed=2026092701,method_id='pamf_t1_e50',source_rxs=[1,3,4,6,8],source_days=[1,2,3],num_classes=6,
            view_comparison={'view_count':3},paper_method='synthetic graph only',repl_row={'group':'main','structure':'SHUFFLE_TIME','L_A':True,'L_B':True,'U':True})
        with patch('torch.load',side_effect=AssertionError('No real checkpoint/IQ')):model=m.native._new_native(cfg,m.native._training_config(cfg,parent),'cpu')
        self.assertEqual(sum(p.numel() for p in model.parameters()),900722);self.assertEqual(sum(p.numel() for p in model.id_backbone.encoder.parameters()),232993)
        self.assertIs(m._model(model),model);model.id_backbone.encoder.register_parameter('wrong_count',nn.Parameter(torch.zeros(1)))
        with self.assertRaises(ValueError):m._model(model)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);(path/'resolved_config.json').write_text(json.dumps(self.config(path)))
            with patch('torch.load',return_value={'schema':'old-joint-OVA'}),patch.object(m.native,'make_trainable_model',side_effect=AssertionError('Reject before constructor')):
                with self.assertRaises(ValueError):m.load_candidate(path,'cpu')


if __name__=='__main__':unittest.main()
