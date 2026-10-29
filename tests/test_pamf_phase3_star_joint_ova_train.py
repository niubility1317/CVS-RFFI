"""Joint OVA orchestration tests without real IQ, checkpoint or GPU access."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import test_pamf_phase3_star_representation_train as fixtures
class FakeModel(fixtures.FakeModel):
    def __init__(self):
        super().__init__();self._phase3_source_handle={'source_contract':'contract.json','split_seed':392005}


class StarJointOvaTrainTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_ova_train')
    def config(self,root,arm='joint'):
        old=fixtures.StarRepresentationTrainTests().config(root,'oe');run='20261029-phase3-star-joint-ova-s2026092701-r01'
        output=str(root/arm);native=dict(old['native_config'],run_id=run,row_id=arm+'-s2026092701',output_root=output)
        return dict(run_id=run,code_commit='a'*40,arm=arm,native_config=native,L_root=old['L_root'],OE_root=old['OE_root'],
            teacher_root=str(root/'teacher'),output_root=output,steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
    def export_config(self,root):
        c=self.config(root);out=str(root/'teacher')
        return {k:c[k] for k in ('run_id','code_commit','L_root','OE_root')}|dict(
            native_config=dict(c['native_config'],row_id='teacher-s2026092701',output_root=out),output_root=out,batch_size=128)
    def roles(self):return {k:fixtures.StarRepresentationTrainTests().role(k) for k in ('L','OE')}
    def teacher(self,root,roles):
        m=self.module();c=self.export_config(root);output=Path(c['output_root']);output.mkdir()
        for role,b in roles.items():
            units=np.zeros((len(b['ids']),160),np.float32);units[:,0]=1
            m._npz(output/(role+'.npz'),dict(ids=b['ids'],unit_features=units))
        m._write(output/'complete.json',dict(schema=m.TEACHER_SCHEMA,status=m.TEACHER_STATUS,config=c,
            source_only=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,
            source_L_count=30,source_OE_count=30,parent_F0_identity=roles['L']['manifest']['parent_F0_identity'],
            dura_identity=roles['L']['manifest']['dura_identity'],cache_handles={k:v['manifest'] for k,v in roles.items()}))
        return output
    def optimizer(self,model,params,**kw):return fixtures.StarRepresentationTrainTests().build_optimizer(model,params,**kw)
    def clip(self,opt,params):return fixtures.StarRepresentationTrainTests().clip(opt,params)

    def test_fixed_configs_forbid_V_target_truth_resume_and_cross_role_output(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=self.config(root);self.assertIs(m.validate_config(c),c)
            e=self.export_config(root);self.assertIs(m.validate_export_config(e),e)
            self.assertEqual(m._native_config(c)['arm'],'supervised_only')
            for key in ('V_root','novel_root','target_root','truth','checkpoint','view_seed'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            for key in ('truth','increment_checkpoint','V_root'):
                bad=copy.deepcopy(c);bad['native_config'][key]='x'
                with self.subTest(native=key),self.assertRaises(ValueError):m.validate_config(bad)
            for defect in ('arm','steps','output'):
                bad=copy.deepcopy(c)
                if defect=='arm':bad['arm']='anchor'
                elif defect=='steps':bad['steps']=222
                else:bad['output_root']=bad['L_root'];bad['native_config']['output_root']=bad['output_root']
                with self.subTest(defect=defect),self.assertRaises(ValueError):m.validate_config(bad)

    def test_teacher_export_smoke_first_P1_only_unit_features_and_marker_last(self):
        m=self.module();roles=self.roles();events=[];reader=FakeModel().eval().id_backbone.encoder
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=self.export_config(root);writes=[];write=m._write
            def load(path):
                name=Path(path).name;events.append(name);return roles[name]
            with patch.object(m,'COUNTS',dict(L=30,OE=30)),patch.object(m.previous,'COUNTS',dict(L=30,OE=30)),\
                 patch.object(m,'_teacher_backend',return_value=({'reader':reader,'reader_identity':roles['L']['manifest']['parent_F0_identity']},{})),\
                 patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),\
                 patch.object(m,'_teacher_smoke',side_effect=lambda *a:events.append('smoke') or dict(status='PASS')),\
                 patch.object(m,'load_L',side_effect=load),patch.object(m,'load_OE',side_effect=load),patch.object(m,'close_role'),\
                 patch.object(m,'_write',side_effect=lambda p,v:writes.append(Path(p).name) or write(p,v)):
                done=m.export(c,'cpu')
            self.assertEqual(events,['smoke','L','OE']);self.assertEqual(writes[-1],'complete.json')
            self.assertEqual(done['F0_feature_rows'],60);self.assertEqual(done['G_forward_rows'],0)
            for name in ('L','OE'):
                with np.load(Path(c['output_root'])/(name+'.npz')) as z:
                    self.assertEqual(set(z.files),{'ids','unit_features'});self.assertEqual(z['unit_features'].dtype,np.float32)
                    np.testing.assert_allclose(np.linalg.norm(z['unit_features'],axis=1),1,atol=1e-6)
            with self.assertRaises(FileExistsError):m.export(c,'cpu')

    def test_teacher_ID_or_parent_mismatch_rejected_and_never_opens_other_roles(self):
        m=self.module();roles=self.roles()
        with tempfile.TemporaryDirectory() as td,patch.object(m,'COUNTS',dict(L=30,OE=30)):
            root=Path(td);c=self.config(root);teacher=self.teacher(root,roles)
            data=m.load_teacher(teacher,roles['L'],roles['OE'],c);self.assertEqual(set(data),{'L','OE','manifest'})
            bad=copy.deepcopy(roles['OE']);bad['ids'][0]='wrong'
            with self.assertRaises(ValueError):m.load_teacher(teacher,roles['L'],bad,c)
            marker=json.loads((teacher/'complete.json').read_text());marker['parent_F0_identity']['epoch']=51
            (teacher/'complete.json').write_text(json.dumps(marker),encoding='utf-8')
            with self.assertRaises(ValueError):m.load_teacher(teacher,roles['L'],roles['OE'],c)

    def test_same_sampling_single_concat_forward_two_optimizers_eval_buffers_and_codec(self):
        m=self.module();roles=self.roles();torch.manual_seed(63);initial=FakeModel().state_dict();indices=[]
        for arm in m.ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
                root=Path(td);c=self.config(root,arm);c['steps']=2;self.teacher(root,roles);events=[]
                model=FakeModel();model.load_state_dict(initial);before=model.id_backbone.encoder.bn.running_mean.clone()
                write=m._write;seen=[];actual=m.forward
                def forward(model,x):seen.append(x.detach().clone());return actual(model,x)
                with patch.object(m,'STEPS',2),patch.object(m,'COUNTS',dict(L=30,OE=30)),patch.object(m.previous,'COUNTS',dict(L=30,OE=30)),\
                     patch.object(m.native,'validate_config',side_effect=lambda c:c),patch.object(m.native,'make_trainable_model',return_value=model),\
                     patch.object(m,'MODEL_PARAMETERS',sum(p.numel() for p in model.parameters())),\
                     patch.object(m,'_smoke',side_effect=lambda *a:events.append('smoke') or dict(status='PASS')),\
                     patch.object(m,'load_L',side_effect=lambda *a:events.append('L') or roles['L']),\
                     patch.object(m,'load_OE',side_effect=lambda *a:events.append('OE') or roles['OE']),patch.object(m,'close_role'),\
                     patch.object(m,'_optimizer',side_effect=self.optimizer),patch.object(m,'_clip',side_effect=self.clip),\
                     patch.object(m,'forward',side_effect=forward),\
                     patch.object(m,'_write',side_effect=lambda p,v:events.append(Path(p).name) or write(p,v)):
                    done=m.run(c,'cpu')
                    with self.assertRaises(FileExistsError):m.run(c,'cpu')
                self.assertEqual(events[:3],['smoke','smoke.json','L']);self.assertEqual(events[-1],'complete.json')
                self.assertEqual([tuple(x.shape) for x in seen],[(256,2,256)]*2);indices.append(seen)
                self.assertEqual(done['encoder_updates'],2);self.assertEqual(done['auxiliary_updates'],2)
                self.assertEqual(done['optimizer_step_calls'],4);self.assertEqual(done['F_forward_calls'],2)
                self.assertEqual(done['L_forward_samples'],256);self.assertEqual(done['OE_forward_samples'],256)
                rows=[json.loads(line) for line in (Path(c['output_root'])/'steps.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['sinc_coordinate_gradient_norm']>0 and r['head_gradient_norm']>0 for r in rows))
                torch.testing.assert_close(model.id_backbone.encoder.bn.running_mean,before)
                payload=torch.load(Path(c['output_root'])/'final.pt',weights_only=False)
                self.assertEqual(set(payload['auxiliary']),{'weight','bias'});self.assertIn('F_optimizer',payload);self.assertIn('aux_optimizer',payload)
                self.assertEqual(payload['source_data_handle'],model._phase3_source_handle)
                with patch.object(m,'STEPS',2),patch.object(m.native,'validate_config',side_effect=lambda c:c),\
                     patch.object(m.native,'make_trainable_model',return_value=FakeModel()),\
                     patch.object(m,'MODEL_PARAMETERS',sum(p.numel() for p in model.parameters())):
                    loaded,aux,identity=m.load_trained_model(c,Path(c['output_root'])/'final.pt','cpu')
                self.assertFalse(loaded.training);self.assertFalse(aux.training);self.assertEqual(identity['arm'],arm)
                self.assertEqual(identity['original_parent_identity'],roles['L']['manifest']['parent_F0_identity'])
                self.assertTrue(all(not p.requires_grad for p in list(loaded.parameters())+list(aux.parameters())))
        for left,right in zip(*indices):torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_teacher_or_training_failure_leaves_no_complete_and_never_uses_forbidden_roles(self):
        m=self.module();roles=self.roles()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c=self.config(root);c['steps']=2;teacher=self.teacher(root,roles)
            with np.load(teacher/'OE.npz') as z:ids=z['ids'].copy();unit=z['unit_features'].copy()
            ids[0]='wrong';np.savez(teacher/'OE.npz',ids=ids,unit_features=unit)
            model=FakeModel()
            with patch.object(m,'STEPS',2),patch.object(m,'COUNTS',dict(L=30,OE=30)),patch.object(m.previous,'COUNTS',dict(L=30,OE=30)),\
                 patch.object(m.native,'validate_config',side_effect=lambda c:c),patch.object(m.native,'make_trainable_model',return_value=model),\
                 patch.object(m,'MODEL_PARAMETERS',sum(p.numel() for p in model.parameters())),patch.object(m,'_smoke',return_value=dict(status='PASS')),\
                 patch.object(m,'load_L',return_value=roles['L']),patch.object(m,'load_OE',return_value=roles['OE']),patch.object(m,'close_role'),\
                 patch.object(m,'_optimizer',side_effect=AssertionError('No optimizer before teacher binding')),self.assertRaises(ValueError):m.run(c,'cpu')
            self.assertFalse((Path(c['output_root'])/'complete.json').exists());self.assertFalse((Path(c['output_root'])/'final.pt').exists())

    def test_initial_aux_seed_and_old_increment_checkpoint_rejection_before_constructor(self):
        m=self.module();rng=torch.get_rng_state().clone();a=m._auxiliary('cpu');b=m._auxiliary('cpu')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));torch.testing.assert_close(a.weight,b.weight,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()),966);self.assertTrue(torch.equal(a.bias,torch.zeros(6)))
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with patch('torch.load',return_value={'schema':'oldE51'}),\
                 patch.object(m.native,'make_trainable_model',side_effect=AssertionError('No old state initialization')),self.assertRaises(ValueError):
                m.load_trained_model(c,'old.pt','cpu')


if __name__=='__main__':unittest.main()
