"""Behavioral/negative tests, source parity and explicit upstream defect checks."""
import ast
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from torch.nn import functional as F
import acceptance
import score
from runtime import (load_diagnostic_input, data_contract_path, expand_head,
                     load_module, verify_source, channel_spectrogram, validate_split)

ROOT=Path(__file__).resolve().parents[2]
ISSL=ROOT/'local_artifacts/external_refs/ISSL_20260930'
LORA=ROOT/'local_artifacts/external_refs/LoRa_RFFI_20260930'
PORT=ROOT/'local_artifacts/external_refs/LoRa_RFFI_Torch_20260930'

def fixture(path, **updates):
    fields=dict(x=np.random.default_rng(123).normal(size=(12,256,2)).astype(np.float32),
                y=np.tile(np.arange(6),2),train=np.arange(12)<6,ids=np.array([f'id{i}' for i in range(12)]))
    fields.update(updates); np.savez(path,**fields)
    return fields

class InputAndScoringTests(unittest.TestCase):
    def test_query_truth_cannot_change_training_or_predictor_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.npz'; b=Path(tmp)/'b.npz'; fields=fixture(a)
            changed=fields['y'].copy(); changed[~fields['train']]=999
            fixture(b,**(fields|{'y':changed}))
            for left,right in zip(load_diagnostic_input(a),load_diagnostic_input(b)):
                np.testing.assert_array_equal(left,right)

    def test_role_mask_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz'; fixture(path,train=np.arange(12)%2)
            with self.assertRaisesRegex(ValueError,'boolean'): load_diagnostic_input(path)

    def test_duplicate_ids_are_rejected_at_runtime_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz'; fixture(path,ids=np.array(['same']*12))
            with self.assertRaisesRegex(ValueError,'Duplicate'): load_diagnostic_input(path)

    def test_nonfinite_iq_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz'; x=np.zeros((12,256,2),np.float32); x[0,0,0]=np.nan
            fixture(path,x=x)
            with self.assertRaisesRegex(ValueError,'finite'): load_diagnostic_input(path)

    def test_empty_query_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz'; fixture(path,train=np.ones(12,dtype=bool))
            with self.assertRaisesRegex(ValueError,'empty'): load_diagnostic_input(path)

    def test_contract_path_points_to_existing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data.npz'; data.touch(); contract=data.with_suffix('.json')
            with self.assertRaises(FileNotFoundError): data_contract_path(data)
            contract.write_text('{}',encoding='utf-8')
            self.assertEqual(data_contract_path(data),contract)

    def test_output_collision_preserves_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel=Path(tmp)/'keep.txt'; sentinel.write_text('keep',encoding='utf-8')
            with mock.patch('sys.argv',['acceptance.py','--method','lora','--data','absent.npz','--output',tmp]):
                with self.assertRaises(FileExistsError): acceptance.main()
            self.assertEqual(sentinel.read_text(encoding='utf-8'),'keep')

    def score_fixture(self,tmp,pred_changes=None,method='issl'):
        data=Path(tmp)/'data.npz'; fields=fixture(data)
        out=Path(tmp)/'result'; out.mkdir()
        payload={'ids':fields['ids'][6:],'A':np.array([0,1,2,0,1,2]),'C':np.arange(6)}
        if method=='issl': payload['C_incremental_baseline']=np.arange(6)
        payload.update(pred_changes or {})
        np.savez(out/'predictions.npz',**payload)
        (out/'acceptance.json').write_text(json.dumps({'method':method,'status':'VERIFIED'}),encoding='utf-8')
        return data,out

    def test_scorer_rejects_invalid_old_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.score_fixture(tmp,{'A':np.array([9,1,2,0,1,2])})
            with mock.patch('sys.argv',['score.py','--data',str(data),'--output',str(out)]):
                with self.assertRaisesRegex(ValueError,'registered'): score.main()

    def test_scorer_rejects_permuted_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.score_fixture(tmp,{'ids':np.array([f'id{i}' for i in range(11,5,-1)])})
            with mock.patch('sys.argv',['score.py','--data',str(data),'--output',str(out)]):
                with self.assertRaisesRegex(ValueError,'ID'): score.main()

    def test_scorer_requires_complete_stage_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.score_fixture(tmp)
            with np.load(out/'predictions.npz') as pred:
                payload={k:pred[k] for k in pred.files if k!='C_incremental_baseline'}
            np.savez(out/'predictions.npz',**payload)
            with mock.patch('sys.argv',['score.py','--data',str(data),'--output',str(out)]):
                with self.assertRaisesRegex(ValueError,'stage'): score.main()

    def test_scorer_derives_K_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.score_fixture(tmp,method='lora')
            with mock.patch('sys.argv',['score.py','--data',str(data),'--output',str(out)]):
                with mock.patch('builtins.print'): score.main()
                metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
                self.assertEqual(metrics['K'],1); self.assertEqual(metrics['query_per_class'],1)
                self.assertEqual(metrics['C']['H'],1.)
                with self.assertRaises(FileExistsError): score.main()

class SourceBehaviorTests(unittest.TestCase):
    def test_pinned_sources_are_unchanged(self):
        for path,name in [(ISSL,'ISSL'),(LORA,'LoRa_RFFI'),(PORT,'LoRa_RFFI_Torch')]: verify_source(path,name)

    def test_source_lock_rejects_other_head_and_edits(self):
        with mock.patch('runtime.subprocess.check_output',side_effect=['wrong\n','']):
            with self.assertRaises(ValueError): verify_source(ISSL,'ISSL')
        with mock.patch('runtime.subprocess.check_output',side_effect=['f58eda144063c7f150fbb7c0d0f31aaccb9d3bbe\n',' M loss.py\n']):
            with self.assertRaises(ValueError): verify_source(ISSL,'ISSL')

    @unittest.skipUnless(torch.cuda.is_available(),'Author losses force CUDA0')
    def test_author_KD_is_detached_but_CE_reaches_student(self):
        loss=load_module(ISSL/'loss.py','source_kd_test')
        student=torch.randn(4,3,device='cuda',requires_grad=True)
        teacher=torch.randn_like(student,requires_grad=True)
        kd=loss.MultiClassCrossEntropy(student,teacher,2)
        expected=-(teacher.detach().div(2).softmax(1)*student.div(2).log_softmax(1)).sum(1).mean()
        self.assertTrue(torch.allclose(kd,expected))
        kd.backward(); self.assertIsNone(student.grad); self.assertIsNone(teacher.grad)
        F.cross_entropy(student,torch.tensor([0,1,2,0],device='cuda')).backward()
        self.assertGreater(float(student.grad.abs().sum()),0)

    @unittest.skipUnless(torch.cuda.is_available(),'Author losses force CUDA0')
    def test_author_queue_reshape_semantics_not_transpose(self):
        loss=load_module(ISSL/'loss.py','source_queue_test')
        q=F.normalize(torch.tensor([[1.,2.,3.],[3.,1.,2.]],device='cuda'),dim=1)
        k=q.clone(); queue=torch.arange(12.,device='cuda').view(4,3)
        actual=loss.loss_function2(q,k,queue,5)
        positive=(q*k).sum(1,keepdim=True)
        expected=F.cross_entropy(torch.cat((positive,q@queue.reshape(3,4)),1)/5,torch.zeros(2,dtype=torch.long,device='cuda'))
        transposed=F.cross_entropy(torch.cat((positive,q@queue.T),1)/5,torch.zeros(2,dtype=torch.long,device='cuda'))
        self.assertTrue(torch.allclose(actual,expected)); self.assertFalse(torch.allclose(actual,transposed))

    def test_crop_and_normalize_match_author_operations(self):
        x=np.random.default_rng(8).normal(size=(2,256)).astype(np.float32)
        ast_tree=ast.parse((ISSL/'augmentations.py').read_text(encoding='utf-8'))
        crop=next(n for n in ast_tree.body if isinstance(n,ast.ClassDef) and n.name=='RandomCrop')
        ns={'random':random}; exec(compile(ast.Module(body=[crop],type_ignores=[]),'author_crop','exec'),ns)
        random.seed(91); expected=ns['RandomCrop'](218)(x)
        expected=(expected-np.mean(expected))/np.std(expected)
        random.seed(91); actual=acceptance.batch_tensor(x.T[None],torch.device('cpu'),crop=218).numpy()[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)

    def test_stage_copy_clears_previous_grads_and_keeps_key_eval(self):
        net=load_module(ISSL/'resnet.py','issl_copy_test').resnet18(num_classes=3)
        net.eval(); net.fc.weight.grad=torch.ones_like(net.fc.weight)
        copied=copy.deepcopy(net); copied,_=expand_head(copied,3); key=copy.deepcopy(copied)
        self.assertIsNone(key.fc.weight.grad); self.assertFalse(key.training)
        copied.train(); self.assertTrue(copied.training); self.assertFalse(key.training)

    def test_two_expansions_preserve_all_existing_rows(self):
        class Net(torch.nn.Module):
            def __init__(self): super().__init__(); self.fc=torch.nn.Linear(4,2)
        net=Net(); net,_=expand_head(net,2); old=net.fc.weight.detach().clone()
        net,teacher=expand_head(net,2)
        self.assertTrue(torch.equal(net.fc.weight[:4],old)); self.assertEqual(teacher.fc.out_features,4)

    def test_original_length_preprocessing_matches_author(self):
        source=load_module(LORA/'Openset_RFFI_TIFS/dataset_preparation.py','author_spectrum_test')
        x=np.random.default_rng(29).normal(size=(3,8192,2))
        expected=source.ChannelIndSpectrogram().channel_ind_spectrogram(x[...,0]+1j*x[...,1])
        np.testing.assert_allclose(channel_spectrogram(x),expected,rtol=1e-5,atol=1e-5)

    def test_stft_rejects_zero_and_nonfinite_signals(self):
        for x in [np.zeros((2,256,2)),np.full((2,256,2),np.nan)]:
            with self.assertRaises(ValueError): channel_spectrogram(x,64,32)

    def test_triplet_formula_and_gradients(self):
        module=load_module(PORT/'Openset_RFFI/deep_learning_models.py','triplet_test')
        a=torch.tensor([[0.,0.],[1.,2.]],requires_grad=True)
        p=torch.tensor([[.2,0.],[3.,2.]],requires_grad=True)
        n=torch.tensor([[.1,0.],[1.,2.]],requires_grad=True)
        expected=torch.tensor([.04-.01+.1,4.+.1]).mean()
        value=module.triplet_loss(a,p,n,.1); self.assertTrue(torch.allclose(value,expected))
        value.backward(); self.assertIsNotNone(a.grad)

    def test_same_padding_and_embedding_shapes_for_both_lengths(self):
        module=load_module(PORT/'Openset_RFFI/deep_learning_models.py','padding_test')
        conv=module.SamePadConv2d(1,2,7,stride=2)
        self.assertEqual(tuple(conv(torch.ones(2,1,27,7)).shape),(2,2,14,4))
        for shape in [(26,6,1),(102,62,1)]:
            net=module.FeatureExtractor(shape); net.eval()
            with torch.no_grad(): result=net(torch.ones(2,*shape))
            self.assertEqual(tuple(result.shape),(2,512))
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(result,dim=1),torch.ones(2),atol=1e-5))

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
