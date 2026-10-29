"""Focused algorithm/interface/protocol checks; no target data or truth access."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_phase2_practical import external_additional as e


def bundle(profile):
    model=e.NativeModel(profile).eval()
    return dict(profile=profile,model=model,features_fn=e.native_features,score_fn=e.native_logits,
        classes=tuple(range(6)),native_state={},resources={})


class AdditionalBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_native_budgets_and_unavailable_M10_are_explicit(self):
        s=e.profiles()
        self.assertEqual((s['issl_fixed_v1']['source_epochs'],s['issl_fixed_v1']['C_SSL_epochs'],s['issl_fixed_v1']['C_transfer_epochs']),(300,2000,300))
        self.assertEqual(s['lora_triplet_knn']['source_epochs'],1000)
        self.assertEqual((s['lora_triplet_knn']['source_early_patience'],s['lora_triplet_knn']['source_lr_patience']),(20,10))
        self.assertEqual(s['feature_separation_hu2024']['B_epochs'],100)
        self.assertEqual(s['ra_collab_single_ft']['source_epochs'],100)
        self.assertTrue(s['tweak_m10']['blocked'])
        with self.assertRaisesRegex(ValueError,'track'):
            e.load_source({},'tweak_m10',1,'cpu')
        self.assertTrue(all(not v.get('source_U',False) for v in s.values()))

    def test_lora_spectrogram_matches_scipy_author_operations(self):
        from scipy import signal
        x=np.random.default_rng(17).normal(size=(3,2,256)).astype('float32')
        z=x[:,0]+1j*x[:,1];z=z/np.sqrt(np.mean(np.abs(z)**2,axis=1,keepdims=True))
        _,_,s=signal.stft(z,window='boxcar',nperseg=64,noverlap=32,nfft=64,
            return_onesided=False,padded=False,boundary=None)
        s=np.fft.fftshift(s,axes=1)
        ref=np.log10(np.abs(s[:,:,1:]/s[:,:,:-1])**2)[:,round(64*.3):round(64*.7)]
        actual=e.channel_spectrogram(torch.tensor(x)).numpy()[:,0]
        np.testing.assert_allclose(actual,ref,atol=8e-5,rtol=8e-5)
        self.assertEqual(actual.shape,(3,26,6))
        with self.assertRaisesRegex(ValueError,'Zero IQ'):
            e.channel_spectrogram(torch.zeros(1,2,256))

    def test_each_native_encoder_is_per_sample_frozen_and_callback_uses_model(self):
        x=torch.randn(2,2,256)
        names=('issl_fixed_v1','issl_author_behavior','lora_triplet_knn','poster_ft',
            'radionet_df_ft','radionet_triplet','feature_separation_hu2024','ra_collab_single_ft','tweak_m1')
        for name in names:
            with self.subTest(name=name):
                b=bundle(name);m=b['model'];before=copy.deepcopy(m.state_dict())
                with torch.no_grad():
                    together=b['features_fn'](m,x)
                    separate=torch.cat([b['features_fn'](m,row[None]) for row in x])
                torch.testing.assert_close(together,separate,atol=2e-5,rtol=2e-4)
                self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        b=bundle('poster_ft');other=copy.deepcopy(b['model'])
        with torch.no_grad():other.network.head.bias.add_(3)
        torch.testing.assert_close(b['score_fn'](other,x)-b['score_fn'](b['model'],x),torch.full((2,6),3.))

    def test_native_head_expansion_preserves_B_and_only_final_affine_is_trainable(self):
        sx=torch.randn(8,2,256);sy=torch.arange(8)
        for name in ('poster_ft','radionet_df_ft','feature_separation_hu2024','ra_collab_single_ft'):
            with self.subTest(name=name):
                b=bundle(name);before=copy.deepcopy(b['model'].state_dict());old=e._head(b['model'])
                result=e.expand_head(b,sx,sy,8,31);m=result['model'];head=e._head(m)
                torch.testing.assert_close(head.weight[:6],old.weight,rtol=0,atol=0)
                torch.testing.assert_close(head.bias[:6],old.bias,rtol=0,atol=0)
                self.assertEqual(head.out_features,8)
                self.assertEqual({id(v) for v in result['trainable_parameters']},{id(v) for v in head.parameters()})
                self.assertEqual({id(v) for v in m.parameters() if v.requires_grad},{id(v) for v in head.parameters()})
                self.assertTrue(all(torch.equal(v,b['model'].state_dict()[k]) for k,v in before.items()))
                changed={k for k,v in before.items() if k not in m.state_dict() or v.shape!=m.state_dict()[k].shape or not torch.equal(v,m.state_dict()[k])}
                self.assertEqual(len(changed),2)
                self.assertTrue(torch.isfinite(result['logits_fn'](m,sx)).all())

    def test_B_keeps_actual_target_budgets_and_method_update_scope(self):
        sx=torch.randn(6,2,256);sy=torch.arange(6)
        for name,epochs in [('poster_ft',10),('radionet_df_ft',30),('feature_separation_hu2024',100),('ra_collab_single_ft',20),('issl_fixed_v1',10)]:
            b=bundle(name);before=copy.deepcopy(b['model'].state_dict())
            with patch.object(e,'_train_ce') as train:
                state=e.fit_B({},b,sx,sy,19)
            self.assertEqual(train.call_args.args[3],epochs)
            self.assertEqual(state['classes'],tuple(range(6)))
            self.assertTrue(torch.equal(state['native_state']['support_x'],sx))
            self.assertTrue(all(torch.equal(v,b['model'].state_dict()[k]) for k,v in before.items()))
            if name=='feature_separation_hu2024':
                self.assertTrue(all(not p.requires_grad for p in state['model'].network.rx_branch.parameters()))
                self.assertTrue(all(not p.requires_grad for p in state['model'].network.rx_classifier.parameters()))
            if name=='poster_ft':
                self.assertFalse(state['model'].network.conv1.weight.requires_grad)
                self.assertTrue(state['model'].network.dense1.weight.requires_grad)

    def test_Tweak_radius_rule_matches_native_algorithm_and_C_appends(self):
        b=dict(profile='tweak_m1',model=torch.nn.Linear(1,1),resources={},native_state={})
        # Class 0: centroid 0/radius 1. Class 1: centroid 10/radius 8.
        features=torch.tensor([[-1.],[1.],[2.],[18.]])
        labels=torch.tensor([0,0,1,1])
        with patch.object(e,'native_features',lambda model,x:x*model.weight):
            with torch.no_grad():b['model'].weight.fill_(1)
            state=e._metric_state(b,features,labels)
            scores=state['score_fn'](b['model'],torch.tensor([[2.5],[-4.]]))
            self.assertEqual(scores.argmax(1).tolist(),[1,0])
            clone=copy.deepcopy(b['model'])
            with torch.no_grad():clone.weight.fill_(10)
            self.assertFalse(torch.equal(scores,state['score_fn'](clone,torch.tensor([[2.5],[-4.]]))))
        b=bundle('tweak_m1');sx=torch.randn(6,2,256);sy=torch.arange(6)
        state=e.fit_B({},b,sx,sy,13)
        c=e.native_C({},b,state,torch.randn(4,2,256),torch.tensor([6,6,7,7]),[6,7],14)
        torch.testing.assert_close(c['native_state']['centroids'][:6],state['native_state']['centroids'],rtol=0,atol=0)
        torch.testing.assert_close(c['native_state']['radii'][:6],state['native_state']['radii'],rtol=0,atol=0)
        self.assertEqual(c['classes'],tuple(range(8)))
        self.assertEqual(len(c['native_state']['enrollment_y']),10)

    def test_lora_native_uniform15_smallK_rule_and_stage_isolation(self):
        b=bundle('lora_triplet_knn');sx=torch.randn(6,2,256);sy=torch.arange(6)
        state=e.fit_B({},b,sx,sy,5)
        self.assertEqual(state['resources']['knn_k'],6)
        with torch.no_grad():scores=state['score_fn'](state['model'],torch.randn(2,2,256))
        torch.testing.assert_close(scores,torch.ones(2,6)/6)
        c=e.native_C({},b,state,torch.randn(4,2,256),torch.tensor([6,6,7,7]),[6,7],6)
        self.assertEqual(c['classes'],tuple(range(8)))
        self.assertEqual(len(c['native_state']['enrollment_y']),10)
        with self.assertRaisesRegex(ValueError,'New class IDs'):
            e.native_C({},b,state,torch.randn(4,2,256),torch.tensor([6,6,7,7]),[6,6,7,7],6)
        with self.assertRaisesRegex(ValueError,'exactly'):
            e.fit_B({},b,sx[:5],sy[:5],5)
        with self.assertRaisesRegex(ValueError,'contiguous'):
            e.native_C({},b,state,torch.randn(1,2,256),torch.tensor([7]),['x'],5)

    def test_radionet_native_mean_signature_distance_knn19(self):
        b=bundle('radionet_triplet')
        state=e._metric_state(b,torch.randn(26,64),torch.arange(26))
        with torch.no_grad():scores=state['score_fn'](state['model'],torch.randn(2,2,256))
        self.assertEqual(tuple(scores.shape),(2,26))
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(scores.sum(1),torch.ones(2))
        self.assertTrue(bool(((scores>0).sum(1)<=19).all()))

    def test_ISSL_C_inherits_B_and_keeps_full_SSL_transfer_configuration(self):
        b=bundle('issl_fixed_v1');b['native_state'].update(source_replay_x=torch.randn(6,2,256),source_replay_y=torch.arange(6))
        with patch.object(e,'_train_ce'):
            state=e.fit_B({},b,torch.randn(6,2,256),torch.arange(6),1)
        before=e._head(state['model']).weight.detach().clone()
        with patch.object(e,'_issl_ssl') as ssl,patch.object(e,'_train_ce') as transfer:
            c=e.native_C({},b,state,torch.randn(4,2,256),torch.tensor([6,6,7,7]),[6,7],2)
        self.assertEqual(ssl.call_args.args[3]['C_SSL_epochs'],2000)
        self.assertEqual(transfer.call_args.args[3],300)
        self.assertEqual(len(ssl.call_args.args[1]),22)  # L_s twice + B/current C support.
        self.assertEqual(len(transfer.call_args.args[1]),16)
        torch.testing.assert_close(e._head(c['model']).weight[:6],before,rtol=0,atol=0)
        self.assertIsNone(e._head(c['model']).bias)

    def test_legacy_source_reuse_checks_real_native_state_contract_and_no_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=dict(role_ids={'L_s':['l'],'U_s':['u'],'V':['v']},source_rxs=[1,3,4,6,8],
                source_days=[1,2,3],counts={'L_s':6300,'U_s':56700,'V':27000},classes=list('abcdef'),
                ratios=[.07,.63,.3],split_seed=392005,physical_roles='EXACT_MATCH',equalized=1,out_len=256,normalize=True)
            def write(name,value):(root/name).write_text(json.dumps(value),encoding='utf-8')
            write('source_contract.json',c);write('initialization.json',dict(scratch_only=True,checkpoint_sources=[],target_contact=False,model_seed=7))
            write('completion.json',dict(status='SOURCE_TRAINED',epoch=200,target_evaluated=False))
            write('resolved_config.json',dict(method='poster',model_seed=7,options={'--epochs':200,'--author_backbone':'poster','--source_only':None}))
            m=e.NativeModel('poster_ft').eval();torch.save(dict(epoch=200,model=m.network.state_dict()),root/'last.pt')
            rt=dict(source_contract=str(root/'source_contract.json'),additional_sources={'poster_ft':{'7':str(root)}})
            loaded=e.load_source(rt,'poster_ft',7,'cpu')
            self.assertTrue(loaded['resources']['source_reused'])
            self.assertTrue(all(torch.equal(v,loaded['model'].state_dict()[k]) for k,v in m.state_dict().items()))
            write('initialization.json',dict(scratch_only=True,checkpoint_sources=[],target_contact=True,model_seed=7))
            with self.assertRaisesRegex(ValueError,'PROVENANCE'):
                e.load_source(rt,'poster_ft',7,'cpu')
            changed=copy.deepcopy(c);changed['role_ids']['L_s']=['different']
            with self.assertRaisesRegex(ValueError,'DATA_CONTRACT'):
                e._contract_match(changed,c)


if __name__=='__main__':
    unittest.main()
