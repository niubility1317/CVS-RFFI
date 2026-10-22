import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import torch

from experiments.cric_dura_comparison.supervised_physics import ARMS, validate_config
from experiments.cric_dura_comparison.supervised_extension import architecture_config, validate_supervised_source


class SupervisedPhysicsTests(unittest.TestCase):
    def config(self, arm='SIG_TIME'):
        return architecture_config(dict(protocol_id='F0',method_id='pamf_t1_e50',
            optimizer_updates=11100,num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3]),
            arm,2026092701,'/project','commit',{})

    def test_all_eight_arms_strict_supervised(self):
        for arm in ARMS:
            c=self.config(arm);validate_config(c)
            from experiments.cric_dura_comparison.f_train import arguments
            # Bootstrap activates the real immutable trainer below.
            a=arguments(c,'cpu')
            self.assertFalse(a.use_unlabeled);self.assertFalse(a.use_ema_teacher)
            self.assertEqual((a.lambda_u,a.lambda_ent),(0.,0.))
            self.assertEqual(a.label_smoothing,0.)

    def test_rejects_U_regime_and_G_LU(self):
        c=self.config();c['U_use']=True
        with self.assertRaises(ValueError):validate_config(c)
        with self.assertRaises(ValueError):
            validate_supervised_source(self.config(),{'raw':None,'DURA-LU':{'kind':'contract_matched','roles':'LU'}})
        with self.assertRaises(ValueError):
            validate_supervised_source({'regime':'native_cric','ablation':None},{'raw':None})

    def test_supervised_loader_never_constructs_U(self):
        from experiments.cric_dura_comparison import f_data
        dataset=type('Dummy',(),{'__len__':lambda self:128})()
        c=self.config();c.update(source_contract='contract',source_cache='cache')
        roles=[]
        def make(root,role,contract):
            if role=='U_s':raise AssertionError('U opened')
            roles.append(role);return dataset
        contract=dict(source_rxs=c['source_rxs'],source_days=c['source_days'],
            counts=dict(L_s=128,U_s=900,V=128),classes=list(range(6)))
        with patch.object(f_data,'read',return_value=contract),patch.object(f_data,'SourceDataset',side_effect=make):
            ctx=f_data.context(c)
        self.assertEqual(roles,['L_s','V']);self.assertIsNone(ctx['unlabeled_loader'])

    def test_architectures_roundtrip_and_real_gradients(self):
        from experiments.cric_dura_comparison.f_train import class_dimension,original_config,arguments
        from experiments.cric_dura_comparison.f_model import install
        from experiments.cvs_phase1_stack.runtime import installed
        models={}
        for arm in (None,)+ARMS:
            c=self.config(arm or 'SIG_TIME');c.pop('supervised_architecture',None)
            if arm:c['supervised_architecture']=arm
            torch.manual_seed(c['model_seed'])
            with class_dimension(c),installed(original_config(c['model_seed'])) as native:
                a=arguments(c,'cpu');a.input_len=256
                model=install(native.build_baseline_model(a,'cpu'),c)
            model.train();x=torch.randn(4,2,256)
            logits=model(x)
            self.assertEqual(tuple(logits.shape),(4,6));self.assertTrue(torch.isfinite(logits).all())
            torch.nn.functional.cross_entropy(logits,torch.arange(4)).backward()
            head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
            self.assertTrue(torch.isfinite(head.grad).all());self.assertGreater(float(head.grad.norm()),0)
            ref=model.id_backbone.encoder
            if arm not in ('NO_SIGNATURE',):
                projection=ref.sequence_pool.output_projection.weight
                self.assertTrue(torch.isfinite(projection.grad).all());self.assertGreater(float(projection.grad.norm()),0)
            state=copy.deepcopy(model.state_dict());model.load_state_dict(state,strict=True)
            model.eval()
            with torch.no_grad():
                one=model(x);two=model(x)
            self.assertTrue(torch.equal(one,two))
            models[arm]=dict(count=sum(p.numel() for p in model.parameters()),state=state)
        for arm in ('NO_TIME','COV_X3','SHUFFLE_TIME','WRONG_REFERENCE'):
            self.assertEqual(models[arm]['count'],models[None]['count'])

    def test_source_geometry_and_noop_intervention(self):
        from experiments.cric_dura_comparison.source_physics import pair_geometry,interventions
        x=torch.randn(4,2,256)
        self.assertTrue(torch.equal(interventions(x)[0][1],x))
        z=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        out=pair_geometry(z,torch.tensor([0,0,1,1]),torch.tensor([1,2,1,2]),torch.ones(4))
        self.assertEqual(out['same_TX_cross_RX_distance'],0.)
        self.assertEqual(out['different_TX_same_RX_distance'],1.)


def setUpModule():
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    root=Path(__file__).resolve().parents[1]/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
    activate(dict(frozen_sources={k:str(root/k) for k in ('F','AF','BASE','FOLD','G')}))
    numerical_policy()


if __name__=='__main__':unittest.main()
