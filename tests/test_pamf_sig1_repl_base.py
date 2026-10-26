import unittest
from types import SimpleNamespace
import torch
from torch import nn
from copy import deepcopy
from unittest.mock import patch
import sys
# Pool unit tests use the actual descriptor code. The unused native backbone
# constructor needs the N607 frozen runtime and is checked by the real smoke.
with patch.dict(sys.modules,{'experiments.cvs_reference_identity.model':SimpleNamespace(ReferenceCVS=nn.Module),
                            'experiments.cvs_e20_cross_style.style':SimpleNamespace(_paths=None)}):
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
    pool_module=sys.modules['experiments.cvs_rdrop_sinc_units.x3_features']
from experiments.pamf_channel_pl.sig1_model import install_single_view,validate_single_view
from experiments.pamf_channel_pl import sig1_repl_base as base
from experiments.pamf_channel_pl.sig1_repl_base import check_state,validate_config


def config(method):
    context=dict(project='/project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
        cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='/old/segment20',
            source_config=dict(ablation='F21',num_classes=6),expected_source_contract='roles') for s in (2026092701,2026092702,2026092703)])
    old=dict(num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='cache',source_contract='roles',hard_pl=dict(weight=6))
    with patch.object(base,'read',return_value=dict(config=old)):return base.config(context,'commit',method)


class SingleViewTests(unittest.TestCase):
    def setUp(self):
        active=patch.dict(sys.modules,{'experiments.cvs_rdrop_sinc_units.x3_features':pool_module})
        active.start();self.addCleanup(active.stop)

    def model(self):
        core=SimpleNamespace(id_backbone=SimpleNamespace(cls_head=nn.Linear(160,6,bias=False)))
        ref=SimpleNamespace(sequence_pool=CompositePool(96,'SIG_TIME_X3',2026092701),core=core)
        return SimpleNamespace(id_backbone=SimpleNamespace(encoder=ref),_feature_learning_report={})

    def test_canonical_single_view_shape_parameter_and_gradient(self):
        model=self.model();head=model.id_backbone.encoder.core.id_backbone.cls_head
        install_single_view(model,2026092701);pool=validate_single_view(model)
        self.assertIs(head,model.id_backbone.encoder.core.id_backbone.cls_head)
        x=torch.randn(3,96,32,requires_grad=True);z=pool(x)
        self.assertEqual(tuple(z.shape),(3,160));self.assertEqual(sum(p.numel() for p in pool.parameters()),18976)
        head(z).square().mean().backward()
        self.assertGreater(float(x.grad.norm()),0);self.assertGreater(float(pool.views[0].local_projection.weight.grad.norm()),0)

    def test_x3_and_missing_signature_checkpoints_rejected(self):
        model=self.model();pool=model.id_backbone.encoder.sequence_pool
        def state(pool):
            return dict(model={**{'id_backbone.encoder.sequence_pool.'+k:v for k,v in pool.state_dict().items()},
                'id_backbone.encoder.core.id_backbone.cls_head.weight':torch.zeros(6,160)})
        with self.assertRaises(ValueError):check_state(state(pool))
        install_single_view(model,2026092701);check_state(state(model.id_backbone.encoder.sequence_pool))
        with self.assertRaises((ValueError,KeyError)):check_state(dict(model={'id_backbone.encoder.core.id_backbone.cls_head.weight':torch.zeros(6,160)}))

    def test_foreign_or_target_config_rejected(self):
        for method in base.METHODS:
            c=config(method);validate_config(c)
            for change in (dict(target_inputs='query'),dict(sig_time_views=3),dict(sig_time_views=0),dict(U_use=not c['U_use']),dict(default_head_joint=False),dict(output_root='/other'),dict(own_EMA=True)):
                with self.subTest(method=method,change=change),self.assertRaises(ValueError):validate_config(dict(c,**change))
            bad=deepcopy(c);bad['repl_row']['structure']='NO_SIGNATURE'
            with self.assertRaises(ValueError):validate_config(bad)

    def test_supervised_cannot_inherit_U_and_PL_uses_own_single_view(self):
        c=config('SIG1-Sup')
        with self.assertRaises(ValueError):validate_config(dict(c,shared_SIG1_E20=base.prefix('/project')))
        c=config('SIG1-PL')
        with self.assertRaises(ValueError):validate_config(dict(c,shared_SIG1_E20='/old/x3/segment20'))
        segment=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_SIG1_E20'],output_root=c['output_root']+'/segment30')
        validate_config(segment)
        for change in (dict(parent_source='/old/NoSIG/segment20'),dict(segment_epochs=11),dict(step_offset=0)):
            with self.assertRaises(ValueError):validate_config(dict(segment,**change))


if __name__=='__main__':unittest.main()
