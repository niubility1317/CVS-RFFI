from copy import deepcopy
from unittest.mock import patch
import unittest
import torch
from experiments.pamf_channel_pl import nosig_rgpl_base as m


def context():
    return dict(project='/project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
        cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='/old/segment20',
            source_config=dict(ablation='F21',num_classes=6),expected_source_contract='roles') for s in (2026092701,2026092702,2026092703)])


def cfg(method):
    old=dict(num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='cache',source_contract='roles',hard_pl=dict(weight=6))
    with patch.object(m,'read',return_value=dict(config=old)):return m.config(context(),'commit',method)


class NoSigBaseTests(unittest.TestCase):
    def test_controls_keep_full_L_and_explicit_U_permission(self):
        for method in m.METHODS:
            c=cfg(method);m.validate_config(c);self.assertEqual(c['U_use'],method=='NoSIG-PL');self.assertEqual(c['repl_row']['structure'],'NO_SIGNATURE')
            self.assertFalse(any(k.startswith('hard_pl') for k in c))
            for change in (dict(target_inputs='query'),dict(no_sigtime=False),dict(U_use=not c['U_use']),dict(default_head_joint=False),dict(output_root='/other'),dict(own_EMA=True)):
                with self.subTest(method=method,change=change),self.assertRaises(ValueError):m.validate_config(dict(c,**change))
            bad=deepcopy(c);bad['repl_row']['structure']='NO_TIME'
            with self.assertRaises(ValueError):m.validate_config(bad)

    def test_supervised_must_not_inherit_hidden_U_or_full_signature_prefix(self):
        c=cfg('NoSIG-Sup')
        with self.assertRaises(ValueError):m.validate_config(dict(c,shared_NoSIG_E20=m.prefix('/project')))
        c=cfg('NoSIG-PL')
        with self.assertRaises(ValueError):m.validate_config(dict(c,shared_NoSIG_E20='/old/full_SIG/segment20'))
        segment=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_NoSIG_E20'],output_root=c['output_root']+'/segment30')
        m.validate_config(segment)
        for change in (dict(parent_source='/full_SIG/segment20'),dict(epoch_offset=30),dict(segment_epochs=11),dict(step_offset=0)):
            with self.assertRaises(ValueError):m.validate_config(dict(segment,**change))

    def test_checkpoint_tensor_identity_rejects_remaining_signature_parameters(self):
        cp=dict(model={'id_backbone.encoder.core.id_backbone.cls_head.weight':torch.zeros(6,160)})
        m.check_state(cp)
        with self.assertRaises(ValueError):m.check_state(dict(model=dict(cp['model'],**{'id_backbone.encoder.sequence_pool.views.0.weight':torch.zeros(2)})))
        cp['model']['id_backbone.encoder.core.id_backbone.cls_head.weight']=torch.zeros(10,160)
        with self.assertRaises(ValueError):m.check_state(cp)


if __name__=='__main__':unittest.main()
