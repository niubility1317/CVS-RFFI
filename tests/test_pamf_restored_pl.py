import copy
import unittest
import numpy as np
import torch
from experiments.pamf_channel_pl.plan import PROTOCOLS,SEEDS
from experiments.pamf_channel_pl.restored_dose_source import RECIPE
from experiments.pamf_restored_pl.plan import rows,make_config,validate_config,source_decision
from experiments.pamf_restored_pl.objective import ramp_weight
from experiments.pamf_restored_pl.cache import SourceCache


def context():
    return dict(cases=[dict(row_id=p+'-ordinary_PL-s'+str(s),protocol=p,model_seed=s,
        parent_source='prefix/'+p+'/s'+str(s)+'/segment20',expected_source_contract='roles/'+p,
        source_config=dict(ablation='F21',num_classes=10 if p=='ManyRx' else 6,model_seed=s,run_id='parent',row_id='parent'+str(s),
            source_contract='roles/'+p,source_cache='source/'+p,source_rxs=[1,3,4,6,8],source_days=[1,2,3]))
        for p in PROTOCOLS for s in SEEDS])


class FixedSourceTests(unittest.TestCase):
    def test_nine_rows_keep_original_main_and_reject_target_head_budget_changes(self):
        self.assertEqual(len(rows()),9)
        for row in rows():
            cfg=make_config(context(),row,'project','commit',.126,'cache')
            self.assertEqual(cfg['restored_pl']['main'],'unchanged_ordinary_self_PL')
            self.assertFalse(cfg['restored_pl']['noise'])
            for key,value in [('target_inputs','target'),('default_head_joint',False),('segment_epochs',19),('native_U_gate',True)]:
                bad=copy.deepcopy(cfg);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)

    def test_only_complete_source_dose_and_excluded_noise_are_admitted(self):
        proof=dict(status='SOURCE_DOSE_NOISE_COMPLETE',recipe=RECIPE,source_V_seen=27000,source_V_gradient_reads=0,
            target_reads=0,U_truth_read=False,optimizer_updates=0,F_G_parameter_updates=0,parameters_and_buffers_unchanged=True,
            noise_feasibility_pass=False,gradient=dict(rows=[{}]*12,model_seed=SEEDS[0],backward_calls=48,V_gradient_reads=0,
                decision=dict(selected_weight=.126,source_gradient_pass=True)))
        self.assertEqual(source_decision(proof),.126)
        for key,value in [('target_reads',1),('source_V_gradient_reads',1),('noise_feasibility_pass',True)]:
            bad=copy.deepcopy(proof);bad[key]=value
            with self.assertRaises(ValueError):source_decision(bad)

    def test_two_epoch_ramp_does_not_reset_at_new_epoch(self):
        self.assertEqual(ramp_weight(.126,6660),0)
        self.assertAlmostEqual(ramp_weight(.126,6882),.063)
        self.assertEqual(ramp_weight(.126,7104),.126)
        self.assertEqual(ramp_weight(.126,11100),.126)

    def test_U_cache_rejects_foreign_physical_samples_and_visible_truth(self):
        cache=SourceCache.__new__(SourceCache);cache.index={(1,3,1,2,4):0}
        cache.probability=np.ones((1,6),np.float32)/6;cache.accepted=np.ones(1,bool)
        cache.iq=np.zeros((3,1,2,256),np.float32)
        meta={k:torch.tensor([v]) for k,v in zip(('eq_i','rx_i','day_i','sig_i','base_index'),(1,3,1,2,4))}
        meta['tx_label_visible']=torch.tensor([False]);q,mask,x=cache.take(meta,0,'cpu',True)
        self.assertFalse(q.requires_grad);self.assertEqual(tuple(x.shape),(1,2,256));self.assertTrue(mask.item())
        meta['base_index']=torch.tensor([5])
        with self.assertRaises(ValueError):cache.take(meta,0,'cpu',True)
        meta['tx_label_visible']=torch.tensor([True])
        with self.assertRaises(ValueError):cache.take(meta,0,'cpu',True)


if __name__=='__main__':unittest.main()
