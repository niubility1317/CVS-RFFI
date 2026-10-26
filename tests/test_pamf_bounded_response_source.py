import unittest
from unittest.mock import patch
import numpy as np
from experiments.pamf_u_history.bounded_response_source import source_rule,paired_results,check_reader_smoke,validate_context,PREDECESSOR,SEEDS,VIEWS,RXS


def rows():
    return [dict(view=v,dimension=d,stratum=s,count=n,delta_correct=0,accuracy_delta_pp=0.) for v in VIEWS for d,s,n in [('overall','ALL',27000),*[('RX',str(r),5400) for r in RXS]]]


class SourceResponseTests(unittest.TestCase):
    def test_context_rejects_target_G_and_cross_seed_parent(self):
        from experiments.pamf_channel_pl.sig1_matched_baselines import prefix
        c=dict(project='/project',python='/python',frozen_sources={k:'/'+k for k in ('F','AF','BASE','FOLD')},channel_code='/physics',source_contract='/contract',source_cache='/cache',
            predecessor_complete='/project/runs/'+PREDECESSOR+'/complete.json',cases=[dict(model_seed=s,baseline_config=dict(model_seed=s,source_contract='/contract',source_cache='/cache',paper_method='SIG1-PL',shared_SIG1_E20=prefix('/project',s),frozen_sources={})) for s in SEEDS])
        with patch('experiments.pamf_channel_pl.sig1_matched_baselines.validate_config',side_effect=lambda x:x):
            validate_context(c)
            with self.assertRaises(ValueError):validate_context(dict(c,tests={}))
            with self.assertRaises(ValueError):validate_context(dict(c,frozen_sources=dict(c['frozen_sources'],G='/G')))
            c['cases'][1]['baseline_config']['shared_SIG1_E20']=prefix('/project',SEEDS[0])
            with self.assertRaises(ValueError):validate_context(c)

    def test_reader_smoke_does_not_require_full_native_student_parameter_total(self):
        import torch
        model=torch.nn.Module();model.sequence_pool=torch.nn.Module();model.sequence_pool.views=[object()]
        model.sequence_pool.register_parameter('signature',torch.nn.Parameter(torch.zeros(18976)))
        model.core=torch.nn.Module();model.core.id_backbone=torch.nn.Module();model.core.id_backbone.cls_head=torch.nn.Linear(160,6,bias=False)
        output=[torch.zeros(2,6),torch.zeros(2,6)]
        signature=check_reader_smoke(model,output)
        self.assertNotEqual(signature['reader_parameters'],863730);self.assertEqual(signature['signature_parameters'],18976)
        model.core.id_backbone.cls_head=torch.nn.Linear(160,7,bias=False)
        with self.assertRaises(ValueError):check_reader_smoke(model,output)

    def test_clean_boundary_and_receiver_damage(self):
        a=rows();a[0]['delta_correct']=-27;self.assertTrue(source_rule(a)['source_necessary_pass'])
        a[0]['delta_correct']=-28;self.assertFalse(source_rule(a)['source_necessary_pass'])
        a=rows();a[1]['accuracy_delta_pp']=-.201;self.assertFalse(source_rule(a)['source_necessary_pass'])
        with self.assertRaises((KeyError,ValueError)):source_rule(rows()[1:])

    def test_H_total_and_individual_views_both_checked(self):
        a=rows();a[6]['delta_correct']=-54;a[12]['delta_correct']=54
        self.assertTrue(source_rule(a)['source_necessary_pass'])
        a[6]['delta_correct']=-55;a[12]['delta_correct']=55;self.assertFalse(source_rule(a)['source_necessary_pass'])
        a=rows();a[6]['delta_correct']=-1;self.assertFalse(source_rule(a)['source_necessary_pass'])

    def test_paired_fixes_and_breaks_are_counted(self):
        n=30;y=np.arange(n)%6;rx=np.asarray(RXS*6);day=np.tile([1,2,3],10);logits=np.zeros((2,4,n,6),np.float32)
        for j in range(2):
            for v in range(4):logits[j,v,np.arange(n),y]=1
        logits[0,:,0,:]=0;logits[0,:,0,1]=2;logits[1,:,1,:]=0;logits[1,:,1,2]=2
        row=next(r for r in paired_results(logits,y,rx,day) if r['view']=='clean' and r['dimension']=='overall')
        self.assertEqual(row['fixed'],1);self.assertEqual(row['broken'],1);self.assertEqual(row['delta_correct'],0)


if __name__=='__main__':unittest.main()
