from copy import deepcopy
import unittest
import torch
from experiments.pamf_channel_pl.anchor_drift_source import validate_final,F50_ROWS
from experiments.pamf_channel_pl.receiver_reference import build_references
from experiments.pamf_channel_pl.set_reference import mean_logits


class AnchorDriftTests(unittest.TestCase):
    def test_known_source_final_and_changed_roles_or_truth_use_rejected(self):
        contract=dict(source_rxs=[1,3,4,6,8],source_days=[1,2,3],role_ids='original')
        row=F50_ROWS[0];c=dict(run_id=row[1],row_id=row[2],model_seed=2026092701,protocol_id='F0',num_classes=6,
            method_id='pamf_t1_e50',default_head_joint=True,posthoc_head_fit=False,source_cache='source',
            source_rxs=contract['source_rxs'],source_days=contract['source_days'])
        d=dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,head_fit_updates=0,config=c)
        validate_final(d,contract,contract,row,'source')
        for key,value in [('epoch',49),('target_access',True),('U_truth_read',True),('head_fit_updates',1)]:
            bad=deepcopy(d);bad[key]=value
            with self.assertRaises(ValueError):validate_final(bad,contract,contract,row,'source')
        for key,value in [('model_seed',2026092702),('default_head_joint',False),('target_truth','truth'),('run_id','other')]:
            bad=deepcopy(d);bad['config'][key]=value
            with self.assertRaises(ValueError):validate_final(bad,contract,contract,row,'source')
        with self.assertRaises(ValueError):validate_final(d,dict(contract,role_ids='other'),contract,row,'source')

    def test_same_frame_control_and_rotated_current_frame(self):
        y=torch.tensor([0,0,1,1]);rx=torch.tensor([1,3,1,3]);original=torch.zeros(4,160)
        original[:2,0]=1;original[2:,1]=1
        fixed,_=build_references(original,y,rx,(1,3),2,'L_s')
        same,_=build_references(original.clone(),y,rx,(1,3),2,'L_s')
        self.assertTrue(torch.equal(mean_logits(original,fixed,rx,(1,3)),mean_logits(original,same,rx,(1,3))))
        current=torch.zeros_like(original);current[:2,1]=1;current[2:,0]=-1
        fresh,_=build_references(current,y,rx,(1,3),2,'L_s')
        self.assertFalse(torch.equal(mean_logits(current,fixed,rx,(1,3)).argmax(1),y))
        self.assertTrue(torch.equal(mean_logits(current,fresh,rx,(1,3)).argmax(1),y))
        self.assertFalse(fresh.requires_grad)


if __name__=='__main__':unittest.main()
