import copy
import unittest
import numpy as np
from experiments.pamf_channel_pl.restored_source import validate_context, validate_e30, paired_counts, policies, SEEDS


def context():
    return dict(project='project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
        cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='own/segment20',
            source_config=dict(ablation='F21',num_classes=6,run_id='old',row_id='row'),expected_source_contract='F0') for s in SEEDS])


class SourceBoundaryTests(unittest.TestCase):
    def test_no_target_or_other_contract_or_duplicate_seed(self):
        validate_context(context())
        c=context();c['target_inputs']='target'
        with self.assertRaises(ValueError):validate_context(c)
        c=context();c['cases'][0]['protocol']='label01'
        with self.assertRaises(ValueError):validate_context(c)
        c=context();c['cases'][1]=copy.deepcopy(c['cases'][0])
        with self.assertRaises(ValueError):validate_context(c)

    def test_e30_requires_own_full_ancestry_and_actual_native_flags(self):
        case=context()['cases'][0];parent=case['parent_source']
        cfg=dict(parent_source=parent,epoch_offset=20,step_offset=4440,model_seed=case['model_seed'],
            ablation='F21',num_classes=6,run_id='old',row_id='row')
        d=dict(config=cfg,status='SOURCE_FROZEN',epoch=10,updates=2220,initialization='own_previous_segment',
            ancestors=[parent],target_access=False,U_truth_read=False,head_fit_updates=0)
        cp=dict(args=dict(baseline_ckpt=parent+'/final_ssdg.pth',teacher_ckpt='',use_unlabeled=False,use_ema_teacher=False),
            run_id='old',candidate_id='row',epoch=10,a1_ema_successful_updates=2220)
        validate_e30(case,d,cp,{}, {})
        for change in ('foreign_parent','target_contact','native_EMA','different_contract'):
            x=copy.deepcopy(d);q=copy.deepcopy(cp);actual={}
            if change=='foreign_parent':x['config']['parent_source']='foreign'
            if change=='target_contact':x['target_access']=True
            if change=='native_EMA':q['args']['use_ema_teacher']=True
            if change=='different_contract':actual={'roles':'other'}
            with self.assertRaises(ValueError):validate_e30(case,x,q,actual,{})


class PairingTests(unittest.TestCase):
    def test_corrected_and_broken_are_not_independent_accuracy_counts(self):
        y=np.array([0,0,0,0]);raw=np.array([0,1,0,1]);g=np.array([0,0,1,1])
        r=paired_counts(raw,g,y)
        self.assertEqual([r[k] for k in ('both_correct','corrected','broken','both_wrong')],[1,1,1,1])
        self.assertEqual(r['net_correct'],0)
        empty=paired_counts(raw,g,y,np.zeros(4,bool));self.assertIsNone(empty['restored_accuracy'])

    def test_consensus_exposes_correlated_wrong_labels_and_coverage(self):
        z=np.zeros((3,4,2));z[:,:,0]=12;z[1,2]=[0,12];z[2,3]=[0,0]
        rows,accept,pseudo=policies(z,np.array([0,1,0,0]))
        self.assertEqual(accept.tolist(),[True,True,False,False])
        self.assertEqual(rows[-1]['correct'],1);self.assertEqual(rows[-1]['precision_percent'],50)
        self.assertFalse(rows[-1]['feasibility_pass'])


if __name__=='__main__':unittest.main()
