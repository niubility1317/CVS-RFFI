import unittest
from copy import deepcopy
from unittest.mock import patch
import torch
from experiments.pamf_u_history import covariance_three_seed as module
from experiments.pamf_channel_pl.sig1_matched_baselines import prefix


def context():
    p='/p';frozen={k:'/'+k for k in ('F','AF','BASE','FOLD')};cases=[]
    for seed in module.SEEDS:
        rid='F0-SIG1-PL-s'+str(seed)
        c=dict(project=p,model_seed=seed,run_id=module.BASELINE,row_id=rid,paper_method='SIG1-PL',algorithm_id='pamf_sig1_matched_baselines_v1',
            method_id='pamf_t1_e50',protocol_id='F0',num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],regime='matched_sat_plain_pl',ablation='F21',
            sig_time_views=1,U_use=True,U_start_epoch=14,shared_SIG1_E20=prefix(p,seed),epochs=50,optimizer_updates=11100,trajectory=[20,10,20],
            default_head_joint=True,plain_PL_threshold=.95,plain_PL_weight=1.,unlabeled_batch=256,labeled_batch=128,frozen_sources=frozen,
            source_contract='/contract',source_cache='/cache',expected_source_contract='/contract',
            output_root=p+'/runs/'+module.BASELINE+'/F/'+rid+'/source',
            repl_row=dict(model_seed=seed,structure='SIG_TIME',L_A=True,L_B=True,U=True,blocks=dict(style=False,joint=False,guard=False,extrapolation=False)))
        cases.append(dict(model_seed=seed,baseline_config=c,parent=dict(path=prefix(p,seed)),completed_reference=c['output_root']))
    return dict(project=p,python='/python',frozen_sources=frozen,cases=cases,channel_code='/physics',U_channel_cache='/H',seed01_masks='/old',source_contract='/contract',source_cache='/cache')


class ThreeSeedCovarianceTests(unittest.TestCase):
    def test_eight_missing_rows_and_no_seed01_retraining(self):
        ctx=context();rows=[]
        for seed in module.NEW_SEEDS:
            for method in module.METHODS:
                c=module.config(ctx,'test',method,seed,1.);rows.append(c)
                self.assertEqual(c['shared_SIG1_E20'],prefix('/p',seed));self.assertNotIn('source_G',c)
                self.assertIn('/s'+str(seed)+'/',c['covariance_masks'])
        self.assertEqual(len({c['output_root'] for c in rows}),8)
        with self.assertRaises(ValueError):module.config(ctx,'test','CovSel-U',module.SEEDS[0],1.)
        c=deepcopy(rows[0]);c['shared_SIG1_E20']=prefix('/p',module.SEEDS[0])
        with self.assertRaises(ValueError):module.validate_train(c)
        c=deepcopy(rows[0]);c['plain_PL_weight']=.5
        with self.assertRaises(ValueError):module.validate_train(c)

    def test_own_basis_mask_and_matched_counts_required(self):
        c=module.config(context(),'test','CovSel-U',module.NEW_SEEDS[0],1.)
        masks={}
        for level,channels in [('time_down',48),('t2.pool',96)]:
            m=torch.zeros((channels,channels),dtype=torch.bool);m[0,1]=True
            masks[level]=dict(selective=m,random=m.clone(),full=torch.ones_like(m).triu(1))
        p=dict(schema='pamf_sig1_selective_covariance_masks_v2_natural_support',recipe=dict(model_seed=c['model_seed']),
            source_prefix=c['shared_SIG1_E20'],source_contract={'same':True},fit_role='L_s',U_truth_read=False,V_used=False,target_used=False,
            masks=masks,diagnostics={k:dict(selected_edges=1) for k in masks})
        with patch('torch.load',return_value=p),patch.object(module,'read',return_value={'same':True}):module.load_masks(c)
        bad=deepcopy(p);bad['recipe']['model_seed']=module.SEEDS[0]
        with patch('torch.load',return_value=bad),patch.object(module,'read',return_value={'same':True}):
            with self.assertRaises(ValueError):module.load_masks(c)

    def test_scoring_remains_source_and_G_free(self):
        c=dict(run_id=module.RUN,method='CovFull-U',model_seed=module.SEEDS[2],row_id='F0-CovFull-U-s'+str(module.SEEDS[2]),protocol_id='F0',source_days=[1,2,3],
            classes=list('abcdef'),readouts=['raw'],tests={t:dict(views_root='/views',truth='/truth') for t in module.TASKS})
        module.validate_evaluation(c,True)
        for key in ('source_output','source_contract','source_cache','frozen_sources','source_G','G','restored_caches'):
            with self.assertRaises(ValueError):module.validate_evaluation(dict(c,**{key:'/bad'}),True)
