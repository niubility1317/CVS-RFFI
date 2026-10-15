import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from .probe import forward,group_errors,group_geometry,summarize
from experiments.cvs_urban_bridge50 import objective
from experiments.cvs_urban_branch50.model import build


class Checks(unittest.TestCase):
    def test_actual_group_measurement_matches_bounded_training_objective_and_excludes_dt(self):
        torch.manual_seed(103);a=torch.randn(12,112);b=torch.randn(12,112)
        scale=dict(increment=.3,feature_area=.5,time_area=.2)
        values=group_errors(a,b,scale);loss,groups=objective.local(a,b,scale,torch.ones(12))
        self.assertAlmostEqual(float(sum(v.mean() for v in values.values())/3),float(loss),places=6)
        for key in values:self.assertTrue(torch.allclose(values[key].mean(),groups[key]))
        changed=a.clone().reshape(12,4,28);changed[:,:,6]+=1000
        other=group_errors(changed.reshape(12,112),b,scale)
        self.assertTrue(all(torch.equal(values[k],other[k]) for k in values))
        geometry=group_geometry(a,b,torch.arange(12)%6)
        self.assertTrue(all(torch.isfinite(v['gap']).all() for v in geometry.values()))
        with self.assertRaises(ValueError):group_geometry(a,b,torch.full((12,),-1))
        with self.assertRaises(ValueError):group_errors(a,b,dict(scale,increment=float('nan')))

    def test_repairs_use_fixed_subset_and_do_not_invent_missing_CE_comparator(self):
        rows=[]
        for fixed,base,ce,correct in ((True,False,False,True),(True,True,True,False),(False,False,False,True)):
            rows.append(dict(fixed_NOMINAL_clean_correct=fixed,NOMINAL_full_correct=base,BRIDGE_CE_full_correct=ce,
                correct=correct,EMA_correct=correct,margin=.2,EMA_margin=.3,
                common_local_group_error=dict(increment=.1,feature_area=.2,time_area=.3)))
        result=summarize(rows)
        self.assertEqual(result['fixed_NOMINAL_clean_correct_count'],2)
        self.assertEqual(result['NOMINAL_errors_repaired'],1);self.assertEqual(result['NOMINAL_correct_new_errors'],1)
        self.assertEqual(result['BRIDGE_CE_errors_repaired'],1);self.assertEqual(result['fixed_NOMINAL_retention'],.5)
        for row in rows:row['BRIDGE_CE_full_correct']=None
        result=summarize(rows);self.assertIsNone(result['BRIDGE_CE_errors_repaired'])
        self.assertEqual(result['BRIDGE_CE_comparison_count'],0)

    def test_final50_and_real_own_calibration_required_before_source_analysis(self):
        configs=d.configs();inner=dict(same_physical_fold=True)
        def read(path):
            c=next(v for v in configs.values() if path.parent==Path(v['output_root']))
            if path.name=='completion.json':return dict(config=c,epoch=50,optimizer_steps=11100,target_access=False,head_fit_updates=0)
            if path.name=='source_frozen.json':return dict(config=c,epoch=50,selection='final_only',target_access=False)
            if path.name=='inner_source_contract.json':return inner
            if path.name=='training_mechanism_complete.json':return dict(status='VERIFIED',updates=11100,main_forward_packets=9031800,head_fit_updates=0,U_truth_read=False)
            if path.name=='bridge_calibration.json':return dict(status='SOURCE_BRIDGE_CALIBRATED',warmup_epoch=5,source_only=True,query_read=False,forward_packets=5760,optimizer_updates=0,EMA_updates=0,head_fit_updates=0)
            raise AssertionError(str(path))
        with patch.object(d,'read',read):self.assertEqual(len(d.eligible_rows()[0]),3)
        def bad(path):
            result=read(path)
            if path.name=='bridge_calibration.json':result['query_read']=True
            return result
        with patch.object(d,'read',bad):
            with self.assertRaisesRegex(ValueError,'source epoch5'):d.eligible_rows()
        self.assertEqual(d.FORWARD_PACKETS,2+720+3*(1440+6*2880))

    def test_real_frozen_default_model_and_SIG_capture_leave_state_and_gradients_unchanged(self):
        torch.set_num_threads(2);model=build(d.training.references()['NOMINAL'],'cpu').eval().requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()};modes={m:m.training for m in model.modules()}
        result=forward(model,torch.randn(12,2,256))
        self.assertEqual(result['D'].shape,(12,112));self.assertEqual(result['logits'].shape,(12,6))
        self.assertEqual(result['J'].shape,(12,160));self.assertTrue(torch.isfinite(result['logits']).all())
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(m.training==mode for m,mode in modes.items()))

    def test_publisher_routes_to_registered_source_module_without_local_checkpoint_access(self):
        from . import publish as pub
        from experiments.cvs_urban_semantic_mechanism import publish as previous
        original_d,original_remote=previous.d,previous.REMOTE
        def inspect(output):
            self.assertIs(previous.d,d)
            self.assertIn('experiments.cvs_urban_bridge_mechanism.probe',previous.REMOTE)
            self.assertIn('experiments/cvs_urban_bridge_mechanism/registration.json',previous.REMOTE)
            return dict(output=output)
        with patch.object(previous,'publish',inspect),patch.object(d,'eligible_rows',side_effect=AssertionError('remote-only inputs')):
            self.assertEqual(pub.publish('placeholder'),dict(output='placeholder'))
        self.assertIs(previous.d,original_d);self.assertEqual(previous.REMOTE,original_remote)


if __name__=='__main__':unittest.main()
