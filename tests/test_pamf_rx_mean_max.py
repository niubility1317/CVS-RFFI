from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from experiments.pamf_u_history.rx_mean_views import MeanBank,Shift
from experiments.pamf_u_history.rx_mean_max_objective import all_styles,choose,RECIPE
from experiments.pamf_u_history.rx_mean_max_source import RUN,PREDECESSOR,SEEDS,config,validate_train,validate_context,prepare,trajectory


def context():
    from test_pamf_rx_mean import RXMeanTests
    c=RXMeanTests().context();c['predecessor_complete']=c['project']+'/runs/'+PREDECESSOR+'/complete.json'
    return c


class RXMeanMaxTests(unittest.TestCase):
    def test_exact_four_styles_reproduce_existing_random_transform(self):
        import numpy as np
        shift=Shift(MeanBank(np.arange(20,dtype=np.float64).reshape(5,4)/7),'cpu')
        x=torch.randn(5,2,256);rx=torch.tensor([1,3,4,6,8]);ids=torch.arange(5);call=9
        views,styles=all_styles(shift,x,rx);random,random_styles=shift(x,rx,ids,call)
        self.assertEqual(tuple(views.shape),(4,5,2,256));self.assertTrue(bool((styles!=rx[None,:]).all()))
        for i in range(5):self.assertEqual(len(set(styles[:,i].tolist())),4)
        offset=(ids+call-1)%4;torch.testing.assert_close(views[offset,torch.arange(5)],random);torch.testing.assert_close(styles[offset,torch.arange(5)],random_styles)

    def test_selection_is_per_packet_and_permutation_equivariant(self):
        q=torch.softmax(torch.randn(7,6),1);logits=torch.randn(4,7,6);selected,losses=choose(q,logits)
        torch.testing.assert_close(losses[selected,torch.arange(7)],losses.max(0).values)
        order=torch.tensor([6,1,4,0,3,2,5]);changed,other=choose(q[order],logits[:,order])
        torch.testing.assert_close(changed,selected[order]);torch.testing.assert_close(other,losses[:,order])
        tie,_=choose(q,torch.zeros(4,7,6));torch.testing.assert_close(tie,torch.zeros(7,dtype=torch.long))
        with self.assertRaises(ValueError):choose(q.requires_grad_(),logits)
        with self.assertRaises(ValueError):choose(q.detach(),logits.requires_grad_())

    def test_shared_PL_and_trajectory_negative_cases(self):
        c=config(context(),'commit',SEEDS[0],.4);validate_train(c)
        for key,value in [('plain_PL_threshold',.9),('sig_time_views',3),('source_G','G'),('target_truth','target')]:
            with self.assertRaises(ValueError):validate_train(dict(c,**{key:value}))
        bad=deepcopy(c);bad['rx_mean_recipe']['candidate_variants']=3
        with self.assertRaises(ValueError):validate_train(bad)
        c.update(segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=c['output_root']+'/segment30',output_root=c['output_root']+'/segment50');validate_train(c)
        with self.assertRaises(ValueError):validate_train(dict(c,parent_source='/foreign'))

    def test_process_local_override_restores_original_bindings(self):
        from experiments.pamf_u_history import rx_mean_source as mean
        old_validate,old_objective=mean.validate_train,mean.Objective
        with trajectory():self.assertIsNot(mean.validate_train,old_validate);self.assertIsNot(mean.Objective,old_objective)
        self.assertIs(mean.validate_train,old_validate);self.assertIs(mean.Objective,old_objective)
        with self.assertRaises(ValueError):validate_context(dict(context(),predecessor_complete='/other'))

    def test_all_three_seeds_wait_for_current_matrix_and_no_source_truth(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);c=root/'context.json';c.write_text(json.dumps(context()))
            e=root/'evaluation.json';e.write_text(json.dumps(dict(classes=list(range(6)),tests={t:dict(views_root='/'+t,truth='/truth') for t in ('cross_rx','cross_both')})))
            rt=dict(run_id=RUN,context=str(c),evaluation_context=str(e),run_root=str(root/'run'),release_root='/release',python='/python',code_commit='commit')
            jobs=prepare(rt);self.assertEqual(len(jobs),12)
            for job in jobs:
                if job['kind']=='train':continue
                value=json.loads(Path(job['config']).read_text())
                if job['kind']=='calibrate':
                    self.assertEqual(job['dependencies'],[context()['predecessor_complete']]);self.assertNotIn('evaluation_context',value)
                if job['kind']=='predict':self.assertNotIn('truth',json.dumps(value));self.assertNotIn('mean_bank',value)
                if job['kind']=='score':self.assertNotIn('source_output',value);self.assertNotIn('frozen_sources',value)


if __name__=='__main__':unittest.main()
