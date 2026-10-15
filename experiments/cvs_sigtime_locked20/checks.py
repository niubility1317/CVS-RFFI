import copy,unittest
from unittest.mock import patch
import torch
from torch import nn
from .mechanism import install_dropout,TailAverage,TAIL_EPOCHS


class Mechanism(unittest.TestCase):
    def fixture(self):
        model=nn.Module();model.t2=nn.Module();model.t2.drop=nn.Dropout(.1)
        model.default_head=nn.Linear(3,2,bias=False);model.gain=nn.Parameter(torch.tensor([.05]))
        model.register_buffer('fixed_coordinates',torch.tensor([25000.]))
        return model

    def set_mode(self,model,mode):
        with patch('experiments.cvs_sigtime_locked20.mechanism.layout',return_value=(model,'core',None)):
            return install_dropout(model,mode)

    def test_locked_masks_preserve_flat_time_channels(self):
        torch.manual_seed(7);model=self.fixture();self.set_mode(model,'LOCKED')
        value=model.t2.drop(torch.ones(128,8,32))
        self.assertTrue(torch.equal(value[:,:,1:],value[:,:,:-1]))
        self.assertTrue(bool((value==0).any()) and bool((value>0).any()))

    def test_original_element_masks_make_time_changes(self):
        torch.manual_seed(7);model=self.fixture();original=model.t2.drop;self.set_mode(model,'ELEMENT')
        self.assertIs(model.t2.drop,original)
        value=model.t2.drop(torch.ones(128,8,32));self.assertTrue(bool((value[:,:,1:]!=value[:,:,:-1]).any()))

    def test_eval_and_state_unchanged(self):
        reference=self.fixture().eval();x=torch.randn(3,8,32)
        for mode in ('ELEMENT','NONE','LOCKED'):
            model=copy.deepcopy(reference);self.set_mode(model,mode)
            self.assertTrue(torch.equal(model.t2.drop(x),x))
            self.assertEqual(set(model.state_dict()),set(reference.state_dict()))

    def test_reject_nonoriginal_site(self):
        model=self.fixture();self.set_mode(model,'LOCKED')
        with self.assertRaises(ValueError):self.set_mode(model,'LOCKED')

    def test_average_includes_head_gain_and_preserves_live(self):
        model=self.fixture();average=TailAverage('seed01');snapshots=[]
        for epoch in TAIL_EPOCHS:
            with torch.no_grad():
                model.gain.fill_(epoch/10);model.default_head.weight.fill_(epoch)
            before=copy.deepcopy(model.state_dict());rng=torch.random.get_rng_state().clone()
            average.add(model,epoch,trajectory_id='seed01');snapshots.append(before)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(value,model.state_dict()[name]) for name,value in before.items()))
        exported=average.state_dict()
        for name in ('gain','default_head.weight'):
            expected=torch.stack([state[name].double() for state in snapshots]).mean(0).float()
            self.assertTrue(torch.equal(exported[name],expected))
        self.assertTrue(torch.equal(exported['fixed_coordinates'],model.fixed_coordinates))

    def test_wrong_trajectory_or_incomplete_window(self):
        model=self.fixture();average=TailAverage('seed01')
        with self.assertRaises(ValueError):average.add(model,17,trajectory_id='seed02')
        with self.assertRaises(ValueError):average.state_dict()
        with self.assertRaises(ValueError):average.add(model,18,trajectory_id='seed01')

    def test_reject_changing_buffers(self):
        model=self.fixture();average=TailAverage('seed01');average.add(model,17,trajectory_id='seed01')
        model.fixed_coordinates.add_(1)
        with self.assertRaises(ValueError):average.add(model,18,trajectory_id='seed01')

    def test_fixed_source_recipe_rejects_changes(self):
        from . import design as d
        for key,value in (('model_seed',2026092702),('test_views',['clean']),('training_recipe','changed')):
            c=copy.deepcopy(d.config('LOCKED'));c[key]=value
            with self.assertRaises(ValueError):d.validate(c)
        c=copy.deepcopy(d.config('LOCKED'));c['training_mechanism']['p']=.2
        with self.assertRaises(ValueError):d.validate(c)

    def test_average_ignores_earlier_epochs(self):
        model=self.fixture();average=TailAverage('seed01')
        average.add(model,16,trajectory_id='seed01');self.assertEqual(average.epochs,[])
        for epoch in TAIL_EPOCHS:average.add(model,epoch,trajectory_id='seed01')
        with self.assertRaises(ValueError):average.add(model,20,trajectory_id='seed01')

if __name__=='__main__':unittest.main(verbosity=2)
