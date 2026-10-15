"""Focused source-only checks for actual shared gradients and diagnostic semantics."""
import copy,unittest
import torch
from . import branches as b,design as d
from .model import build
from .mechanism import gradient_record,contributions,select_developer,summarize,KEYS


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.model=build(d.config(d.rows()[0]),'cpu').eval()

    def test_actual_shared_T_S_gradient_scope_and_frozen_state(self):
        model=copy.deepcopy(self.model);teacher=copy.deepcopy(model).requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(3,2,256)
        logits,target=b.frozen_teacher(teacher,x);y=logits.argmax(1)
        out,values=b.forward(model,x+.2*torch.randn_like(x))
        recipe=dict(coefficients={k:dict(coefficient=.1) for k in (*KEYS,'ALL')},ALL_weights={k:.2 for k in b.BRANCHES})
        record=gradient_record(model,out,values,logits,target,y,'A-S',recipe,'practical_low_urban',0)
        self.assertGreater(record['global_gradient']['auxiliary_weighted_norm'],0.)
        self.assertTrue(any('t2.' in key for key in record['T_S_shared_parameter_names']))
        self.assertFalse(any('local_projection.' in key or 't_proj.' in key for key in record['T_S_shared_parameter_names']))
        self.assertTrue(all(v==0 for v in record['auxiliary_boundary_norms'].values()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_gain_contribution_can_disappear_without_branch_repair(self):
        model=copy.deepcopy(self.model).requires_grad_(False)
        with torch.no_grad():out,values=b.forward(model,torch.randn(4,2,256))
        norm=float(values['S'].norm());before=contributions(model,values)
        model.id_backbone.encoder.sequence_pool.gain_raw.zero_()
        after=contributions(model,values)
        self.assertGreater(before['S']['mean_contribution_norm'],0.)
        self.assertEqual(after['S']['mean_contribution_norm'],0.)
        self.assertEqual(float(values['S'].norm()),norm)

    def test_only_declared_developer_physical_IDs_selected(self):
        ids=[str(i) for i in range(6300)]
        L=dict(ids=ids,x=torch.arange(6300)[:,None],y=torch.zeros(6300),rx=torch.ones(6300),day=torch.ones(6300))
        inner=dict(role_ids=dict(L_train=ids[:5580],L_dev=ids[5580:]),L_dev_updates=False)
        contract=dict(role_ids={'L_s':ids});panel=select_developer(L,inner,contract)
        self.assertEqual(panel['ids'],ids[5580:]);self.assertEqual(int(panel['x'][0]),5580)
        bad=copy.deepcopy(inner);bad['role_ids']['L_dev'][0]='0'
        with self.assertRaises(ValueError):select_developer(L,bad,contract)
        bad=copy.deepcopy(inner);bad['role_ids']['L_dev'][-1]='missing'
        with self.assertRaises(ValueError):select_developer(L,bad,contract)

    def test_repairs_use_fixed_A0_clean_correct_subset(self):
        def item(fixed,correct,baseline):return dict(fixed_A0_clean_correct=fixed,correct=correct,A0_satellite_correct=baseline,
            margin=.2,identity={key:dict(clean_pair_cosine=.5) for key in KEYS})
        value=summarize([item(True,True,False),item(True,False,True),item(False,True,False)])
        self.assertEqual(value['fixed_A0_clean_correct_count'],2)
        self.assertEqual(value['A0_errors_repaired'],1);self.assertEqual(value['A0_correct_new_errors'],1)
        self.assertEqual(value['fixed_A0_retention'],.5)


if __name__=='__main__':unittest.main()
