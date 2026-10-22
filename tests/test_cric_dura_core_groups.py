import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cric_dura_comparison.core_groups import switches,validate_config,check_parent,plan
from experiments.cric_dura_comparison.core_extension import training_config
from experiments.cric_dura_comparison.t1_model import hook

torch.set_num_threads(2)


def config(bits):
    template=dict(method_id='pamf_t1_e50',optimizer_updates=11100,source_rxs=[1,3,4,6,8],
        source_days=[1,2,3],num_classes=6,source_contract='contract',source_cache='cache',
        regime='native_cric',ablation=None,satellite_concat=False)
    return training_config(template,bits,2026092701,dict(project='project',code_commit='test',frozen_sources={}))


class Fake(nn.Module):
    def __init__(self,c):
        super().__init__();self.body=nn.Linear(512,12);self.head=nn.Linear(12,6)
        self._cd_config=c;self._feature_l_steps=1;self._cd_extra_L=0;self._cd_previous_extra=0
        self._cd_As_seen=self._cd_sat_seen=self._cd_sat_applied=self._cd_U_seen=self._cd_U_accepted=self._cd_calls=0
        self._cd_handles=[];self._cd_pending=None;self._cd_phase=None;self.outputs=[]
        self._cd_practical=lambda x,meta,epoch,step:x+.01
    def forward_identity_only(self,x,domain_labels=None):
        h=torch.tanh(self.body(x.flatten(1)));h=F.dropout(h,.2,self.training)
        logits=self.head(h);self.outputs.append(logits)
        maps={'time_down':h[:,:6,None].expand(-1,-1,4),'t2.pool':h[:,6:,None].expand(-1,-1,4)}
        if self._cd_phase=='second':self._cd_pending['second']=maps
        return dict(tx_logits=logits,z_id=h,maps=maps)
    def begin(self,x):
        out=self.forward_identity_only(x)
        self._cd_pending=dict(x=x.detach().clone(),first=out['maps'],second={})
        return out


def objective(c,x,y,rx):
    torch.manual_seed(2026102201);model=Fake(c).train();out=model.begin(x)
    ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
    delta,metrics=hook(model,model=model,labeled_output=out,labels=y,extra=(None,{'rx_i':rx}),
        domains=rx,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
    total=ce+delta;total.backward()
    return model,total.detach(),metrics


class CoreGroupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9);self.x=torch.randn(128,2,256);self.y=torch.arange(128)%6
        self.rx=torch.tensor([1,3,4,6,8])[torch.arange(128)%5]
    def test_full_and_reuse_cells_are_actual_objective_equivalents(self):
        for bits,arm in (('111',None),('011','F16'),('101','F04'),('110','F20')):
            grouped=config(bits);grouped['step_offset']=4440
            legacy=copy.deepcopy(grouped);legacy.pop('core_mechanisms');legacy['ablation']=arm
            left,loss_l,ml=objective(grouped,self.x,self.y,self.rx)
            right,loss_r,mr=objective(legacy,self.x,self.y,self.rx)
            self.assertTrue(torch.equal(loss_l,loss_r),(bits,loss_l,loss_r))
            for p,q in zip(left.parameters(),right.parameters()):
                self.assertTrue(torch.equal(p.grad,q.grad),bits)
            self.assertEqual(ml['T1_L_packets'],768);self.assertEqual(len(left.outputs),6)
    def test_all_off_retains_CE_dose_views_and_gradient(self):
        c=config('000');c['step_offset']=4440
        model,loss,metrics=objective(c,self.x,self.y,self.rx)
        a,b,m,n,r,s=model.outputs;mask=(torch.arange(128)+1)%2!=0
        ce=lambda p,q,y:.5*(F.cross_entropy(p,y,label_smoothing=.01)+F.cross_entropy(q,y,label_smoothing=.01))
        expected=.5*ce(a,b,self.y)+.25*ce(m[~mask],n[~mask],self.y[~mask])\
            +.25*ce(m[mask],n[mask],self.y[mask])+.25*ce(r,s,self.y)
        self.assertTrue(torch.equal(loss,expected.detach()))
        self.assertEqual(metrics['CRIC_supcon_weight'],0.);self.assertEqual(metrics['T1_As_KL_weight'],0.)
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
    def test_U_off_rejects_malformed_flags_before_source_IO(self):
        from experiments.cric_dura_comparison.f_data import context
        c=config('000');c['ablation']=None
        with patch('experiments.cric_dura_comparison.f_data.read',side_effect=AssertionError('Source opened')):
            with self.assertRaisesRegex(ValueError,'U-off'):context(c)
    def test_U_off_does_not_construct_U_dataset(self):
        from experiments.cric_dura_comparison.f_data import context
        c=config('100');opened=[]
        contract=dict(source_rxs=c['source_rxs'],source_days=c['source_days'],classes=list(range(6)),
            counts={'L_s':256,'U_s':512,'V':64})
        def dataset(root,role,contract):
            opened.append(role)
            return torch.utils.data.TensorDataset(torch.zeros(256,2,256),torch.zeros(256,dtype=torch.long))
        with patch('experiments.cric_dura_comparison.f_data.read',return_value=contract),\
                patch('experiments.cric_dura_comparison.f_data.SourceDataset',side_effect=dataset):
            data=context(c)
        self.assertEqual(opened,['L_s','V']);self.assertIsNone(data['unlabeled_loader'])
    def test_parent_must_share_all_three_switches(self):
        with self.assertRaisesRegex(ValueError,'own mechanism'):check_parent(config('000'),config('100'))
        check_parent(config('000'),config('000'))
        c=config('001');c['core_mechanisms']['selective_SSL']=0
        with self.assertRaisesRegex(ValueError,'boolean'):validate_config(c)
    def test_factorial_requires_only_four_missing_cells(self):
        p=plan();self.assertEqual(len(p['rows']),24);self.assertEqual(sum(r['new_F'] for r in p['rows']),12)
        self.assertEqual({r['bits'] for r in p['rows'] if r['new_F']},{'000','001','010','100'})
        self.assertEqual(p['main_table'],['111','011','101','110'])


if __name__=='__main__':unittest.main()
