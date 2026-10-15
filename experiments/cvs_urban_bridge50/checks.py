"""Protocol negatives, physical pairing, and real native-graph wiring."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d,features as f,objective
from .channel import PairedChannel
from .model import build
from .observer import Observer
from experiments.cvs_urban_branch50 import design as old
from experiments.cvs_urban_branch50.channel import PairedChannel as Original
from experiments.cvs_urban_identity50.features import boundary_parameters
from experiments.cvs_urban_bridge.signature import KEYS,teacher_scales


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_full_nominal_wave_exact_and_session_not_TX_and_epoch_refresh(self):
        c=d.config(d.rows()[0]);channel=PairedChannel(c);x=torch.randn(12,2,256)
        meta=dict(base_index=torch.arange(12),rx_i=torch.tensor([1,3,4,6,8,1]*2),
            day_i=torch.ones(12,dtype=torch.long),eq_i=torch.zeros(12,dtype=torch.long),tx_i=torch.arange(12)%6)
        full,hq,records=channel.apply(x,meta,1,1)
        expected,reference=Original(old.config(old.rows()[0])).apply(x,meta,1,1)
        self.assertTrue(torch.equal(full,expected));self.assertFalse(torch.equal(full,hq))
        for a,b in zip(records,reference):
            for key in ('scenario','state_end','quality_snr_db','receiver_lock_status','output_frequency_hz','receiver','agc_gain_db','residual_kernel_length'):
                self.assertEqual(a[key],b[key])
            self.assertTrue(a['bridge']['frozen_fields_identical'] and a['bridge']['frozen_kernels_identical'])
        other=dict(meta,tx_i=torch.full((12,),99));again,bridge2,second=channel.apply(x,other,1,1)
        self.assertTrue(torch.equal(full,again) and torch.equal(hq,bridge2))
        later,_,next_records=channel.apply(x,meta,2,1)
        self.assertFalse(torch.equal(full,later))
        self.assertEqual([r['receiver'] for r in records],[r['receiver'] for r in next_records])
        with self.assertRaises(ValueError):channel.apply(x,meta,1,1,view='U')
        with self.assertRaises(ValueError):channel.apply(x,meta,1,1,scene='practical_low_urban')

    def test_local_is_bounded_balanced_and_constant_dt_ignored(self):
        student=torch.randn(8,112,requires_grad=True);teacher=torch.randn(8,112,requires_grad=True)
        scales=teacher_scales(teacher);weight=torch.ones(8)
        value,groups=objective.local(student,teacher,scales,weight)
        self.assertGreaterEqual(float(value),0.);self.assertLess(float(value),1.)
        self.assertTrue(torch.allclose(value,sum(groups.values())/3))
        shifted=student.detach().clone().reshape(8,4,28);shifted[:,:,6]+=1000
        same,_=objective.local(shifted.reshape(8,112),teacher,scales,weight);self.assertTrue(torch.allclose(value,same))
        torch.autograd.grad(value,student);self.assertIsNone(teacher.grad)
        for change in (dict(scales={k:0. for k in KEYS}),dict(scales={k:float('nan') for k in KEYS}),dict(student=torch.zeros(8,111))):
            args=dict(student=student,teacher=teacher,scales=scales,weight=weight);args.update(change)
            with self.assertRaises(ValueError):objective.local(**args)

    def test_true_L_and_continuous_quality_only_weight_transfer(self):
        y=torch.tensor([0,1,2]);logits=F.one_hot(y,6).float()*5
        records=[dict(scenario='practical_low_urban',q=q) for q in (0.,.4,1.)]
        jw,sw=objective.weights(logits,y,records);self.assertEqual(jw.tolist(),[1.,1.,1.])
        self.assertTrue(torch.allclose(sw,torch.tensor([0.,.4,1.])))
        with self.assertRaises(ValueError):objective.weights(logits,torch.full((3,),-1),records)
        with self.assertRaises(ValueError):objective.weights(logits,y,[dict(scenario='practical_low_urban',q=2.)]*3)
        for row in d.rows():
            c=d.config(row);args=d.make_args(c,'cpu')
            self.assertEqual((args.epochs,args.label_epochs,args.pseudo_epochs),(50,33,17))
            self.assertTrue(args.use_aug and args.use_unlabeled and args.use_ema_teacher and args.use_concat_sat_channel_aug)
            self.assertEqual(c['checkpoint_sources'],[]);self.assertFalse(c['separate_head_fit'])
            bad=copy.deepcopy(c);bad['target_access']=True
            with self.assertRaises(ValueError):d.validate(bad)

    def test_local_reaches_t2_and_local_projection_but_not_last_projection_W_or_gain(self):
        c=d.config(d.rows()[1]);model=build(c,'cpu').train();x=torch.randn(8,2,256)
        _,values=f.forward(model,x);teacher=values['D'].detach()+.2*torch.randn_like(values['D'])
        value,_=objective.local(values['D'],teacher,teacher_scales(teacher),torch.ones(8))
        encoder=model.id_backbone.encoder;core=encoder.core.id_backbone
        gradients=torch.autograd.grad(value,[core.t2.pw.weight,encoder.sequence_pool.views[0].local_projection.weight],retain_graph=True)
        self.assertTrue(all(float(v.norm())>0 for v in gradients))
        params=[encoder.sequence_pool.output_projection.weight,*boundary_parameters(model).values()]
        self.assertTrue(all(v is None for v in torch.autograd.grad(value,params,allow_unused=True)))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_native_CE_HQ_graph_hook_lifetime_and_default_head_joint_updates(self):
        for row in d.rows():
            c=d.config(row);model=build(c,'cpu').train();teacher=copy.deepcopy(model).eval().requires_grad_(False);observer=Observer(c)
            with tempfile.TemporaryDirectory() as temp:
                observer.initialize(model,c,Path(temp))
                try:
                    x=torch.randn(128,2,256);y=torch.arange(128)%6
                    meta=dict(base_index=torch.arange(128),rx_i=torch.tensor([(1,3,4,6,8)[i%5] for i in range(128)]),day_i=torch.ones(128,dtype=torch.long))
                    records=[dict(sample_id=str(i),session_id='rx:day',scenario='practical_low_urban',state_end='PARTIAL',
                        receiver_lock_status='degraded',quality_snr_db=8.,signal_to_added_noise_db=8.,output_frequency_hz=5.,
                        residual_kernel_length=5,q=.8,bridge=dict(bridge_independent_signal_to_added_noise_db=14.)) for i in range(128)]
                    observer.channel.apply=lambda x,*a,**kw:(x+.2*torch.randn_like(x),x+.1*torch.randn_like(x),records)
                    observer.steps=1110;observer.recipe=dict(scales={k:1. for k in KEYS},coefficients={k:dict(coefficient=.1) for k in ('joint','local')})
                    native=SimpleNamespace(_urban_practical_training_hook=None,_source_feature_auxiliary_hook=None,
                        _prepare_concat_sat_batch_for_training=lambda *a,**kw:None,move_batch=lambda b,dev:(b[0],b[1],(None,b[2])))
                    args=d.make_args(c,'cpu')
                    with observer.native_scope(native):
                        native.move_batch((x,y,meta),'cpu');prepared=x+.01
                        xx,yy,_,_,_=native._prepare_concat_sat_batch_for_training(None,prepared,y,None,args=args,epoch=6,batch_idx=1)
                        output=model(xx,y_tx=yy,grl_lambda=1.,return_aux=True,domain_labels=None)
                        self.assertTrue(observer.armed.done)
                        self.assertEqual(observer.armed.result()['D'].shape,(256,112))
                        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
                        steps=model._feature_l_steps;ce=F.cross_entropy(output['tx_logits'],yy,label_smoothing=.01)
                        extra,metrics=observer.observe(model=model,args=args,epoch=6,batch_idx=1,clean_iq=xx,labels=yy,clean_meta=meta,
                            clean_output=output,clean_ce=ce,unlabeled_iq=None,unlabeled_meta={},teacher_prob=None,weak_mask=None,teacher_model=teacher)
                        self.assertEqual(model._feature_l_steps,steps)
                        for key in ('clean_L_packets','satellite_L_packets','HQ_L_packets','teacher_L_clean_packets','teacher_HQ_packets'):
                            self.assertEqual(observer.counts[key],128)
                        expected=.5*(metrics['urban/clean_CE']+metrics['urban/satellite_CE'])+.25*metrics['urban/HQ_CE']
                        self.assertTrue(torch.allclose(ce+extra-metrics['urban/auxiliary_weighted_loss'],expected))
                        self.assertEqual(float(metrics['urban/local_SIG_weighted_loss'])>0,c['auxiliary']['local_SIG'])
                        (ce+extra).backward();head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
                        self.assertGreater(float(head.grad.norm()),0.)
                        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                    self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
                    self.assertIsNone(native._urban_practical_training_hook)
                finally:observer.trace.close()

    def test_calibration_rejection_preserves_pending_grads_modes_states_and_RNG(self):
        from .calibrate import run
        c=d.config(d.rows()[0]);model=build(c,'cpu').train();teacher=copy.deepcopy(model).eval().requires_grad_(False)
        x=torch.randn(720,2,256)
        with torch.no_grad():
            y=torch.cat([teacher.forward_identity_only(x[start:start+90])['tx_logits'].argmax(1) for start in range(0,720,90)])
        panel=dict(x=x,y=y,rx=torch.ones(720,dtype=torch.long),day=torch.ones(720,dtype=torch.long),ids=[f'unit:{i}' for i in range(720)])
        with self.assertRaises(ValueError):run(model,teacher,panel,c)
        model._feature_l_steps=1110
        parameter=next(model.parameters());parameter.grad=torch.randn_like(parameter)
        pending=parameter.grad.clone();rng=torch.get_rng_state().clone()
        before={k:v.clone() for k,v in model.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
        def apply(x,meta,*args,scene=None,**kwargs):
            return x+.25*torch.randn_like(x),x+.12*torch.randn_like(x),[dict(scenario=scene,q=.7) for _ in x]
        with patch('experiments.cvs_urban_bridge50.calibrate.PairedChannel',return_value=SimpleNamespace(apply=apply)):
            # This intentionally untrained synthetic model is not an actual
            # source epoch5 calibration. Its near-zero auxiliary norm must be
            # rejected rather than silently amplified beyond the fixed bound.
            with self.assertRaisesRegex(ValueError,'Calibration outside registered bounded strength'):
                run(model,teacher,panel,c)
        self.assertTrue(model.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(pending,parameter.grad));self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()))


if __name__=='__main__':unittest.main()
