"""Meaningful relation, stop-gradient, source-role and actual-fusion checks."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from .objective import semantic,mask
from . import design as d
from .model import build as new_build
from .observer import Observer
from experiments.cvs_urban_branch50 import design as parent,branches
from experiments.cvs_urban_branch50.model import build
from experiments.cvs_urban_identity50.features import boundary_parameters


class Checks(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(19)
        return dict(clean=torch.randn(8,160,requires_grad=True),urban=torch.randn(8,160,requires_grad=True),
            labels=torch.tensor([0,1,0,1,0,1,0,1]),receiver=torch.tensor([1,1,3,3,4,4,6,6]),
            day=torch.ones(8,dtype=torch.long),quality=torch.ones(8))

    def test_true_TX_and_same_reference_condition_and_no_same_RX_positive(self):
        x=self.inputs();a,p,n=mask(x['labels'],x['receiver'],x['day']).nonzero(as_tuple=True)
        self.assertGreater(len(a),0)
        self.assertTrue(bool((x['labels'][a]==x['labels'][p]).all()))
        self.assertTrue(bool((x['labels'][a]!=x['labels'][n]).all()))
        self.assertTrue(bool((x['receiver'][a]!=x['receiver'][p]).all()))
        self.assertTrue(bool((x['receiver'][p]==x['receiver'][n]).all()))
        self.assertTrue(bool((x['day'][p]==x['day'][n]).all()))

    def test_one_way_stops_clean_keys_and_two_way_updates_both_anchors(self):
        x=self.inputs();one,stats=semantic(**x)
        gc,gu=torch.autograd.grad(one,[x['clean'],x['urban']],allow_unused=True)
        self.assertIsNone(gc);self.assertGreater(float(gu.norm()),0.)
        two,_=semantic(**x,bidirectional=True)
        gc,gu=torch.autograd.grad(two,[x['clean'],x['urban']]);self.assertGreater(float(gc.norm()),0.);self.assertGreater(float(gu.norm()),0.)
        self.assertLessEqual(float(two),2.1);self.assertTrue(torch.isfinite(two))

    def test_CPU_metadata_is_moved_to_label_device_before_triplet_mask(self):
        # Native move_batch keeps RX/day on CPU while labels follow the model.
        # Meta is a distinct device available on the local CPU-only runtime.
        x=self.inputs()
        result=mask(torch.empty(8,device='meta',dtype=torch.long),x['receiver'],x['day'])
        self.assertEqual(result.device.type,'meta')
        self.assertEqual(tuple(result.shape),(8,8,8))

    def test_margin_satisfaction_stops_and_missing_groups_and_U_rejected(self):
        x=self.inputs();q=torch.zeros(8,160);q[torch.arange(8),x['labels']]=1.
        loss,_=semantic(**dict(x,clean=q,urban=q),bidirectional=True);self.assertEqual(float(loss),0.)
        loss,_=semantic(**dict(x,receiver=torch.ones(8,dtype=torch.long)));self.assertEqual(float(loss),0.)
        for change in (dict(labels=torch.full((8,),-1)),dict(quality=torch.ones(8)*1.1),dict(receiver=torch.full((8,),14)),dict(margin=3.)):
            with self.assertRaises(ValueError):semantic(**dict(x,**change))

    def test_actual_joint_gradients_reach_old_features_and_no_aux_W_or_gain(self):
        torch.set_num_threads(2);c=parent.config(parent.rows()[0]);model=build(c,'cpu').train()
        x=self.inputs();out,values=branches.forward(model,torch.randn(16,2,256));z=values['J']
        loss,_=semantic(**dict(x,clean=z[:8],urban=z[8:]),bidirectional=True)
        boundary=boundary_parameters(model);grads=torch.autograd.grad(loss,list(boundary.values()),allow_unused=True,retain_graph=True)
        self.assertTrue(all(g is None or float(g.abs().max())==0. for g in grads))
        core=model.id_backbone.encoder.core.id_backbone
        params=[core.t2.pw.weight,model.id_backbone.encoder.sequence_pool.views[0].local_projection.weight]
        grads=torch.autograd.grad(loss,params,retain_graph=True);self.assertTrue(all(float(g.norm())>0 for g in grads))
        ce=torch.nn.functional.cross_entropy(out['tx_logits'],torch.arange(16)%6)
        self.assertGreater(float(torch.autograd.grad(ce,core.cls_head.weight)[0].norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fixed_matrix_and_archived_controls_are_never_retrained(self):
        self.assertEqual([r['arm'] for r in d.rows()],['D0','D1','D2','D3','D4'])
        self.assertEqual([r['arm'] for r in d.fresh_rows()],['D0','D3','D4'])
        for row in d.rows():
            c=d.config(row)
            if c['reused']:
                self.assertEqual(c['reuse_source_config'],d.reference_config(c['arm']))
                with self.assertRaises(ValueError):d.make_args(c,'cpu')
            else:
                a=d.make_args(c,'cpu');self.assertTrue(a.use_unlabeled and a.use_ema_teacher)
                self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs),(50,33,17))
                self.assertEqual(a.use_concat_sat_channel_aug,c['arm']!='D0')
                self.assertTrue(a.use_aug)
        from . import evaluate
        for arm in ('D1','D2'):
            row=next(v for v in d.rows() if v['arm']==arm)
            with self.assertRaisesRegex(ValueError,'Reuse archived'):evaluate.predict(row['row_id'])
            with self.assertRaisesRegex(ValueError,'Reuse archived'):evaluate.preflight(row['row_id'])

    def test_recovery_rejects_unmatched_original_D0_and_reuses_calibration_without_forward(self):
        import json
        from .calibrate import reuse_original
        original=d.read(Path(d.__file__).with_name('archived_D0_r01.json'))
        self.assertEqual(original['run_id'],d.ORIGINAL_RUN)
        self.assertEqual(original['output_root'],(d.ORIGINAL_BASE/'D0-s2026092701/source').as_posix())
        self.assertFalse(original['reused']);self.assertFalse(original['ordinary_concat'])
        with self.assertRaises(ValueError):d.validate(original)
        with self.assertRaises(ValueError):d.reference_config('D0')
        matched=d.config(next(r for r in d.rows() if r['arm']=='D0'))
        self.assertFalse(matched['reused']);self.assertTrue(d.make_args(matched,'cpu').use_aug)
        expected=(d.parent.BASE/'A0-s2026092701/source/warmup_epoch5.pth').as_posix()
        r=dict(status='SOURCE_SEMANTIC_CALIBRATED',source_only=True,query_read=False,U_truth_read=False,
            V_forward_packets=0,optimizer_updates=0,head_fit_updates=0,source_count=720,warmup_epoch=5,
            feature_parameters=195041,calibration_seed=d.CALIBRATION_SEED,forward_packets=2160,margin=.1,
            model_state_unchanged=True,training_gradients_unchanged=True,source_checkpoint=expected,
            coefficients={arm:dict(coefficient=.2,target_gradient_ratio=.1) for arm in ('D3','D4')})
        copied=reuse_original(r,'original.json')
        self.assertEqual(copied['coefficients'],r['coefficients']);self.assertEqual(copied['new_forward_packets'],0)
        self.assertNotIn('reused_from',r)
        for change in (dict(query_read=True),dict(source_count=6300),dict(warmup_epoch=50),dict(model_state_unchanged=False)):
            with self.assertRaises(ValueError):reuse_original(dict(r,**change),'original.json')

    def test_actual_native_D0_D3_D4_cache_CE_teacher_and_gradient_boundaries(self):
        torch.set_num_threads(2)
        for row in d.fresh_rows():
            c=d.config(row);model=new_build(c,'cpu').train();teacher=copy.deepcopy(model).eval().requires_grad_(False)
            observer=Observer(c)
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory);d.write(out/'native_train_indices.json',dict(base_indices=list(range(128))))
                observer.initialize(model,c,out)
                try:
                    x=torch.randn(128,2,256);y=torch.arange(128)%6
                    meta=dict(base_index=torch.arange(128),rx_i=torch.tensor([(1,3,4,6,8)[(i//6)%5] for i in range(128)]),day_i=torch.ones(128,dtype=torch.long))
                    records=[dict(scenario='practical_low_urban',sample_id=str(i),session_id='rx:day',state_end='PARTIAL',
                        receiver_lock_status='degraded',quality_snr_db=8.,signal_to_added_noise_db=8.,output_frequency_hz=5.,residual_kernel_length=5,q=.8) for i in range(128)]
                    observer.channel.apply=lambda x,*a,**kw:(x+.1*torch.randn_like(x),records)
                    observer.steps=1110;observer.coefficients={arm:dict(coefficient=.1) for arm in ('D3','D4')}
                    native=SimpleNamespace(_urban_practical_training_hook=None,_source_feature_auxiliary_hook=None,
                        _prepare_concat_sat_batch_for_training=lambda *a,**kw:None,move_batch=lambda b,dev:(b[0],b[1],(None,b[2])))
                    args=d.make_args(c,'cpu')
                    with observer.native_scope(native):
                        native.move_batch((x,y,meta),'cpu')
                        augmentor=(lambda z,**kw:z+.01) if args.use_aug else None
                        concat_active=bool(args.use_concat_sat_channel_aug)
                        # Match both actual native placements of generic IQ aug.
                        prepared=torch.nan_to_num(augmentor(x,labels=y)) if augmentor is not None and concat_active else x
                        xx,yy,_,_,_=native._prepare_concat_sat_batch_for_training(None,prepared,y,None,args=args,epoch=6,batch_idx=1)
                        x_l_main=torch.nan_to_num(augmentor(xx,labels=yy)) if augmentor is not None and not concat_active else xx
                        result=model(x_l_main,y_tx=yy,grl_lambda=1.,return_aux=True,domain_labels=None)
                        self.assertTrue(observer.armed.done)
                        self.assertIs(observer.armed.x,x_l_main)
                        self.assertTrue(torch.allclose(x_l_main[:128],x+.01))
                        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
                        ce=torch.nn.functional.cross_entropy(result['tx_logits'],yy,label_smoothing=.01)
                        extra,metrics=observer.observe(model=model,args=args,epoch=6,batch_idx=1,clean_iq=xx,labels=yy,clean_meta=meta,
                            clean_output=result,clean_ce=ce,unlabeled_iq=None,unlabeled_meta={},teacher_prob=None,weak_mask=None,teacher_model=teacher)
                        self.assertEqual(observer.counts['teacher_L_clean_packets'],128)
                        if c['arm']=='D0':self.assertEqual(len(xx),128);self.assertEqual(float(extra),0.)
                        else:self.assertEqual(len(xx),256);self.assertGreater(float(extra),0.)
                        (ce+extra).backward();self.assertGreater(float(model.id_backbone.encoder.core.id_backbone.cls_head.weight.grad.norm()),0.)
                        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                    self.assertIsNone(native._urban_practical_training_hook)
                finally:observer.trace.close()


if __name__=='__main__':unittest.main()
