"""Focused negative/factual checks for the new source fold and native graph wiring."""
import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,branches as b,gradient as g
from .inner_source import partition
from .model import build
from .observer import Observer


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.c=d.config(d.rows()[0]);cls.model=build(cls.c,'cpu')

    def test_global_ratio_of_medians_and_degenerate_rejection(self):
        result=g.coefficient([1.,100.,100.],[1.,1.,100.])
        self.assertAlmostEqual(result['coefficient'],10.,places=8)
        # A median of sample ratios would give .1, a materially different recipe.
        for a,bv in [([1.],[0.]),([float('nan')],[1.]),([1.],[1e-7])]:
            with self.assertRaises(ValueError):g.coefficient(a,bv)
        p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(3))
        value=g.vectors(p.square().sum(),[('p',p),('q',q)])
        self.assertTrue(torch.equal(value['q'],torch.zeros_like(q)))

    def test_internal_source_fold_balanced_disjoint_and_order_independent(self):
        index=[SimpleNamespace(tx_i=tx,rx_i=rx,day_i=day,number=i)
            for tx in range(6) for rx in (1,3,4,6,8) for day in (1,2,3) for i in range(70)]
        def opaque(v):return f'{v.tx_i}/{v.rx_i}/{v.day_i}/{v.number}'
        train,dev=partition(index,opaque);self.assertEqual((len(train),len(dev)),(5580,720))
        self.assertFalse(set(train)&set(dev));held={opaque(index[i]) for i in dev}
        reverse=list(reversed(index));_,new=partition(reverse,opaque)
        self.assertEqual(held,{opaque(reverse[i]) for i in new})
        with self.assertRaises(ValueError):partition(index[:-1],opaque)

    def test_eight_matched_configs_and_frozen_changes_rejected(self):
        self.assertEqual([r['arm'] for r in d.rows()],list(d.ARMS))
        for row in d.rows():
            c=d.config(row);self.assertTrue(c['ordinary_concat']);self.assertEqual(c['auxiliary']['start_epoch'],6)
            self.assertFalse(c['curriculum'] or c['margin'] or c['urban_ssl'])
            for key,value in [('epochs',200),('default_head','retrained'),('source_roles',dict(L_s=6300)),('parent_checkpoint','old.pth')]:
                bad=copy.deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):d.validate(bad)

    def test_source_family_gate_preserves_negative_controls(self):
        from .selection import pilot
        base=dict(clean_accuracy=.9,scene_accuracy={'practical_low_urban':.5},urban_mean=.6,worst_urban_RX=.5)
        records={key:copy.deepcopy(base) for key in d.ARMS}
        self.assertFalse(pilot(records)['family_expandable'])
        records['A-R']['scene_accuracy']['practical_low_urban']=.52
        self.assertTrue(pilot(records)['family_expandable'])
        records['A-R']['clean_accuracy']=.88
        self.assertFalse(pilot(records)['family_expandable'])
        records.pop('A-F')
        with self.assertRaises(ValueError):pilot(records)

    def test_recipe_publication_is_complete_and_does_not_overwrite(self):
        from .calibrate import publish
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recipe.json';publish({'status':'complete','records':list(range(1000))},path)
            self.assertEqual(d.read(path)['status'],'complete')
            self.assertFalse(path.with_suffix('.json.writing').exists())
            with self.assertRaises(FileExistsError):publish({'status':'overwrite'},path)

    def test_native_concat_graph_removed_before_teacher_U_and_aux_boundary(self):
        model=copy.deepcopy(self.model).train();teacher=copy.deepcopy(model).eval().requires_grad_(False)
        c=d.config(next(r for r in d.rows() if r['arm']=='A-S'));observer=Observer(c)
        with tempfile.TemporaryDirectory() as directory,ExitStack() as cleanup:
            observer.initialize(model,c,Path(directory))
            cleanup.callback(observer.trace.close)
            # The channel is independently exercised by existing practical checks;
            # this bounded test isolates the actual native concat/autograd boundary.
            records=[dict(scenario='practical_low_urban',sample_id=str(i),session_id='rx1:day1:eq1',
                state_end='PARTIAL',receiver_lock_status='degraded',quality_snr_db=8.,
                signal_to_added_noise_db=8.,output_frequency_hz=5.,residual_kernel_length=5,q=.5) for i in range(2)]
            observer.channel.apply=lambda x,*args,**kw:(x+.05*torch.randn_like(x),records)
            native=SimpleNamespace(_urban_practical_training_hook=None,_source_feature_auxiliary_hook=None,
                _prepare_concat_sat_batch_for_training=lambda *a,**kw:None,
                move_batch=lambda batch,device:(batch[0],batch[1],(None,batch[2])))
            x=torch.randn(2,2,256);meta=dict(rx_i=torch.tensor([1,1]))
            with torch.no_grad():y=teacher.forward_identity_only(x)['tx_logits'].argmax(1)
            args=d.make_args(c,'cpu');observer.steps=1110
            observer.coefficients={key:dict(coefficient=.1) for key in (*b.BRANCHES,'J','ALL')}
            with observer.native_scope(native):
                native.move_batch((x,y,meta),'cpu')
                xx,yy,dd,_,_=native._prepare_concat_sat_batch_for_training(None,x,y,None,args=args,epoch=6,batch_idx=1)
                before={id(m):len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules()}
                out=model(xx,y_tx=yy,grl_lambda=1.,return_aux=True,domain_labels=None)
                self.assertTrue(observer.armed.done)
                self.assertTrue(all(v.requires_grad for k,v in observer.armed.values.items() if k in b.BRANCHES))
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
                loss=F.cross_entropy(out['tx_logits'],yy,label_smoothing=args.label_smoothing)
                extra,metrics=observer.observe(model=model,args=args,epoch=6,batch_idx=1,
                    clean_iq=xx,labels=yy,clean_meta=meta,clean_output=out,clean_ce=loss,
                    unlabeled_iq=None,unlabeled_meta={},teacher_prob=None,weak_mask=None,teacher_model=teacher)
                self.assertGreater(float(extra),0.);self.assertIsNone(observer.armed)
                (loss+extra).backward()
                core=model.id_backbone.encoder.core.id_backbone
                self.assertGreater(float(core.cls_head.weight.grad.norm()),0.)
                self.assertGreater(float(core.t2.pw.weight.grad.norm()),0.)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            observer.trace.close()
            self.assertIsNone(native._urban_practical_training_hook)


if __name__=='__main__':unittest.main()
