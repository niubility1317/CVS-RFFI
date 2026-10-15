"""Focused subset, no-U, default-head and actual CE graph checks."""
import copy,tempfile,unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,subset
from .model import build
from .observer import Observer
from experiments.cvs_phase1_stack.runtime import installed


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.c=d.config(d.rows()[0]);cls.model=build(cls.c,'cpu')

    def test_equal_subsets_TX_day_and_RX_exclusion_independent_of_index_order(self):
        index=[SimpleNamespace(tx_i=t,rx_i=r,day_i=day,number=i) for t in range(6) for r in (1,3,4,6,8) for day in (1,2,3) for i in range(70)]
        def opaque(v):return f'{v.tx_i}/{v.rx_i}/{v.day_i}/{v.number}'
        selected={}
        for arm in d.ARMS:
            train,dev,unused=subset.select(index,opaque,arm)
            self.assertEqual((len(train),len(dev),len(unused)),(4464,720,1116));self.assertFalse(set(train)&set(dev))
            self.assertEqual(set(Counter((index[i].tx_i,index[i].day_i) for i in train).values()),{248})
            self.assertEqual(set(Counter(index[i].tx_i for i in train).values()),{744})
            reverse=list(reversed(index));other,_,_=subset.select(reverse,opaque,arm)
            self.assertEqual({opaque(index[i]) for i in train},{opaque(reverse[i]) for i in other})
            selected[arm]=[index[i] for i in train]
        self.assertNotIn(1,{v.rx_i for v in selected['EXCLUDE_RX1']})
        self.assertEqual({v.rx_i for v in selected['SAMPLE']},{1,3,4,6,8})
        with self.assertRaises(ValueError):subset.select(index[:-1],opaque,'SAMPLE')

    def test_actual_args_close_U_EMA_augmentation_and_frozen_config_rejects_changes(self):
        a=d.make_args(self.c,'cpu')
        self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs),(50,50,0))
        self.assertFalse(a.use_unlabeled or a.use_ema_teacher or a.use_concat_sat_channel_aug)
        self.assertFalse(a.use_aug)
        self.assertEqual(a.sat_training_mode,'disabled')
        self.assertTrue(all(v==0. for k,v in vars(a).items() if k.startswith('lambda_')))
        old=d.parent.make_args(d.operational_parent(),'cpu')
        self.assertFalse(old.pseudo_domain_gate);self.assertEqual(old.pseudo_temporal_mode,'batch_neighbor')
        print('ACTUAL_OLD_GATE',old.pseudo_threshold_mode,old.pseudo_temporal_window,old.pseudo_temporal_min_conf,old.pseudo_strong_agreement,flush=True)
        for key,value in [('epochs',200),('parent_checkpoint','all-RX.pth'),('default_head','refit')]:
            c=copy.deepcopy(self.c);c[key]=value
            with self.assertRaises(ValueError):d.validate(c)

    def test_fixed222_steps_never_iterate_U(self):
        class ForbiddenU:
            def __len__(self):return 222
            def __iter__(self):raise AssertionError('U was opened')
        with installed(self.c) as native:
            values=list(native._muse_epoch_pairs([1,2],ForbiddenU(),use_muse=False))
        self.assertEqual(len(values),222);self.assertTrue(all(v[1] is None for v in values))

    def test_same_initial_network_and_head_joint_CE_and_exposure_negatives(self):
        other=build(d.config(d.rows()[1]),'cpu')
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),863730)
        self.assertTrue(all(torch.equal(v,other.state_dict()[k]) for k,v in self.model.state_dict().items()))
        c=d.config(d.rows()[1]);observer=Observer(c);model=copy.deepcopy(other).train()
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);d.write(out/'inner_source_contract.json',dict(train_native_base_indices=list(range(128)),train_receivers=[3,4,6,8]))
            observer.initialize(model,c,out)
            try:
                x=torch.randn(128,2,256);y=torch.arange(128)%6;meta={k:torch.tensor(v) for k,v in dict(base_index=list(range(128)),rx_i=[3]*128,day_i=[1]*128).items()}
                result=model(x,y_tx=y,grl_lambda=1.,return_aux=True,domain_labels=None);ce=F.cross_entropy(result['tx_logits'],y,label_smoothing=.01)
                kwargs=dict(model=model,args=d.make_args(c,'cpu'),epoch=1,batch_idx=1,clean_iq=x,labels=y,clean_meta=meta,clean_output=result,clean_ce=ce,
                    unlabeled_iq=None,unlabeled_meta=None,teacher_prob=None,weak_mask=None,teacher_model=None)
                bad=copy.deepcopy(meta);bad['rx_i'][0]=1
                with self.assertRaises(ValueError):observer.observe(**dict(kwargs,clean_meta=bad))
                with self.assertRaises(ValueError):observer.observe(**dict(kwargs,unlabeled_iq=x))
                extra,_=observer.observe(**kwargs);self.assertEqual(float(extra),0.)
                ce.backward();core=model.id_backbone.encoder.core.id_backbone
                self.assertGreater(float(core.cls_head.weight.grad.norm()),0.)
                self.assertGreater(float(core.t2.pw.weight.grad.norm()),0.)
                self.assertGreater(float(model.id_backbone.encoder.sequence_pool.views[0].local_projection.weight.grad.norm()),0.)
            finally:observer.handle.remove();observer.trace.close()


if __name__=='__main__':unittest.main()
