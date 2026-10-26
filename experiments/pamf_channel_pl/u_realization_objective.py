"""Matched original losses on four equally scheduled physical channel realizations."""
from pathlib import Path
from experiments.cric_dura_comparison.io import read
from .feature_task_training import PairedCache,extra as task_extra
from .restoration_bridge import extra as relation_extra
from .asymmetric_hg import extra as joint_extra
from .u_realization_cache import RUN as NEW_CACHE_RUN,KEYS,namespace,hidden
from .restored_source import SCENES

VARIANTS=('GTaskMulti','GGCLMulti','HGCLMulti','GJointMulti')
WEIGHTS=dict(GTaskMulti=.05941532657537612,GGCLMulti=.04677712061808677,
             HGCLMulti=.04677712061808677,GJointMulti=.04221045401136996)
ROUTES=dict(GTaskMulti=('G','G'),GGCLMulti=('G','G_next'),HGCLMulti=('H','G'),GJointMulti=('H','G'))
CONTROLS=dict(GTaskMulti=('20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01','GFeatureHard'),
    GGCLMulti=('20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01','BridgeGG'),
    HGCLMulti=('20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01','BridgeHG'),
    GJointMulti=('20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01','GJoint'))


def schedule(call):
    if type(call) is not int or not 1<=call<=4440:raise ValueError('Twenty complete source U epochs only')
    return ((call-1)//222)%4,(call+6659)%3


def recipe(method):
    if method not in VARIANTS:raise ValueError('Four fixed input-support controls required')
    return dict(method=method,views=list(ROUTES[method]),weight=WEIGHTS[method],fixed_control=list(CONTROLS[method]),
        replicas=[0,1,2,3],replica_schedule='extra_U_epoch_mod4',scene_schedule='original_cumulative_step_mod3',
        extra_forwards=2,held_extra_rx=1,ramp_steps=444,auxiliary_classifier_weight_shield=True,
        pseudo_target='argmax_frozen_three_E30_raw_consensus95',main='unchanged_ordinary_self_PL',
        loss='mean_two_feature_task_CE' if method=='GTaskMulti' else ('G_CE_plus_crossview_crossRX_CL' if method=='GJointMulti' else 'crossview_crossRX_CL'),
        normalization='unchanged_matching_fixed_control',new_head=False,G_updates=0,teacher_updates=0,reference_updates=0)


class MultiCache:
    def __init__(self,c):
        import numpy as np
        self.fixed=PairedCache(c);self.base=self.fixed.base;self.contract=read(c['source_contract'])
        root=Path(c['u_realization_cache']);complete=read(root/'complete.json')
        if (root!=Path(c['project'])/'runs'/NEW_CACHE_RUN or complete['status']!='SOURCE_U_REALIZATION_CACHE_COMPLETE'
                or complete['replicas']!=[1,2,3] or complete['U_physical_records']!=56700
                or any(complete[k] for k in ('F_weight_reads','F_output_reads','F_forward_packets','U_truth_read','V_reads','target_reads','optimizer_updates','G_updates'))
                or not complete['all_G_parameters_and_registered_buffers_unchanged']):
            raise ValueError('Complete G-only hidden-U realization cache required')
        self.h={0:self.fixed.h};self.g={0:self.base.iq}
        expected=np.asarray(list(self.base.index),np.int64)
        for replica in (1,2,3):
            folder=root/('fresh'+str(replica));done=read(folder/'complete.json')
            if (done['status']!='SOURCE_U_HG_REALIZATION_COMPLETE' or done['replica']!=replica or done['namespace']!=namespace(replica)
                    or done['source_contract']!=self.contract or done['G']!=c['source_G'] or done['U_seen']!=56700
                    or done['scenes']!=list(SCENES) or done['physical_keys']!=list(KEYS)
                    or any(done[k] for k in ('F_weight_reads','F_output_reads','F_forward_packets','U_truth_read','V_reads','target_reads','optimizer_updates','G_updates'))
                    or not done['G_parameters_and_registered_buffers_unchanged']):
                raise ValueError('Aligned source-only H/G realization required')
            with np.load(folder/'physical_index.npz',allow_pickle=False) as a:
                if not np.array_equal(a['physical'],expected) or list(a['ids'])!=self.contract['role_ids']['U_s']:
                    raise ValueError('Fresh and original physical packet order differs')
            self.h[replica]=np.load(folder/'channel_U.npy',mmap_mode='r',allow_pickle=False)
            self.g[replica]=np.load(folder/'restored_U.npy',mmap_mode='r',allow_pickle=False)
            if any(a.shape!=(3,56700,2,256) or a.dtype!=np.float32 for a in (self.h[replica],self.g[replica])):
                raise ValueError('Three finite FP32 paired scenes required')

    def take(self,metadata,k,device,replica,next_G=False):
        import numpy as np
        import torch
        if type(replica) is not int or replica not in (0,1,2,3) or k not in (0,1,2):raise ValueError('Registered realization and scene required')
        q,mask,_=self.base.take(metadata,k,device,False)
        indices=[self.base.index[v] for v in zip(*(metadata[name].detach().cpu().tolist() for name in KEYS))]
        def tensor(value):
            a=np.ascontiguousarray(value,np.float32)
            return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
        h=tensor(self.h[replica][k,indices]);g=tensor(self.g[replica][k,indices])
        other=tensor(self.g[replica][(k+1)%3,indices]) if next_G else None
        if not all(bool(torch.isfinite(v).all()) for v in (h,g) if v is not None) or (other is not None and not bool(torch.isfinite(other).all())):
            raise ValueError('Finite actual paired source IQ required')
        return q,mask,h,g,other


def auxiliary(student,views,domain,q,mask,rx,method,check=False):
    if method=='GTaskMulti':
        values=[task_extra(student,v,domain,q,mask,True,check=check) for v in views]
        value=sum(values)/2
        return value,dict(valid_anchors=0,positive_pairs=0,negative_pairs=0,CE_H_accepted=0,CE_G_accepted=2*int(mask.sum()),
            CE_loss=value.detach(),relation_loss=0.)
    if method in ('GGCLMulti','HGCLMulti'):
        value,detail=relation_extra(student,views,domain,q,mask,rx,'bridge')
        detail.update(CE_H_accepted=0,CE_G_accepted=0,CE_loss=0.,relation_loss=value.detach())
        return value,detail
    if method=='GJointMulti':return joint_extra(student,views,domain,q,mask,rx,'GJoint',check=check)
    raise ValueError('Declared matching fixed loss required')


class Objective:
    def __init__(self,c):
        from .u_realization_training import validate_config
        validate_config(c);self.c=c;self.cache=MultiCache(c);self.metrics={};self.weight=WEIGHTS[c['paper_method']]
        self.calls=self.seen=self.raw_accepted=self.extra_accepted=self.held_seen=0
        self.H_forwards=self.G_forwards=self.valid_anchors=self.positive_pairs=self.negative_pairs=0
        self.CE_H_accepted=self.CE_G_accepted=0
        self.replica_seen=[0]*4;self.replica_accepted=[0]*4;self.replica_calls=[0]*4
        self.expected_extra_per_epoch=sum(int(self.cache.base.accepted[i]) for key,i in self.cache.base.index.items() if key[1]!=1)
        if self.expected_extra_per_epoch!=41939:raise ValueError('Original fixed raw consensus acceptance required')

    def loss(self,student,device):
        import torch
        from torch.nn import functional as F
        try:batch=next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        hidden(batch,self.cache.contract)
        x=batch[0].to(device);domain=batch[2].to(device);rx=batch[3]['rx_i'].to(device)
        student.eval()
        with torch.no_grad():confidence,pseudo=student(x).softmax(1).max(1)
        student.train();accepted=confidence>=.95
        logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
        main=(F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean()
        self.calls+=1;replica,k=schedule(self.calls);method=self.c['paper_method']
        q,mask,h,g,gnext=self.cache.take(batch[3],k,device,replica,next_G=method=='GGCLMulti');mask=mask&(rx!=1)
        values=dict(H=h,G=g,G_next=gnext);views=[values[v] for v in ROUTES[method]]
        extra,detail=auxiliary(student,views,domain,q,mask,rx,method);ramp=min(self.calls/444.,1.)
        n=len(x);a=int(mask.sum());self.seen+=n;self.raw_accepted+=int(accepted.sum());self.extra_accepted+=a;self.held_seen+=int((rx==1).sum())
        self.replica_seen[replica]+=n;self.replica_accepted[replica]+=a;self.replica_calls[replica]+=1
        self.H_forwards+=ROUTES[method].count('H')*n;self.G_forwards+=sum(v.startswith('G') for v in ROUTES[method])*n
        self.valid_anchors+=detail['valid_anchors'];self.positive_pairs+=detail['positive_pairs'];self.negative_pairs+=detail['negative_pairs']
        self.CE_H_accepted+=detail['CE_H_accepted'];self.CE_G_accepted+=detail['CE_G_accepted']
        student._cd_U_seen+=n;student._cd_U_accepted+=int(accepted.sum())
        self.metrics=dict(UR_main_loss=main.detach(),UR_extra_loss=extra.detach(),UR_CE_loss=detail['CE_loss'],UR_CL_loss=detail['relation_loss'],
            UR_extra_weight=self.weight,UR_ramp=ramp,UR_replica=replica,UR_scene=k,UR_U_seen=self.seen,UR_raw_accepted=self.raw_accepted,
            UR_extra_accepted=self.extra_accepted,UR_held_seen=self.held_seen,UR_H_forwards=self.H_forwards,UR_G_forwards=self.G_forwards,
            UR_CE_H_accepted=self.CE_H_accepted,UR_CE_G_accepted=self.CE_G_accepted,UR_valid_anchors=self.valid_anchors,
            UR_positive_pairs=self.positive_pairs,UR_negative_pairs=self.negative_pairs,UR_G_updates=0,UR_teacher_updates=0,UR_U_truth=0)
        return main+self.weight*ramp*extra

    def state_dict(self):
        return dict(calls=self.calls,U_seen=self.seen,raw_accepted=self.raw_accepted,extra_accepted=self.extra_accepted,held_seen=self.held_seen,
            extra_student_forwards=2*self.seen,expected_extra_per_epoch=self.expected_extra_per_epoch,H_forwards=self.H_forwards,G_forwards=self.G_forwards,
            valid_anchors=self.valid_anchors,positive_pairs=self.positive_pairs,negative_pairs=self.negative_pairs,
            CE_H_accepted=self.CE_H_accepted,CE_G_accepted=self.CE_G_accepted,
            replica_seen=self.replica_seen,replica_accepted=self.replica_accepted,replica_calls=self.replica_calls,
            reference_updates=0,G_updates=0,teacher_updates=0,U_truth_read=False,target_reads=0,objective_V_reads=0,V_state_updates=0,new_head=False,
            extra_weight=self.weight,auxiliary_classifier_weight_shield=True,views=list(ROUTES[self.c['paper_method']]),
            pseudo_target='hard_argmax_frozen_three_E30_raw_consensus95',fixed_control=list(CONTROLS[self.c['paper_method']]),new_inference_parameters=0)
