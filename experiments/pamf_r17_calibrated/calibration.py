import numpy as np
import torch
from torch import nn
from . import design as d
from experiments.pamf_restoration.runner import array_tensor,read_json
from tools.r17_activation_factorization import InputSecondMoment,factor_rank


def smoke_info():
    return dict(mode='smoke2Ls',examples=2,L_s=2,U_s=0,source_run=d.ACTION_RUN,received_epoch=None,
        selection_seed=None,positions=d.CALIBRATION_POSITIONS,ridge_fraction=d.RIDGE_FRACTION,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,teacher_parameter_updates=0)


def source_batches(parent_action,device):
    if parent_action.name!=d.ACTION_RUN:raise ValueError('Only original R07 training cache allowed before IO')
    cache=parent_action/'cache';done=read_json(cache/'complete.json')
    if any(done.get(k)!=v for k,v in dict(status='ACTION_SOURCE_CACHE_COMPLETE',train_count=63000,
            source_V_count=1800,epochs=20,TX_labels_read=0,target_reads=0).items()):
        raise ValueError('Existing complete original source cache required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Only original complete L/U training roles allowed')
    received=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    if received.shape!=(63000,2,256):raise ValueError('Original epoch01 training received IQ required')
    selected=np.random.default_rng(d.CALIBRATION_SEED).permutation(63000)[:d.CALIBRATION_COUNT]
    info=dict(smoke_info(),mode='source_train512',examples=len(selected),L_s=int((roles[selected]=='L_s').sum()),
        U_s=int((roles[selected]=='U_s').sum()),received_epoch=1,selection_seed=d.CALIBRATION_SEED)
    def batches():
        for offset in range(0,len(selected),d.CALIBRATION_BATCH):
            yield array_tensor(received[selected[offset:offset+d.CALIBRATION_BATCH]],device)
    return batches(),info


@torch.no_grad()
def collect_moments(teacher,batches,fraction):
    if teacher.candidate!='unroll_unshared' or teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Frozen eval original R17 teacher required')
    stats={};handles=[];examples=0
    for stage,core in enumerate(teacher.prior):
        for name,conv in core.named_modules():
            if not isinstance(conv,nn.Conv1d) or conv.kernel_size[0]<=1:continue
            try:factor_rank(conv,fraction)
            except ValueError:continue
            stat=InputSecondMoment(conv,d.CALIBRATION_POSITIONS);stats[stage,name]=stat
            def observe(module,inputs,stat=stat):stat.update(inputs[0])
            handles.append(conv.register_forward_pre_hook(observe))
    if not stats:raise ValueError('Eligible original teacher layers required')
    try:
        for received in batches:
            if received.ndim!=3 or received.shape[1:]!=(2,256) or not torch.isfinite(received).all():
                raise ValueError('Finite source received IQ2x256 required')
            output=teacher(received)
            if output.shape!=received.shape or not torch.isfinite(output).all():raise ValueError('Finite original teacher output required')
            examples+=len(received)
        if not examples:raise ValueError('Actual source calibration examples required')
        moments={key:stat.value() for key,stat in stats.items()}
        report=dict(observed_examples=examples,eligible_layers=len(stats),
                    sampled_patches_min=min(stat.count for stat in stats.values()),sampled_patches_max=max(stat.count for stat in stats.values()))
        return moments,report
    finally:
        for handle in handles:handle.remove()


def validate_calibration(value,arm,epoch):
    if not isinstance(value,dict) or arm not in d.ARMS:raise ValueError('Actual source calibration metadata required')
    smoke=epoch==0;examples=2 if smoke else d.CALIBRATION_COUNT
    required=dict(mode='smoke2Ls' if smoke else 'source_train512',examples=examples,observed_examples=examples,
        source_run=d.ACTION_RUN,received_epoch=None if smoke else 1,selection_seed=None if smoke else d.CALIBRATION_SEED,
        positions=d.CALIBRATION_POSITIONS,ridge_fraction=d.RIDGE_FRACTION,eligible_layers=84 if arm=='activation_half' else 81,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,teacher_parameter_updates=0)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Calibration source/budget/permissions differ')
    if (any(not isinstance(value.get(k),int) or value[k]<0 for k in ('L_s','U_s'))
            or value['L_s']+value['U_s']!=examples or (smoke and (value['L_s'],value['U_s'])!=(2,0))):
        raise ValueError('Actual calibration L/U exposures required')
    if value.get('sampled_patches_min')!=examples*d.CALIBRATION_POSITIONS or value.get('sampled_patches_max')!=examples*d.CALIBRATION_POSITIONS:
        raise ValueError('Actual fixed per-layer patch exposures required')
