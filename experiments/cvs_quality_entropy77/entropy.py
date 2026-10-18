"""Bounded physical target entropy; private permutation and no label-based RNG."""
import hashlib
import json
import math
import numbers
import numpy as np
import torch
from torch.nn import functional as F

BASE_EPSILON=.01
MAX_EPSILON=.20
PERMUTATION_SEED=20261016421
PERMUTATION_PART='quality_entropy77/permutation'


def stable_seed(seed,*parts):
    raw=json.dumps([int(seed),*parts],ensure_ascii=False,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(raw.encode('utf-8')).digest()[:8],'little')


def epsilon_from_snr(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or math.isnan(float(value)):
        raise ValueError('Numeric native added-noise ratio required; NaN/missing is invalid')
    s=float(value);u=s*(math.log(10.)/10.)
    r=math.exp(-abs(u))
    fraction=r/(1.+r) if u>=0 else 1./(1.+r)
    result=BASE_EPSILON+(MAX_EPSILON-BASE_EPSILON)*fraction
    if not math.isfinite(result) or not BASE_EPSILON<=result<=MAX_EPSILON:
        raise FloatingPointError('Bounded physical epsilon calculation failed')
    return result


def resolve_snr(record):
    if not isinstance(record,dict) or 'signal_to_added_noise_db' not in record:
        raise ValueError('Actual native added-noise ratio field required')
    value=record['signal_to_added_noise_db']
    if value is not None:
        epsilon_from_snr(value)
        value=float(value)
        return value,('finite' if math.isfinite(value) else ('explicit_positive_infinity' if value>0 else 'explicit_negative_infinity'))
    power=record.get('measured_added_noise_power_pre_agc')
    if (isinstance(power,(bool,np.bool_)) or not isinstance(power,numbers.Real)
            or not math.isfinite(float(power)) or float(power)<0):
        raise ValueError('A missing native ratio requires finite nonnegative measured pre-AGC added-noise power')
    return (float('-inf'),'zero_signal') if float(power)>0 else (float('inf'),'zero_added_noise')


def from_records(records,namespace,mode):
    resolved=[resolve_snr(record)for record in records]
    result=assignment([value[0]for value in resolved],namespace,mode)
    def encoded(value):
        if value is None:return None
        value=float(value)
        return value if math.isfinite(value) else ('+inf' if value>0 else '-inf')
    kinds=[value[1]for value in resolved]
    result.update(signal_to_added_noise_db_native=[encoded(record['signal_to_added_noise_db'])for record in records],
        snr_boundary_kinds=kinds,snr_finite_count=kinds.count('finite'),
        snr_explicit_infinite_count=kinds.count('explicit_positive_infinity')+kinds.count('explicit_negative_infinity'),
        snr_zero_signal_count=kinds.count('zero_signal'),snr_zero_added_noise_count=kinds.count('zero_added_noise'),
        added_noise_ratio_scope='measured_pre_AGC_signal_to_added_noise_not_total_SNR')
    return result


def assignment(values,namespace,mode):
    if mode not in ('R','Q') or not isinstance(namespace,str) or not namespace.startswith('FcrossRX/train/e') or not namespace.endswith('/L_S') or len(values)!=64:
        raise ValueError('One fixed Q/R source As64 namespace required')
    original=[epsilon_from_snr(value) for value in values]
    seed=stable_seed(PERMUTATION_SEED,namespace,PERMUTATION_PART) if mode=='R' else None
    order=np.random.default_rng(seed).permutation(64).tolist() if mode=='R' else list(range(64))
    assigned=[original[i] for i in order]
    if sorted(original)!=sorted(assigned) or sorted(order)!=list(range(64)):
        raise ValueError('The exact original epsilon multiset must be preserved')
    def encoded(value):
        value=float(value)
        return value if math.isfinite(value) else ('+inf' if value>0 else '-inf')
    return dict(mode=mode,namespace=namespace,signal_to_added_noise_db=[encoded(v) for v in values],
        epsilon_original=original,epsilon_assigned=assigned,permutation=order,permutation_seed=seed,
        permutation_base_seed=PERMUTATION_SEED,permutation_part=PERMUTATION_PART,
        epsilon_multiset_preserved=True,epsilon_same_both_passes=True)


def variable_ce(logits,y,epsilon):
    if (not torch.is_tensor(logits) or logits.ndim!=2 or logits.shape!=(len(y),6) or len(y)==0
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
            or not torch.is_tensor(y) or y.dtype!=torch.long or y.ndim!=1 or y.device!=logits.device
            or not bool(((y>=0)&(y<6)).all()) or not torch.is_tensor(epsilon) or epsilon.shape!=y.shape
            or epsilon.device!=logits.device or epsilon.dtype!=logits.dtype or epsilon.requires_grad
            or not bool(torch.isfinite(epsilon).all()) or not bool(((epsilon>=BASE_EPSILON)&(epsilon<=MAX_EPSILON)).all())):
        raise ValueError('Finite visible L logits/labels and detached bounded epsilon required')
    if bool((epsilon==BASE_EPSILON).all()):
        return F.cross_entropy(logits,y,label_smoothing=BASE_EPSILON)
    logp=F.log_softmax(logits,dim=1)
    nll=-logp.gather(1,y[:,None]).squeeze(1);uniform=-logp.mean(1)
    return ((1.-epsilon)*nll+epsilon*uniform).mean()


def paired_delta(z0,z1,y,epsilon):
    if z0.shape!=z1.shape or z0.device!=z1.device or z0.dtype!=z1.dtype:
        raise ValueError('Two aligned original As default-head outputs required')
    new0,new1=variable_ce(z0,y,epsilon),variable_ce(z1,y,epsilon)
    old0=F.cross_entropy(z0,y,label_smoothing=BASE_EPSILON)
    old1=F.cross_entropy(z1,y,label_smoothing=BASE_EPSILON)
    weighted=.125*((new0-old0)+(new1-old1))
    if not bool(torch.isfinite(weighted)):raise FloatingPointError('Nonfinite weighted As entropy correction')
    return dict(weighted_delta=weighted,nominal_CE0=old0,nominal_CE1=old1,new_CE0=new0,new_CE1=new1,
        all_default=bool((epsilon==BASE_EPSILON).all()),As_CE_passes=2,per_pass_weight=.125)


def summary(mapping,losses,base_total,total):
    original=mapping['epsilon_original'];assigned=mapping['epsilon_assigned']
    mean=math.fsum(original)/len(original);std=math.sqrt(math.fsum((v-mean)**2 for v in original)/len(original))
    result=dict(mode=mapping['mode'],epsilon_min=min(original),epsilon_max=max(original),epsilon_mean=mean,epsilon_std=std,
        assigned_epsilon_min=min(assigned),assigned_epsilon_max=max(assigned),assigned_epsilon_mean=math.fsum(assigned)/len(assigned),
        epsilon_multiset_preserved=mapping['epsilon_multiset_preserved'],epsilon_same_both_passes=True,As_CE_passes=2,
        nominal_CE0=float(losses['nominal_CE0'].detach()),nominal_CE1=float(losses['nominal_CE1'].detach()),
        new_CE0=float(losses['new_CE0'].detach()),new_CE1=float(losses['new_CE1'].detach()),
        weighted_delta=float(losses['weighted_delta'].detach()),base_total_L=base_total,total_L=total,
        per_pass_weight=.125,total_As_weight=.25,all_labeled_CE_retained=True,As_KL=False,
        scope='Only As CE label smoothing; original ordinary/raw CE KL SupCon and U retained')
    result.update(assigned_CE0=result['new_CE0'],assigned_CE1=result['new_CE1'])
    for key in ('snr_finite_count','snr_explicit_infinite_count','snr_zero_signal_count','snr_zero_added_noise_count','added_noise_ratio_scope'):
        if key in mapping:result[key]=mapping[key]
    return result
