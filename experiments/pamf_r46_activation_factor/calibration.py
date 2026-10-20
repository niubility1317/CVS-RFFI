import time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from tools.r46_activation_factorization import SharedInputMoments,eligible_groups
from experiments.pamf_r44_pruned_refine import runtime as parent_runtime
from experiments.pamf_r46_complex_fidelity.runtime import validate_parent_root
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json

SCHEMA='R46_source_activation_moments_v1'


def choose_positions(roles):
    if roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Original TX-hidden6300Ls/56700Us source-training roles required')
    take=[]
    for role in ('L_s','U_s'):
        positions=np.flatnonzero(roles==role)
        take.extend(positions[np.linspace(0,len(positions)-1,d.CAL_PER_ROLE,dtype=np.int64)].tolist())
    if len(set(take))!=d.CAL_UNIQUE:
        raise ValueError('Distinct evenly spaced source-role positions required')
    return np.asarray(take,dtype=np.int64)


def validate_calibration(value,teacher=None):
    expected=dict(schema=SCHEMA,run=d.RUN,config=d.calibration_configuration(),status='SOURCE_CALIBRATED',
        teacher_examples=d.CALIBRATION_IQ,unique_source_positions=d.CAL_UNIQUE,
        L_exposures=d.CAL_PER_ROLE*len(d.CAL_VIEWS),U_exposures=d.CAL_PER_ROLE*len(d.CAL_VIEWS),
        F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True)
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Matching source-only original-parent activation calibration required')
    moments=value.get('moments')
    if not isinstance(moments,dict) or not moments:
        raise ValueError('Actual nonempty kernel input moments required')
    if teacher is not None:
        groups=eligible_groups(teacher,.5)
        if set(moments)!=set(groups):raise ValueError('All and only original-parent eligible moments required')
        for key,group in groups.items():
            conv=group['members'][0];inner=conv.in_channels*conv.kernel_size[0];moment=moments[key]
            if not torch.is_tensor(moment) or moment.shape!=(inner,inner) or moment.device.type!='cpu' or not torch.isfinite(moment).all():
                raise ValueError('Finite CPU moments with original kernel dimensions required')
    return value


def load_calibration(path,teacher=None):
    path=Path(path)
    if path.name!='source_moments.pt' or path.parent.name!=d.RUN:
        raise ValueError('Current run calibration path required before IO')
    return validate_calibration(torch.load(path,map_location='cpu',weights_only=False),teacher)


@torch.no_grad()
def prepare(args):
    if args.output.name!=d.RUN or args.parent_action.name!=d.ACTION_RUN or args.source_cache.name!='source_cache':
        raise ValueError('Fixed current run and source-training handles required before IQ IO')
    if (args.output/'source_moments.pt').exists():raise FileExistsError('Immutable source moments already exist')
    validate_parent_root(args.compact_parent)
    manifest=read_json(args.source_cache/'manifest.json')
    if manifest.get('status')!='READY' or manifest.get('native_role_ids')!='EXACT_MATCH':
        raise ValueError('Reuse existing READY/EXACT_MATCH source data only')
    cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False);take=choose_positions(roles)
    teacher,payload=parent_runtime.load_checkpoint(args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.COMPACT_PARENT_ROW,20,4940):raise ValueError('Original source-final parent required')
    teacher.eval().requires_grad_(False);before={k:v.clone() for k,v in teacher.state_dict().items()};started=time.perf_counter()
    paths=dict(clean=cache/'train/clean.npy',received01=cache/'epochs/01/received.npy',received20=cache/'epochs/20/received.npy')
    with SharedInputMoments(teacher,.5,d.CAL_POSITIONS) as collector:
        for view in d.CAL_VIEWS:
            values=np.load(paths[view],mmap_mode='r',allow_pickle=False)
            if values.shape!=(63000,2,256) or values.dtype!=np.dtype('float32'):raise ValueError('Original native source IQ cache required')
            for offset in range(0,len(take),32):collector.observe(array_tensor(values[take[offset:offset+32]],'cpu'))
    moments,report=collector.finish()
    if report['teacher_examples']!=d.CALIBRATION_IQ or any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('Exactly fixed source observations and unchanged teacher required')
    value=dict(schema=SCHEMA,run=d.RUN,config=d.calibration_configuration(),status='SOURCE_CALIBRATED',
        moments=moments,teacher_examples=report['teacher_examples'],unique_source_positions=len(take),
        L_exposures=d.CAL_PER_ROLE*len(d.CAL_VIEWS),U_exposures=d.CAL_PER_ROLE*len(d.CAL_VIEWS),
        F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
        seconds=time.perf_counter()-started,groups=report['groups'])
    validate_calibration(value,teacher)
    with (args.output/'source_moments.pt').open('xb') as stream:torch.save(value,stream)
    summary={k:v for k,v in value.items() if k!='moments'};write_json(args.output/'calibration.json',summary)
    print('SOURCE_ACTIVATION_CALIBRATION_COMPLETE',report['teacher_examples'],flush=True)
    return summary
