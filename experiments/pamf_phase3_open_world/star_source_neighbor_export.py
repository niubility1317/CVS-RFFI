"""Original F0 source-only P0 export with exact existing P1 teacher reuse."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import time
import numpy as np
import torch

from . import runner
from . import star_relative_source as original
from . import star_joint_ova_train as teacher
from . import star_open_feature_export as feature
from .star_representation_cache import load_role, close_role
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261030-phase3-source-crossrx-neighbors-s2026092701-r01'
SCHEMA='pamf_phase3_source_neighbor_features_v1'
STATUS='SOURCE_NEIGHBOR_FEATURES_COMPLETE'
COUNTS=dict(L=6300,OE=6134)
FIELDS={'run_id','code_commit','reader_config','source_contract','cache_root','teacher_root','output_root','batch_size'}


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c[k],str) or not c[k].strip() for k in ('source_contract','cache_root','teacher_root','output_root'))):
        raise ValueError('Only original-F0 L/OE cached G10 and exact P1 teacher inputs allowed')
    original._reader_config(c)
    out=Path(c['output_root']).resolve()
    for p in [c['cache_root'],c['teacher_root'],c['reader_config']['source_output']]:
        path=Path(p).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError('New feature output must be separate from all inputs')
    return c


def _teacher(root,bundle,role,parent):
    marker=runner._read(Path(root)/'complete.json')
    if (marker.get('schema')!=teacher.TEACHER_SCHEMA or marker.get('status')!=teacher.TEACHER_STATUS
            or marker.get('code_commit')!='5aa041122396d48c49b8cb3d5912649bdc3217e1'
            or marker.get('parent_F0_identity')!=parent or marker.get('dura_identity')!=bundle['manifest']['dura_identity']
            or marker.get('cache_handles',{}).get(role)!=bundle['manifest']
            or marker.get('source_only') is not True or marker.get('model_buffers_unchanged') is not True
            or marker.get('feature')!='unit_fused160' or marker.get('input_path')!='cached_P1_only'
            or any(marker.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads',
                'OE_truth_reads','encoder_updates','identity_head_updates','G_updates'))):
        raise ValueError('Matching frozen original-F0 P1 teacher and cache ancestry required')
    with np.load(Path(root)/(role+'.npz'),allow_pickle=False) as stored:
        if set(stored.files)!={'ids','unit_features'}:raise ValueError('P1 teacher must not contain TX or extra metadata')
        ids=stored['ids']; z=stored['unit_features'].copy()
    if (not np.array_equal(ids,bundle['ids']) or z.shape!=(COUNTS[role],160) or z.dtype!=np.float32
            or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0.,atol=2e-6)):
        raise ValueError('Aligned finite unit P1 features required')
    return z,marker


def export(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('Use a new immutable source feature output')
    start=time.perf_counter();backend=original._backend(c,device,with_dura=False)
    reader=backend['reader'];parent=backend['reader_identity'];snapshot=_snapshot_model(reader)
    if reader.training or any(p.requires_grad for p in reader.parameters()):raise ValueError('Frozen original F0 required')
    feature._features(reader,feature._synthetic(device),True);_check_model(reader,snapshot)
    out.mkdir(parents=True,exist_ok=False)
    runner._write(out/'smoke.json',dict(status='PASS',actual_checkpoint=True,synthetic_G10_P0_F_rows=2,
        real_source_IQ_reads=0,target_reads=0,private_truth_reads=0,G_forward_rows=0))
    print('SOURCE_NEIGHBOR_CHECKPOINT_SMOKE_PASS',flush=True)
    handles={};teacher_marker=None;all_ids=set();dura=None
    for role,count in COUNTS.items():
        bundle=load_role(Path(c['cache_root'])/role,role)
        try:
            m=bundle['manifest'];meta=bundle['metadata'];ids=bundle['ids']
            if (m.get('parent_F0_identity')!=parent or m.get('count')!=count or m.get('role')!=role
                    or m.get('all_new_IQ_satellite') is not True or m.get('restored_input_path')!='P1'
                    or set(meta)!=({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
                    or all_ids.intersection(ids.tolist())):
                raise ValueError('Only aligned, disjoint source L/OE roles and original F0 ancestry allowed')
            if dura is not None and dura!=m['dura_identity']:raise ValueError('Source L/OE DURA ancestry differs')
            dura=m['dura_identity'];all_ids.update(ids.tolist());handles[role]=m
            z1,marker=_teacher(c['teacher_root'],bundle,role,parent)
            if teacher_marker is not None and marker!=teacher_marker:raise ValueError('Single frozen P1 teacher required')
            teacher_marker=marker;rows=[]
            for offset in range(0,count,c['batch_size']):
                rows.append(teacher._teacher_encode(reader,bundle['received'][offset:offset+c['batch_size']],device))
                print('SOURCE_NEIGHBOR_P0_PROGRESS',role,min(offset+c['batch_size'],count),count,flush=True)
            data=dict(ids=ids,rx_i=meta['rx_i'],day_i=meta['day_i'],P0=np.concatenate(rows),P1=z1)
            if role=='L':data['y']=meta['y']
            with (out/(role+'.npz')).open('xb') as stream:np.savez(stream,**data)
            _check_model(reader,snapshot)
        finally:close_role(bundle)
    _check_model(reader,snapshot)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,
        counts=COUNTS,parent_F0_identity=parent,dura_identity=dura,cache_handles=handles,teacher_identity=teacher_marker,
        source_only=True,input_paths=['same_fixed_G10_P0','reused_same_record_P1'],model_states_unchanged=True,
        model_updates=0,optimizer_updates=0,threshold_updates=0,G_forward_rows=0,
        real_F_forward_rows=sum(COUNTS.values()),synthetic_F_forward_rows=2,P1_feature_rows_reused=sum(COUNTS.values()),
        source_V_reads=0,source_novel_reads=0,original_U_s_reads=0,target_reads=0,private_truth_reads=0,
        restored_cache_values_read=0,clean_IQ_reads=0,source_physical_pool=sum(COUNTS.values()),
        final_phase3_complete=False,seconds=time.perf_counter()-start)
    runner._write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    export(runner._read(a.config),a.device)


if __name__=='__main__':main()
