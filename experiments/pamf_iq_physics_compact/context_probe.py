"""Freeze seven source context cells, then score known simulation in CPU process."""
import os
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_compact_physics_probe.analysis import KEYS,validate_estimate,diagnostics,summarize
from experiments.pamf_restoration.runner import array_tensor,tensor_array,read_json
from experiments.pamf_restoration.models import pair
from experiments.pamf_restoration.data import write_json
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference


@torch.no_grad()
def extract_context(args):
    reference=args.parent_action/'cache/validation'
    ids,_,_=source_reference(reference,args.source_cache)
    if len(ids)!=1800:raise ValueError('Fixed source1800 panel required')
    row=args.output/args.arm
    model,value=load_checkpoint(row/d.KIND/'final.pt','cuda:0')
    if (value['arm'],value['epoch'],value['updates'])!=(args.arm,20,4940):raise ValueError('Correct frozen source final required')
    del value
    model.requires_grad_(False).eval();output=row/'source_physics';output.mkdir()
    np.save(output/'physical_ids.npy',ids,allow_pickle=False)
    for view in d.VIEWS:
        received=np.load(reference/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(1800,2,256):raise ValueError('Original fixed source view required')
        pieces={key:[] for key in KEYS}
        for begin in range(0,len(ids),64):
            base,action,variance=model.estimate_context(array_tensor(received[begin:begin+64],'cuda:0'))
            values=dict(coarse=tensor_array(base),kernel=tensor_array(pair(action.kernel)),variance=tensor_array(variance),
                coefficients=tensor_array(torch.stack((action.main.real,action.main.imag,action.image.real,action.image.imag),1)))
            validate_estimate(values,len(variance))
            for key in KEYS:pieces[key].append(values[key])
        values={key:np.concatenate(pieces[key]) for key in KEYS};validate_estimate(values,1800)
        with (output/(view+'.npz')).open('xb') as stream:np.savez(stream,**values)
    write_json(output/'complete.json',dict(status='CONTEXT_ESTIMATES_COMPLETE',arm=args.arm,views=list(d.VIEWS),
        count=1800,TX_labels_read=0,F_access=0,target_reads=0,clean_oracle_reads=0,state_updates=0))


def load_estimates(folder,arm,ids):
    header=read_json(folder/'complete.json')
    expected=dict(status='CONTEXT_ESTIMATES_COMPLETE',arm=arm,views=list(d.VIEWS),count=1800,
        TX_labels_read=0,F_access=0,target_reads=0,clean_oracle_reads=0,state_updates=0)
    if any(header.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('All seven matched source contexts required before oracle')
    values={}
    for view in d.VIEWS:
        with np.load(folder/(view+'.npz'),allow_pickle=False) as data:value={key:data[key] for key in data.files}
        validate_estimate(value,1800);values[view]=value
    return values


def score_context(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU source oracle score required')
    reference=args.parent_action/'cache/validation'
    ids,source,index=source_reference(reference,args.source_cache)
    output=args.output/args.arm/'source_physics';estimates=load_estimates(output,args.arm,ids)
    clean=np.load(reference/'clean.npy',allow_pickle=False)
    rng=np.random.default_rng(2026102040)
    probes=(rng.normal(size=(4,len(ids),256))+1j*rng.normal(size=(4,len(ids),256)))/np.sqrt(2)
    records=[];samples={}
    for view in d.VIEWS:
        folder=args.parent_action/'cache/validation_actions'/view
        oracle={key:np.load(folder/(key+'.npy'),allow_pickle=False) for key in ('kernel','coefficients','variance')}
        received=np.load(reference/(view+'.npy'),allow_pickle=False)
        values=diagnostics(estimates[view],oracle,clean,received,probes)
        records.extend(summarize(values,args.arm,view,source.rx[index],source.day[index]))
        for key,value in values.items():samples[view+'__'+key]=value
    with (output/'sample_metrics.npz').open('xb') as stream:np.savez(stream,**samples)
    write_json(output/'summary.json',dict(status='ANALYZED',arm=args.arm,role='source_V',records=records,
        count=1800,probe_seed=2026102040,probes_per_IQ=4,all_context_estimates_before_oracle=True,
        TX_labels_read=0,F_access=0,target_reads=0,source_fit_updates=0,selected_candidate=None))
