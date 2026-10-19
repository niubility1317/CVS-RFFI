"""Portable residual checkpoint, source-only scale and frozen-head readouts."""
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import NAMES,build,ResidualDiffusion,ResidualFlow
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_reception_restore.runner import load_checkpoint as load_coarse_checkpoint
from experiments.pamf_reception_restore.runner import score
from experiments.pamf_reception_restore.data import route_array
from experiments.pamf_restoration.runner import array_tensor,tensor_array,read_json,predict_batch,predict_row
from experiments.pamf_restoration.data import SCENES,write_json,fixed_noise

ROOT=Path(__file__).resolve().parents[2]


def parent_metadata(parent):
    launch=read_json(parent/'launch.json')
    decision=read_json(parent/'source_decision.json')
    if (parent.name!=d.PARENT_RUN or launch['commit']!=d.PARENT_COMMIT or decision['status']!='SOURCE_SELECTION_FROZEN' or
        decision['waveform_selected']!=d.COARSE_NAME or decision['waveform_selection_TX_F_access'] or
        decision['training_TX_labels_read']!=0 or not decision['no_target_feedback']):
        raise ValueError('Fixed source waveform-only parent selection required')
    row=next(r for r in decision['identity_diagnostic'] if r['candidate']==d.COARSE_NAME)
    if not row['qualified']:raise ValueError('Coarse parent source qualification absent')
    training=read_json(parent/d.COARSE_NAME/'training_complete.json')
    if (training['epochs']!=20 or training['updates']!=4940 or training['TX_labels_read']!=0 or
        training['L_exposures']!=126000 or training['U_exposures']!=1134000):
        raise ValueError('Coarse parent source training contract differs')
    return dict(run=d.PARENT_RUN,commit=d.PARENT_COMMIT,candidate=d.COARSE_NAME,
                selector='source_waveform_A',training_TX_labels_read=0,source_roles='original7/63/30',
                target_training=False,coarse_epochs=20,coarse_updates=4940)


def load_coarse(parent,device):
    metadata=parent_metadata(parent)
    model,payload=load_coarse_checkpoint(parent/d.COARSE_NAME/'final.pt',d.COARSE_NAME,device)
    if payload['epoch']!=20 or payload['updates']!=4940:
        raise ValueError('Fixed final coarse checkpoint required')
    return model.requires_grad_(False).eval(),metadata


def save_checkpoint(path,model,optimizer,epoch,updates,name,parent):
    torch.save(dict(schema='residual_restorer_checkpoint_v3',model=model.state_dict(),
        optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,candidate=name,
        config=d.configuration(),scale=float(model.residual_scale),parent=parent,
        TX_labels_read=0,target_training=False,corrector_initialization='scratch'),path)


def load_checkpoint(path,name,device):
    payload=torch.load(path,map_location=device,weights_only=False)
    parent=payload['parent']
    if (payload['schema']!='residual_restorer_checkpoint_v3' or payload['config']!=d.configuration() or
        payload['candidate']!=name or payload['TX_labels_read']!=0 or payload['target_training'] or
        payload['corrector_initialization']!='scratch' or parent['run']!=d.PARENT_RUN or
        parent['commit']!=d.PARENT_COMMIT or parent['candidate']!=d.COARSE_NAME or
        parent['selector']!='source_waveform_A' or parent['training_TX_labels_read']!=0 or parent['target_training']):
        raise ValueError('Residual checkpoint source/config/provenance mismatch')
    model=build(name,ComplexTFUNet(),payload['scale']).to(device)
    model.load_state_dict(payload['model'],strict=True)
    if abs(float(model.residual_scale)-payload['scale'])>1e-7:
        raise ValueError('Saved fixed residual scale differs from model buffer')
    return model.eval(),payload


def calibration_indices(count):
    return np.random.default_rng(d.DATA_SEED+33).permutation(count)[:d.SCALE_SAMPLES]


@torch.no_grad()
def calibrate(cache,output,parent,device):
    coarse,metadata=load_coarse(parent,device)
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    quantile=np.load(cache/'epochs/01/quantile.npy',allow_pickle=False)
    arrays={key:np.load(cache/'epochs/01'/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('residual','full')}
    selected=calibration_indices(len(clean));total=elements=0
    for offset in range(0,len(selected),d.BATCH_SIZE):
        index=selected[offset:offset+d.BATCH_SIZE]
        route=route_array('unet_mixed',quantile[index]);observed=np.asarray(clean[index]).copy()
        for key,array in arrays.items():observed[route==key]=array[index[route==key]]
        x,y=array_tensor(clean[index],device),array_tensor(observed,device)
        error=x-coarse(y)
        if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite source scale estimate')
        total+=float(error.square().sum());elements+=error.numel()
    rms=math.sqrt(total/elements)
    write_json(output/'residual_scale.json',dict(status='SOURCE_SCALE_FROZEN',scale=max(1e-4,rms),
        raw_rms=rms,count=len(selected),elements=elements,source_training_only=True,
        source_V_reads=0,TX_labels_read=0,target_reads=0,optimizer_updates=0,parent=metadata))


def predict_and_score(name,reference,folder,source_cache,frozen_bundle,full=False):
    row=folder
    if full:
        row=folder/'full_source_diagnostic';row.mkdir()
        (row/'final.pt').symlink_to((folder/'final.pt').resolve())
    predict_row(name,reference,row,'cuda:0',frozen_bundle,source_cache,checkpoint_loader=load_checkpoint)
    score(reference,row,source_cache=source_cache)


@torch.no_grad()
def diagnostics(name,reference,folder,device):
    model,_=load_checkpoint(folder/'final.pt',name,device)
    generative=isinstance(model,(ResidualDiffusion,ResidualFlow))
    ids=np.load(reference/'physical_ids.npy',allow_pickle=False)
    if generative:
        encoder=model.conditioner if isinstance(model,ResidualDiffusion) else model.network.conditioner
        original=encoder.forward;result={}
        for scene in SCENES:
            received=array_tensor(np.load(reference/(scene+'.npy'),allow_pickle=False)[:128],device)
            clean=array_tensor(np.load(reference/'clean.npy',allow_pickle=False)[:128],device)
            noise=fixed_noise(ids[:128],d.EVALUATION_SEED,received.shape,device)
            main=model(received,noise)
            def shuffled(v):return original(v.roll(1,0))
            encoder.forward=shuffled
            try:wrong=model(received,noise)
            finally:encoder.forward=original
            denominator=clean.square().mean((1,2)).clamp_min(1e-4)
            result[scene]=dict(count=128,complete_sampler=True,same_initial_noise=True,
                main_nmse=float(((main-clean).square().mean((1,2))/denominator).mean()),
                shuffled_condition_nmse=float(((wrong-clean).square().mean((1,2))/denominator).mean()),
                mean_output_difference=float((main-wrong).square().mean()))
        write_json(folder/'source_condition_shuffle.json',dict(source_only=True,TX_labels_read=0,
            target_reads=0,selection_use=False,current_coarse_unchanged=True,views=result))
    values=array_tensor(np.load(reference/'practical_low_urban.npy',allow_pickle=False)[:16],device)
    for _ in range(2):predict_batch(model,values[:1],ids[:1])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();durations=[]
    for index in range(16):
        torch.cuda.synchronize();started=time.perf_counter()
        estimate=predict_batch(model,values[index:index+1],ids[index:index+1]);torch.cuda.synchronize()
        durations.append((time.perf_counter()-started)*1000)
        if not torch.isfinite(estimate).all():raise FloatingPointError('Nonfinite batch1 restored IQ')
    write_json(folder/'batch1_profile.json',dict(source_only=True,samples=16,warmup_calls=2,
        dtype='float32',restorer_only=True,includes_coarse=True,includes_ID_noise_creation=generative,
        mean_ms=float(np.mean(durations)),median_ms=float(np.median(durations)),p90_ms=float(np.quantile(durations,.9)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),coarse_forward_calls=1,
        denoiser_field_evaluations=16 if generative else 0,condition_encoder_calls=1 if generative else 0))
