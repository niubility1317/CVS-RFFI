"""Source-only fitting; original frozen default heads are independent readouts."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
from . import design as d
from .models import NAMES,build,MultiscaleDiffusion,MultiscaleFlow
from .data import route_array
from experiments.pamf_restoration.data import SCENES,write_json,fixed_noise
from experiments.pamf_restoration.models import complex_iq
from experiments.pamf_restoration.runner import (array_tensor,tensor_array,read_json,code_commit,
                                                predict_batch,predict_row)


def pair_loss(estimate,clean):
    nmse=(estimate-clean).square().mean((1,2))/clean.square().mean((1,2)).clamp_min(1e-4)
    predicted=torch.fft.fft(complex_iq(estimate),norm='ortho')
    target=torch.fft.fft(complex_iq(clean),norm='ortho')
    spectrum=(torch.log(predicted.abs().square()+1e-3)-torch.log(target.abs().square()+1e-3)).abs().mean(-1)
    return nmse+.1*spectrum,dict(iq_nmse=nmse,log_power=spectrum)


def learning_rate(update,total=4940):
    if not 0<=update<total:raise ValueError('Optimizer update outside fixed budget')
    if update<247:return d.LEARNING_RATE*(update+1)/247
    fraction=(update-247)/(total-247-1)
    return d.LEARNING_RATE*(.1+.9*(1+math.cos(math.pi*fraction))/2)


def save_checkpoint(path,model,optimizer,epoch,updates,candidate):
    torch.save(dict(schema='reception_restorer_checkpoint_v2',model=model.state_dict(),
        optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,config=d.configuration(),
        candidate=candidate,TX_labels_read=0,target_training=False,initialization='scratch'),path)


def load_checkpoint(path,name,device):
    item=torch.load(path,map_location=device,weights_only=False)
    if (item['schema']!='reception_restorer_checkpoint_v2' or item['config']!=d.configuration() or
        item['candidate']!=name or item['TX_labels_read']!=0 or item['target_training'] or item['initialization']!='scratch'):
        raise ValueError('Reception-restorer provenance/config mismatch')
    model=build(name).to(device);model.load_state_dict(item['model'],strict=True)
    return model.eval(),item


def score(reference,folder,source_cache=None,target_truth=None):
    command=[sys.executable,'-X','utf8','-m','experiments.pamf_restoration.runner','score',
             '--reference',str(reference),'--row-output',str(folder)]
    command.extend(['--source-cache',str(source_cache)] if source_cache is not None
                   else ['--target-truth',str(target_truth)])
    subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),check=True)


@torch.no_grad()
def condition_diagnostic(name,reference,folder,device):
    model,_=load_checkpoint(folder/'final.pt',name,device)
    if not isinstance(model,(MultiscaleDiffusion,MultiscaleFlow)):return
    ids=np.load(reference/'physical_ids.npy',allow_pickle=False)[:128]
    result={}
    for scene in SCENES:
        value=array_tensor(np.load(reference/(scene+'.npy'),allow_pickle=False)[:128],device)
        clean=array_tensor(np.load(reference/'clean.npy',allow_pickle=False)[:128],device)
        noise=fixed_noise(ids,d.EVALUATION_SEED,value.shape,device)
        # A complete source-only counterfactual: same observation/initial noise,
        # but condition encodings are supplied from a fixed wrong source record.
        # No output is scored as a formal prediction or used for selection.
        if isinstance(model,MultiscaleDiffusion):
            encoder=model.conditioner
        else:encoder=model.network.conditioner
        original=encoder.forward
        main=model(value,noise)
        def shuffled(v):return original(v.roll(1,0))
        encoder.forward=shuffled
        try:wrong=model(value,noise)
        finally:encoder.forward=original
        result[scene]=dict(count=128,complete_sampler=True,same_initial_noise=True,
            main_nmse=float(((main-clean).square().mean((1,2))/clean.square().mean((1,2)).clamp_min(1e-4)).mean()),
            shuffled_condition_nmse=float(((wrong-clean).square().mean((1,2))/clean.square().mean((1,2)).clamp_min(1e-4)).mean()),
            mean_output_difference=float((main-wrong).square().mean()))
    write_json(folder/'source_condition_shuffle.json',dict(source_only=True,TX_labels_read=0,
               target_reads=0,selection_use=False,views=result))


@torch.no_grad()
def batch_one_profile(name,reference,folder,device):
    model,_=load_checkpoint(folder/'final.pt',name,device)
    value=array_tensor(np.load(reference/'practical_low_urban.npy',allow_pickle=False)[:16],device)
    ids=np.load(reference/'physical_ids.npy',allow_pickle=False)[:16]
    for _ in range(2):predict_batch(model,value[:1],ids[:1])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    durations=[]
    for index in range(16):
        torch.cuda.synchronize();started=time.perf_counter()
        estimate=predict_batch(model,value[index:index+1],ids[index:index+1])
        torch.cuda.synchronize();durations.append((time.perf_counter()-started)*1000)
        if not torch.isfinite(estimate).all():raise FloatingPointError('Nonfinite batch1 profile')
    generative=isinstance(model,(MultiscaleDiffusion,MultiscaleFlow))
    write_json(folder/'batch1_profile.json',dict(source_only=True,samples=16,warmup_calls=2,
        dtype='float32',restorer_only=True,includes_ID_noise_creation=generative,
        mean_ms=float(np.mean(durations)),median_ms=float(np.median(durations)),p90_ms=float(np.quantile(durations,.9)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),restorer_forward_calls=1,
        denoiser_field_evaluations=16 if generative else 0,
        condition_encoder_calls=1 if generative else 0,
        deterministic_UNet_modules=2 if name=='tf_unet_mixed' else 0 if generative else 1))


def predict_and_score(name,reference,folder,source_cache,frozen_bundle,full=False):
    row=folder
    if full:
        row=folder/'full_source_diagnostic';row.mkdir()
        (row/'final.pt').symlink_to((folder/'final.pt').resolve())
    predict_row(name,reference,row,'cuda:0',frozen_bundle,source_cache,checkpoint_loader=load_checkpoint)
    score(reference,row,source_cache=source_cache)


def train(name,cache,folder,source_cache,frozen_bundle):
    folder.mkdir()
    write_json(folder/'resolved_config.json',dict(d.configuration(),candidate=name))
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model=build(name).to('cuda:0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=d.LEARNING_RATE,weight_decay=d.WEIGHT_DECAY)
    write_json(folder/'gpu_ready.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
               physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    permutation=np.random.default_rng(d.DATA_SEED)
    internal=torch.Generator(device='cuda:0').manual_seed(d.AUGMENTATION_SEED+10)
    updates=seen_l=seen_u=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            arrays={route:np.load(cache/'epochs'/f'{epoch:02d}'/(route+'.npy'),mmap_mode='r',allow_pickle=False)
                    for route in ('residual','full')}
            quantile=np.load(cache/'epochs'/f'{epoch:02d}'/'quantile.npy',allow_pickle=False)
            scenes=np.load(cache/'epochs'/f'{epoch:02d}'/'scene.npy',allow_pickle=False)
            with np.load(cache/'epochs'/f'{epoch:02d}'/'diagnostics.npz',allow_pickle=False) as archive:
                metadata={key:archive[key] for key in archive.files}
            order=permutation.permutation(len(clean));sums={};strata={};counts={r:0 for r in ('clean','residual','full')}
            clipped=0;model.train()
            for offset in range(0,len(order),d.BATCH_SIZE):
                selected=order[offset:offset+d.BATCH_SIZE]
                route=route_array(name,quantile[selected])
                batch=np.asarray(clean[selected]).copy()
                for key in ('residual','full'):
                    mask=route==key;batch[mask]=arrays[key][selected[mask]]
                x,y=array_tensor(clean[selected],'cuda:0'),array_tensor(batch,'cuda:0')
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                if isinstance(model,(MultiscaleDiffusion,MultiscaleFlow)):
                    estimate,velocity,diagnostic=model.training_prediction(x,y,internal,return_diagnostics=True)
                    paired,terms=pair_loss(estimate,x);loss=velocity+.2*paired.mean()
                    terms=dict(terms,velocity=diagnostic['velocity_error'])
                else:
                    estimate=model(y);paired,terms=pair_loss(estimate,x);loss=paired.mean()
                reconstruction_weight=.2 if isinstance(model,(MultiscaleDiffusion,MultiscaleFlow)) else 1.
                terms=dict(terms,weighted_iq_nmse=reconstruction_weight*terms['iq_nmse'],
                           weighted_log_power=reconstruction_weight*.1*terms['log_power'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired loss')
                loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(gradient>1)
                optimizer.step();updates+=1
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
                values={key:tensor_array(value) for key,value in dict(terms,total=loss.expand(len(x))).items()}
                for key,value in values.items():sums[key]=sums.get(key,0.)+float(value.sum())
                sums['gradient_norm']=sums.get('gradient_norm',0.)+float(gradient)*len(x)
                for key in counts:counts[key]+=int((route==key).sum())
                # Loss summaries stratify only by source/simulator metadata;
                # no stratum or label is fed into the restorer.
                groups={**{key:route==key for key in counts},
                        'L_s':roles[selected]=='L_s','U_s':roles[selected]=='U_s'}
                groups.update({scene:(scenes[selected]==i)&(route!='clean') for i,scene in enumerate(SCENES)})
                chosen={key:np.where(route=='full',metadata['full_'+key][selected],metadata['residual_'+key][selected])
                        for key in ('lock','state','signal_to_added_noise_db')}
                groups.update({'lock_'+lock:(chosen['lock']==i)&(route!='clean') for i,lock in enumerate(('locked','degraded','unlocked'))})
                groups.update({'state_'+state:(chosen['state']==i)&(route!='clean') for i,state in enumerate(('LOS','PARTIAL','BLOCKED'))})
                bounds=(-np.inf,0.,10.,20.,np.inf)
                for i in range(4):
                    groups['added_signal_noise_bin'+str(i)]=(chosen['signal_to_added_noise_db']>=bounds[i])&(chosen['signal_to_added_noise_db']<bounds[i+1])&(route!='clean')
                if reconstruction_weight==.2:
                    fraction=tensor_array(diagnostic['time_fraction'])
                    for i in range(4):groups['model_time_quartile'+str(i)]=(fraction>=i/4)&(fraction<(i+1)/4)
                paired_cpu=tensor_array(paired)
                for key,mask in groups.items():
                    item=strata.setdefault(key,dict(count=0,paired_sum=0.,velocity_sum=0.))
                    item['count']+=int(mask.sum());item['paired_sum']+=float(paired_cpu[mask].sum())
                    if reconstruction_weight==.2:item['velocity_sum']+=float(values['velocity'][mask].sum())
                if updates==1:
                    path=folder/'initial_source_smoke.pt';save_checkpoint(path,model,optimizer,epoch,updates,name)
                    fresh,_=load_checkpoint(path,name,'cuda:0')
                    model.eval()
                    with torch.no_grad():
                        left=predict_batch(model,y[:2],['source-smoke-0','source-smoke-1'])
                        right=predict_batch(fresh,y[:2],['source-smoke-0','source-smoke-1'])
                        if not torch.allclose(left,right,atol=1e-6,rtol=1e-5) or not torch.isfinite(left).all():raise ValueError('Checkpoint no-query smoke failed')
                    del fresh;model.train()
                    write_json(folder/'smoke.json',dict(status='PASS',real_source_samples=2,TX_labels_read=0,target_reads=0))
            record=dict(epoch=epoch,updates=updates,losses={key:value/len(clean) for key,value in sums.items()},
                learning_rate_last=lr,gradient_clip_fraction=clipped/247,route_exposures=counts,
                strata={key:dict(count=value['count'],paired_mean=value['paired_sum']/max(1,value['count']),
                                velocity_mean=value['velocity_sum']/max(1,value['count'])) for key,value in strata.items()},
                generative_training_reconstruction_is_clean_dependent_proxy=reconstruction_weight==.2,
                L_exposures=seen_l,U_exposures=seen_u,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(dict(event='EPOCH_COMPLETE',candidate=name,**record)),flush=True)
    save_checkpoint(folder/'final.pt',model,optimizer,d.EPOCHS,updates,name)
    write_json(folder/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=name,epochs=d.EPOCHS,
        updates=updates,parameters=sum(p.numel() for p in model.parameters()),TX_labels_read=0,target_reads=0,
        L_exposures=seen_l,U_exposures=seen_u,seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del model,optimizer;torch.cuda.empty_cache()
    predict_and_score(name,cache/'validation',folder,source_cache,frozen_bundle)
    predict_and_score(name,cache/'validation_full',folder,source_cache,frozen_bundle,full=True)
    condition_diagnostic(name,cache/'validation',folder,'cuda:0')
    batch_one_profile(name,cache/'validation',folder,'cuda:0')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('worker','confirm'))
    parser.add_argument('--candidate',choices=NAMES,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-cache',type=Path,required=True)
    parser.add_argument('--frozen-bundle',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--target-reference',type=Path)
    parser.add_argument('--target-truth',type=Path)
    args=parser.parse_args()
    for key in ('cache','output','source_cache','frozen_bundle'):
        setattr(args,key,getattr(args,key).resolve())
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code differs')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='worker':
        train(args.candidate,args.cache,args.output/args.candidate,args.source_cache,args.frozen_bundle)
    else:
        decision=read_json(args.output/'source_decision.json')
        if decision['status']!='SOURCE_SELECTION_FROZEN' or args.candidate not in decision['target_candidates']:
            raise ValueError('Frozen source selection required before target inference')
        if args.target_reference is None or args.target_truth is None:raise ValueError('Fixed target reference/truth paths required')
        row=args.output/'target_confirmation'/args.candidate;row.mkdir(parents=True)
        (row/'final.pt').symlink_to((args.output/args.candidate/'final.pt').resolve())
        predict_row(args.candidate,args.target_reference,row,'cuda:0',args.frozen_bundle,args.source_cache,
                    checkpoint_loader=load_checkpoint)
        score(args.target_reference,row,target_truth=args.target_truth)
        write_json(row/'complete.json',dict(status='ANALYZED',source_selection_frozen=True,
                   original_target_inputs_reused=True,parameter_updates=0,training_TX_labels_read=0))


if __name__=='__main__':main()
