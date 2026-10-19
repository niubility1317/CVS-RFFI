import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import build,ResidualDiffusion,ResidualFlow
from .runtime import load_coarse,save_checkpoint,load_checkpoint,predict_and_score,diagnostics
from experiments.pamf_reception_restore.runner import pair_loss,learning_rate
from experiments.pamf_reception_restore.data import route_array
from experiments.pamf_restoration.runner import array_tensor,tensor_array,read_json,predict_batch
from experiments.pamf_restoration.data import SCENES,write_json


def source_groups(selected,route,scenes,roles,metadata,fraction=None):
    groups={**{key:route==key for key in ('clean','residual','full')},
            'L_s':roles[selected]=='L_s','U_s':roles[selected]=='U_s'}
    groups.update({scene:(scenes[selected]==i)&(route!='clean') for i,scene in enumerate(SCENES)})
    for key,labels in (('lock',('locked','degraded','unlocked')),('state',('LOS','PARTIAL','BLOCKED'))):
        chosen=np.where(route=='full',metadata['full_'+key][selected],metadata['residual_'+key][selected])
        groups.update({key+'_'+label:(chosen==i)&(route!='clean') for i,label in enumerate(labels)})
    snr=np.where(route=='full',metadata['full_signal_to_added_noise_db'][selected],
                 metadata['residual_signal_to_added_noise_db'][selected])
    bounds=(-np.inf,0.,10.,20.,np.inf)
    groups.update({'added_signal_noise_bin'+str(i):(snr>=bounds[i])&(snr<bounds[i+1])&(route!='clean') for i in range(4)})
    if fraction is not None:
        groups.update({'model_time_quartile'+str(i):(fraction>=i/4)&(fraction<(i+1)/4) for i in range(4)})
    return groups


def train(name,cache,output,parent,source_cache,frozen_bundle):
    folder=output/name;folder.mkdir()
    scale=read_json(output/'residual_scale.json')
    if (scale['status']!='SOURCE_SCALE_FROZEN' or scale['count']!=d.SCALE_SAMPLES or
        scale['source_V_reads'] or scale['TX_labels_read'] or scale['target_reads']):
        raise ValueError('Shared source-training-only scale required')
    coarse,metadata=load_coarse(parent,'cuda:0')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model=build(name,coarse,scale['scale']).to('cuda:0')
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=d.LEARNING_RATE,weight_decay=d.WEIGHT_DECAY)
    write_json(folder/'resolved_config.json',dict(d.configuration(),candidate=name,scale=scale['scale']))
    write_json(folder/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
               physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    permutation=np.random.default_rng(d.DATA_SEED)
    internal=torch.Generator(device='cuda:0').manual_seed(d.AUGMENTATION_SEED+10)
    generative=isinstance(model,(ResidualDiffusion,ResidualFlow))
    weight=.2 if generative else 1.
    updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            arrays={key:np.load(cache/'epochs'/f'{epoch:02d}'/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('residual','full')}
            quantile=np.load(cache/'epochs'/f'{epoch:02d}'/'quantile.npy',allow_pickle=False)
            scenes=np.load(cache/'epochs'/f'{epoch:02d}'/'scene.npy',allow_pickle=False)
            with np.load(cache/'epochs'/f'{epoch:02d}'/'diagnostics.npz',allow_pickle=False) as archive:
                extra={key:archive[key] for key in archive.files}
            order=permutation.permutation(len(clean));sums={};strata={};counts={key:0 for key in ('clean','residual','full')}
            clipped=0;model.train()
            for offset in range(0,len(order),d.BATCH_SIZE):
                selected=order[offset:offset+d.BATCH_SIZE]
                route=route_array('unet_mixed',quantile[selected]);observed=np.asarray(clean[selected]).copy()
                for key,array in arrays.items():
                    mask=route==key;observed[mask]=array[selected[mask]]
                x,y=array_tensor(clean[selected],'cuda:0'),array_tensor(observed,'cuda:0')
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                if generative:
                    estimate,velocity,diagnostic=model.training_prediction(x,y,internal,return_diagnostics=True)
                    paired,terms=pair_loss(estimate,x);loss=velocity+.2*paired.mean()
                    terms=dict(terms,velocity=diagnostic['velocity_error'])
                else:
                    estimate=model(y);paired,terms=pair_loss(estimate,x);loss=paired.mean()
                terms=dict(terms,weighted_iq_nmse=weight*terms['iq_nmse'],weighted_log_power=weight*.1*terms['log_power'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual reconstruction/velocity')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
                values={key:tensor_array(value) for key,value in dict(terms,total=loss.expand(len(x))).items()}
                for key,value in values.items():sums[key]=sums.get(key,0.)+float(value.sum())
                sums['gradient_norm']=sums.get('gradient_norm',0.)+float(norm)*len(x)
                for key in counts:counts[key]+=int((route==key).sum())
                fraction=tensor_array(diagnostic['time_fraction']) if generative else None
                groups=source_groups(selected,route,scenes,roles,extra,fraction)
                paired_cpu=tensor_array(paired)
                for key,mask in groups.items():
                    item=strata.setdefault(key,dict(count=0,paired_sum=0.,velocity_sum=0.))
                    item['count']+=int(mask.sum());item['paired_sum']+=float(paired_cpu[mask].sum())
                    if generative:item['velocity_sum']+=float(values['velocity'][mask].sum())
                if updates==1:
                    path=folder/'initial_source_smoke.pt'
                    save_checkpoint(path,model,optimizer,epoch,updates,name,metadata)
                    fresh,_=load_checkpoint(path,name,'cuda:0');model.eval()
                    with torch.no_grad():
                        left=predict_batch(model,y[:2],ids[selected][:2])
                        right=predict_batch(fresh,y[:2],ids[selected][:2])
                        if not torch.allclose(left,right,atol=1e-6,rtol=1e-5) or not torch.isfinite(left).all():
                            raise ValueError('Residual checkpoint real-source roundtrip failed')
                    del fresh;model.train()
                    write_json(folder/'smoke.json',dict(status='PASS',real_source_samples=2,
                               TX_labels_read=0,target_reads=0,coarse_updates=0))
            record=dict(epoch=epoch,updates=updates,losses={key:value/len(clean) for key,value in sums.items()},
                learning_rate_last=lr,gradient_clip_fraction=clipped/247,route_exposures=counts,
                strata={key:dict(count=v['count'],paired_mean=v['paired_sum']/max(1,v['count']),
                                velocity_mean=v['velocity_sum']/max(1,v['count'])) for key,v in strata.items()},
                generative_training_reconstruction_is_clean_dependent_proxy=generative,
                L_exposures=seen_l,U_exposures=seen_u,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='EPOCH_COMPLETE',candidate=name,**record)),flush=True)
    save_checkpoint(folder/'final.pt',model,optimizer,d.EPOCHS,updates,name,metadata)
    write_json(folder/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=name,epochs=d.EPOCHS,
        updates=updates,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
        coarse_parameters=sum(p.numel() for p in model.coarse.parameters()),coarse_updates=0,parent_epochs=20,
        cumulative_restorer_epochs=40,TX_labels_read=0,target_reads=0,L_exposures=seen_l,U_exposures=seen_u,
        shared_scale_calibration_samples=d.SCALE_SAMPLES,seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del model,optimizer,coarse;torch.cuda.empty_cache()
    predict_and_score(name,cache/'validation',folder,source_cache,frozen_bundle)
    predict_and_score(name,cache/'validation_full',folder,source_cache,frozen_bundle,full=True)
    diagnostics(name,cache/'validation',folder,'cuda:0')
