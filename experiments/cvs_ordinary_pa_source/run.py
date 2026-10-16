import gc
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import torch
from . import design as d
from .core import paired,waveform,decisions
from experiments.cvs_ordinary_fold20.streams import stream
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def provenance(rx):
    source=d.origin(rx);done=d.read(source/'completion.json');init=d.read(source/'initialization.json')
    contract=d.read(source/'source_contract.json');freeze=d.read(source/'source_frozen.json');result=d.read(source/'result.json')
    c=done['config'];fit=[r for r in d.RX if r!=rx]
    if (c['arm']!='N0' or c['fold']['held_RX']!=rx or c['fold']['fit_RX']!=fit or c['model_seed']!=d.SEED
        or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
        or done['checkpoint_sources'] or done['ancestors'] or done['target_access'] or done['head_fit_updates']
        or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
        or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
        or result['target_access'] or result['U_truth_read'] or not result['default_head_joint_training']
        or contract['held_RX']!=rx or contract['source_receivers']!=fit
        or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}):
        raise ValueError('Source N0 default-head E20 checkpoint provenance differs')
    return c,contract


def source_L():
    expected=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
    if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']:
        raise ValueError('Existing legal source preparation differs')
    packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
    for actual in (packed['contract'],prepared['source_contract']):
        if any(actual.get(k)!=v for k,v in expected.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':
            raise ValueError('Source carrier roles differ')
    L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
    ids=set(L['ids'])
    if len(ids)!=6300 or len(L['ids'])!=6300 or ids!=set(expected['role_ids']['L_s']) or ids&(set(expected['role_ids']['U_s'])|set(expected['role_ids']['V'])):
        raise ValueError('Only exact original source L6300 permitted')
    if L['x'].shape!=(6300,2,256) or set(L['rx'].tolist())!=set(d.RX) or set(L['y'].tolist())!=set(range(6)) or not torch.isfinite(L['x']).all():
        raise ValueError('Full legal source L shape/metadata differs')
    return L


def load_model(rx,device):
    c,contract=provenance(rx)
    if rx==1:
        from experiments.cvs_ordinary_fold20 import model as api
    else:
        from experiments.cvs_ordinary_confirm20 import model as api
    # The ordinary native builder first installs the payload's original module
    # classes; strict state loading happens after that normal initialization.
    model=api.build(c,device)
    payload=torch.load(d.origin(rx)/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if payload['epoch']!=20 or payload['checkpoint_selection']!='final_only' or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt']:
        raise ValueError('Final20 original own-student checkpoint required')
    model.load_state_dict(payload['model'],strict=True);del payload
    model.eval().requires_grad_(False)
    snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():p,_=decisions(model,torch.zeros(2,2,256,device=device),torch.zeros(2,device=device,dtype=torch.long))
    if len(p)!=2:raise ValueError('Real source checkpoint no-query smoke failed')
    print(f'SMOKE PASS N0 RX{rx} final20/defaulthead; query_read=false',flush=True)
    return model,snapshot,contract


def augmentation(rx):
    from SSDG import train_ssdg as native
    args=d.read(d.origin(rx,'N1')/'actual_native_args.json');args=args.get('args',args)
    if not args['use_aug'] or not args['aug_enable_pa_normal'] or args['aug_enable_class_signature'] or args['aug_p_dac']!=0:
        raise ValueError('Actual original PA recipe differs')
    a=SimpleNamespace(**args);base=native.build_aug_base_cfg(a)
    if base!=d.read(d.origin(rx,'N1')/'ordinary_base_config.json'):raise ValueError('Actual ordinary base configuration differs')
    with stream(d.DRAW_SEED,0):aug=native.make_augmentor(base)
    state=native.configure_augmentor_for_epoch(aug,base,13,a)
    progress=d.read(d.origin(rx,'N1')/'mechanism_progress.json')['records']
    saved=next(v['ordinary']['probabilities'] for v in progress if v['epoch']==13)
    if any(getattr(aug,k)!=v for k,v in saved.items()):raise ValueError('Actual E13 augmentation probabilities differ')
    return aug,dict(base=base,actual_E13_probabilities=saved,schedule_state=state,
        ranges={k:getattr(aug,k) for k in ('max_time_shift','cfo_max','phase_noise_sigma_max','dc_offset_max','taper_alpha_min','taper_alpha_max','mp_delay_max')})


def run():
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
        if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner capacity exceeded')
    torch.set_num_threads(2);device=torch.device('cuda:0');start=time.perf_counter();L=source_L();counts={}
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_ordinary_fold20.design import FULL_FP32_POLICY
    try:
        for row in d.rows():
            c=d.config(row);d.validate(c);rx=c['held_RX'];out=Path(c['output_root']);out.mkdir(exist_ok=False)
            model,snapshot,contract=load_model(rx,device);aug,recipe=augmentation(rx)
            idx=torch.nonzero(L['rx']==rx,as_tuple=False).flatten()
            if len(idx)!=1260 or any(int(((L['y'][idx]==t)&(L['day'][idx]==day)).sum())!=70 for t in range(6) for day in (1,2,3)):
                raise ValueError('Each held source L TX/day cell must have70 packets')
            fit_ids={L['ids'][i] for i in range(6300) if int(L['rx'][i])!=rx}
            if set(contract['role_ids']['L_s'])!=fit_ids or any(L['ids'][int(i)] in fit_ids for i in idx):raise ValueError('Diagnostic L must be excluded from this model training')
            with stream(d.DRAW_SEED,rx):idx=idx[torch.randperm(len(idx))]
            raw=[];packet_forwards=2;paired_batches=0;acted_count=0
            d.write(out/'resolved_config.json',dict(config=c,recipe=recipe,checkpoint_provenance='VERIFIED',
                actual_modules={k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')},
                model_parameters=sum(p.numel() for p in model.parameters()),snapshot_tensors=len(snapshot),
                actual_CWD=os.getcwd(),GPU=gpu,python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                inherited_training_RX=contract['source_receivers'],only_held_L_IQ_used=True,
                cached_other_source_roles_deserialized_but_not_used=True,target_access=False))
            with numerical_context(FULL_FP32_POLICY),torch.no_grad():
                for chunk in idx.split(c['batch_size']):
                    x=L['x'][chunk].to(device);y=L['y'][chunk].long().to(device);pred,margin=decisions(model,x,y);stats=waveform(x);packet_forwards+=len(x)
                    for j,i in enumerate(chunk.tolist()):raw.append(dict(index=i,id=L['ids'][i],label=int(L['y'][i]),RX=rx,day=int(L['day'][i]),
                        prediction=pred[j],margin=margin[j],waveform={k:v[j] for k,v in stats.items()}))
                d.write(out/'RAW.json',dict(records=raw,count=len(raw)))
                with gzip.open(out/'paired.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as f:
                    for draw in range(c['draws']):
                        for batch,chunk in enumerate(idx.split(c['batch_size'])):
                            x=L['x'][chunk].to(device);y=L['y'][chunk].long().to(device)
                            full,no_pa,strength=paired(aug,x,y,d.DRAW_SEED,rx*100000+draw*100+batch)
                            pf,mf=decisions(model,full,y);pn,mn=decisions(model,no_pa,y);packet_forwards+=2*len(x);paired_batches+=1
                            wf,wn=waveform(full),waveform(no_pa);s=strength.tolist();acted_count+=sum(v>0 for v in s)
                            mse=(full-no_pa).square().flatten(1).mean(1).tolist()
                            for j,i in enumerate(chunk.tolist()):
                                f.write(json.dumps(dict(index=i,draw=draw,PA_strength=s[j],PA_MSE=mse[j],
                                    FULL=dict(prediction=pf[j],margin=mf[j],waveform={k:v[j] for k,v in wf.items()}),
                                    NO_PA=dict(prediction=pn[j],margin=mn[j],waveform={k:v[j] for k,v in wn.items()})),allow_nan=False)+'\n')
                        f.flush();d.write(out/'progress.json',dict(draws_completed=draw+1,packet_forwards=packet_forwards,paired_batches=paired_batches,PA_acted=acted_count))
                        print(f'PROGRESS RX{rx} draw={draw+1}/16 forwards={packet_forwards}',flush=True)
            if packet_forwards!=41582 or paired_batches!=160 or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError('Frozen diagnostic budget/state differs')
            if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in snapshot.items()):raise ValueError('Frozen checkpoint parameters/buffers changed')
            result=dict(status='SOURCE_PA_DIAGNOSTIC_COMPLETE',config=c,packet_forwards=packet_forwards,paired_batches=paired_batches,
                RAW_packets=1260,FULL_packets=20160,NO_PA_packets=20160,checkpoint_smoke_packets=2,PA_acted=acted_count,
                upstream_identical=True,non_PA_output_identical=True,parameters_buffers_unchanged=True,
                optimizer_updates=0,head_fit_updates=0,U_used=False,V_used=False,target_access=False)
            d.write(out/'completion.json',result);counts[row['row_id']]=result
            del model,snapshot;gc.collect();torch.cuda.empty_cache()
        d.write(d.BASE/'completion.json',dict(status='SOURCE_PA_DIAGNOSTIC_COMPLETE',rows=list(counts),
            packet_forwards=sum(v['packet_forwards'] for v in counts.values()),checkpoint_smoke_packets=10,
            optimizer_updates=0,new_models=0,target_access=False,U_used=False,V_used=False,source_only=True,
            automatic_training=False,goal_completed=False,wall_seconds=time.perf_counter()-start))
        print('COMPLETE source-only PA diagnostic; no training/target',flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
