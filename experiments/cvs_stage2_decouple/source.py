"""Finite source-only preparation and 40-epoch identity micro-tuning."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_stage2_decouple import design as d
from experiments.cvs_stage2_decouple.common import (build_model,verify_parent,IdentityView,identity_logits,margins,
    positive_huber,parameter_groups,learning_rate,clean)
from experiments.cvs_stage2_decouple.actions import fit_actions,load_actions
from experiments.cvs_stage2_decouple.receiver import fit_receiver,ReceiverBank
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def dump(path,value):d.write(path,clean(value))

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def prepare(seed,device='cuda:0'):
    from experiments.cvs_stage2_decouple.data import build_source_cache
    c=d.config(next(r for r in d.rows() if r['model_seed']==seed))
    out=Path(c['preparation']);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(2);seed_all(seed)
    provenance=verify_parent(c)
    with numerical_context(d.FULL_FP32_POLICY):
        parent=build_model(c,device)
        parent.eval()
        # Construction order is deliberate: strict parent load, cache, actions, R.
        packed=build_source_cache(c,IdentityView(parent),device)
        cache=packed['labeled']
        torch.save(packed,out/'source_cache.pt')
        actions=fit_actions(IdentityView(parent),cache,seed,device,passes=2,batch_size=128)
        torch.save(actions,out/'actions.pt')
        del parent
        if torch.cuda.is_available():torch.cuda.empty_cache()
        receiver=fit_receiver(cache,seed,fit_steps=200)
        torch.save(receiver,out/'receiver.pt')
        dump(out/'receiver_diagnostics.json',dict(folds=[{k:v for k,v in f.items() if k not in ('model','standard_fit_ids','standard_mean','standard_scale')} for f in receiver['folds']],
              diagnostics=receiver['diagnostics'],grouping_decisions=receiver['grouping_decisions'],tau_y=receiver['tau_y']))
        dump(out/'parent_provenance.json',provenance)
        result=dict(status='PREPARATION_COMPLETE',target_access=False,seed=seed,config=c,
                    strict_parent_loaded_before_auxiliary=True,source_count=len(cache['x']),validation_count=len(packed['validation']['x']),
                    parent_clean_forward_packets=len(cache['x']),action_passes=2,receiver_fit_steps=200,
                    elapsed_seconds=time.perf_counter()-started,pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        dump(out/'preparation_complete.json',result)
        print(json.dumps(result),flush=True)
    return result

class SourceStream:
    """Auxiliary-independent ordinary permutation stream and relation RNG."""
    def __init__(self,n,bank,seed):
        self.n=n;self.bank=bank
        self.g=torch.Generator().manual_seed(seed+101)
        self.relation_g=torch.Generator().manual_seed(seed+202)
        self.pool=torch.empty(0,dtype=torch.long)
    def ordinary(self,n):
        while len(self.pool)<n:self.pool=torch.cat((self.pool,torch.randperm(self.n,generator=self.g)))
        result=self.pool[:n];self.pool=self.pool[n:];return result
    def step(self,step):
        if step%8:return self.ordinary(128),None
        relation=self.bank.sample_relations(self.relation_g)
        return torch.cat((self.ordinary(96),relation['indices'])),relation

def objective(c,logits,y,parent_margin,parent_correct,tau,step,aug_logits=None,relation=None,bank=None,action_positions=None):
    u=margins(logits,y);zero=u.sum()*0.
    ce=F.cross_entropy(logits,y,label_smoothing=0.)
    keep=aug=drop=receiver=zero;details={}
    eligible=parent_correct & torch.isfinite(tau)
    if c['enabled']['keep']:
        target=torch.minimum(parent_margin,tau).nan_to_num(0.)
        keep=(positive_huber(target-u-.5)*eligible.to(u.dtype)).mean()
    if aug_logits is not None:
        selected=torch.arange(32,device=y.device) if action_positions is None else action_positions
        aug=F.cross_entropy(aug_logits,y[selected],label_smoothing=0.)
        drop=positive_huber(u[selected].detach()-margins(aug_logits,y[selected])-.5).mean()
    if c['enabled']['receiver'] and relation is not None:
        receiver,details=bank.identity_loss(u,relation,offset=96)
    ramp=min(step/444.,1.)
    weighted=dict(ce=ce,keep=ramp*.1*keep,aug_ce=ramp*.25*aug,drop=ramp*.1*drop,receiver=ramp*.1*receiver)
    return sum(weighted.values()),weighted,dict(ce=ce,keep=keep,aug_ce=aug,drop=drop,receiver=receiver,
              ramp=ramp,keep_eligible=int(eligible.sum()),keep_missing_tau=int((~torch.isfinite(tau)).sum()),
              clean_margin=u.mean(),receiver_details=details)

def gradient_diagnostics(weighted,groups):
    result={}
    for name,loss in weighted.items():
        if not loss.requires_grad:continue
        for group in groups:
            grads=torch.autograd.grad(loss,group['params'],retain_graph=True,allow_unused=True)
            norm=sum(g.detach().float().square().sum() for g in grads if g is not None)
            result[f'grad_{name}_{group["name"]}']=float(torch.sqrt(norm)) if torch.is_tensor(norm) else 0.
    for group in groups:
        key=group['name'];base=result.get('grad_ce_'+key,0.)
        for name in weighted:
            if name!='ce':result[f'grad_ratio_{name}_{key}']=result.get(f'grad_{name}_{key}',0.)/base if base>1e-12 else None
    return result

@torch.no_grad()
def observed_relation_metrics(logits,y,parent_margin,parent_correct,sample):
    """Only existing main-batch logits; no extra forward or R fitting."""
    if sample is None:return []
    u=margins(logits,y);correct=logits.argmax(1)==y;results=[]
    for relation in sample['relations']:
        if not relation['valid']:continue
        value=dict(slot=relation['slot'],active=relation['active'],c=relation['c'],independent_ids=16)
        for side in ('strong','weak'):
            positions=torch.tensor(relation[side+'_positions'],device=y.device)+96
            value.update({side+'_margin':float(u[positions].mean()),side+'_parent_margin':float(parent_margin[positions].mean()),
                side+'_margin_change_from_parent':float((u[positions]-parent_margin[positions]).mean()),
                side+'_accuracy':float(correct[positions].float().mean()),side+'_parent_accuracy':float(parent_correct[positions].float().mean())})
        results.append(value)
    return results

def train(c,device='cuda:0'):
    from experiments.cvs_stage2_decouple.data import cached_loader
    from experiments.cvs_feature_disentangle.validation import source_readonly,evaluate_source_stress
    from experiments.cvs_feature_disentangle.source import source_validation
    d.validate(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    prepared=Path(c['preparation']);pdone=d.read(prepared/'preparation_complete.json')
    if pdone['status']!='PREPARATION_COMPLETE' or pdone['seed']!=c['model_seed'] or pdone['target_access']:
        raise ValueError('Invalid shared source preparation')
    provenance=verify_parent(c)
    packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False)
    cache=packed['labeled'];vloader=cached_loader(packed['validation'],256)
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(packed['contract'],d.read(d.SOURCE))
    if sorted(cache['ids'])!=sorted(packed['contract']['role_ids']['L_s']):raise ValueError('Cached L_s physical IDs differ')
    bank=ReceiverBank(torch.load(prepared/'receiver.pt',map_location='cpu',weights_only=False))
    if bank.payload['source_ids']!=cache['ids']:raise ValueError('R cache physical order mismatch')
    stream=SourceStream(len(cache['x']),bank,c['model_seed'])
    seed_all(c['model_seed']);torch.set_num_threads(2);started=time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        model=build_model(c,device);groups,names=parameter_groups(model)
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        actions=load_actions(torch.load(prepared/'actions.pt',map_location='cpu',weights_only=False),device) if c['enabled']['actions'] else None
        # No parent, teacher, U iterator or satellite generator remains resident.
        resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
            commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'local_check',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),backend_flags=actual_flags(),
            hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
            parameter_groups=names,trainable_parameters=sum(p.numel() for g in groups for p in g['params']),
            total_parameters=sum(p.numel() for p in model.parameters()),
            execution=dict(parent_resident=False,teacher=False,unlabeled_iterator=False,satellite_generator=False,
                           action_forward=c['enabled']['actions'],receiver_network_forward=False,relation_sampling_all_arms=True))
        dump(out/'resolved_config.json',resolved);dump(out/'initialization.json',provenance);dump(out/'source_contract.json',packed['contract'])
        print('RESOLVED '+json.dumps(clean(resolved)),flush=True)
        epoch_rows=[];counts=dict(identity_updates=0,linear=0,temporal=0,relation_batches=0,receiver_attempts=0,receiver_active_relations=0)
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as steps:
            for epoch in range(1,41):
                epoch_start=time.perf_counter();sums={};model.train()
                for local in range(1,223):
                    step=(epoch-1)*222+local
                    indices,relation=stream.step(step)
                    x=cache['x'][indices].to(device);y=cache['y'][indices].to(device)
                    parent_margin=cache['parent_margin'][indices].to(device);correct=cache['parent_correct'][indices].to(device)
                    tau=bank.tau_y[y.cpu()].to(device)
                    for g in optimizer.param_groups:g['lr']=learning_rate(step,g['peak'])
                    optimizer.zero_grad(set_to_none=True)
                    timers={}
                    def mark(name):
                        if str(device).startswith('cuda'):
                            event=torch.cuda.Event(enable_timing=True);event.record();timers[name]=event
                        else:timers[name]=time.perf_counter()
                    mark('main_start')
                    logits=identity_logits(model,x);aug_logits=None;stats={};kind=None;action_positions=None
                    mark('main_end')
                    if actions is not None and step%4==0:
                        kind=c['schedule']['action_order'][(step//4-1)%4]
                        action_positions=torch.arange(96,128,device=y.device) if relation is not None else torch.arange(32,device=y.device)
                        ax,stats=actions.transform(x[action_positions],kind,torch.Generator().manual_seed(c['model_seed']+step*13))
                        # Auxiliary dropout cannot perturb the next clean forward's RNG.
                        devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
                        with torch.random.fork_rng(devices=devices):
                            torch.manual_seed(c['model_seed']+step*31)
                            aug_logits=identity_logits(model,ax)
                        counts[kind]+=1
                    mark('action_end')
                    loss,weighted,parts=objective(c,logits,y,parent_margin,correct,tau,step,aug_logits,relation,bank,action_positions)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no skipped update or auto retry')
                    grad={}
                    if step%888==0:
                        grad=gradient_diagnostics(weighted,groups)
                        raw_grad=gradient_diagnostics({k:parts[k] for k in weighted},groups)
                        grad.update({'unweighted_'+k:v for k,v in raw_grad.items()})
                    if aug_logits is not None:
                        difference=margins(logits,y)[action_positions].detach()-margins(aug_logits,y[action_positions]).detach()
                        stats.update(realized_margin_drop_mean=float(difference.mean()),
                            realized_margin_drop_q90=float(torch.quantile(difference,.9)),
                            augmented_margin_mean=float(margins(aug_logits,y[action_positions]).detach().mean()),
                            source_relation_reused=relation is not None)
                    mark('backward_start');loss.backward()
                    gradnorm=torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],float('inf'),error_if_nonfinite=True)
                    optimizer.step();mark('step_end');counts['identity_updates']+=1
                    if relation is not None:counts['relation_batches']+=1
                    if relation is not None and c['enabled']['receiver']:
                        counts['receiver_attempts']+=1;counts['receiver_active_relations']+=parts['receiver_details']['active_relations']
                    scalars={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items() if k!='receiver_details'}
                    scalars.update({'weighted_'+k:float(v.detach()) for k,v in weighted.items()})
                    scalars.update(total_loss=float(loss.detach()),total_gradient_norm=float(gradnorm),lr_encoder=optimizer.param_groups[0]['lr'],lr_fusion_head=optimizer.param_groups[1]['lr'])
                    def elapsed(a,b):
                        if str(device).startswith('cuda') and not timers[b].query():timers[b].synchronize()
                        return timers[a].elapsed_time(timers[b])/1000 if str(device).startswith('cuda') else timers[b]-timers[a]
                    scalars.update(main_forward_seconds=elapsed('main_start','main_end'),
                        action_transform_and_forward_seconds=elapsed('main_end','action_end'),
                        loss_and_gradient_audit_seconds=elapsed('action_end','backward_start'),
                        backward_optimizer_seconds=elapsed('backward_start','step_end'),
                        parent_correct_count=int(correct.sum()),
                        parent_correct_retained_count=int(((logits.argmax(1)==y)&correct).sum()))
                    for k,v in scalars.items():sums[k]=sums.get(k,0.)+v
                    record=dict(step=step,epoch=epoch,metrics=scalars,gradient_audit=grad,action_kind=kind,action_stats=stats,
                                receiver=parts['receiver_details'],relation_observations=observed_relation_metrics(logits,y,parent_margin,correct,relation),
                                relation_sampling=relation['diagnostics'] if relation else None)
                    steps.write(json.dumps(clean(record),allow_nan=False)+'\n')
                steps.flush()
                with source_readonly(model),torch.no_grad():validation=source_validation(IdentityView(model),vloader,device)
                row=dict(epoch=epoch,optimizer_steps=counts['identity_updates'],epoch_time_s=time.perf_counter()-epoch_start,
                    **{k:v/222 for k,v in sums.items()},**validation,**counts)
                epoch_rows.append(clean(row))
                (out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in epoch_rows),encoding='utf-8')
                flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in epoch_rows]
                with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
                print('EPOCH '+json.dumps(clean(row),allow_nan=False),flush=True)
        if counts['identity_updates']!=8880 or counts['relation_batches']!=1110:raise ValueError('Incomplete identity/relationship budget')
        if actions is not None and (counts['linear']!=1110 or counts['temporal']!=1110):raise ValueError('Incomplete action balance')
        identity_training_seconds=time.perf_counter()-started
        ck=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,run_id=d.RUN,candidate_id=c['row_id'],epoch=40,
                optimizer_steps=8880,checkpoint_selection='final_only',parent_checkpoint=c['parent_checkpoint'],optimizer=optimizer.state_dict(),
                target_access=False,rng=torch.get_rng_state())
        torch.save(ck,out/'final_ssdg.pth')
        model.load_state_dict(torch.load(out/'final_ssdg.pth',map_location=device,weights_only=False)['model'],strict=True)
        stress=evaluate_source_stress(model,vloader,device,c,out)
        result=dict(status='SOURCE_TRAINED',epoch=40,optimizer_steps=8880,logged_steps=8880,config=c,checkpoint=str(out/'final_ssdg.pth'),
              checkpoint_selection='final_only',target_access=False,target_evaluated=False,final_source_metrics=validation,
              mechanism_execution=counts,elapsed_seconds=time.perf_counter()-started,
              preparation_seconds=pdone['elapsed_seconds'],shared_preparation_seed=c['model_seed'],
              peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
              resources=dict(identity_training_including_epoch_validation_seconds=identity_training_seconds,
                  final_source_diagnostics_seconds=time.perf_counter()-started-identity_training_seconds,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
                  trainable_parameters=resolved['trainable_parameters'],total_parameters=resolved['total_parameters'],
                  timing_scope='shared GPU wall; incremental stage2; parent E200 and per-seed shared preparation separately reported'),
              parent_provenance=provenance,source_stress=str(out/'source_stress.json'))
        dump(out/'completion.json',result)
        dump(out.parent/'source_frozen.json',dict(status='SOURCE_FROZEN',run_id=d.RUN,row_id=c['row_id'],config=c,
             checkpoint=str(out/'final_ssdg.pth'),epoch=40,optimizer_steps=8880,target_access=False,target_scores_consumed=False,selection='fixed_final_E40'))
        print('SOURCE_TRAINED '+json.dumps(clean(result)),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--seed',type=int)
    parser.add_argument('--config');parser.add_argument('--device',default='cuda:0');parser.add_argument('--train_stage2',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if args.seed not in d.SEEDS:parser.error('Registered seed required')
        prepare(args.seed,args.device)
    else:
        if not args.config:parser.error('--config required')
        train(d.read(args.config),args.device)
