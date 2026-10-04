"""Source-only, from-scratch, 2x2 original-mechanism transplant experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
import numpy as np
import torch
from baselines.common.practical_source import build_contract_split
from baselines.common.cvs_data import make_cvs_loader
from experiments.cvs_selected_concat.source import source_args, checkpoint_smoke
from experiments.cvs_energy_identity.source import validate
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
from experiments.cvs_phase1_overlay.contract import validate_config, build_model, read, write
from experiments.cvs_phase1_overlay.augmentation import OriginalLEO, loss_for_batch


def train(c):
    validate_config(c)
    with numerical_context(c['numerical_policy']):
        _train(c)


def _train(c):
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    seed = c['model_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.set_num_threads(2)
    device = torch.device(c['device'])
    model = build_model(c['arm']).to(device)
    checkpoint_smoke(model, out/'initial_smoke.pt', device)
    split = build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts'] != {'L_s':6300,'U_s':56700,'V':27000}:
        raise ValueError('Source physical role contract differs')
    train_loader = make_cvs_loader(split.train,batch_size=128,shuffle=True,num_workers=0,device=device,drop_last=False)
    train_loader.sampler.generator = torch.Generator().manual_seed(seed)
    val_loader = make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False)
    initial = dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
        target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    write(out/'initialization.json',initial)
    resolved = dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
        commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL_CHECK',
        torch_version=torch.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        total_parameters=sum(p.numel() for p in model.parameters()),source_counts=split.split_info['counts'],
        steps_per_epoch=len(train_loader),precision='float32',backend_flags=actual_flags(),
        architecture_actual=model.contract(),loader_seed=seed,U_s_use='unused',
        gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',target_access=False)
    write(out/'resolved_config.json',resolved)
    print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=c['lr_min'])
    augment = OriginalLEO(out) if 'leo' in c['arm'] else None
    steps_total = 0; started = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    with (out/'step_metrics.jsonl').open('x',encoding='utf-8') as sf, \
         (out/'epoch_metrics.jsonl').open('x',encoding='utf-8') as ef, \
         (out/'epoch_metrics.csv').open('x',encoding='utf-8',newline='') as cf:
        writer = None
        for epoch in range(1,201):
            tic = time.perf_counter(); model.train(); lr = optimizer.param_groups[0]['lr']
            sums = dict(total_loss=0.,clean_ce=0.,satellite_ce=0.,gradient_norm=0.)
            exposure=concat=channel=changed=calls=0; scenes={}; n=0
            for step,batch in enumerate(train_loader,1):
                x,y,d = (batch[k].to(device) for k in ('iq','label','domain'))
                optimizer.zero_grad(set_to_none=True)
                loss,info = loss_for_batch(model,x,y,d,batch,augment,epoch,step)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]))
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
                used = sum(p.numel() for p in model.parameters() if p.grad is not None)
                response_gradient = model.encoder.response_gain.grad
                record = dict(epoch=epoch,step=steps_total+1,arm=c['arm'],total_loss=float(loss.detach()),
                    **{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in info.items()},
                    clean_weight=1.,gradient_norm=float(norm),gradient_used_parameters=used,
                    response_gain_gradient_norm=float(response_gradient.norm()) if response_gradient is not None else None,
                    learning_rate=lr,source_samples=len(y),mixstyle_calls=model.last_mixstyle_calls,
                    mixstyle_changed_layer_samples=model.last_mixstyle_changed_samples,
                    domain_backbone_active=False,pseudo_labels_active=False,extra_losses_active=False)
                optimizer.step(); n+=1; steps_total+=1; exposure+=len(y)
                concat+=info['concat_samples']; channel+=len(y) if info['channel_applied'] else 0
                calls+=model.last_mixstyle_calls; changed+=model.last_mixstyle_changed_samples
                scenes[info['view']]=scenes.get(info['view'],0)+1
                sf.write(json.dumps(record,allow_nan=False)+'\n')
                for k in sums:
                    if record[k] is not None: sums[k]+=record[k]
            if n!=50 or exposure!=6300: raise ValueError('Unexpected optimizer/data exposure')
            scheduler.step()
            metrics = dict(epoch=epoch,arm=c['arm'],optimizer_steps=n,optimizer_steps_total=steps_total,
                source_sample_exposure=exposure,concat_sample_exposure=concat,channel_changed_sample_exposure=channel,
                **{k:v/n for k,v in sums.items()},satellite_weight=.68 if augment and epoch>=80 else 0.,
                clean_weight=1.,learning_rate=lr,mixstyle_calls=calls,mixstyle_changed_layer_samples=changed,
                scene_batch_counts=scenes,gradient_used_parameters=used,
                response_gain_abs_mean=float(model.encoder.response_gain.tanh().abs().mean()),
                response_gain_gradient_norm=record['response_gain_gradient_norm'],
                **validate(model,val_loader,device),elapsed_seconds=time.perf_counter()-tic,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            if not augment or epoch<80: metrics['satellite_ce']=None
            line=json.dumps(metrics,allow_nan=False)
            print('EPOCH '+line,flush=True); ef.write(line+'\n'); ef.flush(); sf.flush()
            compact={k:v for k,v in metrics.items() if not isinstance(v,dict)}
            if writer is None: writer=csv.DictWriter(cf,fieldnames=list(compact)); writer.writeheader()
            writer.writerow(compact); cf.flush()
    contract=read(out/'source_contract.json')
    torch.save(dict(model=model.state_dict(),method=c['method'],variant=c['variant'],arm=c['arm'],epoch=200,
        classes=contract['classes'],num_classes=6,source_contract=contract,initialization=initial,
        selection='fixed_last_epoch',config=resolved),out/'last.pt')
    from experiments.cvs_clean_design.profile import resource_profile
    resource=resource_profile(model,device)
    resource['training_profile_scope']='Synthetic CE-only backbone step; actual mechanism-inclusive epoch times logged separately'
    resource['resident_buffer_bytes']=sum(b.numel()*b.element_size() for b in model.buffers())
    resource['checkpoint_bytes']=(out/'last.pt').stat().st_size
    write(out/'resource_profile.json',resource)
    write(out/'source_selection.json',dict(status='SOURCE_SELECTION_FROZEN',variant=c['variant'],arm=c['arm'],
        epoch=200,rule='user-fixed architecture, fixed factorial comparison, own last epoch',target_access=False,target_score_used=False))
    write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=steps_total,
        checkpoint=str(out/'last.pt'),elapsed_seconds=time.perf_counter()-started,
        target_access=False,target_evaluated=False,final_source_metrics=validate(model,val_loader,device),backend_flags=actual_flags()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    train(read(a.config))
