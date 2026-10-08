"""Scratch CE+native concat training; source-only residual auxiliaries."""
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
import torch.nn.functional as F
from baselines.common.practical_source import build_contract_split
from baselines.common.cvs_data import make_cvs_loader
from experiments.cvs_selected_concat.source import source_args, checkpoint_smoke
from experiments.cvs_energy_identity.source import validate
from experiments.cvs_phase1_overlay.augmentation import OriginalLEO
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
from experiments.cvs_receiver_residual_v2 import design as d
from experiments.cvs_receiver_residual_v2.objectives import auxiliary_losses, collect_bank


def train(c):
    d.validate_config(c)
    with numerical_context(c['numerical_policy']):
        return _train(c)


def _train(c):
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    seed = c['model_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2); device = torch.device(c['device'])
    model = d.build_model(c['arm']).to(device)
    checkpoint_smoke(model, out/'initial_smoke.pt', device)
    split = build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts'] != {'L_s':6300,'U_s':56700,'V':27000}:
        raise ValueError('Source physical roles/counts differ')
    def loader(data, batch, shuffle, loader_seed):
        result = make_cvs_loader(data, batch_size=batch, shuffle=shuffle,
            num_workers=0, device=device, drop_last=False)
        result.generator = torch.Generator().manual_seed(loader_seed)
        if shuffle: result.sampler.generator = torch.Generator().manual_seed(loader_seed)
        return result
    train_loader = loader(split.train, 128, True, seed)
    stat_loader = loader(split.train, 256, False, seed+1)
    val_loader = loader(split.val, 256, False, seed+2)
    initial = dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
        target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    d.write(out/'initialization.json', initial)
    resolved = dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
        commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL_CHECK',
        torch_version=torch.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        source_counts=split.split_info['counts'],steps_per_epoch=len(train_loader),
        precision='float32',backend_flags=actual_flags(),architecture_actual=model.contract(),
        U_s_use='unused',gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',target_access=False,
        statistics_pass='All arms same extra L-only evaluation pass E40-200; no gradients/EMA; only displacement arms consume targets',
        inference_centroid_state=False,cuda_rng_policy='model_seed preserved; additional head uses CPU-only private generator')
    d.write(out/'resolved_config.json', resolved)
    print('RESOLVED_CONFIG '+json.dumps(resolved), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=c['lr_min'])
    augment = OriginalLEO(out); bank = None; steps_total = 0; started = time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with (out/'step_metrics.jsonl').open('x',encoding='utf-8') as sf, \
         (out/'epoch_metrics.jsonl').open('x',encoding='utf-8') as ef, \
         (out/'epoch_metrics.csv').open('x',encoding='utf-8',newline='') as cf, \
         (out/'centroid_diagnostics.jsonl').open('x',encoding='utf-8') as bf:
        writer = None
        for epoch in range(1,201):
            tic=time.perf_counter(); model.train(); strength=d.ramp(epoch)
            model.strength.fill_(strength if c['arm']!='baseline' else 1.)
            lr=optimizer.param_groups[0]['lr']
            sums={k:0. for k in ('total_loss','clean_ce','satellite_ce','displacement_loss','gauge_loss',
                'contribution_loss','gradient_norm','correction_norm_ratio','gate_abs_delta')}
            exposure=channel=pairs=n=0; scenes={}
            for step,batch in enumerate(train_loader,1):
                x,y,rx=(batch[k].to(device) for k in ('iq','label','receiver'))
                optimizer.zero_grad(set_to_none=True)
                view=augment(x,batch,epoch,step)
                details=model.details(torch.cat((x,view.x)))
                clean,sat_logits=details['logits'].split(len(x))
                ce=F.cross_entropy(clean,y)
                sat=F.cross_entropy(sat_logits,y) if epoch>=80 else None
                aux,info=auxiliary_losses(model,details,y,rx,bank,strength)
                loss=ce+(.68*sat if sat is not None else 0.)+aux
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                grads=[p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
                norm=torch.linalg.vector_norm(torch.stack(grads))
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
                used=sum(p.numel() for p in model.parameters() if p.grad is not None)
                def head_norm(name):
                    values=[p.grad.norm() for n,p in model.named_parameters() if n.startswith(name+'.') and p.grad is not None]
                    return float(torch.stack(values).norm()) if values else None
                record=dict(epoch=epoch,step=steps_total+1,arm=c['arm'],total_loss=float(loss.detach()),
                    clean_ce=float(ce.detach()),satellite_ce=float(sat.detach()) if sat is not None else None,
                    **{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in info.items()},
                    clean_weight=1.,satellite_weight=.68 if sat is not None else 0.,
                    displacement_weight=strength*d.RECIPE['displacement_weight'] if hasattr(model,'displacement') else 0.,
                    gauge_weight=strength*d.RECIPE['gauge_weight'] if hasattr(model,'displacement') else 0.,
                    contribution_weight=strength*d.RECIPE['contribution_weight'] if hasattr(model,'contribution') else 0.,
                    residual_strength=strength,correction_norm_ratio=float(details['correction_norm_ratio'].detach()),
                    gate_abs_delta=float(details['gate_abs_delta'].detach()),gradient_norm=float(norm),
                    displacement_gradient_norm=head_norm('displacement'),contribution_gradient_norm=head_norm('contribution'),
                    response_gain_gradient_norm=float(model.encoder.response_gain.grad.norm()),
                    gradient_used_parameters=used,learning_rate=lr,source_samples=len(y),concat_samples=len(y),
                    channel_applied=view.applied,view=view.scenario,view_probability=view.view_prob,
                    domain_backbone_active=False,pseudo_labels_active=False,ema_active=False,
                    auxiliary_active=strength>0 and c['arm']!='baseline')
                optimizer.step(); n+=1; steps_total+=1; exposure+=len(y)
                channel+=len(y) if view.applied else 0; pairs+=info['cross_tx_pair_count']
                scenes[view.scenario]=scenes.get(view.scenario,0)+1
                sf.write(json.dumps(record,allow_nan=False)+'\n')
                if step==1 or step%10==0 or step==len(train_loader):
                    print('TRAIN '+json.dumps(record,allow_nan=False),flush=True)
                for k in sums:
                    if record[k] is not None: sums[k]+=record[k]
            if n!=50 or exposure!=6300: raise ValueError('Optimizer/source exposure mismatch')
            scheduler.step(); bank_seconds=0.; diagnostics=None
            if epoch>=40:
                tb=time.perf_counter(); bank,diagnostics=collect_bank(model,stat_loader,device)
                bank_seconds=time.perf_counter()-tb
                bf.write(json.dumps(dict(epoch=epoch,**diagnostics),allow_nan=False)+'\n');bf.flush()
            metrics=dict(epoch=epoch,arm=c['arm'],optimizer_steps=n,optimizer_steps_total=steps_total,
                source_sample_exposure=exposure,concat_sample_exposure=exposure,
                satellite_ce_sample_exposure=exposure if epoch>=80 else 0,
                channel_changed_sample_exposure=channel,centroid_readonly_L_exposure=6300 if epoch>=40 else 0,
                centroid_pass_seconds=bank_seconds,cross_tx_pair_count=pairs,
                **{k:v/n for k,v in sums.items()},learning_rate=lr,residual_strength=strength,
                displacement_weight=record['displacement_weight'],gauge_weight=record['gauge_weight'],
                contribution_weight=record['contribution_weight'],satellite_weight=record['satellite_weight'],
                clean_weight=1.,scene_batch_counts=scenes,gradient_used_parameters=used,
                response_gain_abs_mean=float(model.encoder.response_gain.tanh().abs().mean()),
                displacement_gradient_norm=record['displacement_gradient_norm'],
                contribution_gradient_norm=record['contribution_gradient_norm'],
                **validate(model,val_loader,device),elapsed_seconds=time.perf_counter()-tic,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                cross_tx_relative_prediction_error=diagnostics['cross_tx_relative_prediction_error'] if diagnostics else None,
                cross_tx_direction_cosine=diagnostics['cross_tx_direction_cosine'] if diagnostics else None,
                domain_backbone_active=False,pseudo_labels_active=False,ema_active=False,
                auxiliary_active=strength>0 and c['arm']!='baseline')
            if epoch<80: metrics['satellite_ce']=None
            line=json.dumps(metrics,allow_nan=False)
            print('EPOCH '+line,flush=True);ef.write(line+'\n');ef.flush();sf.flush()
            compact={k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}
            if writer is None: writer=csv.DictWriter(cf,fieldnames=list(compact));writer.writeheader()
            writer.writerow(compact);cf.flush()
    contract=d.read(out/'source_contract.json')
    torch.save(dict(model=model.state_dict(),method=d.METHOD,variant=c['variant'],arm=c['arm'],epoch=200,
        classes=contract['classes'],num_classes=6,source_contract=contract,initialization=initial,
        selection='fixed_last_epoch',config=resolved),out/'last.pt')
    from experiments.cvs_clean_design.profile import resource_profile
    profile=resource_profile(model,device)
    profile.update(training_profile_scope='Synthetic CE-only step; actual auxiliary/centroid-inclusive epoch times separately logged',
        resident_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        checkpoint_bytes=(out/'last.pt').stat().st_size,centroid_state_transferred_bytes=0)
    d.write(out/'resource_profile.json',profile)
    d.write(out/'source_selection.json',dict(status='SOURCE_SELECTION_FROZEN',variant=c['variant'],arm=c['arm'],
        epoch=200,rule='fixed factorial comparison; all16 ownE200 tested',target_access=False,target_score_used=False))
    d.write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=steps_total,
        checkpoint=str(out/'last.pt'),elapsed_seconds=time.perf_counter()-started,
        target_access=False,target_evaluated=False,final_source_metrics=validate(model,val_loader,device),backend_flags=actual_flags()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    train(d.read(p.parse_args().config))
