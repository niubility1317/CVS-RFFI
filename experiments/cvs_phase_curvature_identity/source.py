"""No target construction, no augmentation, no teacher, one CE update/batch."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.cvs_phase_curvature_identity.model import build, VARIANTS, curvature_contract
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
from experiments.cvs_identity_ce.source import source_args, checkpoint_smoke, write
from experiments.cvs_phase_curvature_identity.physics import frozen_synthetic_diagnostics
from experiments.cvs_phase_curvature_identity.telemetry import memory_state,memory_gradients,memory_step_metrics,memory_epoch_metrics
from baselines.common.practical_source import build_contract_split
from baselines.common.cvs_data import make_cvs_loader


def validate_config(c):
    for key in ('checkpoint','resume','teacher','initial_checkpoint','target_inputs','target_truth','p1_truth','p1_capsule'):
        if c.get(key): raise ValueError('Source-only scratch training rejects ' + key)
    required = dict(method='cvs_phase_curvature_identity', epochs=200, batch_size=128, lr=.0002,
        lr_min=1e-6, weight_decay=.0001, drop_last=False, augmentation=False,
        domain_backbone=False, extra_losses=[], selection='fixed_last_epoch', split_seed=392005)
    if any(c.get(k)!=v for k,v in required.items()): raise ValueError('Clean matched training contract mismatch')
    if c['variant'] not in VARIANTS: raise ValueError('Unknown variant')
    if c.get('curvature')!=curvature_contract(c['variant']):raise ValueError('Causal volterra-envelope input contract mismatch')
    if c.get('numerical_policy')!=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest'):
        raise ValueError('Unregistered numerical policy')
    return c


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); correct=count=0; total=0.; groups={}
    for batch in loader:
        y=batch['label'].to(device); logits=model(batch['iq'].to(device))
        ok=(logits.argmax(1)==y).cpu()
        total+=float(F.cross_entropy(logits,y,reduction='sum')); correct+=int(ok.sum()); count+=len(y)
        for rx in batch['receiver'].unique().tolist():
            mask=batch['receiver']==rx
            g=groups.setdefault(str(rx),[0,0]);g[0]+=int(ok[mask].sum());g[1]+=int(mask.sum())
    rates={rx:hit/n for rx,(hit,n) in groups.items()}
    return dict(source_val_count=count, source_val_accuracy=correct/count, source_val_ce=total/count,
        source_val_rx_accuracy=rates, source_val_worst_rx=min(rates.values()))


@torch.no_grad()
def final_source_diagnostics(model,loader,device):
    model.eval();groups={};count=0
    for batch in loader:
        partial,omega,valid,coherence,alpha=model.coordinates(batch['iq'].to(device))
        emb=model.core.features(partial)
        post_omega,post_valid,_=model.synchronizer.estimate(partial)
        nominal_hz=((1-alpha)*omega*25000000/(2*torch.pi)).cpu().double()
        post_hz=(post_omega*25000000/(2*torch.pi)).cpu().double()
        formula_error=(post_hz-nominal_hz).abs();post_valid=post_valid.cpu()
        hz=(omega*25000000/(2*torch.pi)).cpu().double()
        valid=valid.cpu();coherence=coherence.cpu().double()
        logits=model.classify_features(emb)
        ok=(logits.argmax(1).cpu()==batch['label'])
        normalized=F.normalize(emb,dim=1,eps=1e-4).cpu().double()
        for j in range(len(emb)):
            key=(int(batch['label'][j]),int(batch['receiver'][j]),int(batch['day'][j]))
            g=groups.setdefault(key,dict(count=0,correct=0,embedding_sum=torch.zeros(160,dtype=torch.float64),square_sum=0.,
                relative_cfo_hz_sum=0.,relative_cfo_hz_min=float('inf'),relative_cfo_hz_max=-float('inf'),fallback_count=0,coherence_sum=0.,residual_hz_sum=0.,nominal_hz_sum=0.,formula_error_max=0.,formula_eligible_count=0,residual_valid_count=0))
            g['count']+=1;g['correct']+=int(ok[j]);g['embedding_sum']+=normalized[j];g['square_sum']+=float(normalized[j].square().sum())
            g['relative_cfo_hz_sum']+=float(hz[j]);g['relative_cfo_hz_min']=min(g['relative_cfo_hz_min'],float(hz[j]));g['relative_cfo_hz_max']=max(g['relative_cfo_hz_max'],float(hz[j]))
            g['residual_hz_sum']+=float(post_hz[j]);g['nominal_hz_sum']+=float(nominal_hz[j]);g['residual_valid_count']+=int(post_valid[j])
            if valid[j] and post_valid[j]:
                g['formula_eligible_count']+=1;g['formula_error_max']=max(g['formula_error_max'],float(formula_error[j]))
            g['fallback_count']+=int(not valid[j]);g['coherence_sum']+=float(coherence[j])
        count+=len(emb)
    if count!=27000:raise ValueError('IncompletefinalsourceVdiagnostics')
    records=[];tx_groups={};rx_groups={}
    for (tx,rx,day),g in sorted(groups.items()):
        centroid=g['embedding_sum']/g['count']
        records.append(dict(tx=tx,receiver=rx,day=day,count=g['count'],accuracy=g['correct']/g['count'],
                            unit_embedding_mean=centroid.tolist(),unit_embedding_trace_variance=max(0.,g['square_sum']/g['count']-float(centroid.square().sum())),
                            relative_cfo_hz_mean=g['relative_cfo_hz_sum']/g['count'],relative_cfo_hz_min=g['relative_cfo_hz_min'],relative_cfo_hz_max=g['relative_cfo_hz_max'],
                            alignment_strength=float(model.alignment_strength()),residual_estimated_cfo_hz_mean=g['residual_hz_sum']/g['count'],nominal_residual_cfo_hz_mean=g['nominal_hz_sum']/g['count'],residual_formula_max_error_hz=g['formula_error_max'] if g['formula_eligible_count'] else None,residual_formula_eligible_count=g['formula_eligible_count'],residual_valid_fraction=g['residual_valid_count']/g['count'],coherence_mean=g['coherence_sum']/g['count'],fallback_fraction=g['fallback_count']/g['count']))
        for key,collection in ((tx,tx_groups),((tx,rx),rx_groups)):
            h=collection.setdefault(key,dict(count=0,total=torch.zeros(160,dtype=torch.float64)))
            h['count']+=g['count'];h['total']+=g['embedding_sum']
    tx_means={k:g['total']/g['count'] for k,g in tx_groups.items()}
    between=[float((tx_means[a]-tx_means[b]).square().sum()) for a in tx_means for b in tx_means if a<b]
    within=[float((g['total']/g['count']-tx_means[tx]).square().sum()) for (tx,rx),g in rx_groups.items()]
    return dict(role='V',count=count,groups=records,mean_between_tx_centroid_squared_distance=sum(between)/len(between),
                mean_within_tx_rx_centroid_squared_distance=sum(within)/len(within),
                target_access=False,used_for_training=False,used_for_selection=False,
                claim='Aggregateidentityembeddinggeometryonly;not TX hardwareparameterrecovery or calibratedRXseparation.')


def train(c):
    validate_config(c)
    with numerical_context(c.get('numerical_policy')):
        return _train(c)


def _train(c):
    validate_config(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    seed=c['model_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);device=torch.device(c.get('device','cuda:0'))
    model=build(c['variant']).to(device)
    physical=model
    if physical.contract()!=c['curvature']:raise ValueError('Actual RF operator differs from registered configuration')
    checkpoint_smoke(model,out/'initial_smoke.pt',device)
    split=build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:
        raise ValueError('Source roles changed or target constructed')
    train_loader=make_cvs_loader(split.train,batch_size=128,shuffle=True,num_workers=0,device=device,drop_last=False)
    train_loader.sampler.generator=torch.Generator().manual_seed(seed)
    val_loader=make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
        target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    write(out/'initialization.json',initial)
    resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
        commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'LOCAL',
        torch_version=torch.__version__,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        source_counts=split.split_info['counts'],steps_per_epoch=len(train_loader),
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        U_s_use='unused',augmentation=False,target_access=False,precision='float32',gradient_clipping=None,
        optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,curvature_actual=physical.contract(),feature_curvature_active=True,classifier_scale=30.0)
    if 'numerical_policy' in c:resolved['backend_flags']=actual_flags()
    write(out/'resolved_config.json',resolved);print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=1e-6)
    started=time.perf_counter();step=0
    with (out/'step_metrics.jsonl').open('x',encoding='utf-8') as sf, (out/'epoch_metrics.jsonl').open('x',encoding='utf-8') as ef, (out/'epoch_compact.jsonl').open('x',encoding='utf-8') as jf, (out/'epoch_metrics.csv').open('x',encoding='utf-8',newline='') as cf:
        writer=None
        for epoch in range(1,201):
            memory_grad_sums=[0.]*6;memory_abs_sums=[0.]*6
            tic=time.perf_counter();model.train();ce_sum=norm_sum=alignment_norm_sum=0.;mix_grad_sum=[0.,0.];mix_norm_sum=0.;exposure=0;n=0;lr=optimizer.param_groups[0]['lr'];grad_params=0
            for batch in train_loader:
                x,y=batch['iq'].to(device),batch['label'].to(device);optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x),y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CE')
                loss.backward();grads=[p.grad.detach() for p in model.parameters() if p.grad is not None]
                norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                grad_params=sum(p.numel() for p in model.parameters() if p.grad is not None)
                gate=model.core.behavior[0].mix_raw
                if gate.grad is None or not torch.isfinite(gate.grad).all():raise ValueError('Missing/nonfinite residual gate gradients')
                mix_grad=gate.grad.detach().tolist();mix_before=gate.detach().tolist();mix_coeff_before=gate.detach().tanh().tolist()
                if grad_params!=202561:raise ValueError('Some identity parameters did not receive CE gradients')
                for j in range(2):mix_grad_sum[j]+=mix_grad[j]
                mix_norm_sum+=float(gate.grad.norm())
                memory_before=memory_state(model);memory_grads=memory_gradients(model)
                for i,g in enumerate(memory_grads):memory_grad_sums[i]+=g;memory_abs_sums[i]+=abs(g)
                alignment_trainable=False
                alignment_grad=model.alignment_logit.grad if alignment_trainable else None
                if alignment_trainable and alignment_grad is None:raise ValueError('Alignment scalar was not used')
                alignment_norm=float(alignment_grad.abs()) if alignment_trainable else None
                if alignment_trainable:alignment_norm_sum+=alignment_norm
                alpha_before=float(model.alignment_strength().detach())
                optimizer.step();mix_after=gate.detach().tolist();mix_coeff_after=gate.detach().tanh().tolist();step+=1;n+=1;exposure+=len(y);ce_sum+=float(loss.detach());norm_sum+=float(norm)
                rec=dict(epoch=epoch,step=step,clean_ce=float(loss.detach()),total_loss=float(loss.detach()),ce_weight=1.0,
                    learning_rate=lr,gradient_norm=float(norm),gradient_used_parameters=grad_params,alignment_gradient_used_parameters=int(alignment_trainable),alignment_gradient_norm=alignment_norm,alignment_strength_before=alpha_before,alignment_strength_after=float(model.alignment_strength().detach()),
                    mixture_gradient_used_parameters=2,mix_gradient3=mix_grad[0],mix_gradient5=mix_grad[1],mix_gradient_norm=float(gate.grad.norm()),
                    mix_raw3_before=mix_before[0],mix_raw5_before=mix_before[1],mix_raw3_after=mix_after[0],mix_raw5_after=mix_after[1],
                    mix_coefficient3_before=mix_coeff_before[0],mix_coefficient5_before=mix_coeff_before[1],mix_coefficient3_after=mix_coeff_after[0],mix_coefficient5_after=mix_coeff_after[1],
                    source_samples=len(y),satellite_samples=0,augmentation_active=False,domain_backbone_active=False,
                    pseudo_labels_active=False,extra_losses_active=False,feature_curvature_active=True)
                if 'numerical_policy' in c:rec['cudnn_allow_tf32']=torch.backends.cudnn.allow_tf32
                rec.update(memory_step_metrics(memory_before,memory_state(model),memory_grads))
                sf.write(json.dumps(rec,allow_nan=False)+'\n')
            if n!=50 or exposure!=6300:raise ValueError('Matched 50-step/6300-sample budget changed')
            scheduler.step();validation=validate(model,val_loader,device)
            stats=dict(epoch=epoch,optimizer_steps=n,optimizer_steps_total=step,source_sample_exposure=exposure,
                clean_ce=ce_sum/n,total_loss=ce_sum/n,ce_weight=1.,learning_rate=lr,gradient_norm=norm_sum/n,
                gradient_used_parameters=grad_params,alignment_gradient_used_parameters=int(alignment_trainable),alignment_gradient_norm=alignment_norm_sum/n if alignment_trainable else None,alignment_strength=float(model.alignment_strength().detach()),alignment_gradient_note='Measured scalar gradient;fixed version N/A',augmentation_active=False,satellite_sample_exposure=0,
                domain_backbone_active=False,extra_losses_active=False,pseudo_labels_active=False,
                mixture_gradient_used_parameters=2,mix_gradient3=mix_grad_sum[0]/n,mix_gradient5=mix_grad_sum[1]/n,mix_gradient_norm=mix_norm_sum/n,
                mix_raw3=mix_after[0],mix_raw5=mix_after[1],mix_coefficient3=mix_coeff_after[0],mix_coefficient5=mix_coeff_after[1],
                feature_curvature_active=True,curvature_diagnostics=model.diagnostics(x),energy_diagnostic_scope='last source batch of epoch',
                **validation,elapsed_seconds=time.perf_counter()-tic,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            if 'numerical_policy' in c:stats['cudnn_allow_tf32']=torch.backends.cudnn.allow_tf32
            stats.update(memory_epoch_metrics(memory_state(model),memory_grad_sums,memory_abs_sums,n))
            line=json.dumps(stats,allow_nan=False);print('EPOCH '+line,flush=True);ef.write(line+'\n');ef.flush();sf.flush()
            compact={k:v for k,v in stats.items() if not isinstance(v,dict)}
            jf.write(json.dumps(compact,allow_nan=False)+'\n');jf.flush()
            if writer is None:writer=csv.DictWriter(cf,fieldnames=list(compact));writer.writeheader()
            writer.writerow(compact);cf.flush()
    contract=json.loads((out/'source_contract.json').read_text(encoding='utf-8'))
    torch.save(dict(model=model.state_dict(),method='cvs_phase_curvature_identity',variant=c['variant'],epoch=200,
        classes=contract['classes'],num_classes=6,source_contract=contract,initialization=initial,
        selection='fixed_last_epoch',config=resolved),out/'last.pt')
    write(out/'source_final_diagnostics.json',final_source_diagnostics(model,val_loader,device))
    physical_check=frozen_synthetic_diagnostics(model)
    write(out/'source_physical_diagnostics.json',physical_check)
    print('FROZEN_PHYSICS '+json.dumps(physical_check,allow_nan=False),flush=True)
    from experiments.cvs_clean_design.profile import resource_profile
    profile=resource_profile(model,device)
    profile.update(curvature_contract=physical.contract(),all_identity_paths_shared_energy_normalization=True,
        normalization_state='Per-packet complex RMS only; no mutable source/query moments',
        base_architecture=c['curvature']['identity_core'])
    write(out/'resource_profile.json',profile)
    complete=dict(status='SOURCE_TRAINED',epoch=200,steps=step,checkpoint=str(out/'last.pt'),
        elapsed_seconds=time.perf_counter()-started,target_access=False,target_evaluated=False,final_source_metrics=validation)
    if 'numerical_policy' in c:complete['backend_flags']=actual_flags()
    write(out/'completion.json',complete)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    train(json.loads(Path(a.config).read_text(encoding='utf-8')))
