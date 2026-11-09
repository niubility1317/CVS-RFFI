"""F-free 30-view source rotation with fixed simulator response preservation."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare, extract_context
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.phase1_dura_orthogonal.counterfactual_response import response_loss, DIRECTIONS
from experiments.pamf_action_restore.operators import Action
from experiments.phase1_joint_particle.train import _loss
from experiments.phase1_cumulant_context.model import CumulantContextRecovery, cumulant_loss, CUMULANT_WEIGHT, LAGS
from experiments.phase1_cumulant_context.train import EPOCHS, BATCH, WIDTH, LAYERS, MODEL_SEED, ORDER_SEED
from experiments.phase1_cumulant_diverse.train import SCHEDULE, schedule_contract
from experiments.phase1_cumulant_response10.train import scaled_perturbation, EPSILON, SCALE, RESPONSE_WEIGHT

SCHEMA = 'phase1_cumulant_diverse_response10_source_v1'


@torch.no_grad()
def prepare_changed(fresh, clean, selected, values, group, dura):
    """Cache source-only counterfactual using matched action and fixed receiver state."""
    count = len(selected)
    if len(group) != 3 or len(values[0]) != 3*count:
        raise ValueError('Three matched source views required')
    chunks = []
    for view, epoch in enumerate(group):
        folder = fresh/'epochs'/f'{epoch:02d}'
        meta = core.read(folder/'complete.json')
        if (meta.get('count') != core.COUNT or meta.get('namespace') != f'learned-action/train/epoch{epoch:02d}'
                or any(meta.get(k) != 0 for k in ('TX_labels_read','F_outputs_read','target_reads'))):
            raise ValueError('Matched legal source simulator cache required')
        received = np.load(folder/'received.npy', mmap_mode='r', allow_pickle=False)
        kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
        coeff = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
        if (received.shape != clean.shape or received.dtype != np.float32 or kernel.shape != (core.COUNT,2,64)
                or coeff.shape != (core.COUNT,4,256) or kernel.dtype != np.float32 or coeff.dtype != np.float32):
            raise ValueError('Complete matching source simulation arrays required')
        for start in range(0, count, BATCH):
            ids = selected[start:start+BATCH]
            offset = view*count + start
            x, y = (values[i][offset:offset+len(ids)] for i in (0,4))
            if not torch.equal(x,core.tensor(received[ids],'cpu')) or not torch.equal(y,core.tensor(clean[ids],'cpu')):
                raise ValueError('DURA context must match source observation and clean IQ')
            x, y = x.to('cuda:0'), y.to('cuda:0')
            action = true_action(Action,core.tensor(kernel[ids],'cuda:0'),core.tensor(coeff[ids],'cuda:0'))
            delta = scaled_perturbation(y,offset)
            effect = action(delta)
            changed = x + torch.stack((effect.real,effect.imag),1)
            anchor,direction,variance = extract_context(dura,changed)
            fields = (changed,anchor,direction,variance,torch.stack((delta.real,delta.imag),1))
            if not all(torch.isfinite(z).all() for z in fields) or (variance <= 0).any():
                raise FloatingPointError('Finite source counterfactual context required')
            chunks.append(tuple(z.detach().cpu() for z in fields))
        print('DIVERSE_RESPONSE10_PREP',epoch,count,flush=True)
    return tuple(torch.cat(z,0) for z in zip(*chunks))


def run(root,out,commit,smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    training_environments('cuda:0',os.environ)
    if out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and selected GPU required')
    schedule_contract()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pilot,_,_,_,_,_=source_inputs(root)
    cache,fresh,clean,fit=load_source_training(root,pilot,fit_size=32768)
    if smoke:
        roles=np.load(cache/'train/roles.npy',allow_pickle=False)
        selected=np.concatenate((fit[roles[fit]=='L_s'][:32],fit[roles[fit]=='U_s'][:32]))
        if len(selected)!=64:raise ValueError('Mixed legal source roles required for smoke')
    else:
        selected=fit
    dura=frozen_dura(root,'cuda:0')
    if any(p.requires_grad for p in dura.parameters()):raise ValueError('Frozen source DURA required')
    torch.manual_seed(MODEL_SEED)
    model=CumulantContextRecovery(width=WIDTH,layers=LAYERS).to('cuda:0').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    rng=np.random.default_rng(ORDER_SEED)
    cpu_groups, group_counts, losses, responses = {}, {}, [], []
    out.mkdir(parents=True)
    started=time.monotonic()
    for epoch in range(1,2 if smoke else EPOCHS+1):
        group=SCHEDULE[epoch-1]
        if group not in cpu_groups:
            values,counts=prepare(root,dura,clean,fresh,selected,'cuda:0',BATCH,
                                  architecture='large_context_stable',source_epochs=group)
            changed=prepare_changed(fresh,clean,selected,values,group,dura)
            cpu_groups[group]=(values,changed)
            group_counts[str(group)]=counts
        original,augmented=(tuple(z.to('cuda:0') for z in pair) for pair in cpu_groups[group])
        order=rng.permutation(len(original[0]))
        total=torch.zeros((),device='cuda:0');response_total=torch.zeros((),device='cuda:0')
        for start in range(0,len(order),BATCH):
            idx=torch.tensor(order[start:start+BATCH].tolist(),dtype=torch.long,device='cuda:0')
            received,anchor,direction,variance,target=(z[idx] for z in original)
            aug_received,aug_anchor,aug_direction,aug_variance,delta=(z[idx] for z in augmented)
            output=model(received,anchor,direction,variance)
            response=response_loss(output,model(aug_received,aug_anchor,aug_direction,aug_variance),delta)
            loss=_loss(output,anchor,target)+CUMULANT_WEIGHT*cumulant_loss(output,target)+RESPONSE_WEIGHT*response
            if not torch.isfinite(loss):raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if start==0 and (model.out.weight.grad is None or model.out.weight.grad.abs().sum()<=0):
                raise ValueError('Finite nonzero source gradient required')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=loss.detach()*len(idx);response_total+=response.detach()*len(idx)
        torch.cuda.synchronize()
        losses.append(float(total.cpu())/len(order));responses.append(float(response_total.cpu())/len(order))
        print('DIVERSE_RESPONSE10_SOURCE_EPOCH',epoch,'views',group,'loss',losses[-1],
              'response',responses[-1],'elapsed',round(time.monotonic()-started,1),flush=True)
        del original,augmented
    common=dict(schema=SCHEMA,commit=commit,epochs=1 if smoke else EPOCHS,
        fit_ids=len(fit),actual_fit_ids=len(selected),actual_source_view_count=3*len(selected),views_per_record=3,
        source_epochs_schedule=[list(x) for x in (SCHEDULE[:1] if smoke else SCHEDULE)],
        width=WIDTH,layers=LAYERS,model_seed=MODEL_SEED,order_seed=ORDER_SEED,lags=list(LAGS),
        cumulant_weight=CUMULANT_WEIGHT,response_weight=RESPONSE_WEIGHT,response_epsilon=EPSILON,
        response_scale=SCALE,response_directions=list(DIRECTIONS),response_source_action='matched_source_simulator',
        simulator_action_inference_reads=0,trainable_parameters=sum(p.numel() for p in model.parameters()),
        source_loss='normalized IQ + .02 DURA change + .1 bounded moment + .01 10pct true-action response',
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,DURA_frozen=True,
        source_counts_by_view_group=group_counts,elapsed_seconds=time.monotonic()-started,
        losses=losses,response_losses=responses)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as f:
        torch.save(dict(**common,model=model.eval().cpu().state_dict()),f)
    core.write(out/('smoke.json' if smoke else 'source_summary.json'),dict(
        status='SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED',**common))
    print('DIVERSE_RESPONSE10_SOURCE_DONE',commit,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--device',choices=('cuda:0',),default='cuda:0');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.root,a.out,a.commit,a.smoke)

if __name__ == '__main__':main()
