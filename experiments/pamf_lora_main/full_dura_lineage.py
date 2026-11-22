"""Train the intact WiSig DURA lineage and read it out on matched LoRa IQ."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import STAGES, construct, load, objective
from experiments.cric_dura_comparison.g_train import train
from experiments.cric_dura_comparison.io import read, tensor, write
from .data import SOURCE, TARGET, VIEWS
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f, wait_for
from .full_dura_lineage_data import LoRaView, aligned_teacher, build, in_observed_phase, source_iq
from .full_dura_r69_lora import R69_SEEDS, R69_TRACE_MODES, load_model, load_utility
from .full_dura_trace_source import TRACE_RUN
from .full_packet_data import LIMIT, PROTOCOL
from .full_restore_draw import RUN as DRAW_RUN
from .full_supervised import construct as recognizer
from .full_supervised import handles
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate


RUN='20261122-phase1-lora-dura-wisig-lineage-m1-r02'
PHASE_RUN='20261122-phase1-lora-dura-wisig-observed-phase-m1-r01'
RUNS=(RUN,PHASE_RUN)
MODES=(*R69_TRACE_MODES,'lineage_native','lineage_guarded')
SEED,=R69_SEEDS


@torch.no_grad()
def prepare_model(model,stage):
    if stage=='action':
        impulse=model.estimator.initial_kernel
        impulse.zero_()
        impulse.reshape(-1)[0]=1.


def predict_iq(g,b):
    z=g(b.flatten(0,1)).reshape_as(b)
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite native DURA output')
    return z


def smoke(rt):
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    l,_,_=source_iq(cache)
    clean=np.stack([l[0,i,i] for i in range(6)])
    examples=[aligned_teacher((x,v,SOURCE[0],f'lineage-smoke-{i}',i))[0]
              for i,(x,v) in enumerate(zip(clean,VIEWS[1:],strict=True))]
    received,kernel,coefficients,variance=(np.stack(a) for a in zip(*examples,strict=True))
    use_phase=root.name==PHASE_RUN
    if use_phase:
        from experiments.pamf_action_restore.operators import Action
        from experiments.pamf_restoration.models import complex_iq

        def simulate(ground,coef):
            a=tensor(coef)
            action=Action(complex_iq(tensor(kernel)),torch.complex(a[:,0],a[:,1]),
                          torch.complex(a[:,2],a[:,3]))
            return action(complex_iq(tensor(ground)))

        raw=simulate(clean,coefficients)
        clean,coefficients=in_observed_phase(clean,received,coefficients)
        if not torch.allclose(raw,simulate(clean,coefficients),atol=2e-6,rtol=2e-5):
            raise ValueError('Rotating the reference changed the source action')
    x=tensor(clean)
    received,kernel,coefficients,variance=(tensor(a) for a in (received,kernel,coefficients,variance))
    coarse=construct('coarse')
    action=construct('action',coarse)
    prepare_model(action,'action')
    loss=objective(action,None,'action',received,x,1,(kernel,coefficients,variance),
                   torch.Generator().manual_seed(20261122))
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Original DURA failed aligned LoRa action supervision')
    loss.mean().backward()
    gradient=torch.nn.utils.clip_grad_norm_(action.estimator.parameters(),1.,error_if_nonfinite=True)
    if gradient<=0 or any(p.grad is not None for p in action.coarse.parameters()):
        raise ValueError('Action supervision should update only the estimator')
    # Inference smoke only. The new lineage starts from the original coarse initializer.
    previous=load_model(root.parent/TRACE_RUN,c,SEED,'student','cpu')
    with torch.no_grad():
        predict_iq(previous,tensor(l[1,:2]))
    write(root/'smoke.json',dict(status='PASS',seed=SEED,scenes=list(VIEWS[1:]),
          action_gradient=float(gradient),observed_phase=use_phase,F_reads=0,TX_reads=0,target_reads=0))


def stage_train(rt):
    source,_,_=handles(rt)
    root=Path(rt['run_root'])
    out=root/'G'/'lineage'
    use_phase=root.name==PHASE_RUN
    train(dict(output_root=str(out),model_seed=SEED,code_commit=rt['code_commit'],
          cache=dict(source_contract=str(source/'contract.json'),roles='LU'),observed_phase=use_phase),
          view=LoRaView(rt,observed_phase=use_phase),prepare_model=prepare_model)


def frozen_g(root,c,device):
    p=root/'G'/'lineage'
    receipt=read(p/'complete.json')
    if (receipt['status']!='SOURCE_FROZEN' or receipt['stages']!=list(STAGES)
            or receipt['updates']!=39520 or receipt['allowed_roles']!='LU'
            or any(receipt[k] for k in ('F_reads','TX_labels_read','target_reads'))):
        raise ValueError('DURA full source lineage is incomplete')
    model,item=load(p/'r69'/'final.pt',c,'LU',device)
    if item['model_seed']!=SEED or item['ancestors']!=[str(p/s/'final.pt') for s in STAGES[:-1]]:
        raise ValueError('Unexpected DURA lineage ancestry')
    return model.eval().requires_grad_(False)


@torch.no_grad()
def source_audit(rt):
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    device='cuda:0'
    g=frozen_g(root,c,device)
    a=np.load(cache/'V.npy',allow_pickle=False,mmap_mode='r')
    records={}
    for i,view in enumerate(VIEWS):
        total=0
        error=torch.zeros(2,device=device)
        for start_at in range(0,600,32):
            ids=np.arange(start_at,min(600,start_at+32))*20
            clean=tensor(a[0,ids],device)
            received=tensor(a[i,ids],device)
            output=predict_iq(g,received)
            for j,z in enumerate((received,output)):
                error[j]+=(((z-clean).square().mean((1,2,3)))/
                            clean.square().mean((1,2,3)).clamp_min(1e-8)).sum()
            total+=len(ids)
        records[view]=dict(n=total,mean_nmse=dict(zip(('direct','lineage_native'),
                                                   (error/total).tolist(),strict=True)))
    write(root/'source_wave'/f'{SEED}.json',dict(seed=SEED,records=records,
                                               F_reads=0,TX_reads=0,target_reads=0))


@torch.no_grad()
def predict(rt):
    _,_,c=handles(rt)
    root=Path(rt['run_root'])
    target=Path(rt['target_views'])
    if (target!=Path(rt['parent_root'])/'target_views'
            or read(target/'complete.json')!=dict(status='TARGET_READY',protocol=PROTOCOL,
                   target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS))):
        raise ValueError('LoRa blind target view is incomplete')
    reference=root.parent/TRACE_RUN/'predictions'/str(SEED)
    prior=read(reference/'complete.json')
    activation=prior.pop('activation')
    expected=dict(status='PREDICTIONS_FROZEN',seed=SEED,target_truth_read=False,
                  receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS))
    if prior!=dict(expected,modes=list(R69_TRACE_MODES)):
        raise ValueError('Frozen LoRa comparison is incomplete')
    device='cuda:0'
    g=frozen_g(root,c,device)
    gate=load_utility(root.parent/TRACE_RUN,c,SEED,device)
    f=recognizer(device).eval().requires_grad_(False)
    checkpoint=complete_f(root.parent/SET_RUN,c,case(SEED,SET_METHOD))/'model.pt'
    f.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    out=root/'predictions'/str(SEED)
    out.mkdir(parents=True,exist_ok=False)
    for rx in TARGET:
        iq=np.load(target/f'{rx}.npy',allow_pickle=False,mmap_mode='r')
        if iq.shape!=(len(VIEWS),LIMIT,6,2,256) or iq.dtype!=np.float32:
            raise ValueError('LoRa target receiver IQ differs')
        for j,v in enumerate(VIEWS):
            controls=[np.load(reference/f'{rx}__{v}__{mode}.npy',allow_pickle=False)
                      for mode in R69_TRACE_MODES]
            if any(a.shape!=(LIMIT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=10
                   for a in controls):
                raise ValueError('Invalid frozen LoRa DURA baseline')
            native=controls[0].copy()
            guarded=controls[0].copy()
            accepted=0
            for offset in range(0,LIMIT,48):
                b=tensor(iq[j,offset:offset+48],device)
                use=gate.impairment(b).gt(0).cpu().numpy()
                logits=f(predict_iq(g,b))[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite original DURA plus frozen Tri logits')
                labels=logits.argmax(1).cpu().numpy()
                native[offset:offset+len(b)]=labels
                guarded[offset+np.flatnonzero(use)]=labels[use]
                accepted+=int(use.sum())
            book=activation[f'{rx}/{v}']
            if accepted!=book['trace_active']:
                raise ValueError('Fixed source gate activation differs from its LoRa control')
            for mode,array in zip(MODES,(*controls,native,guarded),strict=True):
                np.save(out/f'{rx}__{v}__{mode}.npy',array)
            book['lineage_active']=LIMIT
        print('LORA_DURA_LINEAGE_PREDICTED',SEED,rx,flush=True)
    write(out/'complete.json',dict(expected,modes=list(MODES),activation=activation))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name,extra_keys=('code_commit',))
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        wait_for(root.parent/DRAW_RUN,'SOURCE_DRAW_READY')
        wait_for(root.parent/TRACE_RUN,'PREDICTIONS_FROZEN')
        if launch(rt,'smoke','source').wait():
            raise RuntimeError('LoRa DURA lineage source smoke failed')
        if root.name==PHASE_RUN:
            bank=root.parent/RUN/'source_action'
            if read(bank/'complete.json')['status']!='SOURCE_ACTION_READY':
                raise ValueError('Both LoRa curricula require the same completed action bank')
            (root/'source_action').symlink_to(bank,target_is_directory=True)
        else:
            write(state,dict(status='BUILDING_SOURCE_ACTION'),exclusive=False)
            if launch(rt,'build','source').wait():
                raise RuntimeError('LoRa source action generation failed')
        write(state,dict(status='SOURCE_DURA_LINEAGE'),exclusive=False)
        schedule(rt,[('train',str(SEED))],launch)
        write(state,dict(status='SOURCE_AUDIT'),exclusive=False)
        schedule(rt,[('audit',str(SEED))],launch)
        wait_for(root.parent/SET_RUN,'PREDICTIONS_FROZEN')
        write(state,dict(status='PREDICTING'),exclusive=False)
        schedule(rt,[('predict',str(SEED))],launch)
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=('dispatch','smoke','build','train','audit','predict'))
    parser.add_argument('runtime',type=Path)
    parser.add_argument('name',nargs='?')
    args=parser.parse_args()
    runtime=read(args.runtime)
    if args.kind!='dispatch':
        activate(runtime)
    dict(dispatch=dispatch,smoke=smoke,build=build,train=stage_train,audit=source_audit,predict=predict)[args.kind](runtime)
