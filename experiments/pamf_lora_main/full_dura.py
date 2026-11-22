"""Full LoRa DURA comparison, anchored to the same three frozen adaptive Tri F."""
import argparse
from functools import partial
from pathlib import Path
from shutil import copy2, copytree

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .dura_gate import RiskDura
from .dura_repair import complex_iq, power
from .full_adaptive import RUN as TRI, SET_METHOD, SET_RUN, WARMUP_RUN, case, complete_f, wait_for
from .full_dura_core import (COVERAGE_RECIPES, Dose, dose_features, g_record, projected_record, protected_shapes,
                            read_gate, receiver_loss, train, train_gate)
from .full_dura_detail import (CONFIRM_NEW_SEEDS, DETAIL_CONFIRM_RUN, DETAIL_DOSE_MODES,
                               DETAIL_DOSE_RUN, DETAIL_MODES, DETAIL_RUN, DETAIL_SEEDS)
from .full_dura_r69_lora import (R69_ACTION_MODES, R69_ACTION_RUN, R69_LORA_RUN, R69_MODES,
                                 R69_SEEDS, R69_TRACE_MODES)
from .full_dura_trace_source import TRACE_RUN
from .full_native_joint import JOINT_MODES, JOINT_RUN, JOINT_SEEDS
from .full_packet_data import LIMIT, PROTOCOL
from .full_supervised import SEEDS, construct, handles
from .packet_runner import schedule
from .refine_g import aligned_target
from .refine_runner import start
from .runner import activate
from .full_restore_draw import RUN as DRAW_RUN
from .full_tri_repl_lora import REPL_RUN

RUN='20261122-phase1-lora-full-dura-dose-m3-r01'
MODES=('direct','fixed','dose')
SWEEP_RUN='20261122-phase1-lora-full-dura-range-m6-r01'
SWEEP={'direct':0.,'fixed':1.,'dose_150':1.5,'dose_200':2.}
SHAPE_RUN='20261122-phase1-lora-dura-preserve-shape-m9-r01'
SHAPE_MODES=('direct','fixed','protect_shape','individual_only')
PROJECTED_RUN='20261122-phase1-lora-dura-train-protected-m3-r01'
PROJECTED_MODES=('direct','fixed','protect_shape','trained_protect')
COVERAGE_RUN='20261122-phase1-lora-dura-source-coverage-m6-r01'
COVERAGE_MODES=(*PROJECTED_MODES,*COVERAGE_RECIPES)
STACK_RUN='20261122-phase1-lora-tri-set-dura-stack-m9-r01'
STAGED_RUN='20261122-phase1-lora-dura-source-distillation-m6-r01'
STAGED_MODES=(*PROJECTED_MODES,'staged_protect')
DETAIL_RUNS=(DETAIL_RUN,DETAIL_DOSE_RUN,DETAIL_CONFIRM_RUN)
R69_RUNS=(R69_LORA_RUN,R69_ACTION_RUN,TRACE_RUN)


def modes_for(root):
    if root.name==JOINT_RUN:
        return JOINT_MODES
    if root.name==R69_ACTION_RUN:
        return R69_ACTION_MODES
    if root.name==TRACE_RUN:
        return R69_TRACE_MODES
    if root.name==R69_LORA_RUN:
        return R69_MODES
    if root.name==DETAIL_RUN:
        return DETAIL_MODES
    if root.name in (DETAIL_DOSE_RUN,DETAIL_CONFIRM_RUN):
        return DETAIL_DOSE_MODES
    if root.name==STAGED_RUN:
        return STAGED_MODES
    if root.name==COVERAGE_RUN:
        return COVERAGE_MODES
    if root.name in (PROJECTED_RUN,STACK_RUN):
        return PROJECTED_MODES
    if root.name==SHAPE_RUN:
        return SHAPE_MODES
    return tuple(SWEEP) if root.name==SWEEP_RUN else MODES


def recognizer_run(root):
    if root.name in (STACK_RUN,STAGED_RUN,JOINT_RUN,*R69_RUNS,*DETAIL_RUNS):
        return SET_RUN
    return WARMUP_RUN if root.name in (SHAPE_RUN,PROJECTED_RUN,COVERAGE_RUN) else TRI


def engines(root,c,seed,dev):
    path=root/'G'/str(seed)
    if read(path/'complete.json')!=g_record(c,seed):
        raise ValueError('DURA is not frozen on the exact full source contract')
    gate=read_gate(root/'G'/'gate',c,dev)
    net=RiskDura(gate,False).to(dev).eval()
    dose=Dose().to(dev).eval()
    weights=torch.load(path/'model.pt',map_location=dev,weights_only=True)
    net.net.load_state_dict(weights['G'],strict=True)
    dose.load_state_dict(weights['dose'],strict=True)
    return net,dose


def scaled(x,z,strength):
    # The registered sweep never exceeds twice DURA's 25% RMS bound.
    return x+strength*(z-x)


def projected_engine(root,c,seed,gate,dev,recipe='protected_shape'):
    path=root/'G'/str(seed) if recipe=='protected_shape' else root/'G'/recipe/str(seed)
    if read(path/'complete.json')!=projected_record(c,seed,recipe):
        raise ValueError('Missing source-only projected DURA weights')
    g=RiskDura(gate,False).to(dev).eval()
    g.net.load_state_dict(torch.load(path/'model.pt',map_location=dev,weights_only=True),strict=True)
    return g


def projected_output(g,x,z):
    return protected_shapes(g,x,z)[0]


def learned_engines(root,c,seed,gate,dev):
    if root.name not in (PROJECTED_RUN,COVERAGE_RUN,STACK_RUN,STAGED_RUN):
        return []
    prior=root if root.name==PROJECTED_RUN else root.parent/PROJECTED_RUN
    nets=[projected_engine(prior,c,seed,gate,dev)]
    if root.name==COVERAGE_RUN:
        nets.extend(projected_engine(root,c,seed,gate,dev,recipe) for recipe in COVERAGE_RECIPES)
    if root.name==STAGED_RUN:
        from .full_dura_stage import load_stage
        nets.append(load_stage(root,c,seed,gate,dev,'student'))
    return nets


@torch.no_grad()
def shape_smoke(rt):
    _,views,c=handles(rt)
    root=Path(rt['run_root'])
    x=tensor(np.load(views/'L_s.npy',mmap_mode='r',allow_pickle=False)[0,:12])
    b=tensor(np.load(views/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)[::80])
    records={}
    for seed in SEEDS:
        g,_=engines(root.parent/RUN,c,seed,'cpu')
        clips,mask=g(x,with_gate=True)
        if mask.any() or not torch.equal(clips,x):
            raise ValueError('Real source clean smoke did not bypass G')
        active=b[g.strength(b)>0][:8]
        if not len(active):
            raise ValueError('Real source shape smoke has no active practical signal')
        fixed=g(active)
        cap=power(complex_iq(fixed-active))
        for proposed in protected_shapes(g,active,fixed):
            if (not torch.isfinite(proposed).all() or torch.equal(proposed,active)
                    or (power(complex_iq(proposed-active))>cap+1e-7).any()):
                raise ValueError('Real source shape protection violated the correction budget')
        records[str(seed)]=len(active)
    write(root/'smoke.json',dict(status='PASS',active=records,clean_exact=len(x),TX_reads=0,F_reads=0,target_reads=0))


def smoke(rt):
    _,views,c=handles(rt)
    net=RiskDura(read_gate(Path(rt['run_root'])/'G'/'gate',c,'cpu'),False)
    x=tensor(np.load(views/'L_s.npy',mmap_mode='r',allow_pickle=False)[0,:12])
    y,g=net(x,with_gate=True)
    if not torch.equal(x,y) or g.any():
        raise ValueError('Full DURA clean bypass smoke failed')
    sample=np.load(views/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)
    b=tensor(sample[::80])
    elig=net.strength(b)>0
    if not elig.any():
        raise ValueError('DURA real source smoke has no active practical signal')
    b=b[elig][:3]
    r=net(b)
    loss=(r-b).square().mean()
    loss.backward()
    grad=float(torch.nn.utils.clip_grad_norm_(net.net.parameters(),5.,error_if_nonfinite=True))
    if not torch.isfinite(r).all() or not grad>0:
        raise ValueError('DURA source-only backwards failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',source_IQ_only=True,gradient=grad,
          clean_exact=len(x),TX_reads=0,target_reads=0,F_reads=0))


@torch.no_grad()
def source_wave(rt,seed):
    if Path(rt['run_root']).name==JOINT_RUN:
        root=Path(rt['run_root'])
        name=f'{seed}.json'
        return write(root/'source_wave'/name,read(root.parent/R69_LORA_RUN/'source_wave'/name))
    if Path(rt['run_root']).name in R69_RUNS:
        from .full_dura_r69_lora import r69_audit
        return r69_audit(rt,seed)
    if Path(rt['run_root']).name in DETAIL_RUNS:
        from .full_dura_detail import source_wave_detail
        return source_wave_detail(rt,seed)
    _,views,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    sweep=root.name==SWEEP_RUN
    shape=root.name in (SHAPE_RUN,PROJECTED_RUN,COVERAGE_RUN,STACK_RUN,STAGED_RUN)
    m,d=engines(root.parent/RUN if sweep or shape else root,c,seed,dev)
    learned=learned_engines(root,c,seed,m.gate,dev)
    arr=np.load(views/'V.npy',mmap_mode='r',allow_pickle=False)
    n=1000
    y=tensor(arr[0,::12],dev)
    rows={}
    for i,v in enumerate(VIEWS):
        x=tensor(arr[i,::12],dev)
        sums=torch.zeros(len(modes_for(root)),device=dev)
        total=active=0
        for clean,b in zip(y.split(48),x.split(48),strict=True):
            p,g=m(b,with_gate=True)
            keep=g>0
            if not torch.equal(p[~keep],b[~keep]):
                raise ValueError('Clean DURA bypass is not an exact copy')
            if shape:
                restored=[p.clone() for _ in range(len(modes_for(root))-2)]
                if keep.any():
                    x_active=b[keep]
                    parts=protected_shapes(m,x_active,p[keep])
                    if learned:
                        parts=[parts[0],*(projected_output(g,x_active,g(x_active)) for g in learned)]
                    for result,part in zip(restored,parts,strict=True):
                        result[keep]=part
                wave=(b,p,*restored)
            elif sweep:
                wave=(b,p,scaled(b,p,1.5),scaled(b,p,2.))
            else:
                recovered=p.clone()
                if keep.any():
                    a=d(dose_features(m,b[keep],p[keep]))
                    recovered[keep]=scaled(b[keep],p[keep],a[:,None,None,None])
                wave=(b,p,recovered)
            ref=complex_iq(aligned_target(clean,b,m.reference,'chirp'))
            sums+=torch.stack([receiver_loss(complex_iq(w)-ref).sum() for w in wave])
            active+=int(keep.sum())
            total+=len(b)
        if total!=n:
            raise ValueError('Source waveform audit count differs')
        rows[v]=dict(n=total,active=active,aligned_error=(sums/total).tolist())
    write(root/'source_wave'/f'{seed}.json',dict(seed=seed,records=rows,target_reads=0,TX_reads=0,F_reads=0))


@torch.no_grad()
def predict(rt,seed):
    _,_,c=handles(rt)
    root=Path(rt['run_root'])
    pred=root.parent/recognizer_run(root)
    target=Path(rt['target_views'])
    if (read(pred/'state.json')['status']!='PREDICTIONS_FROZEN'
            or target!=Path(rt['parent_root'])/'target_views'
            or read(target/'complete.json')!=dict(status='TARGET_READY',protocol=PROTOCOL,target_truth_read=False,
                              receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS))):
        raise ValueError('Blind full pair is not ready')
    if root.name==JOINT_RUN:
        from .full_native_joint import predict_joint
        return predict_joint(root,target,c,seed,'cuda:0')
    method=SET_METHOD if pred.name==SET_RUN else 'adaptive_tri'
    name=case(seed,method)
    base=pred/'predictions'/name
    if read(base/'complete.json')!=dict(status='PREDICTIONS_FROZEN',seed=seed,method=method,
           target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS),modes=['direct']):
        raise ValueError('Direct Tri control is not fixed')
    dev='cuda:0'
    f=construct(dev).eval()
    f.load_state_dict(torch.load(complete_f(pred,c,name)/'model.pt',map_location=dev,weights_only=True),strict=True)
    if root.name in R69_RUNS:
        from .full_dura_r69_lora import predict_r69
        predict_r69(root,target,c,seed,f,dev)
        return
    if root.name in DETAIL_RUNS:
        from .full_dura_detail import predict_detail
        predict_detail(root,target,c,seed,f,dev)
        return
    if root.name in (COVERAGE_RUN,STAGED_RUN):
        predict_added(root,target,c,seed,f,dev)
        return
    sweep=root.name==SWEEP_RUN
    shape=root.name in (SHAPE_RUN,PROJECTED_RUN,STACK_RUN,STAGED_RUN)
    modes=modes_for(root)
    m,d=engines(root.parent/RUN if sweep or shape else root,c,seed,dev)
    learned=learned_engines(root,c,seed,m.gate,dev)
    out=root/'predictions'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    total={}
    for rx in TARGET:
        a=np.load(target/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        if a.shape!=(7,LIMIT,6,2,256) or a.dtype!=np.float32:
            raise ValueError('DURA target IQ differs')
        for j,v in enumerate(VIEWS):
            raw=np.load(base/f'{rx}__{v}__direct.npy',allow_pickle=False)
            if raw.shape!=(LIMIT,) or raw.dtype.kind not in 'iu' or raw.min()<0 or raw.max()>=10:
                raise ValueError('Frozen raw predictions are incomplete')
            altered=[raw.copy() for _ in modes[1:]]
            start_at=active=0
            gain=0.
            for x in tensor(a[j],dev).split(32):
                z,g=m(x,with_gate=True)
                use=g>0
                if not torch.equal(z[~use],x[~use]):
                    raise ValueError('Fixed bypass changed original target IQ')
                if use.any():
                    xx,zz=x[use],z[use]
                    if shape:
                        inputs=[zz,*protected_shapes(m,xx,zz)]
                        if learned:
                            inputs=[*inputs[:2],*(projected_output(g,xx,g(xx)) for g in learned)]
                    elif sweep:
                        inputs=[zz,scaled(xx,zz,1.5),scaled(xx,zz,2.)]
                    else:
                        dose=d(dose_features(m,xx,zz))
                        if not torch.isfinite(dose).all():
                            raise ValueError('Nonfinite full DURA dosage')
                        inputs=[zz,scaled(xx,zz,dose[:,None,None,None])]
                        gain+=float(dose.sum())
                    pp=f(torch.cat(inputs))[0]
                    if not torch.isfinite(pp).all():
                        raise ValueError('Nonfinite full DURA prediction')
                    k=torch.where(use)[0].cpu().tolist()
                    k=np.asarray(k)+start_at
                    for result,labels in zip(altered,pp.argmax(1).chunk(len(altered)),strict=True):
                        result[k]=labels.tolist()
                    active+=len(k)
                start_at+=len(x)
            for mode,z in zip(modes,(raw,*altered),strict=True):
                np.save(out/f'{rx}__{v}__{mode}.npy',z)
            total[f'{rx}/{v}']=dict(n=LIMIT,IQ_exact=LIMIT-active,active=active,
                         mean_dose=(float(active>0) if sweep or shape else gain/max(active,1)))
        print('FULL_DURA_PREDICTED',seed,rx,flush=True)
    write(out/'complete.json',dict(status='PREDICTIONS_FROZEN',seed=seed,modes=list(modes),target_truth_read=False,
          receivers=list(TARGET),views=list(VIEWS),per_receiver=LIMIT,activation=total))


def predict_added(root,target,c,seed,reader,dev):
    coverage=root.name==COVERAGE_RUN
    parent=root.parent/(PROJECTED_RUN if coverage else STACK_RUN)
    if read(parent/'state.json')['status']!='PREDICTIONS_FROZEN':
        raise ValueError('Matched protected DURA controls must be frozen')
    base=parent/'predictions'/str(seed)
    proof=read(base/'complete.json')
    activation=proof.pop('activation')
    expected=dict(status='PREDICTIONS_FROZEN',seed=seed,modes=list(PROJECTED_MODES),target_truth_read=False,
                  receivers=list(TARGET),views=list(VIEWS),per_receiver=LIMIT)
    if proof!=expected or set(activation)!={f'{rx}/{v}' for rx in TARGET for v in VIEWS}:
        raise ValueError('Mismatch in the frozen protected DURA control matrix')
    base_model,_=engines(root.parent/RUN,c,seed,dev)
    if coverage:
        learners=[projected_engine(root,c,seed,base_model.gate,dev,recipe) for recipe in COVERAGE_RECIPES]
    else:
        from .full_dura_stage import load_stage
        learners=[load_stage(root,c,seed,base_model.gate,dev,'student')]
    modes=modes_for(root)
    out=root/'predictions'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    for rx in TARGET:
        a=np.load(target/f'{rx}.npy',allow_pickle=False,mmap_mode='r')
        if a.shape!=(7,LIMIT,6,2,256) or a.dtype!=np.float32:
            raise ValueError('DURA target IQ differs')
        for j,v in enumerate(VIEWS):
            existing=[np.load(base/f'{rx}__{v}__{mode}.npy',allow_pickle=False) for mode in PROJECTED_MODES]
            if any(x.shape!=(LIMIT,) or x.dtype.kind not in 'iu' or x.min()<0 or x.max()>=10 for x in existing):
                raise ValueError('Reused protected DURA predictions differ')
            results=[existing[0].copy() for _ in learners]
            active=0
            for off in range(0,LIMIT,32):
                x=tensor(a[j,off:off+32],dev)
                strength=base_model.strength(x)
                keep=strength>0
                if keep.any():
                    x=x[keep]
                    wave=[projected_output(g,x,g.bounded(x,g.proposal(x),strength[keep])) for g in learners]
                    logits=reader(torch.cat(wave))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite expanded coverage prediction')
                    ids=np.asarray(torch.where(keep)[0].tolist())+off
                    for result,labels in zip(results,logits.argmax(1).chunk(len(learners)),strict=True):
                        result[ids]=labels.tolist()
                    active+=len(ids)
            if activation[f'{rx}/{v}']['active']!=active:
                raise ValueError('Training coverage changed the deployed DURA gate')
            for mode,values in zip(modes,(*existing,*results),strict=True):
                np.save(out/f'{rx}__{v}__{mode}.npy',values)
        print('FULL_DURA_ADDED_PREDICTED',seed,rx,flush=True)
    write(out/'complete.json',dict(expected,modes=list(modes),activation=activation))


def copy_detail_control(root):
    """Retain the existing seed-02 dose outcomes without opening target data."""
    prior=root.parent/DETAIL_DOSE_RUN
    seed,=DETAIL_SEEDS
    source=prior/'predictions'/str(seed)
    receipt=read(source/'complete.json')
    if (receipt['status']!='PREDICTIONS_FROZEN' or receipt['seed']!=seed
            or receipt['target_truth_read'] or receipt['modes']!=list(DETAIL_DOSE_MODES)
            or receipt['receivers']!=list(TARGET) or receipt['views']!=list(VIEWS)
            or receipt['per_receiver']!=LIMIT):
        raise ValueError('Prior LoRa dose experiment differs from the fixed seed-02 matrix')
    copytree(source,root/'predictions'/str(seed))
    copy2(prior/'source_wave'/f'{seed}.json',root/'source_wave'/f'{seed}.json')


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name)
    state=root/'state.json'
    if root.name==DETAIL_CONFIRM_RUN:
        seeds=CONFIRM_NEW_SEEDS
    elif root.name in R69_RUNS:
        seeds=R69_SEEDS
    elif root.name==JOINT_RUN:
        seeds=JOINT_SEEDS
    else:
        seeds=DETAIL_SEEDS if root.name in DETAIL_RUNS else SEEDS
    try:
        if root.name==JOINT_RUN:
            write(state,dict(status='WAITING_FOR_NATIVE_F_AND_G'))
            for prior in (R69_LORA_RUN,REPL_RUN):
                wait_for(root.parent/prior,'PREDICTIONS_FROZEN')
            write(state,dict(status='SOURCE_SMOKE'),exclusive=False)
            if launch(rt,'native_joint_smoke','source').wait():
                raise RuntimeError('Frozen LoRa native joint source smoke failed')
        elif root.name in R69_RUNS:
            write(state,dict(status='SOURCE_SMOKE'))
            wait_for(root.parent/DRAW_RUN,'SOURCE_DRAW_READY')
            if root.name in (R69_ACTION_RUN,TRACE_RUN):
                wait_for(root.parent/R69_LORA_RUN,'PREDICTIONS_FROZEN')
            if root.name==TRACE_RUN:
                write(state,dict(status='BUILDING_SOURCE_PHYSICAL_TRACES'),exclusive=False)
                if launch(rt,'r69_trace_source','source').wait():
                    raise RuntimeError('LoRa physical trace source generation failed')
                write(state,dict(status='SOURCE_SMOKE'),exclusive=False)
            if launch(rt,'r69_smoke','source').wait():
                raise RuntimeError('Real-IQ original DURA smoke failed')
            for stage in ('teacher','student','gate'):
                write(state,dict(status=f'SOURCE_R69_{stage.upper()}'),exclusive=False)
                schedule(rt,[(f'r69_{stage}',str(s)) for s in seeds],launch)
            wait_for(root.parent/DETAIL_CONFIRM_RUN,'PREDICTIONS_FROZEN')
        elif root.name in (SHAPE_RUN,PROJECTED_RUN,COVERAGE_RUN,STACK_RUN,STAGED_RUN,*DETAIL_RUNS):
            write(state,dict(status='SOURCE_SMOKE'))
            if root.name in (DETAIL_RUN,DETAIL_CONFIRM_RUN):
                wait_for(root.parent/DRAW_RUN,'SOURCE_DRAW_READY')
            if root.name==DETAIL_DOSE_RUN:
                wait_for(root.parent/DETAIL_RUN,'PREDICTIONS_FROZEN')
            if root.name==DETAIL_CONFIRM_RUN:
                wait_for(root.parent/DETAIL_DOSE_RUN,'PREDICTIONS_FROZEN')
            check='detail_smoke' if root.name in DETAIL_RUNS else 'shape_smoke'
            if launch(rt,check,'source').wait():
                raise RuntimeError('Frozen LoRa restoration source smoke failed')
            if root.name in (PROJECTED_RUN,COVERAGE_RUN):
                write(state,dict(status='SOURCE_G_TRAINING'),exclusive=False)
                if root.name==PROJECTED_RUN:
                    schedule(rt,[('train_projected',str(s)) for s in SEEDS],launch)
                else:
                    schedule(rt,[('train_coverage',f'{mode}-{s}') for s in SEEDS for mode in COVERAGE_RECIPES],launch)
            if root.name==STAGED_RUN:
                write(state,dict(status='WAITING_FOR_SOURCE_DRAW'),exclusive=False)
                wait_for(root.parent/DRAW_RUN,'SOURCE_DRAW_READY')
                write(state,dict(status='SOURCE_G_TEACHER_TRAINING'),exclusive=False)
                schedule(rt,[('stage_teacher',str(s)) for s in SEEDS],launch)
                write(state,dict(status='SOURCE_G_STUDENT_TRAINING'),exclusive=False)
                schedule(rt,[('stage_student',str(s)) for s in SEEDS],launch)
            if root.name in (DETAIL_RUN,DETAIL_CONFIRM_RUN):
                write(state,dict(status='SOURCE_DETAIL_TRAINING'),exclusive=False)
                schedule(rt,[('train_detail',str(s)) for s in seeds],launch)
        elif root.name==SWEEP_RUN:
            # The source G and raw F are both unchanged in this preregistered sensitivity check.
            write(state,dict(status='SOURCE_AUDIT'))
        else:
            write(state,dict(status='SOURCE_GATE'))
            schedule(rt,[('train_gate','source')],launch)
            if launch(rt,'smoke','source').wait():
                raise RuntimeError('DURA source-only smoke failed')
            write(state,dict(status='SOURCE_G_TRAINING'),exclusive=False)
            schedule(rt,[('train',str(s)) for s in SEEDS],launch)
        if root.name in (COVERAGE_RUN,STACK_RUN,STAGED_RUN):
            write(state,dict(status='WAITING_FOR_FROZEN_CONTROLS'),exclusive=False)
            wait_for(root.parent/PROJECTED_RUN,'PREDICTIONS_FROZEN')
        write(state,dict(status='SOURCE_AUDIT'),exclusive=False)
        schedule(rt,[('wave_audit',str(s)) for s in seeds],launch)
        write(state,dict(status='WAITING_FOR_FROZEN_TRI'),exclusive=False)
        wait_for(root.parent/recognizer_run(root),'PREDICTIONS_FROZEN')
        if root.name in (STAGED_RUN,*DETAIL_RUNS):
            wait_for(root.parent/STACK_RUN,'PREDICTIONS_FROZEN')
        write(state,dict(status='PREDICTING'),exclusive=False)
        if root.name==DETAIL_CONFIRM_RUN:
            copy_detail_control(root)
        schedule(rt,[('predict',str(s)) for s in seeds],launch)
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('dispatch','train_gate','smoke','shape_smoke','train','train_projected',
                                   'train_coverage','stage_teacher','stage_student','wave_audit','predict',
                                   'detail_smoke','train_detail','r69_smoke','r69_teacher','r69_student','r69_gate',
                                   'native_joint_smoke','r69_trace_source'))
    p.add_argument('runtime',type=Path)
    p.add_argument('name',nargs='?')
    a=p.parse_args()
    rt=read(a.runtime)
    root=Path(rt['run_root'])
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if a.kind=='train_gate':
            train_gate(rt,root/'G'/'gate')
        elif a.kind=='smoke':
            smoke(rt)
        elif a.kind=='shape_smoke':
            shape_smoke(rt)
        elif a.kind=='train':
            train(rt,root/'G'/a.name,root/'G'/'gate',int(a.name))
        elif a.kind=='train_projected':
            train(rt,root/'G'/a.name,root.parent/RUN/'G'/'gate',int(a.name),projected=True)
        elif a.kind=='train_coverage':
            mode,seed=a.name.split('-')
            train(rt,root/'G'/mode/seed,root.parent/RUN/'G'/'gate',int(seed),projected=True,coverage=mode)
        elif a.kind in ('stage_teacher','stage_student'):
            from .full_dura_stage import train_stage
            train_stage(rt,int(a.name),a.kind.removeprefix('stage_'))
        elif a.kind=='detail_smoke':
            from .full_dura_detail import detail_smoke
            detail_smoke(rt)
        elif a.kind=='train_detail':
            from .full_dura_detail import train_detail
            train_detail(rt,int(a.name))
        elif a.kind=='r69_smoke':
            from .full_dura_r69_lora import r69_smoke
            r69_smoke(rt)
        elif a.kind in ('r69_teacher','r69_student'):
            from .full_dura_r69_lora import train_r69
            train_r69(rt,int(a.name),a.kind.removeprefix('r69_'))
        elif a.kind=='r69_gate':
            from .full_dura_r69_lora import utility
            utility(rt,int(a.name))
        elif a.kind=='native_joint_smoke':
            from .full_native_joint import smoke as joint_smoke
            joint_smoke(rt)
        elif a.kind=='r69_trace_source':
            from .full_dura_trace_source import build
            build(rt)
        else:
            {'wave_audit':source_wave,'predict':predict}[a.kind](rt,int(a.name))
