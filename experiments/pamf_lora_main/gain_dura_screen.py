"""Single-receiver, two-view source validation of the LoRa and WiSig fusion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, perf_counter, sleep

import numpy as np
import torch
from torch import nn

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.gain_dura_fusion import fuse, companion_only
from experiments.pamf_lora_main.w24_native_dura import capture_scale, iq_error
from experiments.pamf_lora_main.w24_native_spectral_innovation import SpectralInnovation
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_restoration.models import complex_iq, pair


RUN = '20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r01'
SCORE_RUN = RUN.removesuffix('r01') + 'r02'
FACTOR_RUN = '20261128-phase1-lora-wisig-gain-dura-factor-s2026113051-r01'
SEED, COUNT, STEPS, BATCH = 2026113047, 192, 8000, 24
FACTOR_SEED, FACTOR_COUNT = 2026113051, 160
DEV = 'cuda:0'
LO_RA = '20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
LO_RA_COMMIT = '38c7b18d779e4ee5815a2ef3b49ef2c8d23297bf'
ARCHIVE = '20261127-phase1-lora-wisig-packet-dura-s2026113021-r03'
WISIG_REF = '20261125-phase1-wisig-native-fir-source-m4-r01'
MODES = ('direct', 'dura', 'real_gain', 'fusion')
FACTOR_MODES = (*MODES,'data_only','companion_only')
SCREEN_VIEWS = ('clean', 'practical_low_urban')
RX = {'lora': 0, 'wisig': 1}


def screen_rows(ids, receivers, receiver, *, excluded=(), limit=COUNT, seed=SEED):
    if len(ids) != len(receivers) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Expected one blind receiver record per physical ID')
    rows = np.setdiff1d(np.flatnonzero(receivers == receiver), excluded)
    if len(rows) < limit:
        raise ValueError('Insufficient fixed receiver coverage')
    return np.sort(np.random.default_rng(seed).choice(rows,limit,replace=False))


def paired_panel(change, draws):
    return {view: dict(repairs=int((diff>0).sum()),harms=int((diff<0).sum()),
               net=int(diff.sum()),ci95=np.quantile(diff[draws].sum(1),(.025,.975)).tolist())
            for view,diff in zip(SCREEN_VIEWS,change,strict=True)}


def score_panel(predictions, labels, *, count=COUNT, seed=SEED):
    if predictions.shape != (len(SCREEN_VIEWS), len(MODES), count) or labels.shape != (count,):
        raise ValueError('Incomplete same-packet screen')
    hit = predictions == labels
    correct = hit.sum(-1)
    idx = np.random.default_rng(seed+2).integers(0,count,size=(4000,count))
    candidate = hit[:,-1].astype(np.int8)
    paired = {name:paired_panel(candidate-hit[:,i],idx) for i,name in enumerate(MODES[:-1])}
    clean_pass = bool(correct[0,-1] >= correct[0,:-1].max())
    difficult_pass = all(out[SCREEN_VIEWS[-1]]['net'] >= 8 and out[SCREEN_VIEWS[-1]]['ci95'][0] > 0
                         for out in paired.values())
    return dict(correct={v:dict(zip(MODES,c.tolist(),strict=True))
                         for v,c in zip(SCREEN_VIEWS,correct,strict=True)},
                paired=paired,clean_pass=clean_pass,low_urban_pass=difficult_pass,
                promote=clean_pass and difficult_pass)


def score_factor_panel(predictions, labels):
    if predictions.shape != (2,len(FACTOR_MODES),FACTOR_COUNT) or labels.shape!=(FACTOR_COUNT,):
        raise ValueError('Incomplete independent factor screening matrix')
    candidates = {mode:score_panel(predictions[:,[0,1,2,i]],labels,count=FACTOR_COUNT,seed=FACTOR_SEED)
                  for i,mode in enumerate(FACTOR_MODES) if i>=3}
    hit = (predictions==labels).astype(np.int8)
    correct = {v:dict(zip(FACTOR_MODES,c.tolist(),strict=True))
               for v,c in zip(SCREEN_VIEWS,hit.sum(-1),strict=True)}
    draws = np.random.default_rng(FACTOR_SEED+2).integers(0,FACTOR_COUNT,size=(4000,FACTOR_COUNT))
    pairs = {(f'{FACTOR_MODES[a]}_vs_{FACTOR_MODES[b]}'):paired_panel(hit[:,a]-hit[:,b],draws)
             for a,b in ((3,4),(3,5),(4,5))}
    return dict(correct=correct,candidates=candidates,factor_pairs=pairs)


def panel_dimensions(rt):
    return (FACTOR_COUNT,FACTOR_MODES) if rt.get('factor_from') else (COUNT,MODES)


def panel_rows(rt, dataset, ids, rx, *, excluded=()):
    ids,rx = np.asarray(ids),np.asarray(rx)
    if not rt.get('factor_from'):
        return screen_rows(ids,rx,RX[dataset],excluded=excluded)
    root = Path(rt['factor_from'])
    if root != Path(rt['run_root']).parent/RUN:
        raise ValueError('Expected the exact prior screening root')
    folder = root/dataset
    if read(folder/'complete.json') != panel_receipt(dataset,COUNT,MODES):
        raise ValueError('Expected the earlier blind panel for exclusion')
    with np.load(folder/'predictions.npz',allow_pickle=False) as old:
        past = np.flatnonzero(np.isin(ids,old['ids']))
    if len(past)!=COUNT or not (rx[past]==RX[dataset]).all():
        raise ValueError('Previous screen physical ID mismatch')
    return screen_rows(ids,rx,RX[dataset],excluded=np.union1d(excluded,past),limit=FACTOR_COUNT,seed=FACTOR_SEED)


def fusion_outputs(rt, observed, anchor, companion, action, variance):
    ranked = (fuse(observed,anchor,companion,action,variance),)
    if rt.get('factor_from'):
        ranked += (fuse(observed,anchor,anchor,action,variance),companion_only(observed,anchor,companion))
    if not all(bool(torch.isfinite(v).all()) for v in ranked):
        raise ValueError('Nonfinite fusion')
    return ranked


class WiSigRealGain(nn.Module):
    """WiSig-specific repeated-preamble coordinates and shared phase-aware real gain."""

    def __init__(self):
        super().__init__()
        self.coarse = PacketUNet(across_chirps=True,center=32)
        self.gain = SpectralInnovation(phase_aware=True,samples=32)
        self.register_buffer('time',torch.arange(256,dtype=torch.float32),persistent=False)

    def forward(self, raw):
        adjacent = (raw[:,100:200] * raw[:,80:180].conj()).sum(-1,keepdim=True)
        rotation = torch.polar(torch.ones_like(raw.real), adjacent.angle()*self.time/20)
        framed = (raw*rotation.conj()).unflatten(-1,(8,32))
        scale = capture_scale(framed)
        observed = framed/scale
        coherence = (observed[...,1:]*observed[...,:-1].conj()).mean(-1,keepdim=True).abs().clamp(max=1.)
        edges = torch.cat((framed[...,:24],framed[...,-24:]),dim=-1)
        coarse = self.coarse(observed,coherence,scale,edges)[-1]
        state = observed + .5*(coarse-observed)
        first = raw + ((state-observed)*scale).flatten(1)*rotation
        final = first + (self.gain(observed,state)*scale).flatten(1)*rotation
        return first,final


def archive_replay(rt):
    package = importlib.import_module('experiments.pamf_lora_main')
    archived = Path(rt['archive_root'])/'experiments'/'pamf_lora_main'
    package.__path__.append(str(archived))
    return importlib.import_module('experiments.pamf_lora_main.packet_dura_crossdata')


def wifi_context(rt):
    replay = archive_replay(rt)
    config = replay.reference(rt,'wisig')
    return replay.wisig_deps(config),config


def fit_wifi(folder, original, source):
    data, ids, smoke_rows = original.core.source_iq(source)
    if len(ids) != 52604 or len(np.intersect1d(ids,smoke_rows)):
        raise ValueError('Expected legal WiSig gain fit mask and disjoint source smoke')
    torch.manual_seed(SEED)
    model = WiSigRealGain().to(DEV).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=2e-5)
    draws = np.random.default_rng(SEED)
    rng = torch.Generator(device=DEV).manual_seed(SEED+1)

    def objective(clean):
        clean = clean/capture_scale(clean)
        options = dict(device=DEV,generator=rng)
        n = torch.randn(clean.shape,dtype=clean.dtype,**options)
        echo = torch.randn((len(clean),1),dtype=clean.dtype,**options)
        shifted = torch.nn.functional.pad(clean[:,:-3],(3,0))
        snr = -8. + 32.*torch.rand((len(clean),1),**options)
        y = clean + .08*echo*shifted + 10.**(-snr/20)*n/capture_scale(n)
        p,x = model(y)
        _,identity = model(clean[:6])
        e = torch.fft.fft(x-clean,norm='ortho').abs().square()
        e = (e/(.03+torch.fft.fft(clean,norm='ortho').abs().square())).log1p().mean()
        return iq_error(x,clean) + .35*iq_error(p,clean) + .3*e + 1.5*iq_error(identity,clean[:6])

    smoke = objective(complex_iq(tensor(data[smoke_rows[:8]],DEV)))
    grad, = torch.autograd.grad(smoke,model.gain.gain[-1].weight)
    if not bool(torch.isfinite(grad).all() and grad.norm()>0):
        raise ValueError('Real source smoke has no gain gradient')
    write(folder/'fit-smoke.json',dict(status='PASS',gain_gradient=float(grad.norm()),target_reads=0,F_reads=0))
    rng.manual_seed(SEED+1)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for step in range(1,STEPS+1):
        clean = complex_iq(tensor(data[draws.choice(ids,BATCH,replace=False)],DEV))
        loss = objective(clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step==1 or step%100==0:
            print('WISIG_REAL_GAIN',step,float(loss.detach()),flush=True)
    torch.cuda.synchronize()
    with (folder/'real_gain.pt').open('xb') as stream:
        torch.save(model.state_dict(),stream)
    write(folder/'fit.json',dict(status='SOURCE_FROZEN',seed=SEED,steps=STEPS,batch=BATCH,
          training_ids=len(ids),F_reads=0,TX_reads=0,V_reads=0,target_reads=0,
          parameters=sum(p.numel() for p in model.parameters()),seconds=perf_counter()-started,
          peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    return model.eval().requires_grad_(False)


def lora_panel(rt, joint, source, contract):
    old,_,_ = joint.panel(source,contract)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as metadata:
        ids,rx = metadata['ids'].copy(),metadata['rx_i'].copy()
    rows = panel_rows(rt,'lora',ids,rx,excluded=old)
    return rows,ids[rows]


def panel_receipt(dataset, count, modes):
    return dict(dataset=dataset,receiver=RX[dataset],views=list(SCREEN_VIEWS),modes=list(modes),n=count,
                source_truth_reads=0,target_reads=0,status='PREDICTIONS_FROZEN')


def freeze(folder, dataset, ids, decisions, *, count=COUNT, modes=MODES):
    if len(ids)!=count or set(decisions)!=set(SCREEN_VIEWS):
        raise ValueError('Incomplete preregistered screen')
    for d in decisions.values():
        if d.shape!=(len(modes),count):
            raise ValueError('Wrong primary and comparison coverage')
    with (folder/'predictions.npz').open('xb') as stream:
        np.savez(stream,ids=ids,**decisions)
    write(folder/'complete.json',panel_receipt(dataset,count,modes))


def classify_lora(joint, reader, wave):
    packets = np.asarray(wave.contiguous().view(torch.float32).cpu().tolist(),dtype=np.float32).view(np.complex64)
    return joint.classify_aperture(reader,joint.received_aperture(packets,DEV))


@torch.no_grad()
def predict_lora(rt, folder):
    count,modes = panel_dimensions(rt)
    config = read(Path(rt['lora_ref'])/'runtime.json')
    if (config['run_id'],config['code_commit']) != (LO_RA,LO_RA_COMMIT):
        raise ValueError('Wrong frozen LoRa 8192-point real gain version')
    w24 = Path(config['w24_release_root'])
    sys.path.insert(0,str(w24))
    archive_replay(rt).archive_imports(config,extra=(w24,))
    from experiments.pamf_lora_main import w24_joint_pamf as joint, w24_native_training as native

    source = Path(config['source_root'])
    contract,reader,_,_ = joint.boot(config,DEV)
    stage = native.PHASE_ARMS[0]
    gain = native.task_model(None,DEV)
    gain.core.innovation = native.spectral_innovation(stage,DEV)
    native.load_stage(Path(config['run_root']),stage,contract,gain.core,DEV)
    gain.eval().requires_grad_(False)
    dura = native.dura_model(Path(rt['project'])/'runs'/native.DURA_RUN,contract,DEV).eval().requires_grad_(False)

    def apply(packets):
        y = tensor(np.ascontiguousarray(packets).view(np.float32),DEV).view(torch.complex64)
        scale = capture_scale(y)
        received = pair(y/scale)
        coarse,action,variance = dura.estimate(received)
        recovered = complex_iq(dura.unroll(received,coarse,action,variance))*scale
        anchor = gain(y)
        solved = fusion_outputs(rt,y/scale,anchor/scale,recovered/scale,action,variance)
        values = (y,recovered,anchor,*(z*scale for z in solved))
        return np.stack([classify_lora(joint,reader,z) for z in values])

    sample = joint.source_role(source,contract,'U_s')[0][:2]
    if apply(sample).shape != (len(modes),2):
        raise ValueError('LoRa no-query checkpoint smoke failed')
    write(folder/'smoke.json',dict(status='PASS',source_count=2,source_V_reads=0,target_reads=0))
    rows,ids = lora_panel(rt,joint,source,contract)
    x = joint.native(source,'V',contract['counts']['V'])[rows]
    jobs = list(zip(rows.tolist(),[joint.SOURCE[RX['lora']]]*count,x,strict=True))
    decisions = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for view in SCREEN_VIEWS:
            packets = np.stack(list(pool.map(partial(joint.source_receive,view),jobs)))
            decisions[view] = np.concatenate([apply(packets[i:i+8]) for i in range(0,count,8)],axis=1)
    freeze(folder,'lora',ids,decisions,count=count,modes=modes)


def wifi_predictor(rt, original, config, gain):
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    engine = original.tri.core.original_r69(Path(config['parent_root']))
    base = engine.pipeline.model

    @torch.no_grad()
    def apply(block):
        iq = tensor(block,DEV)
        y = complex_iq(iq)
        (_,action,variance) = base.estimate_context(iq)
        (output,) = engine(iq)
        anchor = complex_iq(output)
        companion = gain(y)[-1]
        solved = fusion_outputs(rt,y,anchor,companion,action,variance)
        return np.asarray([original.classify(reader,z).argmax(1).tolist()
                           for z in (y,anchor,companion,*solved)],dtype=np.int16)

    return apply


def smoke_wifi(rt, folder):
    original,config = wifi_context(rt)
    source = Path(config['source_root'])
    gain = WiSigRealGain().to(DEV).eval().requires_grad_(False)
    data,_,rows = original.core.source_iq(source)
    if wifi_predictor(rt,original,config,gain)(data[rows[:4]]).shape != (len(MODES),4):
        raise ValueError('Original R69 and Tri-SIG default head smoke failed')
    write(folder/'smoke.json',dict(status='PASS',source_count=4,source_V_reads=0,target_reads=0))


def predict_wifi(rt, folder, runtime):
    count,modes = panel_dimensions(rt)
    if not rt.get('factor_from'):
        subprocess.run([sys.executable,'-u','-m',__spec__.name,'wifi-smoke',str(runtime)],check=True)
    original,config = wifi_context(rt)
    source = Path(config['source_root'])
    if rt.get('factor_from'):
        prior = Path(rt['factor_from'])/'wisig'
        proof = read(prior/'fit.json')
        if (proof['status'],proof['steps'],proof['seed'],proof['training_ids'])!=('SOURCE_FROZEN',STEPS,SEED,52604) \
                or any(proof[k] for k in ('TX_reads','V_reads','F_reads','target_reads')):
            raise ValueError('Missing legal frozen WiSig real-gain ancestor')
        gain = WiSigRealGain().to(DEV).eval().requires_grad_(False)
        checkpoint = prior/'real_gain.pt'
    else:
        gain = fit_wifi(folder,original,source)
        checkpoint = folder/'real_gain.pt'
    gain.load_state_dict(torch.load(checkpoint,map_location=DEV,weights_only=True),strict=True)
    apply = wifi_predictor(rt,original,config,gain)
    data,_,smoke_rows = original.core.source_iq(source)
    if apply(data[smoke_rows[:4]]).shape != (len(modes),4):
        raise ValueError('WiSig no-query checkpoint smoke failed')
    write(folder/'frozen-smoke.json',dict(status='PASS',source_count=4,source_V_reads=0,target_reads=0))
    cache,_,_,_,ids,rx,_ = original.tri.core.handles(config)
    selected = panel_rows(rt,'wisig',ids,rx)
    decisions={}
    for view in SCREEN_VIEWS:
        array=np.load(cache/'validation_full'/f'{view}.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(len(ids),2,256) or array.dtype!=np.float32:
            raise ValueError('Mismatched WiSig source V capture')
        decisions[view]=np.concatenate([apply(array[selected[i:i+32]]) for i in range(0,count,32)],axis=1)
    freeze(folder,'wisig',np.asarray(ids)[selected],decisions,count=count,modes=modes)


def independent_score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Expected separate CPU truth connection')
    root = Path(rt['run_root'])
    count,modes = panel_dimensions(rt)
    predictions = root
    if rt.get('score_from'):
        predictions = Path(rt['score_from'])
        if predictions != root.parent/RUN or read(predictions/'state.json')['status'] != 'FAILED':
            raise ValueError('Expected the immutable, previously failed score stage')
    rows={}
    for dataset in RX:
        folder = predictions/dataset
        proof = read(folder/'complete.json')
        if proof!=panel_receipt(dataset,count,modes):
            raise ValueError('Both screening matrices must freeze before source truth')
        with np.load(folder/'predictions.npz',allow_pickle=False) as data:
            ids = data['ids'].copy()
            matrix = np.stack([data[v] for v in SCREEN_VIEWS])
        classes = 10 if dataset=='lora' else 6
        if (matrix.shape!=(2,len(modes),count) or matrix.dtype.kind not in 'iu'
                or not ((0<=matrix)&(matrix<classes)).all() or len(set(ids.tolist()))!=count):
            raise ValueError('Invalid frozen predictions')
        rows[dataset] = ids,matrix

    # Activate the WiSig archive before importing any LoRa package that can load
    # another pamf_unrolled_restore namespace.
    original,wifi=wifi_context(rt)
    _,_,wifi_source,_,ids,rx,order=original.tri.core.handles(wifi)
    selected=panel_rows(rt,'wisig',ids,rx)
    if not np.array_equal(rows['wisig'][0],np.asarray(ids)[selected]):
        raise ValueError('WiSig screen order drift')

    lora = read(Path(rt['lora_ref'])/'runtime.json')
    source = Path(lora['source_root'])
    lora_old = read(Path(lora['run_root'])/'full_native'/'source-predictions.json')
    if lora_old['status']!='PREDICTIONS_FROZEN' or lora_old['target_reads'] or lora_old['source_truth_reads']:
        raise ValueError('Expected the fixed earlier source panel exclusion')
    with np.load(source/'V_metadata.npz',allow_pickle=False) as data:
        names,rx = data['ids'].copy(),data['rx_i'].copy()
        role = read(source/'contract.json')['role_ids']['V']
        if names.tolist()!=role:
            raise ValueError('LoRa source validation roles differ')
        exclude = np.flatnonzero(np.isin(names,lora_old['physical_ids']))
        if len(exclude)!=300:
            raise ValueError('The original 300 source validation packets must be excluded')
        index = panel_rows(rt,'lora',names,rx,excluded=exclude)
        if not np.array_equal(rows['lora'][0],names[index]):
            raise ValueError('LoRa screen order drift')
        lora_truth=data['y'][index]
    wifi_truth=np.load(wifi_source/'V/labels.npy',allow_pickle=False)[np.asarray(order)[selected]]
    calculate = score_factor_panel if rt.get('factor_from') else score_panel
    outputs={k:calculate(rows[k][1],y) for k,y in (('lora',lora_truth),('wisig',wifi_truth))}
    write(root/'scores.json',dict(status='ANALYZED',truth_last=True,target_reads=0,n_per_view=count,
                                  modes=list(modes),views=list(SCREEN_VIEWS),results=outputs))


def dispatch(runtime, rt):
    from experiments.pamf_lora_main.runner import available

    root=Path(rt['run_root'])
    root.mkdir()
    state=root/'state.json'

    def launch(action, gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (root/f'{action}.log').open('x',encoding='utf-8') as log:
            return subprocess.Popen([sys.executable,'-u','-m',__spec__.name,action,str(runtime)],
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)

    try:
        if rt.get('score_from'):
            job = launch('score',None)
            write(state,dict(status='SCORING_FROZEN',pid=job.pid),exclusive=False)
            if job.wait():
                raise RuntimeError('Independent scoring of the previously frozen screen failed')
            write(state,dict(status='ANALYZED',target_reads=0,reused_predictions=rt['score_from']),exclusive=False)
            return
        deadline=monotonic()+600
        while not (cards:=available(rt['allowed_gpus'],max_compute_processes=8)):
            if monotonic()>=deadline:
                raise RuntimeError('No registered GPU slot')
            sleep(10)
        jobs={}
        for i,dataset in enumerate(('wisig','lora')):
            card=cards[min(i,len(cards)-1)]
            if i and len(cards)==1 and jobs['wisig'].wait():
                raise RuntimeError('WiSig fit or prediction failed')
            jobs[dataset]=launch(dataset,card)
            write(state,dict(status='RUNNING',workers={k:p.pid for k,p in jobs.items()}),exclusive=False)
        exits={k:p.wait() for k,p in jobs.items()}
        if any(exits.values()) or launch('score',None).wait():
            raise RuntimeError(f'Screen pipeline failed after worker exits {exits}')
        write(state,dict(status='ANALYZED',target_reads=0),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('publish','dispatch','lora','wisig','wifi-smoke','score'))
    parser.add_argument('runtime',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ssh-config',type=Path)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--retry-score',action='store_true')
    modes.add_argument('--factor-screen',action='store_true')
    args=parser.parse_args()
    rt=read(args.runtime)
    if args.action=='publish':
        from experiments.pamf_lora_main.publish import release,release_commit
        project=rt['project']
        releases=f'{project}/releases'
        run = SCORE_RUN if args.retry_score else FACTOR_RUN if args.factor_screen else RUN
        rt={k:rt[k] for k in ('project','python','frozen_sources')}
        rt.update(run_id=run,code_commit=release_commit(Path.cwd()),runner=__spec__.name,
                  release_root=f'{releases}/{run}-release',run_root=f'{project}/runs/{run}',
                  lora_ref=f'{releases}/{LO_RA}-release',wisig_ref=f'{releases}/{WISIG_REF}-release',
                  archive_root=f'{releases}/{ARCHIVE}-release',allowed_gpus=[5,2,6,3,0,7])
        rt['external_releases']=[rt[key] for key in ('lora_ref','wisig_ref','archive_root')]
        if args.retry_score:
            rt['score_from']=f'{project}/runs/{RUN}'
            rt['external_releases'].append(rt['score_from'])
        if args.factor_screen:
            rt['factor_from']=f'{project}/runs/{RUN}'
            rt['external_releases'].append(rt['factor_from'])
        release(Path.cwd(),args.output,rt,args.ssh_config)
    elif args.action=='dispatch':
        dispatch(args.runtime,rt)
    elif args.action=='score':
        independent_score(rt)
    elif args.action=='wifi-smoke':
        numerical_policy()
        smoke_wifi(rt,Path(rt['run_root'])/'wisig')
    else:
        numerical_policy()
        folder=Path(rt['run_root'])/args.action
        folder.mkdir()
        if args.action=='lora':
            predict_lora(rt,folder)
        else:
            predict_wifi(rt,folder,args.runtime)


if __name__=='__main__':
    main()
