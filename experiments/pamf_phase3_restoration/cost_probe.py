"""One source-only, zero-update GPU cost comparison with the actual Phase3 F0."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

RUN = '20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.cost_probe'
MODEL_SEED = 2026092701


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_inputs(c):
    from .source_pair import validate_source
    validate_source(c)
    return c


def gpu_choice():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    pids = {}
    for row in compute.splitlines():
        uuid, pid = [x.strip() for x in row.split(',')]
        pids.setdefault(uuid,set()).add(int(pid))
    eligible = []
    for row in rows.splitlines():
        index, uuid, free = [x.strip() for x in row.split(',')]
        if int(index) in (4,6,7) and int(free)>=12000 and len(pids.get(uuid,set()))<=2:
            eligible.append(dict(GPU=int(index), free_memory_MB=int(free), existing_compute_PIDs=len(pids.get(uuid,set()))))
    return sorted(eligible,key=lambda a:(a['existing_compute_PIDs'],-a['free_memory_MB'],a['GPU']))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Controller must have empty CUDA visibility')
    from .source_pair import read
    c=validate_inputs(read(args.inputs)); out=args.root/'runs'/RUN
    out.mkdir(); (out/'logs').mkdir()
    write(out/'inputs.json',c)
    write(out/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit))
    # Previous run handles must really be terminal; never stop them or others.
    for pid in (2054412,2054413,2054711,2054717):
        cmd=Path('/proc')/str(pid)/'cmdline'
        if cmd.exists() and b'43577dc71dfdab6382dcf94e75c3b699b2c5122e' in cmd.read_bytes():
            raise ValueError('Previous owned full-rate run is still active; preserve it')
    choices=gpu_choice()
    while not choices:
        print('WAITING_FOR_ALLOWED_GPU',flush=True); time.sleep(20); choices=gpu_choice()
    selected=choices[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(selected['GPU']))
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(out/'inputs.json')]
    with (out/'logs/worker.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=child.pid,command=command,CWD=os.getcwd(),admission=selected))
    code=child.wait()
    write(out/'controller_complete.json',dict(status='WORKER_FINISHED' if code==0 else 'WORKER_FAILED_PRESERVED',exit_code=code,pid=child.pid))
    if code:raise RuntimeError('Source cost worker failed; preserve outputs and inspect exact layer')


def worker(args):
    import copy
    import statistics
    import numpy as np
    import torch
    from .model import scratch_model, architecture
    from .objective import PairedObjective, feature_guide, INPUT_GRADIENT_CAP
    from .source_pair import prepare, read
    from experiments.pamf_phase3_open_world import star_channel
    from experiments.pamf_phase3_open_world import star_relative_source as source
    started=time.perf_counter();out=args.root/'runs'/RUN;c=validate_inputs(read(args.inputs))
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Exactly one admitted physical GPU required')
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('Actual release Git commit differs')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';star_channel._channel_api()  # Resolve canonical channel before F0 bootstrap.
    load_started=time.perf_counter();b=source._backend(c,device,with_dura=True)
    original=read(Path(c['reader_config']['source_output'])/'complete.json')
    cfg=original['config']
    if (original.get('status')!='SOURCE_FROZEN' or original.get('epoch')!=50 or original.get('updates')!=11100
            or original.get('target_access') is not False or cfg.get('run_id')!='20261027-phase1-pamf-repl-shuffled-views-m16-r02'
            or cfg.get('row_id')!='main_split-repl-shuffled_time-3view-s2026092701'
            or cfg.get('code_commit')!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'):
        raise ValueError('Strict original Tri-SIG-TIME SSL F0 E50 ancestry required')
    F0=b['reader'];new=scratch_model(MODEL_SEED).to(device).eval()
    reference=copy.deepcopy(b['dura'][0]).eval().requires_grad_(True)
    if sum(p.numel() for p in reference.parameters())!=266951:
        raise ValueError('Complete DURA reference graph required')
    objective=PairedObjective().to(device)
    models=dict(TF=new,DURA_reference=reference)
    before=[(m,{k:v.detach().clone() for k,v in m.state_dict().items()}) for m in [F0,b['dura'][0],new,reference]]
    model_load_seconds=time.perf_counter()-load_started
    counts=dict(F0_feature_rows=0,F0_head_rows=0,TF_rows=0,DURA_reference_rows=0,DURA_production_logical_rows=0)
    def tensor(a):
        a=np.ascontiguousarray(a,dtype=np.float32)
        return torch.frombuffer(memoryview(a),dtype=torch.float32).reshape(a.shape).clone().to(device)
    class CountedReader:
        last=None
        def features(self,value):
            counts['F0_feature_rows']+=len(value);self.last=F0.features(value);return self.last
    reader=CountedReader()
    def invoke(name,x):
        counts[name+'_rows']+=len(x);return models[name](x)
    def unchanged():
        for m, state in before:
            now=m.state_dict()
            if set(now)!=set(state) or any(not torch.equal(v,now[k]) for k,v in state.items()):
                raise ValueError('Zero-update diagnostic changed parameters or buffers')
        if any(p.grad is not None for p in F0.parameters()) or any(p.grad is not None for p in b['dura'][0].parameters()):
            raise ValueError('Frozen F0/DURA reference received parameter gradients')
    def smoke(clean_values,received_values):
        smoke_started=time.perf_counter()
        x,y=tensor(received_values),tensor(clean_values)
        with torch.no_grad():target=reader.features(y).detach()
        norms={}
        for name,m in models.items():
            restored=invoke(name,x)
            if name=='TF':torch.testing.assert_close(restored,x,rtol=0,atol=0)
            else:
                with torch.no_grad():
                    production,valid=b['dura'][1](x);counts['DURA_production_logical_rows']+=len(x)
                    if not valid:raise ValueError('Fixed production DURA is invalid')
                    torch.testing.assert_close(restored.detach(),production,rtol=3e-5,atol=3e-5)
            physical,terms=objective(restored,y);guide,info=feature_guide(reader,restored,y,target)
            with torch.no_grad():
                logits=F0.classify_features(reader.last.detach());counts['F0_head_rows']+=len(x)
                if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Original default six-class head invalid')
            actual=torch.autograd.grad(guide.sum(),restored,retain_graph=True)[0].flatten(1).norm(dim=1)
            if not torch.isfinite(actual).all() or not actual.sum()>0 or (actual>INPUT_GRADIENT_CAP*info['IQ_gradient_norm']*(1+2e-5)).any():
                raise ValueError('Actual source F0 input gradient/cap failed')
            (physical+guide).mean().backward()
            grads=[p.grad for p in m.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
                raise ValueError('Actual paired-source G gradient missing')
            if name=='TF' and not m.output.weight.grad.abs().sum()>0:raise ValueError('New zero output head has no source gradient')
            norms[name]=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True))
            m.zero_grad(set_to_none=True);reader.last=None
        unchanged()
        write(out/'smoke.json',dict(status='PASS',real_L_s=2,paired_G10_received=True,TF_identity=True,
            actual_original_F0_input_VJP=True,F0_default_head=True,output_gradient_nonzero=True,gradient_norms=norms,
            actual_input_gradient_cap=.5,native_production_DURA_match=True,models_unchanged=True,
            counts=dict(counts),TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0,
            seconds=time.perf_counter()-smoke_started))
        print('REAL_PHASE3_F0_SOURCE_PAIR_SMOKE_PASS',flush=True)
    prepare_started=time.perf_counter();pair=prepare(c,smoke=smoke);pair_seconds=time.perf_counter()-prepare_started
    write(out/'source_pair.json',pair['evidence'])
    with (out/'source_pair.npz').open('xb') as stream:
        np.savez(stream,**{k:pair[k] for k in ['clean','received','ids','rx','day','roles']})
    x,y=tensor(pair['received']),tensor(pair['clean'])
    target_started=time.perf_counter()
    with torch.no_grad():target=reader.features(y).detach()
    torch.cuda.synchronize();target_seconds=time.perf_counter()-target_started;reader.last=None
    # Interleave the source diagnostic's 128 L_s and 384 U_s. It is not the
    # future full-pool 10/90 training ratio, and these are repeated cost probes.
    order=np.column_stack((np.arange(128),np.arange(128,256),np.arange(256,384),np.arange(384,512))).reshape(-1)
    batches=[torch.tensor(order[start:start+256],device=device,dtype=torch.long) for start in (0,256)]
    rows=[];bench_started=time.perf_counter()
    for guided in (False,True):
        for block in range(3):
            for iteration in range(8):
                indices=batches[(block+iteration)%2];a,z,t=x[indices],y[indices],target[indices]
                names=['DURA_reference','TF'] if iteration%2==0 else ['TF','DURA_reference']
                for name in names:
                    m=models[name];m.zero_grad(set_to_none=True);torch.cuda.synchronize();wall=time.perf_counter()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                    restored=invoke(name,a);loss,terms=objective(restored,z)
                    if guided:guide,_=feature_guide(reader,restored,z,t);loss=loss+guide
                    loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
                    end.record();torch.cuda.synchronize();wall_ms=1000*(time.perf_counter()-wall)
                    if not torch.isfinite(loss).all():raise ValueError('Nonfinite source objective')
                    rows.append(dict(model=name,guided=guided,block=block,iteration=iteration,warmup=iteration<2,
                        batch_size=256,wall_ms=wall_ms,CUDA_event_ms=begin.elapsed_time(end),loss=float(loss.mean()),gradient_norm=float(norm)))
                    reader.last=None
    torch.cuda.synchronize();bench_seconds=time.perf_counter()-bench_started
    for m in models.values():m.zero_grad(set_to_none=True)
    unchanged();summaries={}
    for guided in (False,True):
        key='IQ_plus_F0' if guided else 'IQ_only';summary={}
        for name in models:
            selected=[r for r in rows if r['guided']==guided and r['model']==name and not r['warmup']]
            summary[name]=dict(timed_steps=len(selected),median_wall_ms=statistics.median(r['wall_ms'] for r in selected),
                median_CUDA_event_ms=statistics.median(r['CUDA_event_ms'] for r in selected),
                block_medians=[statistics.median(r['wall_ms'] for r in selected if r['block']==block) for block in range(3)])
        d,tf=summary['DURA_reference'],summary['TF']
        summary['TF_wall_reduction_percent']=100*(1-tf['median_wall_ms']/d['median_wall_ms'])
        summary['blocks_TF_faster']=sum(a<b for a,b in zip(tf['block_medians'],d['block_medians']))
        summary['atleast10percent_and_all3blocks_positive']=summary['TF_wall_reduction_percent']>=10 and summary['blocks_TF_faster']==3
        summaries[key]=summary
    assert len(rows)==96 and counts['TF_rows']==counts['DURA_reference_rows']==12290 and counts['F0_feature_rows']==12806
    result=dict(status='ANALYZED_SOURCE_COST_ONLY',run_id=RUN,code_commit=args.commit,architecture=architecture(),
        original_F0_identity=b['reader_identity']|dict(source_run=cfg['run_id'],source_row=cfg['row_id'],source_code_commit=cfg['code_commit'],source_epochs=50,source_updates=11100),
        DURA_reference_identity=b['dura_identity'],DURA_reference_is_copy_of_frozen_final_weights_not_TF_ancestor=True,
        zero_update_comparison=dict(new_TF='random identity initialization',DURA_reference='trained R69 weights copied only for cost; eval with parameter gradients enabled'),
        source_pair=pair['evidence'],source_preparation_seconds=pair_seconds,source_preparation_includes_real_smoke=True,model_load_seconds=model_load_seconds,
        source_clean_F0_cache_rows=512,source_clean_F0_cache_bytes=512*160*4,source_clean_F0_cache_seconds=target_seconds,
        counts=counts,summaries=summaries,timing_rows=rows,benchmark_seconds=bench_seconds,
        source_step_includes='resident paired IQ forward/physical objective/optional full F0 feature-input-VJP/backward/clip1; eval normalization state',
        source_step_excludes='optimizer.step/IO/CPU-GPU transfer/cache production/loading/evaluation; startup separately reported',
        physical_objective='IQNMSE+.05mean3STFTmagnitude_relativeL1',F0_guide='0.1cosinefused160/detached per-record IQ-gradientcap.5',
        source_benchmark_role_ratio='128L_s/384U_s interleaved; diagnostic25/75, not full-pool training10/90',
        resident_system_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_name=torch.cuda.get_device_name(),
        models_unchanged=True,optimizer_updates=0,G_training_steps=0,TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,
        cached_DURA_IQ_reads=0,actual_training_quality_verified=False,causal_equal_quality_speed_claim=False,
        default_DURA_changed=False,goal_complete=False,seconds=time.perf_counter()-started)
    write(out/'results.json',result);write(out/'complete.json',dict(status=result['status'],run_id=RUN,code_commit=args.commit,optimizer_updates=0,target_inputs=0))
    print(json.dumps(dict(status=result['status'],summaries=summaries,seconds=result['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    if re.fullmatch('[0-9a-f]{40}',args.commit) is None:raise ValueError('One actual execution Git commit required')
    print(json.dumps(dict(event='P3_RESTORE_COST_PROCESS',pid=os.getpid(),CWD=os.getcwd(),argv=sys.argv,CUDA=os.environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
    if args.mode=='run':controller(args)
    else:
        try:worker(args)
        except Exception as exc:
            out=args.root/'runs'/RUN
            write(out/'failure.json',dict(status='FAILED_PRESERVED',error=str(exc),type=type(exc).__name__,target_inputs=0,optimizer_updates=0))
            raise


if __name__=='__main__':main()
