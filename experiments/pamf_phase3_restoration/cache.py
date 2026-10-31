"""One full original L_s/U_s clean-pair and separate-use F0 target cache."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from .source_pair import read, validate_source, metadata, open_received, EPISODE, COUNTS
from .cost_probe import gpu_choice, write

RUN = '20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.cache'
SCHEMA = 'p3_restore_source_cache_v1'
CAPSULE = 'p3restore-LU392005-G10MIX01-clean-pairs-20261101-r01'
SPLIT = '392005-LsUs-source5rx-3day-eq1'
N = 63000


def tensor(a, device):
    import torch
    a=np.array(a,copy=True,dtype=np.float32,order='C')
    return torch.frombuffer(memoryview(a),dtype=torch.float32).reshape(a.shape).to(device)


def load_F0(c, device):
    from experiments.pamf_phase3_open_world import star_channel
    from experiments.pamf_phase3_open_world import star_relative_source as source
    star_channel._channel_api()
    b=source._backend(c,device,with_dura=False)
    done=read(Path(c['reader_config']['source_output'])/'complete.json');cfg=done['config']
    if (done.get('status')!='SOURCE_FROZEN' or done.get('epoch')!=50 or done.get('updates')!=11100
            or done.get('target_access') is not False or cfg.get('run_id')!='20261027-phase1-pamf-repl-shuffled-views-m16-r02'
            or cfg.get('row_id')!='main_split-repl-shuffled_time-3view-s2026092701'
            or cfg.get('code_commit')!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'):
        raise ValueError('Actual original source-only Tri-SIG-TIME SSL F0 E50 required')
    return b['reader'],dict(b['reader_identity'],source_run=cfg['run_id'],source_row=cfg['row_id'],
                           source_code_commit=cfg['code_commit'])


def check_F0_state(model, before):
    import torch
    current=model.state_dict()
    if (set(current)!=set(before) or any(not torch.equal(v,current[k]) for k,v in before.items())
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Frozen F0 state or parameter gradients changed')


def real_smoke(F0, clean, received, device):
    import torch
    from .model import scratch_model
    from .objective import PairedObjective,feature_guide
    started=time.perf_counter();G=scratch_model(2026092701).to(device).eval()
    before={k:v.detach().clone() for k,v in G.state_dict().items()}
    fbefore={k:v.detach().clone() for k,v in F0.state_dict().items()}
    y,x=tensor(clean,device),tensor(received,device)
    with torch.no_grad():target=F0.features(y).detach()
    class Reader:
        last=None
        def features(self,value):
            self.last=F0.features(value);return self.last
    reader=Reader();restored=G(x);torch.testing.assert_close(restored,x,rtol=0,atol=0)
    physical,_=PairedObjective().to(device)(restored,y)
    guide,info=feature_guide(reader,restored,y,target)
    gradient=torch.autograd.grad(guide.sum(),restored,retain_graph=True)[0].flatten(1).norm(dim=1)
    if not torch.isfinite(gradient).all() or not gradient.sum()>0 or (gradient>.5*info['IQ_gradient_norm']*(1+2e-5)).any():
        raise ValueError('Actual original F0 IQ input VJP/cap failed')
    with torch.no_grad():logits=F0.classify_features(reader.last.detach())
    if logits.shape!=(2,6) or not torch.isfinite(logits).all():
        raise ValueError('Actual original default head invalid')
    (physical+guide).mean().backward()
    if G.output.weight.grad is None or not torch.isfinite(G.output.weight.grad).all() or not G.output.weight.grad.abs().sum()>0:
        raise ValueError('Actual source residual output gradient missing')
    G.zero_grad(set_to_none=True)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):
        raise ValueError('No-update source smoke changed G state')
    check_F0_state(F0,fbefore)
    return dict(status='PASS',real_L_s=2,G_rows=2,F0_feature_rows=4,F0_head_rows=2,
                G_initial_identity=True,actual_F0_input_VJP=True,IQ_gradient_cap=.5,
                models_unchanged=True,optimizer_updates=0,TX_labels_read=0,OE_reads=0,V_reads=0,
                target_inputs=0,seconds=time.perf_counter()-started)


def build(args):
    import torch
    from experiments.pamf_phase3_open_world import star_channel as channel
    from experiments.pamf_phase3_open_world import source_us_cache as locator
    from experiments.pamf_phase3_open_world import star_relative_source as source
    started=time.perf_counter();c=validate_source(read(args.inputs));out=args.root/'runs'/RUN
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('Actual release code commit differs')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('One admitted physical GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';api,_=channel._channel_api();mixture=channel.validate_mixture(read(c['mixture_config_path']))
    F0,identity=load_F0(c,device);before={k:v.detach().clone() for k,v in F0.state_dict().items()}
    contract,meta=metadata(c,full=True)
    left,lm=open_received(c,'L_s',contract);right,um=open_received(c,'U_s',contract)
    folder=out/'cache';folder.mkdir()
    clean=np.lib.format.open_memmap(folder/'clean.npy',mode='w+',dtype=np.float32,shape=(N,2,256))
    received=np.lib.format.open_memmap(folder/'received.npy',mode='w+',dtype=np.float32,shape=(N,2,256))
    prep=time.perf_counter();maximum_audit_difference=0.;smoke=None
    payload=source._load_raw(c['raw_mainsig'])
    records,enumerated,rxnames,days=locator._locate(payload,contract,meta)
    try:
        for i,(tx,rx,day,eq,sig) in enumerate(records):
            value=np.asarray(left[i] if i<6300 else right[i-6300])
            if not np.isfinite(value).all():raise ValueError('Original source P0 is nonfinite; no reselection')
            raw=source.raw_packet(payload['data'][tx][rx][day][eq][sig])
            if raw.dtype!=np.float64:raise ValueError('Original raw source required; normalized cache is not raw')
            r=raw.astype(np.float32);z=r[0].astype(np.float64)+1j*r[1].astype(np.float64)
            if i<2:
                audit,info=channel.observe(raw,physical_id=str(meta['ids'][i]),receiver=rxnames[rx],day=days[day],episode=EPISODE,mixture=mixture)
                maximum_audit_difference=max(maximum_audit_difference,float(np.max(np.abs(audit-value))))
                if not np.array_equal(audit,value):raise ValueError('Cached original source P0 fails canonical G10 reproduction')
                normalized=z/info['input_rms']
            else:normalized,_=api.normalize_input(z)
            clean[i]=np.stack((normalized.real,normalized.imag)).astype(np.float32);received[i]=value
            if i==1:
                smoke=real_smoke(F0,np.array(clean[:2]),np.array(received[:2]),device);write(out/'smoke.json',smoke)
                print('REAL_P3_FULL_CACHE_SOURCE_SMOKE_PASS',flush=True)
                process_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                own=[line for line in process_rows.splitlines() if line.split(',')[1].strip()==str(os.getpid())]
                write(out/'GPU_runtime.json',dict(pid=os.getpid(),CUDA=os.environ['CUDA_VISIBLE_DEVICES'],owned_nvidia_rows=own,GPU_name=torch.cuda.get_device_name()))
            if (i+1)%8192==0 or i+1==N:print('RAW_SOURCE_PAIR_PROGRESS',i+1,N,flush=True)
    finally:
        left._mmap.close();right._mmap.close();clean.flush();received.flush()
    del payload,records
    pair_seconds=time.perf_counter()-prep
    for name,value in dict(ids=meta['ids'],roles=np.asarray(['L_s']*6300+['U_s']*56700),
                           rx=meta['rx_i'],day=meta['day_i']).items():
        with (folder/(name+'.npy')).open('xb') as stream:np.save(stream,value,allow_pickle=False)
    feature=np.lib.format.open_memmap(folder/'F0_features.npy',mode='w+',dtype=np.float32,shape=(N,160))
    cache_started=time.perf_counter()
    try:
        for start in range(0,N,512):
            stop=min(start+512,N)
            with torch.no_grad():v=F0.features(tensor(clean[start:stop],device)).detach()
            if v.shape!=(stop-start,160) or v.dtype!=torch.float32 or not torch.isfinite(v).all():
                raise ValueError('Finite actual F0 clean fused160 cache required')
            feature[start:stop]=np.asarray(v.cpu().tolist(),np.float32)
            if stop%8192==0 or stop==N:print('SOURCE_F0_CACHE_PROGRESS',stop,N,flush=True)
    finally:
        feature.flush();feature._mmap.close();clean._mmap.close();received._mmap.close()
    check_F0_state(F0,before);cache_seconds=time.perf_counter()-cache_started
    manifest=dict(schema=SCHEMA,status='VALIDATED_ONCE',capsule_id=CAPSULE,split_id=SPLIT,
        run_id=RUN,code_commit=args.commit,count=N,counts={'L_s':6300,'U_s':56700},shape=[N,2,256],dtype='float32',
        order='originalL_s_then_originalU_s',episode=EPISODE,received_path='originalG10P0',clean_path='rawFP64_to_FP32_to_oneRMS',
        arrays=['clean','received','ids','roles','rx','day'],optional_guidance_array='F0_features',
        F0_feature_shape=[N,160],F0_feature_bytes=N*160*4,F0_identity=identity,
        source_contract_handle={k:contract[k] for k in ['classes','source_rxs','source_days','split_seed','counts']},
        source_L_producer=lm['run_id'],source_Us_producer=um['run_id'],raw_ManySig_whole_file_read=True,
        raw_payload_includes_other_roles=True,only_original_Ls_Us_values_indexed=True,private_locator_exported=False,
        original_RMS_calls=N,canonical_observe_audit_calls=2,post_channel_RMS_calls=0,maximum_G10_audit_difference=maximum_audit_difference,
        TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0)
    write(folder/'manifest.json',manifest)
    result=dict(status='P3_RESTORE_SOURCE_CACHE_COMPLETE',run_id=RUN,code_commit=args.commit,cache_root=str(folder),
        manifest=manifest,smoke=smoke,raw_coordinate_enumeration=enumerated,pair_preparation_including_smoke_seconds=pair_seconds,
        F0_clean_cache_seconds=cache_seconds,F0_cache_rows=N,F0_total_feature_rows=N+4,F0_head_rows=2,G_smoke_rows=2,
        models_unchanged=True,optimizer_updates=0,target_inputs=0,new_G_training=False,
        no_DURA_model_or_IQ_teacher_loaded=True,no_legacy_normalized_IQ_opened=True,
        seconds=time.perf_counter()-started)
    write(out/'results.json',result);write(out/'complete.json',dict(status=result['status'],run_id=RUN,code_commit=args.commit,cache_root=str(folder),optimizer_updates=0,target_inputs=0))
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],pair_seconds=pair_seconds,F0_seconds=cache_seconds)),flush=True)


def load_cache(root, *, guided=False):
    """Pure-IQ mode does not open, inspect or return the F0 feature array."""
    root=Path(root);m=read(root/'manifest.json')
    expected=dict(schema=SCHEMA,status='VALIDATED_ONCE',capsule_id=CAPSULE,split_id=SPLIT,count=N,counts={'L_s':6300,'U_s':56700},
                  shape=[N,2,256],dtype='float32',order='originalL_s_then_originalU_s',episode=EPISODE,
                  TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Completed original L_s/U_s paired cache required')
    names=['clean','received','ids','roles','rx','day']+(['F0_features'] if guided else [])
    result={'manifest':m}
    try:
        for name in names:result[name]=np.load(root/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if any(result[k].shape!=(N,2,256) or result[k].dtype!=np.float32 for k in ['clean','received']):
            raise ValueError('Source pair dtype or shape differs')
        if result['ids'].shape!=(N,) or len(set(result['ids'].tolist()))!=N or not np.array_equal(result['roles'],np.asarray(['L_s']*6300+['U_s']*56700)):
            raise ValueError('Paired source ID/role order differs')
        if any(result[k].shape!=(N,) for k in ['rx','day']):
            raise ValueError('Aligned source receiver/day metadata required')
        expected_F0=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,
            model_seed=2026092701,source_run='20261027-phase1-pamf-repl-shuffled-views-m16-r02',
            source_row='main_split-repl-shuffled_time-3view-s2026092701',
            source_code_commit='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d')
        if guided and any(m.get('F0_identity',{}).get(k)!=v for k,v in expected_F0.items()):
            raise ValueError('Original source-only F0 guidance identity differs')
        if guided and (result['F0_features'].shape!=(N,160) or result['F0_features'].dtype!=np.float32):
            raise ValueError('Actual aligned F0 guidance cache shape differs')
    except Exception:
        close_cache(result);raise
    return result


def close_cache(bundle):
    for value in bundle.values():
        if isinstance(value,np.memmap) and not value._mmap.closed:value._mmap.close()


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller must have empty CUDA visibility')
    c=validate_source(read(args.inputs));out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();write(out/'inputs.json',c)
    write(out/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit))
    available=gpu_choice()
    while not available:print('WAITING_ALLOWED_GPU',flush=True);time.sleep(20);available=gpu_choice()
    admission=available[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(admission['GPU']))
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(out/'inputs.json')]
    with (out/'logs/worker.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=child.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=child.wait();write(out/'controller_complete.json',dict(status='WORKER_FINISHED' if code==0 else 'WORKER_FAILED_PRESERVED',exit_code=code,pid=child.pid))
    if code:raise RuntimeError('Source cache worker failed; preserve outputs')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args()
    if re.fullmatch('[0-9a-f]{40}',a.commit) is None:raise ValueError('Actual execution Git commit required')
    print(json.dumps(dict(event='P3_SOURCE_CACHE_PROCESS',pid=os.getpid(),CWD=os.getcwd(),argv=sys.argv,CUDA=os.environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
    if a.mode=='run':controller(a)
    else:
        try:build(a)
        except Exception as e:
            write(a.root/'runs'/RUN/'failure.json',dict(status='FAILED_PRESERVED',error=str(e),type=type(e).__name__,optimizer_updates=0,target_inputs=0));raise


if __name__=='__main__':main()
