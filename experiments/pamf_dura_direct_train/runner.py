"""Single-stage random-initialized source training; automatic seven-view truth-last.

Training imports no identity reader or truth loader. A separate no-query smoke
and post-freeze evaluation are the only F consumers.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .model import ARMS, scratch_model, supervised_action, direct_objective, learning_rate
from experiments.pamf_dura_diagnostics.runner import tensor, numpy_values

RUN = '20261030-phase1-DURA-direct-joint-s2026092701-r01'
PARENT = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
ACTION = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
F_RUN = '20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paths(root):
    return dict(output=root/'runs'/RUN, cache=root/'runs'/ACTION/'cache',
        parent=root/'runs'/PARENT/'stage2/native_map_prior/final.pt',
        source_cache=root/'data_caches/phase1_manysig_reference_20261019/source_cache',
        reader=root/'runs'/F_RUN/'CROSS_RX-s2026092701/source',
        formal=root/'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference',
        raw=root/'runs/20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09/prediction',
        dura=root/'runs/20261021-phase1-restoration-stage2-graph-s2026092701-r01/stage2_residual_graph/target_confirmation/native_map_prior/prediction',
        truth=root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')


def configuration():
    return dict(run=RUN,arms=list(ARMS),parameters=266951,initialization='scratch_no_checkpoint',
        ancestor_checkpoints=[],teacher_model=None,training_stages=1,inference_stages=2,
        source_contract='original7/63/30',model_seed=2026092701,data_seed=2026103004,
        augmentation_seed=2026101912,evaluation_seed=2027,epochs=20,updates=4940,batch=256,
        L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,
        optimizer='resetAdamW',peak_lr=2e-4,final_lr=2e-5,warmup=247,weight_decay=1e-4,
        objective='finalIQNMSE+.1mean2internalpriorIQNMSE+.05multiscaleSTFT+.05mean2sourceactionprobeNMSE+.001logvarianceMSE',
        source_supervision='PairedcleanIQ and actualsource simulator action/noise;0TXlabels or learnedteacher',
        prior_calls=2,CG_updates=4,FP32=True,TF32=False,CPU_threads=2,
        source_freeze='fixedfinalE20/4940;noearlystopor source ranking',
        training_TX_labels=0,training_F_access=False,V_training=False,target_training=False,
        target_count=168000,views=list(VIEWS),all_versions_automatic_formal=True,
        allowed_GPUs=list(range(8)),max_own_workers=1,max_own_workers_per_GPU=1,
        max_goal_GPU_workers=2,max_total_visible_GPU_processes=2,minimum_free_MB=12000,
        reference_DURA=PARENT,reference_has_120_component_epochs=True,new_cumulative_epochs=20,
        stop_rule='Badcontract/roles/collision/nonfinite/checkpoint/smoke/predictioncoverage:preservefailedrow,noautoretry;no performance stop',
        goal_complete=False,target_feedback=False)


def parent_model(p, device):
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    model, payload = load_checkpoint(p['parent'], device)
    if (payload['arm'], payload['epoch'], payload['updates'], payload['data_contract'],
        payload['training_F_access'], payload['training_TX_labels_read'], payload['target_training']) != (
            'stage2', 20, 4940, 'original7/63/30', False, 0, False):
        raise ValueError('Matching source-final default DURA provenance required')
    if sum(v.numel() for v in model.parameters()) != 266951:
        raise ValueError('Only original full two-stage DURA allowed')
    return model, payload


def save_model(path, model, arm, epoch, updates, commit):
    from experiments.pamf_r65_four_route.runtime import validate_aliases
    validate_aliases(model)
    value=dict(schema='dura_direct_joint_v1', arm=arm, epoch=epoch, updates=updates,
        code_commit=commit, config=configuration(), source_contract='original7/63/30',
        parent_run=None, parent_arm=None, initialization='scratch_no_checkpoint', training_TX_labels_read=0,
        training_F_access=False, V_training=False, target_training=False, model=model.state_dict())
    with path.open('xb') as stream:
        torch.save(value, stream)


def load_model(path, device, *, arm, commit, epoch=20):
    value=torch.load(path, map_location='cpu', weights_only=False)
    expected=dict(schema='dura_direct_joint_v1', arm=arm, epoch=epoch, updates=epoch*247,
        code_commit=commit, config=configuration(), source_contract='original7/63/30',
        parent_run=None, parent_arm=None, initialization='scratch_no_checkpoint', training_TX_labels_read=0,
        training_F_access=False, V_training=False, target_training=False)
    if any(value.get(key)!=item for key,item in expected.items()):
        raise ValueError('This run fixed-source checkpoint required')
    from experiments.pamf_r65_four_route.runtime import validate_aliases
    model=scratch_model(2026092701)
    validate_aliases(model,value['model'])
    model.load_state_dict(value['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)


def source_training(p):
    done=read_json(p['cache']/'complete.json')
    if (done.get('status'),done.get('train_count'),done.get('TX_labels_read'),done.get('target_reads')) != ('ACTION_SOURCE_CACHE_COMPLETE',63000,0,0):
        raise ValueError('Reuse original complete source cache required')
    clean=np.load(p['cache']/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(p['cache']/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or clean.dtype!=np.float32 or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Original TX-hidden source roles and IQ required')
    return clean,roles


def smoke(args,p):
    clean,roles=source_training(p)
    take=np.flatnonzero(roles=='L_s')[:2]
    folder=p['cache']/'epochs/01'
    def take_array(key):
        return tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu')
    x=tensor(clean[take],'cpu');y=take_array('received')
    action=supervised_action(take_array('kernel'),take_array('coefficients'))
    noise=take_array('variance')
    student=scratch_model(2026092701)
    if student.initialization_report.get('clean_identity_initialization'):
        with torch.no_grad():
            torch.testing.assert_close(student(x),x,atol=2e-6,rtol=2e-6)
    with torch.no_grad():expected=student(y).clone()
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms=direct_objective(student,y,x,action,noise,probes)
    loss.mean().backward()
    grads=[value.grad for value in student.parameters() if value.grad is not None]
    if not grads or not all(torch.isfinite(v).all() for v in grads) or not any(v.abs().sum()>0 for v in grads):
        raise ValueError('Finite nonzero real-source direct training gradients required')
    checkpoint=p['output']/'smoke-scratch_joint.pt'
    save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(y),expected):raise ValueError('Strict randomgraph checkpoint reload changed IQ')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        logits=reader(expected)
        if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual originalF50/defaulthead noquery smoke failed')
    write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,source_initialization_checkpoint_loads=0,
        source_teacher_forwards=0,TX_labels_read=0,target_reads=0,training_F_access=False,F_updates=0,
        head_fit_updates=0,loss=float(loss.detach().mean()),initialization=student.initialization_report))
    print(json.dumps(dict(event='REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS',source_initialization='scratch')),flush=True)


def train(args,p):
    clean,roles=source_training(p)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=scratch_model(2026092701).to('cuda:0')
    before={key:value.detach().cpu().clone() for key,value in student.state_dict().items()}
    optimizer=torch.optim.AdamW(student.parameters(),lr=2e-4,weight_decay=1e-4)
    row=p['output']/args.arm;row.mkdir()
    write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    write_json(row/'resolved_config.json',configuration())
    write_json(row/'initialization.json',student.initialization_report)
    rng=np.random.default_rng(2026103004)
    updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,21):
            folder=p['cache']/'epochs'/f'{epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
            expected=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(value.shape!=expected[key] or value.dtype!=np.float32 for key,value in arrays.items()):
                raise ValueError('Fixed matching full sourceIQ/action epoch required')
            totals=dict.fromkeys(('loss','iq_nmse','spectral_error','prior_nmse','action_probe_nmse','log_variance_mse'),0.)
            clipped=0;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256]
                x=tensor(clean[take],'cuda:0')
                values={key:tensor(value[take],'cuda:0') for key,value in arrays.items()}
                action=supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates)
                loss,terms=direct_objective(student,values['received'],x,action,values['variance'],probes)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source direct objective')
                loss.mean().backward()
                norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,clip_fraction=clipped/247,seconds=time.perf_counter()-started,
                F_inputs=0,TX_labels_read=0,learned_teacher_inputs=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(event='SOURCE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Incomplete fixed source budget')
    changed=sum(not torch.equal(value.detach().cpu(),before[key]) for key,value in student.state_dict().items())
    if not changed:raise ValueError('Random initialized model did not train')
    save_model(row/'final.pt',student,args.arm,20,updates,args.commit)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=20,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_updates=0,learned_teacher_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,
        cumulative_epochs=20,head_fit_updates=0,parameters=266951,changed_state_tensors=changed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started))


def command(args,mode,arm=None,role=None):
    result=[sys.executable,'-u','-m','experiments.pamf_dura_direct_train.runner',mode,
            '--root',str(args.root),'--commit',args.commit]
    if arm is not None:result+=['--arm',arm]
    if role is not None:result+=['--role',role]
    return result


def worker(args,p):
    try:
        train(args,p)
        import gc
        gc.collect();torch.cuda.empty_cache()
        from .evaluation import predict,profile
        profile(args,p)
        for role in ('source','target'):
            predict(args,p,role)
            # Independent truth-side process; training state is already gone.
            import fcntl
            with (p['output']/'score.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                subprocess.run(command(args,'score',args.arm,role),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(p['output']/args.arm/'complete.json',dict(status='ANALYZED',all_seven_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        row=p['output']/args.arm
        if row.exists():write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True))
        raise


def capacity():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    result={}
    for line in gpu.strip().splitlines():
        index,uuid,free=[v.strip() for v in line.split(',')]
        result[int(index)]=dict(uuid=uuid,free_mb=int(free),pids=set())
    for line in procs.strip().splitlines():
        if ',' not in line:continue
        uuid,pid=[v.strip() for v in line.split(',')]
        for value in result.values():
            if value['uuid']==uuid:value['pids'].add(int(pid))
    return result


def dispatch(args,p):
    import fcntl
    if p['output'].exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable root and CPU-only launch controller required')
    p['output'].mkdir();(p['output']/'logs').mkdir()
    write_json(p['output']/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,owner='codex/root/dura-direct-single-stage'))
    write_json(p['output']/'plan.json',configuration())
    smoke(args,p)
    pending=list(ARMS);active={};failed=[];last=0.
    while pending or active:
        for arm,(process,gpu) in list(active.items()):
            if process.poll() is None:continue
            if process.returncode or not (p['output']/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
            del active[arm]
        with (args.root/'runs/receiver_residual_capacity4.lock').open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            state=capacity();used={gpu for _,gpu in active.values()}
            free=[gpu for gpu,row in state.items() if gpu in range(8) and gpu not in used and row['free_mb']>=12000 and len(row['pids'])<2]
            free.sort(key=lambda gpu:-state[gpu]['free_mb'])
            commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
            tokens=('pamf_dura_prior_refine.runner worker','pamf_dura_prior_recovery.runner worker',
                'pamf_dura_direct_train.runner worker','pamf_dura_direct_train.fast_lr worker',
                'pamf_dura_direct_train.identity_init worker','pamf_dura_direct_train.no_variance worker',
                'pamf_dura_direct_train.fast_interp worker','pamf_dura_direct_train.strong_action worker',
                'pamf_dura_direct_train.f_guided worker','pamf_dura_direct_train.head_guided worker',
                'pamf_dura_efficiency.gpu_profile worker',
                'pamf_dura_efficiency.interp_profile worker')
            goal_workers=sum(any(name in line for name in tokens) for line in commands)
            for line in commands:
                if not any(name in line for name in tokens):continue
                pid=line.split(None,1)[0]
                try:environment=(Path('/proc')/pid/'environ').read_bytes().split(b'\0')
                except FileNotFoundError:continue
                for item in environment:
                    if item.startswith(b'CUDA_VISIBLE_DEVICES='):
                        used.update(int(v) for v in item.split(b'=',1)[1].split(b',') if v.isdigit())
            free=[gpu for gpu in free if gpu not in used]
            if goal_workers>=2:free=[]
            while pending and free and len(active)<1:
                arm=pending.pop(0);gpu=free.pop(0)
                with (p['output']/'logs'/(arm+'.log')).open('xb') as log:
                    process=subprocess.Popen(command(args,'worker',arm),cwd=Path.cwd(),
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=(process,gpu)
                write_json(p['output']/('submitted-'+arm+'.json'),dict(pid=process.pid,GPU=gpu,command=command(args,'worker',arm)))
        if time.monotonic()-last>30:
            print(json.dumps(dict(event='QUEUE_STATE',pending=pending,active={a:dict(pid=v[0].pid,GPU=v[1]) for a,v in active.items()},failed=failed)),flush=True)
            last=time.monotonic()
        if pending or active:time.sleep(5)
    write_json(p['output']/'complete.json',dict(status='FAILED_TECHNICAL' if failed else 'ANALYZED',failed=failed,target_feedback=False,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker','score'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--role',choices=('source','target'))
    args=parser.parse_args();args.root=args.root.resolve()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:
        raise ValueError('Pinned release commit required')
    if args.mode!='run' and args.arm is None:parser.error('Fixed arm required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=paths(args.root)
    if args.mode=='run':dispatch(args,p)
    elif args.mode=='worker':worker(args,p)
    else:
        if args.role is None:parser.error('Scoring role required')
        from .evaluation import score
        score(args,p,args.role)


if __name__=='__main__':main()
