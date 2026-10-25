"""CPU-only frozen source-V seed-consensus falsification; never trains or tests."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .plan import SEEDS,validate_context
from .source_quality import validate_prefix_headers

RUN='20261025-phase1-PAMF-PL-consensus-source-m6-r01'
PROTOCOLS=('F0','label01')


def protocol(rt,context,pid,root):
    import torch
    import numpy as np
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from .consensus import policies,summarize,THRESHOLD
    start=time.perf_counter();folder=root/pid;folder.mkdir(exist_ok=False)
    by=validate_context(context);models=[];buffers=[];smokes=[];contract=None;source_cache=None
    for seed in SEEDS:
        case=by[pid,seed];parent=Path(case['parent_source']);done=read(parent/'complete.json');c=done['config']
        expected=read(case['expected_source_contract']);actual=read(c['source_contract'])
        payload=torch.load(parent/'final_ssdg.pth',map_location='cpu',weights_only=False)
        validate_prefix_headers(case,done,payload,actual,expected)
        if contract is not None and actual!=contract:raise ValueError('Three identical source contracts required')
        contract=actual;source_cache=c['source_cache']
        model=reader_from_payload(c,payload,'cpu');del payload
        if contract['source_rxs']!=c['source_rxs'] or contract['source_days']!=[1,2,3]:raise ValueError('Source RX/day differs')
        labeled=SourceDataset(source_cache,'L_s',contract)
        with torch.no_grad():logits=model(labeled.x[:2])
        if logits.shape!=(2,c['num_classes']) or not bool(torch.isfinite(logits).all()):raise ValueError('Actual CPU source checkpoint smoke failed')
        smokes.append(dict(model_seed=seed,status='PASS',source_packets=2,query_reads=0,default_head_classes=c['num_classes']))
        models.append(model);buffers.append({k:v.detach().clone() for k,v in model.named_buffers()});del labeled
    write(folder/'smoke.json',dict(status='PASS',models=smokes,device='cpu',new_parameters=0,model_updates=0))
    validation=SourceDataset(source_cache,'V',contract)
    loader=torch.utils.data.DataLoader(validation,batch_size=256,shuffle=False,num_workers=0)
    chunks=[[] for _ in models];truth=[]
    for i,batch in enumerate(loader):
        with torch.no_grad():
            for j,model in enumerate(models):
                z=model(batch[0])
                if z.shape!=(len(batch[0]),len(contract['classes'])) or not bool(torch.isfinite(z).all()):raise ValueError('Finite original source head required')
                chunks[j].append(z.detach().clone())
        truth.append(batch[1])
        if i%5==0 or i+1==len(loader):
            write(folder/'progress.json',dict(status='SOURCE_INFERENCE',protocol=pid,batches=i+1,
                batches_total=len(loader),source_V_seen=min((i+1)*256,len(validation)),models=3,
                seconds=time.perf_counter()-start,target_reads=0,model_updates=0),exclusive=False)
            print('CPU_SOURCE_CONSENSUS',pid,i+1,len(loader),flush=True)
    logits=torch.stack([torch.cat(v) for v in chunks]);truth=torch.cat(truth);del chunks
    if len(truth)!=contract['counts']['V'] or truth.min()<0 or truth.max()>=len(contract['classes']):raise ValueError('Full labeled source V required')
    for model,before in zip(models,buffers):
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen source diagnostic must not fit parameters')
        if any(not torch.equal(v,before[k]) for k,v in model.named_buffers()):raise ValueError('Diagnostic changed source buffers')
    results=[];choices=policies(logits)
    rx=torch.tensor(validation.metadata['rx_i'].tolist());day=torch.tensor(validation.metadata['day_i'].tolist())
    for name,choice in choices.items():
        results.append(dict(policy=name,dimension='overall',stratum='ALL',**summarize(choice,truth)))
        for dimension,values in [('true_class',truth),('predicted_class',choice['pseudo']),('receiver',rx),('day',day)]:
            for label in values.unique().tolist():
                results.append(dict(policy=name,dimension=dimension,stratum=str(label),**summarize(choice,truth,values==label)))
    errors=logits.argmax(-1)!=truth;overlap=[]
    for i in range(3):
        for j in range(i+1,3):
            overlap.append(dict(seed_a=SEEDS[i],seed_b=SEEDS[j],both_wrong=int((errors[i]&errors[j]).sum()),
                only_a_wrong=int((errors[i]&~errors[j]).sum()),only_b_wrong=int((~errors[i]&errors[j]).sum())))
    # Source-only arrays remain private; no NumPy bridge is required by Torch2.1.
    np.savez(folder/'source_V_predictions.npz',logits=np.asarray(logits.tolist(),np.float32),
        y=np.asarray(truth.tolist(),np.int64),ids=validation.metadata['ids'],rx=validation.metadata['rx_i'],day=validation.metadata['day_i'])
    result=dict(status='SOURCE_CONSENSUS_COMPLETE',protocol=pid,threshold=THRESHOLD,source_V_seen=len(truth),
        model_seeds=list(SEEDS),prefix_epoch=20,ordinary_PL_prefix_U_epochs=7,results=results,error_overlap=overlap,
        all_three_wrong=int(errors.all(0).sum()),device='cpu',CPU_threads=2,model_updates=0,head_fit_updates=0,
        source_V_training_reads=0,U_truth_read=False,target_reads=0,G_reads=0,seconds=time.perf_counter()-start,
        scope='Frozen E20 source-V counterfactual only,not trained SSL or independent three-student target results.')
    write(folder/'complete.json',result);return result


def dispatch(rt):
    if (rt['run_id']!=RUN or rt['allowed_gpus']!=[] or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise ValueError('Only this fixed CPU/no-GPU diagnostic is authorized')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    torch.set_num_threads(2)
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);context=read(rt['context'])
    if any(context.get(k) for k in ('truth','target_truth','tests','target_inputs','views_root','dataset_path')):
        raise ValueError('Source-only context required')
    state=dict(status='SOURCE_CONSENSUS_RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],device='cpu',
        completed=[],target_reads=0,model_updates=0)
    write(root/'pipeline_state.json',state)
    results=[]
    for pid in PROTOCOLS:
        results.append(protocol(rt,context,pid,root));state['completed'].append(pid)
        write(root/'pipeline_state.json',state,exclusive=False)
    write(root/'complete.json',dict(status='SOURCE_CONSENSUS_COMPLETE',protocols=results,code_commit=rt['code_commit'],
        parents=6,new_models=0,model_updates=0,target_reads=0,U_truth_read=False,device='cpu'))
    state['status']='SOURCE_CONSENSUS_COMPLETE';write(root/'pipeline_state.json',state,exclusive=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch',));p.add_argument('--config',required=True)
    args=p.parse_args();dispatch(read(args.config))
