"""One frozen E20 L/V diagnostic; no training, U IQ/truth, target or restoration."""
import argparse,ctypes,os,time
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read,write
from . import receiver_balance_source as parent
from . import rx_consensus as method

RUN='20261027-diagnostic-PAMF-SIG1-RXConsensus-m1-r01'
RELEASE='20261027-diagnostic-PAMF-SIG1-RXConsensus-release-r01'
OWNER='codex-pamf-rx-consensus-source-20261027'


def validate_context(c):return parent.validate_context(c)


def validate_runtime(rt):
    expected={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
    if (set(rt)!=expected or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN
        or rt['release_root']!=rt['project']+'/releases/'+RELEASE or rt['context']!=rt['release_root']+'/context.json'
        or rt['runtime_file']!=rt['release_root']+'/runtime.json' or rt['launch_owner']!=OWNER
        or rt['allowed_gpus']!=list(range(8)) or set(rt['frozen_sources'])!={'F','AF','BASE','FOLD'}):
        raise ValueError('One immutable source L/V diagnostic with four F code roots only')


def role_input(c,contract,role):
    if role not in ('L_s','V'):raise ValueError('This diagnostic never reads U or target IQ')
    from experiments.cric_dura_comparison.source_iq import aligned_source_iq
    with np.load(Path(c['source_cache'])/(role+'_metadata.npz'),allow_pickle=False) as f:
        allowed={'rx_i','day_i','eq_i','sig_i','base_index','domain','ids','y'}
        if set(f.files)!=allowed:raise ValueError('Original source metadata required')
        ids=f['ids'].copy();rx=f['rx_i'].copy()
        # Never request the V y array in the GPU prediction process.
        labels=f['y'].copy() if role=='L_s' else None
    if ids.tolist()!=contract['role_ids'][role] or len(ids)!=contract['counts'][role]:raise ValueError('Physical source role/order differs')
    x,binding=aligned_source_iq(c['source_cache'],role,ids)
    if x.shape!=(len(ids),2,256):raise ValueError('Original clean source IQ shape differs')
    return x,ids,rx,labels,binding


def copy_fp32(t):
    import torch
    if t.dtype!=torch.float32:raise ValueError('Exact FP32 features/logits required')
    host=t.detach().cpu().contiguous()
    return np.frombuffer(ctypes.string_at(host.data_ptr(),host.numel()*4),dtype=np.float32).reshape(tuple(host.shape))


def extract(rt,device='cuda:0'):
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx)
    c=parent.template(ctx,rt['code_commit'],'REPL_RB_RXTX') # Recipe metadata only; no failed E50 weights.
    parent.base.activate(c)
    import torch
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.cric_dura_comparison.io import tensor
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done,payload=parent.base.load_parent(c,parent.validate_config);contract=read(c['source_contract'])
    if contract['counts']!={'L_s':6300,'U_s':56700,'V':27000} or contract['source_rxs']!=[1,3,4,6,8]:raise ValueError('Original F0 roles only')
    pc=done['config']
    with reader_structure(pc):reader=reader_from_payload(pc,payload,device)
    del payload
    pool=reader.sequence_pool;head=reader.core.id_backbone.cls_head.weight
    if len(pool.views)!=1 or pool.arm!='SIG_TIME' or pool.modes!=('SIG_TIME',) or tuple(head.shape)!=(6,160):raise ValueError('Original full PAMF/single SIG_TIME/default joint head required')
    root=Path(rt['run_root']);started=time.monotonic();lx,lids,lrx,ly,lbinding=role_input(c,contract,'L_s')
    before=parent.base.snapshot(reader)
    with torch.inference_mode():
        z=reader.features(tensor(lx[:2],device));q=reader.classify_features(z)
    if z.shape!=(2,160) or q.shape!=(2,6) or not bool(torch.isfinite(z).all() and torch.isfinite(q).all()) or not parent.base.unchanged(reader,before):
        raise ValueError('Actual E20 checkpoint no-query smoke failed')
    write(root/'smoke.json',dict(status='PASS',checkpoint_source=c['shared_SIG1_E20'],epoch=20,updates=4440,signature_views=1,
        default_head=[6,160],L_packets=2,optimizer_updates=0,U_reads=0,target_reads=0,V_truth_reads=0))
    print('REAL_CHECKPOINT_SMOKE_PASS; extracting L_s then V',flush=True)
    values={}
    for role,inputs in [('L_s',(lx,lids,lrx,ly,lbinding)),('V',role_input(c,contract,'V'))]:
        x,ids,rx,y,binding=inputs;features=np.empty((len(ids),160),np.float32);logits=np.empty((len(ids),6),np.float32)
        for start in range(0,len(ids),256):
            stop=min(start+256,len(ids))
            with torch.inference_mode():
                z=reader.features(tensor(x[start:stop],device));q=reader.classify_features(z)
            features[start:stop]=copy_fp32(z);logits[start:stop]=copy_fp32(q)
            if not np.isfinite(features[start:stop]).all() or not np.isfinite(logits[start:stop]).all():raise ValueError('Nonfinite frozen output')
            if start%4096==0 or stop==len(ids):
                write(root/'progress.json',dict(status='EXTRACTING',role=role,complete=stop,total=len(ids),V_truth_reads=0,target_reads=0),exclusive=False)
                print(role,stop,'/',len(ids),flush=True)
        values[role]=(method.unit(features),logits,ids,rx,y,binding)
    if not parent.base.unchanged(reader,before):raise ValueError('Frozen model parameters/buffers changed')
    l,v=values['L_s'],values['V'];centers,counts=method.bank(l[0],l[4],l[3],c['source_rxs'])
    pred=method.decisions(v[0],v[1],v[3],centers,c['source_rxs'])
    np.savez(root/'L_features.npz',features=l[0],ids=l[2],rx=l[3],labels=l[4],prototypes=centers,counts=counts)
    np.savez(root/'V_predictions.npz',features=v[0],logits=v[1],ids=v[2],rx=v[3],**pred)
    write(root/'predictions_complete.json',dict(status='RX_CONSENSUS_PREDICTIONS_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],
        L_count=6300,V_count=27000,prototype_counts=counts.tolist(),signature_views=1,head_shape=[6,160],
        model_unchanged=True,optimizer_updates=0,U_reads=0,target_reads=0,V_truth_reads=0,seconds=time.monotonic()-started,
        source_bindings={'L_s':l[5],'V':v[5]}))


def score_source(c):
    if set(c)!={'run_root','source_cache','source_contract'} or Path(c['run_root']).name!=RUN:raise ValueError('Separate source scorer without F/G/target handles required')
    root=Path(c['run_root']);done=read(root/'predictions_complete.json');contract=read(c['source_contract'])
    if done['status']!='RX_CONSENSUS_PREDICTIONS_COMPLETE' or done['V_truth_reads'] or done['target_reads'] or not done['model_unchanged']:raise ValueError('Complete frozen source predictions before V truth')
    with np.load(root/'V_predictions.npz',allow_pickle=False) as f:pred={k:f[k].copy() for k in f.files}
    if pred['ids'].tolist()!=contract['role_ids']['V'] or len(pred['logits'])!=27000:raise ValueError('Complete original V physical order required')
    with np.load(Path(c['source_cache'])/'V_metadata.npz',allow_pickle=False) as f:
        if f['ids'].tolist()!=contract['role_ids']['V']:raise ValueError('Source scorer physical IDs differ')
        y=f['y'].copy();rx=f['rx_i'].copy();day=f['day_i'].copy()
    result=method.score(pred,y);strata=[]
    for kind,groups in [('RX',rx),('day',day),('TX',y)]:
        for group in np.unique(groups):
            mask=groups==group
            for rule in method.RULES:
                extra=pred[rule]&mask;n=int(extra.sum());ok=int((extra&(pred['consensus_label']==y)).sum())
                strata.append(dict(kind=kind,group=int(group),rule=rule,count=int(mask.sum()),extra_accepted=n,extra_correct=ok,
                    extra_precision_percent=100*ok/n if n else None))
    write(root/'strata.json',strata)
    write(root/'complete.json',dict(status='RX_CONSENSUS_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=done['code_commit'],
        model_seed=2026092701,checkpoint_epoch=20,checkpoint_updates=4440,optimizer_updates=0,new_model_training=0,
        U_IQ_reads=0,U_truth_reads=0,target_reads=0,result=result))
    print(result,flush=True)


def dispatch(rt):
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx);c=parent.template(ctx,rt['code_commit'],'REPL_RB_RXTX')
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    configs=root/'configs';configs.mkdir();write(configs/'extract.json',rt)
    write(configs/'score.json',dict(run_root=rt['run_root'],source_cache=c['source_cache'],source_contract=c['source_contract']))
    common=dict(execution_release=rt['release_root'],execution_python=rt['python'],entrypoint=__name__ if __name__!='__main__' else 'experiments.pamf_channel_pl.rx_consensus_source')
    jobs=[dict(common,id='extract',kind='extract',config=str(configs/'extract.json'),completion=str(root/'predictions_complete.json'),successful_statuses=['RX_CONSENSUS_PREDICTIONS_COMPLETE']),
        dict(common,id='score',kind='score',config=str(configs/'score.json'),completion=str(root/'complete.json'),successful_statuses=['RX_CONSENSUS_DIAGNOSTIC_COMPLETE'])]
    write(root/'jobs.json',jobs);state=dict(status='WAITING_FOR_CAPACITY',pid=os.getpid(),code_commit=rt['code_commit'],jobs={})
    for job in jobs:
        gpu=None
        if job['kind']=='extract':
            while True:
                available=[];pending=processes()
                for index in rt['allowed_gpus']:
                    pids,free=capacity(index);load=len(pids|{pid for pid,g in pending.items() if g==index})
                    if load<2 and free>=10000:available.append((load,-free,index))
                if available:gpu=min(available)[2];break
                write(root/'pipeline_state.json',state,exclusive=False);time.sleep(12)
        else:
            while foreign_CPU_score(set()):
                state['status']='WAITING_FOR_CPU_SCORER';write(root/'pipeline_state.json',state,exclusive=False);time.sleep(12)
        child,item=launch(rt,job,gpu);state['jobs'][job['id']]=item;state['status']='RUNNING'
        write(root/'pipeline_state.json',state,exclusive=False)
        code=child.wait();good=code==0 and done(job)
        state['jobs'][job['id']]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)
        if not good:state['status']='STOPPED_FOR_TECHNICAL_FAILURE';break
    else:state['status']='RX_CONSENSUS_DIAGNOSTIC_COMPLETE'
    state['updated_at']=time.time();write(root/'pipeline_state.json',state,exclusive=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--config',required=True);p.add_argument('--kind',choices=['extract','score'])
    a=p.parse_args();c=read(a.config)
    if a.action=='dispatch':dispatch(c)
    elif a.kind=='extract':extract(c)
    else:score_source(c)
