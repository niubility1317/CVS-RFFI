"""Six fixed training-only RX-mean SSL rows, three seeds, direct-F formal tests."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import statistics
import time
from types import MethodType
from experiments.cric_dura_comparison.io import read,write
from .covariance_seed_masks import SEEDS,activate
from .rx_mean_objective import METHODS,RECIPE,Objective
from .rx_mean_views import RECIPE as VIEW_RECIPE

RUN='20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r01'
RELEASE='20261027-phase1-PAMF-RXMean-SIG1-threeSeed-release-r01'
BASELINE='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'
TASKS=('cross_rx','cross_both')


def validate_context(c):
    if (set(c)!={'project','python','frozen_sources','cases','channel_code','source_contract','source_cache'}
        or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'} or len(c['cases'])!=3 or {r['model_seed'] for r in c['cases']}!=set(SEEDS)):raise ValueError('Three actual source E20 parents, original handles and signal simulation only')
    from experiments.pamf_channel_pl.sig1_matched_baselines import validate_config
    for row in c['cases']:
        a=validate_config(row['baseline_config'])
        if (a['model_seed']!=row['model_seed'] or a['source_cache']!=c['source_cache'] or a['source_contract']!=c['source_contract']
            or a['paper_method']!='SIG1-PL' or 'source_G' in a or set(a['frozen_sources'])!={'F','AF','BASE','FOLD'}):raise ValueError('Own source-only matched baseline context required')
    return c


def config(ctx,commit,method,seed,alpha,calibration=False):
    validate_context(ctx)
    if seed not in SEEDS or method not in METHODS:raise ValueError('Two scopes times three seeds required')
    c=deepcopy(next(r['baseline_config'] for r in ctx['cases'] if r['model_seed']==seed));rid='F0-'+method+'-s'+str(seed)
    c.update(run_id=RUN,row_id=rid,paper_method=method,algorithm_id='pamf_sig1_rx_mean_training_v1',code_commit=commit,
        rx_mean_recipe=deepcopy(RECIPE),rx_mean_coefficient=alpha,rx_mean_calibration=calibration,channel_code=ctx['channel_code'],
        matched_baseline_source=ctx['project']+'/runs/'+BASELINE+'/F/F0-SIG1-PL-s'+str(seed)+'/source',
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source')
    c['repl_row'].update(row_id=rid,method=method,group='rx_mean_three_seed')
    return validate_train(c)


def validate_train(c):
    from experiments.pamf_channel_pl.sig1_matched_baselines import validate_config
    seed=c.get('model_seed');method=c.get('paper_method');cal=c.get('rx_mean_calibration');alpha=c.get('rx_mean_coefficient')
    rid='F0-'+str(method)+'-s'+str(seed);root=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if (seed not in SEEDS or method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid
        or c.get('algorithm_id')!='pamf_sig1_rx_mean_training_v1' or c.get('rx_mean_recipe')!=RECIPE
        or type(cal) is not bool or type(alpha) not in (int,float) or not (alpha==0 if cal else 0<alpha<=1)
        or c.get('matched_baseline_source')!=c['project']+'/runs/'+BASELINE+'/F/F0-SIG1-PL-s'+str(seed)+'/source'
        or set(c.get('frozen_sources',{}))!={'F','AF','BASE','FOLD'} or any(k in c for k in ('source_G','G','restored_cache','restored_caches'))
        or c.get('repl_row',{}).get('model_seed')!=seed or c.get('repl_row',{}).get('group')!='rx_mean_three_seed'):raise ValueError('Fixed training-only RX-mean ordinaryPL extension required')
    # Project back to the reviewed control validator for every shared invariant.
    b=deepcopy(c);bid='F0-SIG1-PL-s'+str(seed)
    b.update(run_id=BASELINE,row_id=bid,paper_method='SIG1-PL',algorithm_id='pamf_sig1_matched_baselines_v1',
        output_root=c['project']+'/runs/'+BASELINE+'/F/'+bid+'/source')
    b['repl_row'].update(row_id=bid,method='SIG1-PL',group='sig1_matched_baselines')
    for key in ('segment_epochs','epoch_offset','step_offset','parent_source','_calibration_base_override'):b.pop(key,None)
    validate_config(b)
    if '_calibration_base_override' in c and (not cal or c['_calibration_base_override'] not in ('raw','H')):raise ValueError('Calibration-only paired base override required')
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset);parent=c['shared_SIG1_E20'] if offset==20 else root+'/segment30'
        if cal or length is None or c['segment_epochs']!=length or c.get('step_offset')!=offset*222 or c.get('parent_source')!=parent or c['output_root']!=root+'/segment'+str(offset+length):raise ValueError('Own E20+10+20 trajectory required')
    elif c['output_root']!=root or any(k in c for k in ('epoch_offset','step_offset','parent_source')):raise ValueError('One immutable source row root required')
    return c


def load_prefix(c):
    from experiments.pamf_channel_pl.sig1_matched_baselines import load_prefix as load
    validate_train(c);b=read(Path(c['matched_baseline_source'])/'complete.json')['config'];b.pop('source_G',None)
    b['frozen_sources']={k:v for k,v in b['frozen_sources'].items() if k!='G'}
    if b['model_seed']!=c['model_seed'] or b['source_cache']!=c['source_cache'] or read(b['source_contract'])!=read(c['source_contract']):raise ValueError('Actual same-seed original data handles differ')
    return load(b)


@contextmanager
def installed(c,holder):
    from experiments.cric_dura_comparison import t1_model
    from experiments.pamf_repl_completion.objectives import loader_scope
    from .model import install_single_view
    original,old_pl=t1_model.install,t1_model.plain_pl
    def build(model,cfg):
        validate_train(cfg);model=install_single_view(original(model,cfg),cfg['model_seed']);obj=Objective(cfg);model._rx_mean=obj
        for name in ('history','cache','shift'):
            if name in holder:setattr(obj,name,holder[name])
        old_hook=model._source_labeled_feature_hook
        def hook(self,**kwargs):
            with obj.observe(self):
                correction,metrics=old_hook(**kwargs);extra=obj.extra(self,kwargs)
            return correction+extra,dict(metrics,**obj.last)
        hook._accepts_labeled_inputs=True;model._source_labeled_feature_hook=MethodType(hook,model)
        model._feature_learning_report.update(rx_mean_recipe=cfg['rx_mean_recipe'],sig_time_views=1)
        holder['model']=model;return model
    t1_model.install=build;t1_model.plain_pl=lambda m,d:m._rx_mean.plain(m,d) if hasattr(m,'_rx_mean') else old_pl(m,d)
    try:
        with loader_scope():yield
    finally:t1_model.install,t1_model.plain_pl=original,old_pl


def fit_bank(rt):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from .receiver_response import Response,RXS
    from .rx_mean_views import fit,transfers
    c=validate_context(read(rt['context']));contract=read(c['source_contract']);data=SourceDataset(c['source_cache'],'U_s',contract)
    if len(data)!=56700 or 'y' in data.metadata:raise ValueError('Complete hiddenU only')
    root=Path(rt['run_root'])/'mean_bank';root.mkdir(parents=True,exist_ok=False);response=Response();values=[];torch.set_num_threads(2)
    for begin in range(0,len(data),256):
        end=min(begin+256,len(data));coeff,_=response(np.asarray(data.x[begin:end].tolist(),np.float64));values.append(coeff)
        write(root/'progress.json',dict(U_seen=end,total=56700),exclusive=False);print('RX_MEAN_FIT',end,flush=True)
    bank,counts=fit(np.concatenate(values),data.metadata['rx_i'],role='U_s');_,bounds=transfers(bank)
    if set(counts)!=set(map(str,RXS)) or set(counts.values())!={11340}:raise ValueError('Five balanced RX training populations required')
    with (root/'bank.npz').open('xb') as f:np.savez(f,means=bank.means)
    write(root/'complete.json',dict(status='RX_MEAN_BANK_FROZEN',recipe=VIEW_RECIPE,source_contract=contract,counts=counts,
        U_seen=56700,model_seed=None,optimizer_updates=0,F_G_weight_reads=0,V_reads=0,target_reads=0,bounds=bounds))


def calibrate(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison.f_data import SourceDataset,FixedEpochLoader
    from .receiver_probe import gradient_summary
    ctx=validate_context(read(rt['context']));activate(ctx);seed=rt['model_seed'];root=Path(rt['output_root']);root.mkdir(parents=True,exist_ok=False)
    c=config(ctx,rt['code_commit'],METHODS[0],seed,0.,True);parent,cp=load_prefix(c);holder={};contract=read(c['source_contract'])
    l=SourceDataset(c['source_cache'],'L_s',contract);u=SourceDataset(c['source_cache'],'U_s',contract)
    L=iter(FixedEpochLoader(l,128,2026102741,steps=12,drop_last=True));U=FixedEpochLoader(u,256,2026102741,steps=12)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rows=[]
    with installed(c,holder):
        with f_train.class_dimension(c),native_installed(f_train.original_config(seed)) as native:
            args=f_train.arguments(c,'cuda:0');args.input_len=256;args.baseline_ckpt=args.teacher_ckpt='';args.from_scratch=args.a1_scratch_only=True
            model=t1_model.install(native.build_baseline_model(args,'cuda:0'),c).to('cuda:0')
        model.load_state_dict(cp['model'],strict=True);del cp;before={k:v.detach().clone() for k,v in model.state_dict().items()}
        model._cd_U_loader=U;model._cd_U_iter=iter(U);named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        for step in range(12):
            c['_calibration_base_override']='raw' if step%2==0 else 'H'
            # The native labeled context increments this counter on entry.
            batch=next(L);x,y,d=[v.to('cuda:0') for v in batch[:3]];model.train();model.zero_grad(set_to_none=True);model._feature_l_steps=4440+step
            with model._phase1_labeled_feature_context():out=model(x,return_aux=True,domain_labels=d)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,extra=(None,batch[3]),domains=d,
                optimizer_step=model._feature_l_steps,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            primary=ce+correction;g=torch.autograd.grad(primary,[p for _,p in named],retain_graph=True,allow_unused=True);obj=model._rx_mean
            if metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or metrics['T1_PL_seen']!=(step+1)*256:raise ValueError('Original fullL/ordinaryPL must execute once')
            if not torch.equal(obj.candidates[METHODS[0]],obj.candidates[METHODS[1]]):raise ValueError('Scope must change gradient only, not scalar loss')
            summaries={m:gradient_summary(obj.candidates[m],g,named) for m in METHODS}
            if any(not 0<v['norm']<float('inf') or not 0<v['base_norm']<float('inf') for v in summaries.values()):raise ValueError('Finite nonzero actual auxiliary and primary gradients required')
            ul,ll=obj.calibration_nodes;gu=torch.autograd.grad(obj.candidates[METHODS[0]],(ul,ll),retain_graph=True,allow_unused=True)
            gl=torch.autograd.grad(obj.candidates[METHODS[1]],(ul,ll),retain_graph=True,allow_unused=True)
            if gu[0] is None or gu[0].norm()<=0 or gu[1] is not None or gl[0] is None or gl[1] is None or gl[1].norm()<=0:raise ValueError('Actual U/LU gradient scope differs from declaration')
            if step==0:
                head=[a for (name,_),a in zip(named,g) if name.endswith('core.id_backbone.cls_head.weight')]
                sig=sum(float(a.detach().square().sum()) for (name,_),a in zip(named,g) if 'sequence_pool.' in name and a is not None)**.5
                if len(head)!=1 or head[0] is None or not torch.isfinite(head[0]).all() or head[0].norm()<=0 or sig<=0 or sum(p.numel() for p in model.parameters())!=863730:raise ValueError('FullPAMF actual defaulthead/SIG1 joint smoke failed')
                write(root/'smoke.json',dict(status='PASS',actual_parent=parent['folder'],actual_model_parameters=863730,L_packets=768,U=256,
                    signature_gradient_norm=sig,head_gradient_norm=float(head[0].norm()),L_scope_isolated=True,no_new_model_parameters=True,optimizer_updates=0,U_truth_read=False,target_reads=0))
            rows.append(dict(step=step+1,base=c['_calibration_base_override'],gradients=summaries));print('RX_MEAN_CALIBRATION',seed,step+1,flush=True)
        if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise ValueError('Source gradient calibration changed parameters/buffers')
    base=statistics.median(v['gradients'][METHODS[0]]['base_norm'] for v in rows)
    largest=max(statistics.median(v['gradients'][m]['norm'] for v in rows) for m in METHODS);alpha=min(1.,.1*base/largest)
    write(root/'calibration.json',dict(status='RX_MEAN_GRADIENT_CALIBRATED',model_seed=seed,coefficient=alpha,formula=RECIPE['coefficient_rule'],
        steps=rows,L_seen=1536,U_seen=3072,raw_U=1536,H_U=1536,optimizer_updates=0,V_reads=0,target_reads=0,parameters_unchanged=True))
    for method in METHODS:write(Path(rt['run_root'])/'configs'/('train__'+method+'-s'+str(seed)+'.json'),config(ctx,rt['code_commit'],method,seed,alpha))
    write(root/'complete.json',dict(status='RX_MEAN_CALIBRATED',model_seed=seed,coefficient=alpha,configs_frozen_before_training=True,optimizer_updates=0,target_reads=0))


def train(c):
    import shutil
    import torch
    from experiments.cric_dura_comparison.f_train import train as segment
    from experiments.pamf_channel_pl.sig1_matched_baselines import check_state
    validate_train(c)
    if c['rx_mean_calibration']:raise ValueError('Calibration config cannot train')
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c);start=time.monotonic()
    old,cp=load_prefix(c);del cp;chain=[old];previous=c['shared_SIG1_E20'];holder={}
    for epochs,offset in ((10,20),(20,30)):
        cfg=dict(c,output_root=str(root/('segment'+str(offset+epochs))),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        with installed(cfg,holder):segment(cfg)
        obj=holder['model']._rx_mean
        for key in ('history','cache','shift'):holder[key]=getattr(obj,key)
        done=read(Path(cfg['output_root'])/'complete.json');cp=torch.load(Path(cfg['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
        if (done['status']!='SOURCE_FROZEN' or done['updates']!=epochs*222 or done['target_access'] or done['U_truth_read']
            or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt'] or cp['args']['baseline_ckpt']!=previous+'/final_ssdg.pth'
            or cp['a1_ema_successful_updates']!=epochs*222):raise ValueError('Own complete source continuation required')
        del cp;chain.append(dict(folder=cfg['output_root'],**done));previous=cfg['output_root'];write(Path(previous)/'rx_mean_state.json',obj.history.state())
        write(root/'progress.json',dict(epoch=offset+epochs,updates=(offset+epochs)*222,seconds=time.monotonic()-start),exclusive=False)
    state=holder['history'].state();check_exposure(state)
    write(root/'rx_mean_state.json',state);shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=chain,
        source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in chain],target_access=False,
        U_truth_read=False,head_fit_updates=0,signature_views=1,signature_parameters=18976,rx_mean=state,seconds=time.monotonic()-start,
        cumulative_training_seconds=sum(v['seconds'] for v in chain)))


def check_exposure(s):
    if (s['calls']!=6660 or s['seen']!=1701000 or s['unique_seen']!=56700 or s['min_visits']!=30 or s['max_visits']!=30
        or s['raw_base_U']!=850500 or s['H_base_U']!=850500 or s['mean_shift_L_pairs']!=426240 or s['nonzero_shift_U']!=1701000):raise ValueError('AllU thirty visits,15raw/15H and nonzero shifts required')


def validate_evaluation(c,scoring=False):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    if (c.get('run_id')!=RUN or c.get('model_seed') not in SEEDS or c.get('method') not in METHODS or c.get('readouts')!=['raw'] or c.get('protocol_id')!='F0'
        or c.get('source_days')!=[1,2,3] or c.get('row_id')!='F0-'+c['method']+'-s'+str(c['model_seed']) or set(c['tests'])!=set(TASKS) or len(c['classes'])!=6
        or any(k in c for k in ('G','source_G','restored_cache','restored_caches','teacher_checkpoint','mean_bank'))):raise ValueError('Original raw-F inference only; no augmentation bank or G')
    handles={'source_output','source_contract','source_cache','frozen_sources'}
    if scoring:
        if handles&set(c):raise ValueError('Independent scorer cannot read models/source')
    else:
        reject_truth(c)
        if not handles.issubset(c) or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'}:raise ValueError('Frozen F handles only')
    return c


def check_frozen(c):
    done=read(Path(c['source_output'])/'complete.json');cfg=validate_train(done['config']);check_exposure(read(Path(c['source_output'])/'rx_mean_state.json'))
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
        or cfg['row_id']!=c['row_id'] or cfg['model_seed']!=c['model_seed'] or cfg['source_cache']!=c['source_cache'] or read(cfg['source_contract'])!=read(c['source_contract'])):raise ValueError('Own fixed final full-budget F required')
    return cfg


def prepare(rt):
    ctx=validate_context(read(rt['context']));evaluation=read(rt['evaluation_context']);root=Path(rt['run_root']);jobs=[]
    def add(key,kind,path,completion,status,deps):
        jobs.append(dict(id=key,kind=kind,config=str(path),completion=str(completion),successful_statuses=[status],dependencies=[str(v) for v in deps],
            execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.rx_mean_source'))
    # No truth paths are copied into fit/calibration/train configurations.
    clean={k:v for k,v in rt.items() if k!='evaluation_context'}
    path=root/'configs/mean_bank.json';write(path,clean);add('mean_bank','fit',path,root/'mean_bank/complete.json','RX_MEAN_BANK_FROZEN',[])
    for seed in SEEDS:
        path=root/'configs'/('calibrate-s'+str(seed)+'.json');calroot=root/'calibration'/('s'+str(seed));write(path,dict(clean,model_seed=seed,output_root=str(calroot)))
        add('calibrate-s'+str(seed),'calibrate',path,calroot/'complete.json','RX_MEAN_CALIBRATED',[root/'mean_bank/complete.json'])
        for method in METHODS:
            rid='F0-'+method+'-s'+str(seed);source=root/'F'/rid/'source';path=root/'configs'/('train__'+method+'-s'+str(seed)+'.json')
            add('train__'+method+'-s'+str(seed),'train',path,source.parent/'source_validation/complete.json','MULTISEED_SOURCE_EVALUATED',[calroot/'complete.json'])
            out=root/'readouts'/rid
            p=dict(run_id=RUN,row_id=rid,method=method,model_seed=seed,protocol_id='F0',source_days=[1,2,3],classes=evaluation['classes'],readouts=['raw'],
                tests={t:dict(views_root=evaluation['tests'][t]['views_root']) for t in TASKS},source_output=str(source),source_contract=ctx['source_contract'],
                source_cache=ctx['source_cache'],frozen_sources=ctx['frozen_sources'],code_commit=rt['code_commit'],output_root=str(out/'predictions'))
            validate_evaluation(p);path=root/'configs'/('predict__'+rid+'.json');write(path,p)
            add('predict__'+rid,'predict',path,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[source.parent/'source_validation/complete.json'])
            s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:evaluation['tests'][t] for t in TASKS})
            for k in ('source_output','source_contract','source_cache','frozen_sources'):s.pop(k)
            validate_evaluation(s,True);path=root/'configs'/('score__'+rid+'.json');write(path,s)
            add('score__'+rid,'score',path,out/'scores/complete.json','SCORED_COMPLETE',[out/'predictions/complete.json'])
    write(root/'reused_controls.json',dict(run_id=BASELINE,methods=['SIG1-PL','SIG1-Sup'],model_seeds=list(SEEDS),new_control_training=0,new_control_predictions=0))
    write(root/'jobs.json',jobs);return jobs


def worker(c,kind):
    if kind=='fit':return fit_bank(c)
    if kind=='calibrate':return calibrate(c)
    if kind=='score':
        validate_evaluation(c,True)
        from experiments.cric_dura_comparison import evaluation
        evaluation.tasks=lambda _:list(TASKS);return evaluation.score(c)
    if kind=='predict':
        from .fonly_confirm import activate_raw,predict
        activate_raw(c);return predict(c,validator=validate_evaluation,source_validator=check_frozen)
    if kind!='train':raise ValueError('Only fixed source training and independent frozen readouts')
    validate_train(c);activate(dict(frozen_sources=c['frozen_sources'],channel_code=c['channel_code']));train(c)
    from .covariance_three_seed import evaluate_source
    evaluate_source(c)


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.practical_full_2x2.cache import atomic
    from .covariance_queue import admissible,POLICY
    if rt['run_id']!=RUN or not rt.get('cpu_only') or rt.get('queue_policy')!=POLICY:raise ValueError('One immutable six-model/three-seed matrix')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs};active={};children={};stop=False
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs})
    while True:
        for key,item in list(active.items()):
            code=children[key].poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key];del children[key]
        pending=processes()
        for job in sorted(jobs,key=lambda j:dict(fit=0,score=1,predict=2,calibrate=3,train=4)[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v['pid'] for v in active.values()}):continue
            elif job['kind']!='fit':
                choices=[]
                for card in rt['allowed_gpus']:
                    pids,memory=capacity(card)
                    if admissible(card,pids,pending,active,memory):choices.append((-memory,card))
                if not choices:continue
                gpu=min(choices)[1]
            child,item=launch(rt,job,gpu);active[key]=item;children[key]=child;state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'RX_MEAN_THREE_SEED_COMPLETE';atomic(root/'pipeline_state.json',state)
    if not stop:write(root/'complete.json',dict(status='RX_MEAN_THREE_SEED_COMPLETE',methods=list(METHODS),model_seeds=list(SEEDS),models=6,raw_only=True,truth_last=True,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
