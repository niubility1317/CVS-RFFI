"""Three fixed L/U restored-view interventions; existing control reused without training."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
from types import MethodType
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,VIEWS,snapshot,unchanged,source_channel
from .sig1_role_objective import METHODS,ROLES,RestoredRoles
from .sig1_repl_base import prefix,check_state

RUN='20261026-phase1-PAMF-SIG1-REPL-RestoredRoles-source-m3-r01'
RELEASE='20261026-phase1-PAMF-SIG1-REPL-RestoredRoles-source-release-r01'
SEED=2026092701
OLD_RUN='20261026-phase1-PAMF-SIG1-REPL-RXOffset-source-m3-r01'
REFERENCES=('Reference_PL_X3','Reference_REPL_SIG1','REPL_NoStyleNoRXE')
COST_ORDER=('REPL_G_L','REPL_G_U','REPL_G_LU')


def config(ctx,commit,method):
    from .sig1_repl_offset_source import config as old_config
    if method not in METHODS:raise ValueError('Exactly three role interventions required')
    c=old_config(ctx,commit,'REPL_NoStyleNoRXE');rid='F0-'+method+'-s'+str(SEED)
    for key in ('receiver_extrapolation_mode','rx_initial_features'):c.pop(key,None)
    c.update(run_id=RUN,row_id=rid,paper_method=method,algorithm_id='pamf_sig1_restored_roles_v1',
        source_context=ctx['project']+'/releases/'+RELEASE+'/context.json',code_commit=commit,
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',
        restored_roles=dict(enable_L=ROLES[method][0],enable_U=ROLES[method][1],ordinary_half_only=True,noise=False,extra_F=0))
    c['repl_row'].update(row_id=rid,method=method,group='sig1_restored_roles')
    return validate_config(c)


def validate_config(c):
    method=c.get('paper_method');rid='F0-'+str(method)+'-s'+str(SEED);r=c.get('hard_pl',{});row=c.get('repl_row',{})
    base=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid or c.get('model_seed')!=SEED
            or c.get('algorithm_id')!='pamf_sig1_restored_roles_v1' or c.get('method_id')!='pamf_t1_e50' or c.get('protocol_id')!='F0'
            or c.get('source_context')!=c['project']+'/releases/'+RELEASE+'/context.json' or c.get('num_classes')!=6
            or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3] or c.get('sig_time_views')!=1
            or c.get('regime')!='matched_sat_plain_pl' or c.get('ablation')!='F21' or c.get('U_use') is not True or c.get('U_start_epoch')!=14
            or c.get('shared_SIG1_E20')!=prefix(c['project']) or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
            or c.get('plain_PL_weight')!=6. or c.get('plain_PL_threshold')!=.95 or c.get('labeled_batch')!=128 or c.get('unlabeled_batch')!=256
            or not c.get('default_head_joint') or r.get('method')!='HPLT1U' or r.get('weight')!=6. or r.get('threshold')!=.95 or c.get('hard_pl_no_uas') is not True
            or c.get('hard_pl_mixstyle',{}).get('probability_per_level')!=0. or c.get('hard_pl_gradient_guard',{}).get('U_to_L_norm_cap')!=1.
            or c.get('hard_pl_joint_contrast',{}).get('weight')!=.05
            or any(c.get(k) for k in ('no_sigtime','own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
            or c.get('restored_roles')!=dict(enable_L=ROLES[method][0],enable_U=ROLES[method][1],ordinary_half_only=True,noise=False,extra_F=0)
            or row.get('structure')!='SIG_TIME' or row.get('L_A') is not True or row.get('L_B') is not True or row.get('U') is not True
            or row.get('blocks')!=dict(style=False,joint=True,guard=True,extrapolation=False)):
        raise ValueError('Fixed matched single-view L/U roles and original full objectives required')
    if any(k in c for k in ('truth','target_truth','target_inputs','views_root','tests','dataset_path','checkpoint','teacher_checkpoint','supervised_architecture','core_mechanisms','rx_initial_features')):
        raise ValueError('Source role training rejects target or unrelated mechanisms')
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset);parent=prefix(c['project']) if offset==20 else base+'/segment30'
        if length is None or c['segment_epochs']!=length or c.get('step_offset')!=offset*222 or c.get('parent_source')!=parent or c['output_root']!=base+'/segment'+str(offset+length):
            raise ValueError('Own immutable E20+10+20 segments required')
    elif c['output_root']!=base or any(k in c for k in ('parent_source','epoch_offset','step_offset')):raise ValueError('One role row root required')
    return c


@contextmanager
def installed(c,holder):
    import torch
    from experiments.cric_dura_comparison import t1_model
    from experiments.pamf_repl_completion.objectives import loader_scope,portable_scope
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .sig1_model import install_single_view
    validate_config(c);contract=read(c['source_contract']);g=load_G(c['source_G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    warm=SourceDataset(c['source_cache'],'L_s',contract).x[:2].to('cuda:0')
    with torch.no_grad():engine(warm.clone())
    original,plain=t1_model.install,t1_model.plain_pl
    def build(model,cfg):
        validate_config(cfg);model=install_single_view(original(model,cfg),cfg['model_seed']);obj=RestoredRoles(cfg,g,engine);holder.update(model=model,objective=obj)
        model._feature_learning_report.update(sig_time_views=1,source_U_IQ_used=True,supervised_architecture='SIG_TIME',restored_roles=cfg['restored_roles'])
        old_hook=model._source_labeled_feature_hook
        def hook(self,**kwargs):
            if self is not kwargs['model'] or not self.training or kwargs['labeled_count']!=128 or kwargs['optimizer_step']!=self._feature_l_steps:raise ValueError('Ordered actual L128 source context required')
            with obj.labeled(self,kwargs):value,metrics=old_hook(**kwargs)
            value=obj.guide(kwargs['identity_loss']+value,self)-kwargs['identity_loss']
            return value,dict(metrics,**obj.metrics)
        hook._accepts_labeled_inputs=True;model._source_labeled_feature_hook=MethodType(hook,model);return model
    t1_model.install=build;t1_model.plain_pl=lambda model,device:holder['objective'].loss(model,device)
    try:
        with loader_scope(),portable_scope():yield
    finally:t1_model.install,t1_model.plain_pl=original,plain


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
            or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
            or rt['context']!=rt['release_root']+'/context.json' or rt['runtime_file']!=rt['release_root']+'/runtime.json'
            or rt['allowed_gpus']!=list(range(8)) or rt['launch_owner']!='codex-pamf-sig1-restored-roles-source-20261026'):
        raise ValueError('One immutable source-only new-candidate run required')



def load_parent(c):
    import torch
    from experiments.pamf_repl_completion.source import parent
    validate_config(c);done=parent(c,c['shared_SIG1_E20']);cp=torch.load(Path(done['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
    check_state(cp)
    if (done['epoch']!=20 or done['updates']!=4440 or done['initialization']!='scratch' or done['ancestors']!=[]
            or read(c['source_contract'])!=read(c['expected_source_contract']) or cp['checkpoint_selection']!='final_only'
            or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']):raise ValueError('Known completed single-view ordinaryPL E20 source ancestry required')
    return done,cp



def smoke(rt,configs):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    _,cp=load_parent(configs[0]);rows=[]
    for c in configs:
        cfg=dict(c,output_root=c['output_root']+'/segment30',segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_SIG1_E20'],candidate_smoke=True)
        holder={}
        with installed(cfg,holder):
            ctx=f_train.context(cfg)
            with f_train.class_dimension(cfg),native_installed(f_train.original_config(SEED)) as native:
                args=f_train.arguments(cfg,'cuda:0');args.input_len=256;args.baseline_ckpt=args.teacher_ckpt='';args.from_scratch=args.a1_scratch_only=True
                model=t1_model.install(native.build_baseline_model(args,'cuda:0'),cfg).to('cuda:0')
            model.load_state_dict(cp['model'],strict=True);model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,d=[v.to('cuda:0') for v in batch[:3]];model.train();model._feature_l_steps=0
            with model._phase1_labeled_feature_context():out=model(x,return_aux=True,domain_labels=d)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,extra=(None,batch[3]),domains=d,
                optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();obj=holder['objective'];state=obj.state_dict();check_state(dict(model=model.state_dict()))
            if (not bool(torch.isfinite(total)) or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or state['calls']!=1
                    or state['seen']!=256 or state['student_views']!=1536 or state['accepted']<=0 or any(state['U_style']['applied'].values())
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters())):
                raise ValueError('Actual single-view new REPL full-L/U joint backward smoke failed')
            roles=state['restored_roles'];l,u=ROLES[c['paper_method']]
            if roles['L_replaced']!=128*int(l) or roles['U_replaced']!=256*int(u) or roles['G_packets']!=128*int(l)+256*int(u) or not roles['source_G_unchanged']:
                raise ValueError('Actual role replacement counts or frozen G differ')
            if set(obj.last_diversity)!=({v for v,on in [('L',l),('U',u)] if on}):raise ValueError('Actual independent G views required')
            rows.append(dict(method=c['paper_method'],status='PASS',U_seen=256,U_accepted=state['accepted'],U_student_views=1536,L_packets=768,
                style_applied=state['U_style']['applied'],actual_U_gradient_norms=obj.smoke_U_gradient_norms,
                restored_roles=roles,actual_G_pair_relative_differences=obj.last_diversity,optimizer_updates=0,U_truth_reads=0,target_reads=0))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',rows=rows,actual_completed_SIG1_E20=True,optimizer_updates=0,target_reads=0))



def train(c):
    import shutil
    import torch
    from experiments.cric_dura_comparison.f_train import train as segment
    validate_config(c);old,cp=load_parent(c);del cp;root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    chain=[old];previous=c['shared_SIG1_E20'];states=[]
    for epochs,offset in ((10,20),(20,30)):
        folder=root/('segment'+str(offset+epochs));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        holder={}
        with installed(cfg,holder):segment(cfg)
        obj=holder['objective'];state=obj.state_dict();done=read(folder/'complete.json');payload=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(payload)
        if (state['calls']!=epochs*222 or state['seen']!=epochs*56700 or state['student_views']!=6*state['seen'] or state['EMA_updates'] or state['U_truth_read']
                or any(state['U_style']['applied'].values()) or done['epoch']!=epochs or done['updates']!=epochs*222 or done['target_access'] or done['U_truth_read']
                or payload['args']['baseline_ckpt']!=previous+'/final_ssdg.pth' or payload['args']['teacher_ckpt'] or payload['args']['use_unlabeled'] or payload['args']['use_ema_teacher']):
            raise ValueError('Fixed single-view E20+10+20/U6/default-head candidate execution differs')
        role_state=state['restored_roles'];l,u=ROLES[c['paper_method']]
        if (role_state['L_calls']!=epochs*222 or role_state['L_replaced']!=epochs*222*128*int(l)
                or role_state['U_replaced']!=epochs*56700*int(u) or not role_state['source_G_unchanged'] or role_state['extra_F_forwards']):
            raise ValueError('Exact restored role exposure, unchanged F budget and frozen G required')
        write(folder/'repl_state.json',state);states.append(state);chain.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,updates=(offset+epochs)*222,states=len(states),target_reads=0),exclusive=False)
    if len(chain)!=3 or sum(v['updates'] for v in chain)!=11100:raise ValueError('Original cumulative E50 budget required')
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=chain,
        source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in chain],
        target_access=False,U_truth_read=False,head_fit_updates=0,signature_views=1,signature_parameters=18976,actual_REPL_states=states,
        own_incremental_epochs=30,own_incremental_updates=6660,source_diagnostic_only=True))



def evaluate(c):
    import numpy as np
    import torch
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    ctx=read(c['context']);validate_context(ctx);source=Path(c['source_output']);done=read(source/'complete.json');cfg=done['config'];contract=read(cfg['source_contract'])
    expected=read(next(v['expected_source_contract'] for v in ctx['cases'] if v['model_seed']==SEED))
    if contract!=expected or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']:raise ValueError('Frozen completed source-only model and original contract required')
    model,_=load_reader(source,contract,'cuda:0');g=load_G(ctx['G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    views=len(model.sequence_pool.views)
    if views!=c['signature_views'] or tuple(model.core.id_backbone.cls_head.weight.shape)!=(6,160):raise ValueError('Declared completed-model structure differs')
    data=SourceDataset(cfg['source_cache'],'V',contract);n=len(data);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    if n!=27000:raise ValueError('Complete original source V required')
    logits=np.empty((8,n,6),np.float32);offset=0
    with torch.inference_mode():engine(data.x[:2].to('cuda:0'))
    before=snapshot(model);gbefore=snapshot(g)
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');inputs=[x,engine(x.clone())[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract);inputs.extend((h,engine(h.clone())[0].clone()))
            for k,value in enumerate(inputs):
                z=model(value)
                if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite original default-head source predictions required')
                logits[k,offset:offset+len(x)]=fp32_copy(z)
            offset+=len(x);write(out/'progress.json',dict(status='SOURCE_VALIDATING',seen=offset,total=n,target_reads=0),exclusive=False)
    if offset!=n or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Complete frozen model/source coverage required')
    y=data.metadata['y'];pred=logits.argmax(-1);strata=[('overall','ALL',np.ones(n,bool))]
    for dim,values in [('TX',y),('RX',data.metadata['rx_i']),('day',data.metadata['day_i'])]:strata.extend((dim,str(v),values==v) for v in np.unique(values))
    rows=[dict(method=c['method'],view=view,dimension=dim,stratum=s,n=int(mask.sum()),correct=int((pred[k,mask]==y[mask]).sum()),accuracy=100*float((pred[k,mask]==y[mask]).mean())) for k,view in enumerate(VIEWS) for dim,s,mask in strata]
    np.savez(out/'source_V_predictions.npz',logits=logits,y=y,ids=data.metadata['ids'],views=np.asarray(VIEWS))
    write(out/'complete.json',dict(status='SIG1_REPL_SOURCE_EVALUATED',method=c['method'],source_output=c['source_output'],signature_views=views,source_V_seen=n,
        results=rows,parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_reads=0,target_reads=0,
        F_logical_packets=216000,G_logical_packets=108002,reference_model_inference_only=c['method'] in REFERENCES))



def source_score(done):
    if (done.get('status')!='SIG1_REPL_SOURCE_EVALUATED' or done.get('source_V_seen')!=27000
            or not done.get('parameters_and_registered_buffers_unchanged') or done.get('optimizer_updates')!=0
            or done.get('U_truth_reads')!=0 or done.get('target_reads')!=0):raise ValueError('Complete frozen source-only evaluation required')
    rows={r['view']:r for r in done['results'] if r['dimension']=='overall'}
    if set(rows)!=set(VIEWS) or any(r['n']!=27000 for r in rows.values()):raise ValueError('Original full source eight views required')
    return dict(raw_correct=rows['raw']['correct'],H3_correct=sum(rows[s]['correct'] for s in SCENES),G3_correct=sum(rows['G_'+s]['correct'] for s in SCENES))


def choose_source(scores):
    maxima={key:max(scores[m][key] for m in REFERENCES) for key in ('raw_correct','H3_correct','G3_correct')};decisions=[]
    for method in METHODS:
        s=scores[method];passed=s['raw_correct']>=maxima['raw_correct']-27 and s['H3_correct']>=maxima['H3_correct']-81 and s['G3_correct']>=maxima['G3_correct']+243
        decisions.append(dict(method=method,**s,source_pass=passed,delta_raw=100*(s['raw_correct']-maxima['raw_correct'])/27000,
            delta_H3=100*(s['H3_correct']-maxima['H3_correct'])/81000,delta_G3=100*(s['G3_correct']-maxima['G3_correct'])/81000))
    accepted=[r for r in decisions if r['source_pass']]
    chosen=max(accepted,key=lambda r:(r['G3_correct'],r['raw_correct'],r['H3_correct'],-COST_ORDER.index(r['method'])))['method'] if accepted else None
    return decisions,chosen


def prepare(rt):
    ctx=read(rt['context']);validate_context(ctx);root=Path(rt['run_root']);jobs=[];refs={}
    expected=read(next(v['expected_source_contract'] for v in ctx['cases'] if v['model_seed']==SEED))
    for method in REFERENCES:
        path=Path(rt['project'])/'runs'/OLD_RUN/'evaluations'/method/'complete.json';done=read(path)
        source=read(Path(done['source_output'])/'complete.json')
        if (source['status']!='SOURCE_FROZEN' or source['epoch']!=50 or source['updates']!=11100 or source['target_access']
                or source['U_truth_read'] or source['head_fit_updates'] or read(source['config']['source_contract'])!=expected):raise ValueError('Existing reference source contract differs')
        refs[method]=dict(path=str(path),score=source_score(done))
    write(root/'reference_summaries.json',refs)
    for method in METHODS:
        c=config(ctx,rt['code_commit'],method);path=root/'configs'/(method+'.json');write(path,c)
        jobs.append(dict(id=method,kind='train_eval',config=str(path),completion=str(root/'evaluations'/method/'complete.json'),successful_statuses=['SIG1_REPL_SOURCE_EVALUATED'],
            dependencies=[str(root/'smoke.json')],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.sig1_role_training'))
    path=root/'configs/smoke.json';write(path,dict(runtime=rt,train_configs=[j['config'] for j in jobs]))
    smoke_job=dict(id='smoke',kind='smoke',config=str(path),completion=str(root/'smoke.json'),successful_statuses=['PASS'],dependencies=[],gpu_pool=list(range(8)),
        execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.sig1_role_training')
    jobs=[smoke_job]+jobs;write(root/'jobs.json',jobs);return jobs


def analyze(rt):
    root=Path(rt['run_root']);scores={k:v['score'] for k,v in read(root/'reference_summaries.json').items()}
    for method in METHODS:scores[method]=source_score(read(root/'evaluations'/method/'complete.json'))
    decisions,chosen=choose_source(scores)
    write(root/'complete.json',dict(status='SIG1_RESTORED_ROLES_SOURCE_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,
        source_scores=scores,decisions=decisions,source_selected=chosen,U_truth_reads=0,target_reads=0,scope='Single-seed source falsification,not target or three-seed success;existing controls reused'))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt)
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop=stop or not good;del active[key]
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            available=[]
            for gpu in job['gpu_pool']:
                pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
                if load<2 and free>=10000:available.append((load,-free,gpu))
            if not available:continue
            gpu=min(available)[2];child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:analyze(rt)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'SIG1_RESTORED_ROLES_SOURCE_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    from experiments.pamf_repl_completion.runner import activate
    activate(c.get('runtime',c))
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='smoke':return smoke(c['runtime'],[read(p) for p in c['train_configs']])
    if kind=='evaluate':return evaluate(c)
    if kind=='train_eval':
        train(c)
        return evaluate(dict(method=c['paper_method'],context=c['source_context'],source_output=c['output_root'],signature_views=1,
            output_root=c['project']+'/runs/'+RUN+'/evaluations/'+c['paper_method'],frozen_sources=c['frozen_sources']))
    raise ValueError('Only new-candidate training or frozen source evaluation required')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
