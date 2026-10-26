"""Four matched SIG1 covariance source pilots; no PL/Sup comparator training."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import statistics
import time
from types import MethodType
from experiments.cric_dura_comparison.io import read,write
from .covariance_objective import METHODS,RECIPE
from .source import prefix,check_state,SEED

RUN='20261026-phase1-PAMF-SIG1-Covariance-source-m4-r01'
RELEASE='20261026-phase1-PAMF-SIG1-Covariance-source-release-r01'
FIT_RUN='20261026-diagnostic-PAMF-SIG1-CovMask-L-s2026092701-r01'
BASELINE_RUN='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'


def config(ctx,commit,method,coefficient,calibration=False):
    from .source import config as template
    if method not in METHODS:raise ValueError('Four fixed covariance extensions only')
    c=template(ctx,commit,'PL-History');c.pop('u_history_recipe');rid='F0-'+method+'-s'+str(SEED)
    c.update(run_id=RUN,row_id=rid,paper_method=method,algorithm_id='pamf_sig1_selective_covariance_v1',
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',covariance_recipe=deepcopy(RECIPE),
        covariance_masks=ctx['project']+'/runs/'+RUN+'/mask_preparation/masks.pt',
        covariance_coefficient=coefficient,covariance_calibration=calibration)
    c['repl_row'].update(row_id=rid,method=method,group='sig1_covariance')
    return validate_config(c)


def validate_config(c):
    method=c.get('paper_method');row=c.get('repl_row',{});rid='F0-'+str(method)+'-s'+str(SEED)
    alpha=c.get('covariance_coefficient');cal=c.get('covariance_calibration')
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid or c.get('model_seed')!=SEED
        or c.get('algorithm_id')!='pamf_sig1_selective_covariance_v1' or c.get('method_id')!='pamf_t1_e50'
        or c.get('covariance_recipe')!=RECIPE or type(cal) is not bool or type(alpha) not in (int,float)
        or not (alpha==0 if cal else 0<alpha<=1.)
        or c.get('covariance_masks')!=c['project']+'/runs/'+RUN+'/mask_preparation/masks.pt'
        or c.get('num_classes')!=6 or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
        or c.get('sig_time_views')!=1 or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
        or c.get('shared_SIG1_E20')!=prefix(c['project']) or c.get('regime')!='matched_sat_plain_pl' or c.get('ablation')!='F21'
        or c.get('U_use') is not True or c.get('U_start_epoch')!=14 or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
        or c.get('labeled_batch')!=128 or c.get('unlabeled_batch')!=256 or not c.get('default_head_joint')
        or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
        or row.get('structure')!='SIG_TIME' or any(not row.get(k) for k in ('L_A','L_B','U')) or any(row.get('blocks',{}).values())):
        raise ValueError('Fixed fullPAMF SIG1 ordinaryPL covariance pilot required')
    if any(k.startswith('hard_pl') or k in ('tests','truth','target_truth','target_inputs','views_root','dataset_path','checkpoint','teacher_checkpoint','core_mechanisms','supervised_architecture') for k in c):
        raise ValueError('Source covariance worker rejects target,foreign objective or head replacement')
    base=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset)
        parent=c['shared_SIG1_E20'] if offset==20 else base+'/segment30'
        if cal or length is None or c['segment_epochs']!=length or c.get('step_offset')!=offset*222 or c.get('parent_source')!=parent or c['output_root']!=base+'/segment'+str(offset+length):
            raise ValueError('Own E20+10+20 covariance trajectory required')
    elif c['output_root']!=base or any(k in c for k in ('epoch_offset','step_offset','parent_source')):raise ValueError('One immutable covariance row root required')
    return c


def load_prefix(c):
    import torch
    validate_config(c);p=Path(c['shared_SIG1_E20']);done=read(p/'complete.json');pc=done['config'];contract=read(c['source_contract'])
    cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp);a=cp['args']
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=20 or done['updates']!=4440 or done['initialization']!='scratch'
        or done['ancestors']!=[] or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
        or read(pc['source_contract'])!=contract or contract!=read(c['expected_source_contract']) or pc['source_cache']!=c['source_cache']
        or pc['model_seed']!=SEED or pc['ablation']!='F21' or pc['repl_row']['structure']!='SIG_TIME'
        or not all(pc['repl_row'][k] for k in ('L_A','L_B','U')) or cp['run_id']!=pc['run_id'] or cp['candidate_id']!=pc['row_id']
        or cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440 or cp['checkpoint_selection']!='final_only'
        or a['use_unlabeled'] or a['use_ema_teacher'] or a['teacher_ckpt'] or a['baseline_ckpt']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: actual scratch SIG1 ordinaryPL E20 required')
    return dict(folder=str(p),**done),cp


@contextmanager
def installed(c,holder):
    from experiments.cric_dura_comparison import t1_model
    from experiments.pamf_repl_completion.objectives import loader_scope
    from .model import install_single_view
    from .covariance_objective import Objective
    original,old_pl=t1_model.install,t1_model.plain_pl
    def build(model,cfg):
        validate_config(cfg);model=install_single_view(original(model,cfg),cfg['model_seed']);obj=Objective(cfg);model._covariance=obj
        if 'exposure' in holder:obj.history=holder['exposure']
        if 'cache' in holder:obj.cache=holder['cache']
        old_hook=model._source_labeled_feature_hook
        def hook(self,**kwargs):
            with obj.observe(self):
                correction,metrics=old_hook(**kwargs);extra=obj.extra(self,kwargs)
            return correction+extra,dict(metrics,**obj.last)
        hook._accepts_labeled_inputs=True;model._source_labeled_feature_hook=MethodType(hook,model)
        model._feature_learning_report.update(covariance_recipe=cfg['covariance_recipe'],sig_time_views=1)
        holder['model']=model;return model
    t1_model.install=build;t1_model.plain_pl=lambda m,d:m._covariance.plain(m,d) if hasattr(m,'_covariance') else old_pl(m,d)
    try:
        with loader_scope():yield
    finally:t1_model.install,t1_model.plain_pl=original,old_pl


def calibration(rt):
    import torch
    from torch.nn import functional as F
    from .covariance_support import derive
    from .receiver_probe import gradient_summary
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison.f_data import SourceDataset,FixedEpochLoader
    root=Path(rt['run_root']);old=Path(rt['project'])/'runs'/FIT_RUN
    derive(old/'statistics.pt',old/'result.json',root/'mask_preparation',rt['code_commit'])
    c=config(read(rt['context']),rt['code_commit'],'CovSel-U',0.,True);_,cp=load_prefix(c);holder={}
    contract=read(c['source_contract']);l=SourceDataset(c['source_cache'],'L_s',contract);u=SourceDataset(c['source_cache'],'U_s',contract)
    L=iter(FixedEpochLoader(l,128,2026102631,steps=9,drop_last=True));U=FixedEpochLoader(u,256,2026102631,steps=9)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    rows=[]
    with installed(c,holder):
        with f_train.class_dimension(c),native_installed(f_train.original_config(SEED)) as native:
            args=f_train.arguments(c,'cuda:0');args.input_len=256;args.baseline_ckpt=args.teacher_ckpt='';args.from_scratch=args.a1_scratch_only=True
            model=t1_model.install(native.build_baseline_model(args,'cuda:0'),c).to('cuda:0')
        model.load_state_dict(cp['model'],strict=True);del cp;before={k:v.detach().clone() for k,v in model.state_dict().items()}
        model._cd_U_loader=U;model._cd_U_iter=iter(U);named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        for step in range(9):
            batch=next(L);x,y,d=[v.to('cuda:0') for v in batch[:3]];model.train();model.zero_grad(set_to_none=True);model._feature_l_steps=4440+step
            with model._phase1_labeled_feature_context():out=model(x,return_aux=True,domain_labels=d)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,extra=(None,batch[3]),domains=d,
                optimizer_step=model._feature_l_steps,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            base=ce+correction;g=torch.autograd.grad(base,[p for _,p in named],retain_graph=True,allow_unused=True)
            obj=model._covariance
            if metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or metrics['T1_PL_seen']!=(step+1)*256:
                raise ValueError('Actual fullL and ordinaryPL must execute once per source step')
            if not torch.equal(obj.candidates['CovSel-U'],obj.candidates['CovSel-LU']):raise ValueError('L gradient scope changed scalar objective')
            summaries={m:gradient_summary(obj.candidates[m],g,named) for m in METHODS}
            if any(v['norm']<=0 or not 0<v['base_norm']<float('inf') for v in summaries.values()):raise ValueError('Nonzero finite covariance/primary gradients required')
            if step==0:
                head=[a for (name,_),a in zip(named,g) if name.endswith('core.id_backbone.cls_head.weight')]
                sig=sum(float(a.detach().square().sum()) for (name,_),a in zip(named,g) if 'sequence_pool.' in name and a is not None)**.5
                level='time_down';nodes=[obj.maps['L_raw'][level],obj.maps['U_raw'][level]]
                gu=torch.autograd.grad(obj.candidates['CovSel-U'],nodes,retain_graph=True,allow_unused=True)
                gl=torch.autograd.grad(obj.candidates['CovSel-LU'],nodes,retain_graph=True,allow_unused=True)
                if (len(head)!=1 or head[0] is None or not torch.isfinite(head[0]).all() or head[0].norm()<=0 or sig<=0
                    or gu[0] is not None and bool((gu[0]!=0).any()) or gu[1] is None or gu[1].norm()<=0 or gl[0] is None or gl[0].norm()<=0):
                    raise ValueError('Actual defaulthead/SIG1/Lscope/Ucovariance smoke failed')
                write(root/'smoke.json',dict(status='PASS',actual_SIG1_E20=True,L_packets=768,U=256,head_gradient_norm=float(head[0].norm()),
                    signature_gradient_norm=sig,L_scope_isolated=True,all_four_covariance_gradients_nonzero=True,optimizer_updates=0,U_truth_read=False,target_reads=0))
            rows.append(dict(step=step+1,gradients=summaries))
            print('COVARIANCE_CALIBRATION_STEP',step+1,flush=True)
        if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise ValueError('Calibration changed model state')
    base_norm=statistics.median(v['gradients']['CovSel-U']['base_norm'] for v in rows)
    largest=max(statistics.median(v['gradients'][m]['norm'] for v in rows) for m in METHODS)
    alpha=min(1.,.1*base_norm/largest)
    write(root/'calibration.json',dict(status='COVARIANCE_CALIBRATED',coefficient=alpha,formula=RECIPE['coefficient_rule'],
        median_primary_norm=base_norm,max_method_median_covariance_norm=largest,steps=rows,L_seen=1152,U_seen=2304,
        V_reads=0,target_reads=0,optimizer_updates=0,parameters_unchanged=True))


def train(c):
    import shutil
    import torch
    from experiments.cric_dura_comparison.f_train import train as segment
    validate_config(c)
    if c['covariance_calibration']:raise ValueError('Calibration config cannot train')
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    start=time.monotonic();old,cp=load_prefix(c);del cp;chain=[old];previous=c['shared_SIG1_E20'];holder={}
    for epochs,offset in ((10,20),(20,30)):
        cfg=dict(c,output_root=str(root/('segment'+str(offset+epochs))),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        with installed(cfg,holder):segment(cfg)
        obj=holder['model']._covariance;holder['exposure']=obj.history;holder['cache']=obj.cache
        done=read(Path(cfg['output_root'])/'complete.json');cp=torch.load(Path(cfg['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
        if (done['status']!='SOURCE_FROZEN' or done['updates']!=epochs*222 or done['target_access'] or done['U_truth_read']
            or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt']
            or cp['args']['baseline_ckpt']!=previous+'/final_ssdg.pth' or cp['a1_ema_successful_updates']!=epochs*222):raise ValueError('Own complete E20+10+20 required')
        del cp;chain.append(dict(folder=cfg['output_root'],**done));previous=cfg['output_root']
        write(Path(previous)/'covariance_state.json',obj.history.state())
        write(root/'progress.json',dict(epoch=offset+epochs,updates=(offset+epochs)*222,seconds=time.monotonic()-start),exclusive=False)
    state=holder['exposure'].state()
    if state['calls']!=6660 or state['seen']!=1701000 or state['unique_seen']!=56700 or state['min_visits']!=30 or state['max_visits']!=30:raise ValueError('Complete allU covariance exposure required')
    write(root/'covariance_state.json',state);shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=chain,
        source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in chain],
        target_access=False,U_truth_read=False,head_fit_updates=0,signature_views=1,signature_parameters=18976,
        covariance=state,seconds=time.monotonic()-start,cumulative_training_seconds=sum(v['seconds'] for v in chain)))


def evaluate(c,output):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.frozen_inference import load_F,load_G
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from experiments.pamf_channel_pl.restored_source import source_channel,SCENES,VIEWS,snapshot,unchanged
    contract=read(c['source_contract']);done=read(Path(c['output_root'])/'complete.json')
    if done['config']!=c or done['epoch']!=50 or done['updates']!=11100 or done['target_access']:raise ValueError('Fixedfinal complete sourceF required')
    with reader_structure(c):model,_=load_F(c['output_root'],contract,'cuda:0')
    g=load_G(c['source_G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    data=SourceDataset(c['source_cache'],'V',contract);n=len(data)
    if n!=27000:raise ValueError('Full fixed sourceV required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);logits=np.empty((8,n,6),np.float32);offset=0
    with torch.inference_mode():engine(data.x[:2].to('cuda:0'))
    before=snapshot(model);gbefore=snapshot(g)
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');views=[x,engine(x.clone())[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract);views.extend((h,engine(h.clone())[0].clone()))
            for k,view in enumerate(views):
                p=model(view)
                if p.shape!=(len(x),6) or not torch.isfinite(p).all():raise ValueError('Finite actual defaulthead source logits required')
                logits[k,offset:offset+len(x)]=fp32_copy(p)
            offset+=len(x);write(out/'progress.json',dict(seen=offset,total=n,target_reads=0),exclusive=False)
    if not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Source evaluation changed F/G state')
    np.savez(out/'source_V_predictions.npz',logits=logits,y=data.metadata['y'],ids=data.metadata['ids'],views=np.asarray(VIEWS),rx=data.metadata['rx_i'],day=data.metadata['day_i'])
    rows=[dict(view=v,n=n,correct=int((logits[k].argmax(1)==data.metadata['y']).sum())) for k,v in enumerate(VIEWS)]
    write(out/'complete.json',dict(status='COVARIANCE_SOURCE_EVALUATED',method=c['paper_method'],model_seed=SEED,results=rows,
        source_V_seen=n,parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_read=False,target_reads=0))


def prepared_jobs(rt,alpha):
    ctx=read(rt['context']);root=Path(rt['run_root']);jobs=[]
    for method in METHODS:
        c=config(ctx,rt['code_commit'],method,alpha);path=root/'configs'/(method+'.json');write(path,c)
        jobs.append(dict(id=method,kind='train_eval',config=str(path),completion=str(Path(c['output_root']).parent/'source_validation/complete.json'),successful_statuses=['COVARIANCE_SOURCE_EVALUATED'],dependencies=[],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.covariance_source'))
    parent=Path(rt['project'])/'runs'/BASELINE_RUN/'F/F0-SIG1-PL-s2026092701/source';done=read(parent/'complete.json')
    from experiments.pamf_channel_pl.sig1_matched_baselines import validate_config as validate_baseline
    validate_baseline(done['config'])
    c=dict(baseline_config=done['config'],evaluation_output=str(root/'baseline_source'),frozen_sources=rt['frozen_sources'])
    path=root/'configs/baseline.json';write(path,c)
    jobs.append(dict(id='baseline_source',kind='baseline',config=str(path),completion=str(root/'baseline_source/complete.json'),successful_statuses=['COVARIANCE_SOURCE_EVALUATED'],dependencies=[],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.covariance_source'))
    return jobs


def summarize(rt,jobs):
    root=Path(rt['run_root']);control={r['view']:r['correct'] for r in read(root/'baseline_source/complete.json')['results']};decisions=[]
    scenes=('practical_low_suburban','practical_mid_urban','practical_low_urban')
    for job in jobs:
        if job['kind']!='train_eval':continue
        r=read(job['completion']);counts={v['view']:v['correct'] for v in r['results']};delta={k:counts[k]-control[k] for k in counts}
        passed=delta['raw']>=-27 and sum(delta[s] for s in scenes)>=0 and sum(delta['G_'+s] for s in scenes)>=0
        decisions.append(dict(method=r['method'],source_necessary_pass=passed,correct_count_delta=delta,
            scope='source-only necessary condition;not goal success or target-selected candidate'))
    write(root/'source_decisions.json',dict(status='SOURCE_PILOT_ANALYZED',decisions=decisions,matched_baseline_reused=True,baseline_training=0,target_reads=0))


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done
    from experiments.practical_full_2x2.cache import atomic
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('One fixed fourrow covariance run required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);path=root/'calibration_config.json';write(path,rt)
    first=dict(id='calibration',kind='calibration',config=str(path),completion=str(root/'calibration.json'),successful_statuses=['COVARIANCE_CALIBRATED'],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.covariance_source')
    jobs=[first];state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={'calibration':dict(status='QUEUED')});active={};expanded=False;stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            job=next(j for j in jobs if j['id']==key);good=code==0 and done(job);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key]
        if not stop and not expanded and state['jobs']['calibration']['status']=='COMPLETE':
            new=prepared_jobs(rt,read(root/'calibration.json')['coefficient']);jobs.extend(new)
            state['jobs'].update({j['id']:dict(status='QUEUED') for j in new});expanded=True;write(root/'jobs.json',jobs)
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED':continue
            free=[]
            for gpu in job['gpu_pool']:
                pids,memory=capacity(gpu)
                if not (pids|{pid for pid,g in pending.items() if g==gpu}) and memory>=10000:free.append((-memory,gpu))
            if free:
                gpu=min(free)[1];child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or expanded and all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:summarize(rt,jobs)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'COVARIANCE_SOURCE_PILOT_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='calibration':return calibration(c)
    if kind=='train_eval':train(c);return evaluate(c,Path(c['output_root']).parent/'source_validation')
    if kind=='baseline':return evaluate(c['baseline_config'],c['evaluation_output'])
    raise ValueError('Only source covariance phases allowed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
