"""Matched NoSIG supervised/ordinary-PL controls and an own E30 graph teacher."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,VIEWS,snapshot,unchanged

RUN='20261026-phase1-PAMF-NoSIG-RGPL-base-source-m2-r01'
RELEASE='20261026-phase1-PAMF-NoSIG-RGPL-base-source-release-r01'
PREFIX_RUN='20261024-phase1-PAMF-REPL-completion-m201-r01'
METHODS=('NoSIG-PL','NoSIG-Sup')
SEED=2026092701


def prefix(project,seed=SEED):
    if seed!=SEED:raise ValueError('Initial source falsification uses student seed01 only')
    return project+'/runs/'+PREFIX_RUN+'/prefixes/F0-A1B1-NO_SIGNATURE-F21-s'+str(seed)+'/segment20'


def config(ctx,commit,method):
    validate_context(ctx)
    if method not in METHODS:raise ValueError('Two matched NoSIG controls required')
    c=deepcopy(read(Path(prefix(ctx['project']))/'complete.json')['config'])
    for k in list(c):
        if k.startswith('hard_pl') or k in ('shared_prefix','parent_source','step_offset','epoch_offset','segment_epochs','core_mechanisms','supervised_architecture'):
            c.pop(k)
    use=method=='NoSIG-PL';rid='F0-'+method+'-s'+str(SEED)
    c.update(run_id=RUN,row_id=rid,protocol_id='F0',model_seed=SEED,paper_method=method,algorithm_id='pamf_nosig_rgpl_base_v1',method_id='pamf_t1_e50',
        repl_row=dict(row_id=rid,protocol_id='F0',model_seed=SEED,method=method,group='nosig_rgpl_base',L_A=True,L_B=True,U=use,structure='NO_SIGNATURE',
            blocks=dict(style=False,joint=False,guard=False,extrapolation=False)),
        no_sigtime=True,source_G=ctx['G'],frozen_sources=ctx['frozen_sources'],project=ctx['project'],
        expected_source_contract=next(v['expected_source_contract'] for v in ctx['cases'] if v['model_seed']==SEED),
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',code_commit=commit,
        shared_NoSIG_E20=prefix(ctx['project']) if use else None,regime='matched_sat_plain_pl' if use else 'matched_sat_supervised',ablation='F21' if use else 'F20',
        U_use=use,U_start_epoch=14,unlabeled_batch=256,labeled_batch=128,own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,
        satellite_concat=False,satellite_CE_weight=0.,satellite_consistency_weight=0.,plain_PL_threshold=.95,plain_PL_weight=1.,
        default_head_joint=True,posthoc_head_fit=False,epochs=50,optimizer_updates=11100,trajectory=[20,10,20])
    return validate_config(c)


def validate_config(c):
    method=c.get('paper_method');use=method=='NoSIG-PL';rid='F0-'+str(method)+'-s'+str(SEED);row=c.get('repl_row',{})
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid or c.get('model_seed')!=SEED
            or c.get('algorithm_id')!='pamf_nosig_rgpl_base_v1' or c.get('method_id')!='pamf_t1_e50' or c.get('protocol_id')!='F0'
            or c.get('num_classes')!=6 or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
            or c.get('regime')!=('matched_sat_plain_pl' if use else 'matched_sat_supervised') or c.get('ablation')!=('F21' if use else 'F20')
            or c.get('U_use') is not use or c.get('no_sigtime') is not True or c.get('U_start_epoch')!=14
            or c.get('shared_NoSIG_E20')!=(prefix(c['project']) if use else None)
            or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
            or not c.get('default_head_joint') or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
            or c.get('unlabeled_batch')!=256 or c.get('labeled_batch')!=128
            or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
            or row.get('structure')!='NO_SIGNATURE' or row.get('U') is not use or row.get('L_A') is not True or row.get('L_B') is not True
            or any(row.get('blocks',{}).values())):
        raise ValueError('Matched NoSIG full-L default-head controls and original budgets required')
    if any(k.startswith('hard_pl') or k in ('tests','truth','target_truth','target_inputs','views_root','dataset_path','checkpoint','teacher_checkpoint','core_mechanisms','supervised_architecture') for k in c):
        raise ValueError('NoSIG controls reject target,REPL,new head or foreign objectives')
    base=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={0:20,20:10,30:20}.get(offset)
        parent=(c['shared_NoSIG_E20'] if use else base+'/segment20') if offset==20 else base+'/segment30' if offset==30 else None
        if (length is None or (use and offset==0) or c['segment_epochs']!=length or c.get('step_offset')!=offset*222
                or c.get('parent_source')!=parent or c['output_root']!=base+'/segment'+str(offset+length)):
            raise ValueError('Own immutable NoSIG segments and parents required')
    elif c['output_root']!=base or any(k in c for k in ('parent_source','step_offset','epoch_offset')):
        raise ValueError('One immutable NoSIG row output required')
    return c


def check_state(cp):
    state=cp['model']
    if any('sequence_pool.' in name for name in state) or tuple(state['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape)!=(6,160):
        raise ValueError('Actual NoSequence state and original six-class joint default head required')


def load_prefix(c):
    import torch
    validate_config(c)
    if c['paper_method']!='NoSIG-PL':raise ValueError('Supervised control must not inherit a U-trained prefix')
    p=Path(c['shared_NoSIG_E20']);done=read(p/'complete.json');pc=done['config'];contract=read(c['source_contract'])
    cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);a=cp['args'];check_state(cp)
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=20 or done['updates']!=4440 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or done['initialization']!='scratch' or done['ancestors']!=[]
            or pc['model_seed']!=SEED or pc['ablation']!='F21' or pc['repl_row']['structure']!='NO_SIGNATURE' or not pc['repl_row']['L_A'] or not pc['repl_row']['L_B']
            or not pc['U_use'] or pc.get('parent_source') or read(pc['source_contract'])!=contract or contract!=read(c['expected_source_contract'])
            or pc['source_cache']!=c['source_cache'] or cp['run_id']!=pc['run_id'] or cp['candidate_id']!=pc['row_id'] or cp['epoch']!=20
            or cp['a1_ema_successful_updates']!=4440 or cp['checkpoint_selection']!='final_only' or a['use_unlabeled'] or a['use_ema_teacher'] or a['teacher_ckpt'] or a['baseline_ckpt']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: scratch ordinary-PL NoSIG E20 source prefix required')
    return dict(folder=str(p),**done),cp


@contextmanager
def installed(c):
    from experiments.cric_dura_comparison import t1_model
    from experiments.cric_dura_comparison.structural import NoSequence
    from experiments.pamf_repl_completion.objectives import loader_scope
    original=t1_model.install
    def build(model,cfg):
        validate_config(cfg);model=original(model,cfg);model.id_backbone.encoder.sequence_pool=NoSequence()
        if any('sequence_pool.' in n for n,_ in model.named_parameters()):raise ValueError('No trainable signature branch permitted')
        model._feature_learning_report.update(supervised_architecture='NO_SIGNATURE',source_U_IQ_used=cfg['U_use'],no_sigtime=True)
        return model
    t1_model.install=build
    try:
        with loader_scope():yield
    finally:t1_model.install=original


def smoke(configs,output):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload,load_G
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    c=next(v for v in configs if v['paper_method']=='NoSIG-PL');_,payload=load_prefix(c);rows=[]
    with reader_structure(c):reader=reader_from_payload(c,payload,'cuda:0')
    contract=read(c['source_contract']);g=load_G(c['source_G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    labeled=SourceDataset(c['source_cache'],'L_s',contract)
    with torch.inference_mode():
        x=labeled.x[:2].to('cuda:0');restored=engine(x.clone())[0].clone();logits=reader(restored)
        if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Actual NoSIG E20 and same G source L2 smoke failed')
    for cfg in configs:
        with installed(cfg):
            ctx=f_train.context(cfg)
            if (ctx['unlabeled_loader'] is None)!=(cfg['paper_method']=='NoSIG-Sup'):raise ValueError('Exact U-off supervised loader required')
            with f_train.class_dimension(cfg),native_installed(f_train.original_config(SEED)) as native:
                a=f_train.arguments(cfg,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
                student=t1_model.install(native.build_baseline_model(a,'cuda:0'),cfg).to('cuda:0')
            if cfg['paper_method']=='NoSIG-PL':student.load_state_dict(payload['model'],strict=True)
            student._cd_U_loader=ctx['unlabeled_loader'];student._cd_U_iter=iter(ctx['unlabeled_loader']) if ctx['unlabeled_loader'] else None
            batch=next(iter(ctx['train_loader']));x,y,d=[v.to('cuda:0') for v in batch[:3]];student.train()
            # The smoke samples the E21 hook, without advancing any optimizer or checkpoint.
            student._feature_l_steps=4440
            with student._phase1_labeled_feature_context():out=student(x,return_aux=True,domain_labels=d)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=out,labels=y,extra=(None,batch[3]),domains=d,
                optimizer_step=4441,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();head=student.id_backbone.encoder.core.id_backbone.cls_head.weight
            expected=256 if cfg['U_use'] else 0
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all()) or float(head.grad.norm())<=0
                    or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or student._cd_U_seen!=expected
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in student.parameters())):
                raise ValueError('Actual NoSIG L/U joint backward or supervised U-off smoke failed')
            rows.append(dict(method=cfg['paper_method'],status='PASS',actual_U_seen=expected,L_forward_packets=768,head_finite_nonzero=True,
                actual_signature_parameters=0,optimizer_updates=0,U_truth_read=False,query_reads=0))
    write(output,dict(status='PASS',results=rows,actual_same_contract_NoSIG_E20_default_head_and_G=True,checkpoint_state_verified=True,optimizer_updates=0,target_reads=0))


def train(c):
    import shutil
    import torch
    from experiments.cric_dura_comparison.f_train import train as segment
    validate_config(c);root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c);start=time.perf_counter()
    chain=[];previous=None;use=c['U_use']
    if use:
        old,cp=load_prefix(c);del cp;chain.append(old);previous=c['shared_NoSIG_E20']
    for epochs,offset in (((10,20),(20,30)) if use else ((20,0),(10,20),(20,30))):
        folder=root/('segment'+str(offset+epochs));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        with installed(cfg):segment(cfg)
        done=read(folder/'complete.json');cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
        if (done['status']!='SOURCE_FROZEN' or done['epoch']!=epochs or done['updates']!=epochs*222 or done['target_access'] or done['U_truth_read']
                or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt']
                or cp['args']['baseline_ckpt']!=('' if previous is None else previous+'/final_ssdg.pth') or cp['a1_ema_successful_updates']!=epochs*222):
            raise ValueError('Exact own NoSIG trajectory and source permissions required')
        del cp;chain.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,updates=(offset+epochs)*222,signature_parameters=0,U_use=use,seconds=time.perf_counter()-start),exclusive=False)
    if len(chain)!=3 or sum(v['updates'] for v in chain)!=11100:raise ValueError('Three original NoSIG segments required')
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,method_id=c['paper_method'],epoch=50,updates=11100,segments=chain,
        source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in chain],
        target_access=False,U_truth_read=False,head_fit_updates=0,signature_parameters=0,seconds=time.perf_counter()-start,source_diagnostic_only=True))


def source_evaluate(c,validator=None):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload,load_G
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .restored_source import source_channel
    (validator or validate_config)(c);source=Path(c['output_root']);done=read(source/'complete.json');contract=read(c['source_contract'])
    if done['config']!=c or done['epoch']!=50 or done['updates']!=11100:raise ValueError('Complete own NoSIG control required')
    cp=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
    with reader_structure(c):model=reader_from_payload(c,cp,'cuda:0')
    del cp;g=load_G(c['source_G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    data=SourceDataset(c['source_cache'],'V',contract);n=len(data)
    if n!=27000:raise ValueError('Complete original source V required')
    out=source.parent/'source_validation';out.mkdir(exist_ok=False);logits=np.empty((8,n,6),np.float32);offset=0
    with torch.inference_mode():engine(data.x[:2].to('cuda:0'))
    before=snapshot(model);gbefore=snapshot(g)
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');views=[x,engine(x.clone())[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract);views.extend((h,engine(h.clone())[0].clone()))
            for k,view in enumerate(views):
                value=model(view)
                if value.shape!=(len(x),6) or not bool(torch.isfinite(value).all()):raise ValueError('Finite NoSIG default-head source predictions required')
                logits[k,offset:offset+len(x)]=fp32_copy(value)
            offset+=len(x);write(out/'progress.json',dict(status='SOURCE_VALIDATING',seen=offset,total=n,target_reads=0),exclusive=False)
    if offset!=n or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Source coverage or frozen NoSIG/G state differs')
    y=data.metadata['y'];pred=logits.argmax(-1);rx=data.metadata['rx_i'];strata=[('overall','ALL',np.ones(n,bool)),('extra_RX_holdout','held',rx==1),('extra_RX_holdout','trained',rx!=1)]
    for dim,values in [('TX',y),('RX',rx),('day',data.metadata['day_i'])]:strata.extend((dim,str(v),values==v) for v in np.unique(values))
    rows=[dict(method=c['paper_method'],model_seed=SEED,view=view,dimension=dim,stratum=s,n=int(mask.sum()),correct=int((pred[k,mask]==y[mask]).sum()),accuracy=100*float((pred[k,mask]==y[mask]).mean())) for k,view in enumerate(VIEWS) for dim,s,mask in strata]
    np.savez(out/'source_V_predictions.npz',logits=logits,y=y,ids=data.metadata['ids'],views=np.asarray(VIEWS))
    write(out/'complete.json',dict(status='NOSIG_SOURCE_CONTROL_EVALUATED',method=c['paper_method'],model_seed=SEED,source_V_seen=n,results=rows,
        signature_parameters=0,parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_read=False,target_reads=0,scope='Single student source control;not target performance'))


def prepare(rt):
    ctx=read(rt['context']);validate_context(ctx);root=Path(rt['run_root']);jobs=[]
    for method in METHODS:
        c=config(ctx,rt['code_commit'],method);path=root/'configs'/(method+'.json');write(path,c)
        jobs.append(dict(id=method,kind='train_eval',config=str(path),completion=str(Path(c['output_root']).parent/'source_validation/complete.json'),successful_statuses=['NOSIG_SOURCE_CONTROL_EVALUATED'],
            dependencies=[str(root/'smoke.json')],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.nosig_rgpl_base',row_id=c['row_id']))
    path=root/'configs/smoke.json';write(path,dict(train_configs=[j['config'] for j in jobs],output=str(root/'smoke.json'),frozen_sources=rt['frozen_sources']))
    job=dict(id='smoke',kind='smoke',config=str(path),completion=str(root/'smoke.json'),successful_statuses=['PASS'],dependencies=[],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.nosig_rgpl_base')
    jobs=[job]+jobs;write(root/'jobs.json',jobs);return jobs


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed NoSIG base source matrix required')
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
    if not stop:write(root/'complete.json',dict(status='NOSIG_SOURCE_CONTROLS_COMPLETE',results=[read(j['completion']) for j in jobs if j['kind']=='train_eval'],target_reads=0,graph_or_candidate_training=False))
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'NOSIG_SOURCE_CONTROLS_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='smoke':return smoke([read(p) for p in c['train_configs']],c['output'])
    if kind=='train_eval':train(c);return source_evaluate(c)
    raise ValueError('Only fixed NoSIG smoke and source controls required')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
