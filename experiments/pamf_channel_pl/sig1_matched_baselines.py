"""Full PAMF, single SIG_TIME view: matched supervised and ordinary-PL controls."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,VIEWS,snapshot,unchanged

RUN='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'
RELEASE='20261026-phase1-PAMF-SIG1-matched-baselines-release-r01'
PREFIX_RUN='20261024-phase1-PAMF-REPL-completion-m201-r01'
METHODS=('SIG1-PL','SIG1-Sup')
SEEDS=(2026092701,2026092702,2026092703)


def prefix(project,seed):
    if seed not in SEEDS:raise ValueError('Three original model seeds required')
    return project+'/runs/'+PREFIX_RUN+'/prefixes/F0-A1B1-SIG_TIME-F21-s'+str(seed)+'/segment20'


def config(ctx,commit,method,seed):
    if seed not in SEEDS:raise ValueError("Unknown model seed")
    validate_context(ctx)
    if method not in METHODS:raise ValueError('Two matched SIG1 controls required')
    c=deepcopy(read(Path(prefix(ctx['project'],seed))/'complete.json')['config'])
    for k in list(c):
        if k.startswith('hard_pl') or k in ('shared_prefix','parent_source','step_offset','epoch_offset','segment_epochs','core_mechanisms','supervised_architecture'):
            c.pop(k)
    use=method=='SIG1-PL';rid='F0-'+method+'-s'+str(seed)
    c.update(run_id=RUN,row_id=rid,protocol_id='F0',model_seed=seed,paper_method=method,algorithm_id='pamf_sig1_matched_baselines_v1',method_id='pamf_t1_e50',
        repl_row=dict(row_id=rid,protocol_id='F0',model_seed=seed,method=method,group='sig1_matched_baselines',L_A=True,L_B=True,U=use,structure='SIG_TIME',
            blocks=dict(style=False,joint=False,guard=False,extrapolation=False)),
        sig_time_views=1,source_G=ctx['G'],frozen_sources=ctx['frozen_sources'],project=ctx['project'],
        expected_source_contract=next(v['expected_source_contract'] for v in ctx['cases'] if v['model_seed']==seed),
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',code_commit=commit,
        shared_SIG1_E20=prefix(ctx['project'],seed) if use else None,regime='matched_sat_plain_pl' if use else 'matched_sat_supervised',ablation='F21' if use else 'F20',
        U_use=use,U_start_epoch=14,unlabeled_batch=256,labeled_batch=128,own_EMA=False,native_U_gate=False,native_unlabeled_loop=False,
        satellite_concat=False,satellite_CE_weight=0.,satellite_consistency_weight=0.,plain_PL_threshold=.95,plain_PL_weight=1.,
        default_head_joint=True,posthoc_head_fit=False,epochs=50,optimizer_updates=11100,trajectory=[20,10,20])
    return validate_config(c)


def validate_config(c):
    seed=c.get('model_seed');method=c.get('paper_method');use=method=='SIG1-PL';rid='F0-'+str(method)+'-s'+str(seed);row=c.get('repl_row',{})
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid or seed not in SEEDS
            or c.get('algorithm_id')!='pamf_sig1_matched_baselines_v1' or c.get('method_id')!='pamf_t1_e50' or c.get('protocol_id')!='F0'
            or c.get('num_classes')!=6 or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
            or c.get('regime')!=('matched_sat_plain_pl' if use else 'matched_sat_supervised') or c.get('ablation')!=('F21' if use else 'F20')
            or c.get('U_use') is not use or c.get('sig_time_views')!=1 or c.get('U_start_epoch')!=14
            or c.get('shared_SIG1_E20')!=(prefix(c['project'],seed) if use else None)
            or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
            or not c.get('default_head_joint') or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
            or c.get('unlabeled_batch')!=256 or c.get('labeled_batch')!=128
            or any(c.get(k) for k in ('no_sigtime','own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
            or row.get('structure')!='SIG_TIME' or row.get('U') is not use or row.get('L_A') is not True or row.get('L_B') is not True
            or any(row.get('blocks',{}).values())):
        raise ValueError('Matched SIG1 full-L default-head controls and original budgets required')
    if any(k.startswith('hard_pl') or k in ('tests','truth','target_truth','target_inputs','views_root','dataset_path','checkpoint','teacher_checkpoint','core_mechanisms','supervised_architecture') for k in c):
        raise ValueError('SIG1 controls reject target,REPL,new head or foreign objectives')
    base=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={0:20,20:10,30:20}.get(offset)
        parent=(c['shared_SIG1_E20'] if use else base+'/segment20') if offset==20 else base+'/segment30' if offset==30 else None
        if (length is None or (use and offset==0) or c['segment_epochs']!=length or c.get('step_offset')!=offset*222
                or c.get('parent_source')!=parent or c['output_root']!=base+'/segment'+str(offset+length)):
            raise ValueError('Own immutable SIG1 segments and parents required')
    elif c['output_root']!=base or any(k in c for k in ('parent_source','step_offset','epoch_offset')):
        raise ValueError('One immutable SIG1 row output required')
    return c


def check_state(cp):
    state=cp['model']
    prefix='id_backbone.encoder.sequence_pool.'
    required=(prefix+'views.0.local_projection.weight',prefix+'output_projection.weight','id_backbone.encoder.core.id_backbone.cls_head.weight')
    if any(k not in state for k in required):raise ValueError('Missing actual single-view SIG_TIME or default-head state')
    if (tuple(state[prefix+'views.0.local_projection.weight'].shape)!=(6,96,1)
            or any(prefix+'views.'+str(i)+'.' in name for i in (1,2) for name in state)
            or tuple(state[prefix+'output_projection.weight'].shape)!=(160,112)
            or tuple(state['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape)!=(6,160)):
        raise ValueError('Actual one-view SIG_TIME and original six-class joint default head required')


def load_prefix(c):
    import torch
    validate_config(c)
    if c['paper_method']!='SIG1-PL':raise ValueError('Supervised control must not inherit a U-trained prefix')
    p=Path(c['shared_SIG1_E20']);done=read(p/'complete.json');pc=done['config'];contract=read(c['source_contract'])
    cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);a=cp['args'];check_state(cp)
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=20 or done['updates']!=4440 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or done['initialization']!='scratch' or done['ancestors']!=[]
            or pc['model_seed']!=c['model_seed'] or pc['ablation']!='F21' or pc['repl_row']['structure']!='SIG_TIME' or not pc['repl_row']['L_A'] or not pc['repl_row']['L_B']
            or not pc['U_use'] or pc.get('parent_source') or read(pc['source_contract'])!=contract or contract!=read(c['expected_source_contract'])
            or pc['source_cache']!=c['source_cache'] or cp['run_id']!=pc['run_id'] or cp['candidate_id']!=pc['row_id'] or cp['epoch']!=20
            or cp['a1_ema_successful_updates']!=4440 or cp['checkpoint_selection']!='final_only' or a['use_unlabeled'] or a['use_ema_teacher'] or a['teacher_ckpt'] or a['baseline_ckpt']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: scratch ordinary-PL SIG1 E20 source prefix required')
    return dict(folder=str(p),**done),cp


@contextmanager
def installed(c):
    from experiments.cric_dura_comparison import t1_model
    from .sig1_model import install_single_view
    from experiments.pamf_repl_completion.objectives import loader_scope
    original=t1_model.install
    def build(model,cfg):
        validate_config(cfg);model=original(model,cfg);model=install_single_view(model,cfg['model_seed'])
        model._feature_learning_report.update(supervised_architecture='SIG_TIME',source_U_IQ_used=cfg['U_use'],sig_time_views=1)
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
    c=next(v for v in configs if v['paper_method']=='SIG1-PL');_,payload=load_prefix(c);rows=[]
    with reader_structure(c):reader=reader_from_payload(c,payload,'cuda:0')
    contract=read(c['source_contract']);g=load_G(c['source_G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    labeled=SourceDataset(c['source_cache'],'L_s',contract)
    with torch.inference_mode():
        x=labeled.x[:2].to('cuda:0');restored=engine(x.clone())[0].clone();logits=reader(restored)
        if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Actual SIG1 E20 and same G source L2 smoke failed')
    for cfg in configs:
        if cfg['U_use']:
            _,payload=load_prefix(cfg)
        with installed(cfg):
            ctx=f_train.context(cfg)
            if (ctx['unlabeled_loader'] is None)!=(cfg['paper_method']=='SIG1-Sup'):raise ValueError('Exact U-off supervised loader required')
            with f_train.class_dimension(cfg),native_installed(f_train.original_config(cfg['model_seed'])) as native:
                a=f_train.arguments(cfg,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
                student=t1_model.install(native.build_baseline_model(a,'cuda:0'),cfg).to('cuda:0')
            if cfg['paper_method']=='SIG1-PL':student.load_state_dict(payload['model'],strict=True)
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
                raise ValueError('Actual SIG1 L/U joint backward or supervised U-off smoke failed')
            sig_grad=sum(float(p.grad.detach().square().sum()) for name,p in student.named_parameters() if 'sequence_pool.' in name and p.grad is not None)**.5
            if sig_grad<=0:raise ValueError('Single SIG_TIME view must receive actual training gradients')
            rows.append(dict(model_seed=cfg['model_seed'],signature_gradient_norm=sig_grad,method=cfg['paper_method'],status='PASS',actual_U_seen=expected,L_forward_packets=768,head_finite_nonzero=True,
                actual_signature_parameters=18976,actual_signature_views=1,optimizer_updates=0,U_truth_read=False,query_reads=0))
    write(output,dict(status='PASS',results=rows,actual_same_contract_SIG1_E20_default_head_and_G=True,checkpoint_state_verified=True,optimizer_updates=0,target_reads=0))


def train(c):
    import shutil
    import torch
    from experiments.cric_dura_comparison.f_train import train as segment
    validate_config(c);root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c);start=time.perf_counter()
    chain=[];previous=None;use=c['U_use']
    if use:
        old,cp=load_prefix(c);del cp;chain.append(old);previous=c['shared_SIG1_E20']
    for epochs,offset in (((10,20),(20,30)) if use else ((20,0),(10,20),(20,30))):
        folder=root/('segment'+str(offset+epochs));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        with installed(cfg):segment(cfg)
        done=read(folder/'complete.json');cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
        if (done['status']!='SOURCE_FROZEN' or done['epoch']!=epochs or done['updates']!=epochs*222 or done['target_access'] or done['U_truth_read']
                or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt']
                or cp['args']['baseline_ckpt']!=('' if previous is None else previous+'/final_ssdg.pth') or cp['a1_ema_successful_updates']!=epochs*222):
            raise ValueError('Exact own SIG1 trajectory and source permissions required')
        del cp;chain.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,updates=(offset+epochs)*222,signature_parameters=18976,signature_views=1,U_use=use,seconds=time.perf_counter()-start),exclusive=False)
    if len(chain)!=3 or sum(v['updates'] for v in chain)!=11100:raise ValueError('Three original SIG1 segments required')
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,method_id=c['paper_method'],epoch=50,updates=11100,segments=chain,
        source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in chain],
        target_access=False,U_truth_read=False,head_fit_updates=0,signature_parameters=18976,signature_views=1,seconds=time.perf_counter()-start,
        cumulative_training_seconds=sum(v['seconds'] for v in chain),reused_prefix_seconds=chain[0]['seconds'] if use else 0.))


TASKS=('cross_rx','cross_both')
READOUTS=['raw','DURA-LU']


def validate_evaluation(c,scoring=False):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    if not scoring:reject_truth(c)
    seed=c.get('model_seed');method=c.get('method');rid='F0-'+str(method)+'-s'+str(seed)
    if (c.get('run_id')!=RUN or c.get('row_id')!=rid or seed not in SEEDS or method not in METHODS
            or c.get('protocol_id')!='F0' or set(c.get('tests',{}))!=set(TASKS)
            or c.get('readouts')!=READOUTS or len(c.get('classes',[]))!=6 or c.get('source_days')!=[1,2,3]):
        raise ValueError('Fixed F0 six-model raw/G-LU clean+six-practical matrix required')
    if not scoring:
        if set(c.get('restored_caches',{}))!={'LU'} or c.get('U') is not (method=='SIG1-PL'):
            raise ValueError('Own supervised/PL source and same G-LU required')
    elif any(k in c for k in ('source_output','source_cache','restored_caches','frozen_sources')):
        raise ValueError('Independent scorer must have no model or source IQ handles')


def predict(c,device='cuda:0'):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.cric_dura_comparison.evaluation import handle,VIEWS as TARGET_VIEWS
    from experiments.pamf_repl_completion.source import load_reader
    validate_evaluation(c);contract=read(c['source_contract']);src=read(Path(c['source_output'])/'complete.json')
    validate_config(src['config'])
    if src['config']['row_id']!=c['row_id'] or src['config']['U_use'] is not c['U']:
        raise ValueError('Wrong frozen single-view F')
    for segment in src['segments']:
        if segment['config']['U_use'] is not c['U'] or segment.get('U_truth_read') or segment.get('target_access'):
            raise ValueError('Single-view source ancestry differs')
    reader,_=load_reader(c['source_output'],contract,device,require_no_unlabeled=not c['U'])
    cache=Path(c['restored_caches']['LU']);done=read(cache/'complete.json')
    if (done['status']!='DURA_IQ_CACHE_COMPLETE' or done['protocol_id']!='F0' or done['readout']!='DURA-LU'
            or done['truth_read'] or done['G']!=c['same_G'] or read(done['G']['source_contract'])!=contract):
        raise ValueError('Same original frozen G-LU cache required')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);counts={};started=time.monotonic();before=snapshot(reader)
    for task in TASKS:
        manifest,ids,positions=handle(c['tests'][task]['views_root'],c['classes'])
        if len(ids)!={'cross_rx':126000,'cross_both':42000}[task]:raise ValueError('Original F0 test population differs')
        with np.load(cache/task/'index.npz',allow_pickle=False) as ix:
            if not np.array_equal(ix['ids'],ids):raise ValueError('Frozen G physical ID order differs')
        values={}
        for view in TARGET_VIEWS:
            raw=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            arrays=[np.load(raw,mmap_mode='r',allow_pickle=False),np.load(cache/task/(view+'.npy'),mmap_mode='r',allow_pickle=False)]
            for route,array in zip(READOUTS,arrays):
                result=[]
                with torch.inference_mode():
                    for begin in range(0,len(ids),128):
                        ix=positions[begin:begin+128] if route=='raw' and positions is not None else slice(begin,begin+128)
                        x=tensor(np.ascontiguousarray(array[ix],np.float32),device)
                        logits=reader.classify_features(reader.features(x))
                        if logits.shape!=(len(x),6) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Invalid frozen single-view default head')
                        result.extend(logits.argmax(1).cpu().tolist())
                if len(result)!=len(ids):raise ValueError('Incomplete frozen predictions')
                values[route+'__'+view]=np.asarray(result,np.int64)
            write(out/'progress.json',dict(status='PREDICTING',task=task,view=view,count=len(ids),truth_read=False),exclusive=False)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(TARGET_VIEWS),readouts=READOUTS)
    if not unchanged(reader,before):raise ValueError('Frozen F changed during target prediction')
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=RUN,row_id=c['row_id'],tests=counts,
        views=list(TARGET_VIEWS),readouts=READOUTS,truth_read=False,query_fit=False,parameters_and_buffers_unchanged=True,seconds=time.monotonic()-started))


def score(c):
    validate_evaluation(c,scoring=True)
    from experiments.cric_dura_comparison import evaluation as baseline
    baseline.tasks=lambda _:list(TASKS)
    baseline.score(c)


def prepare(rt):
    ctx=read(rt['context']);validate_context(ctx);target=read(rt['evaluation_context'])
    if set(target)!= {'tests','restored_caches','classes','same_G'} or target['same_G']!=ctx['G'] or set(target['tests'])!=set(TASKS):
        raise ValueError('Fixed evaluation handles and source G required')
    root=Path(rt['run_root']);jobs=[];train_configs=[]
    def add(kind,rid,c,completion,status,deps):
        path=root/'configs'/(kind+'__'+rid+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+rid,kind=kind,config=str(path),completion=str(completion),successful_statuses=[status],
            dependencies=deps,gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],
            entrypoint='experiments.pamf_channel_pl.sig1_matched_baselines',row_id=rid))
        return str(path)
    for seed in SEEDS:
        for method in METHODS:
            c=config(ctx,rt['code_commit'],method,seed);rid=c['row_id'];source=Path(c['output_root'])
            train_configs.append(add('train',rid,c,source/'complete.json','SOURCE_FROZEN',[str(root/'smoke.json')]))
            out=root/'readouts'/rid
            p=dict(run_id=RUN,row_id=rid,protocol_id='F0',model_seed=seed,method=method,U=c['U_use'],source_days=[1,2,3],
                source_output=str(source),source_contract=c['source_contract'],classes=target['classes'],same_G=target['same_G'],
                tests={t:dict(views_root=target['tests'][t]['views_root']) for t in TASKS},
                restored_caches=target['restored_caches'],readouts=READOUTS,frozen_sources=rt['frozen_sources'],
                code_commit=rt['code_commit'],output_root=str(out/'predictions'))
            validate_evaluation(p);add('predict',rid,p,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(source/'complete.json')])
            s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests=target['tests'])
            for k in ('source_output','restored_caches','frozen_sources'):s.pop(k,None)
            validate_evaluation(s,scoring=True);add('score',rid,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
    add('smoke','all',dict(train_configs=train_configs,output=str(root/'smoke.json'),frozen_sources=rt['frozen_sources']),root/'smoke.json','PASS',[])
    write(root/'jobs.json',jobs);return jobs


def analyze(rt,jobs):
    import csv
    import statistics
    rows=[]
    for job in jobs:
        if job['kind']!='score':continue
        c=read(job['config']);data=read(Path(c['output_root'])/'scores.json')
        if not data['truth_last'] or not data['target_feedback_forbidden']:raise ValueError('Truth-last scores required')
        for route in READOUTS:
            cells=[v for v in data['results'] if v['readout']==route and v['dimension']=='overall']
            for population in (*TASKS,'overall_168000'):
                selected=[v for v in cells if population=='overall_168000' or v['test']==population]
                for metric in ('clean','six_practical'):
                    values=[v for v in selected if (v['view']=='clean')==(metric=='clean')]
                    expected=(168000 if population=='overall_168000' else {'cross_rx':126000,'cross_both':42000}[population])*(1 if metric=='clean' else 6)
                    denominator=sum(v['count'] for v in values)
                    if denominator!=expected:raise ValueError('Incomplete formal metric cells')
                    correct=sum(sum(v['confusion'][i][i] for i in range(6)) for v in values)
                    rows.append(dict(method=c['method'],model_seed=c['model_seed'],readout=route,population=population,metric=metric,
                        correct=correct,count=denominator,accuracy=100*correct/denominator))
    summary=[]
    for method in METHODS:
        for route in READOUTS:
            for population in (*TASKS,'overall_168000'):
                for metric in ('clean','six_practical'):
                    values=[v['accuracy'] for v in rows if (v['method'],v['readout'],v['population'],v['metric'])==(method,route,population,metric)]
                    if len(values)!=3:raise ValueError('Complete three-seed baseline required')
                    summary.append(dict(method=method,readout=route,population=population,metric=metric,seeds=3,mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
    paired=[];paired_summary=[]
    for route in READOUTS:
        for population in (*TASKS,'overall_168000'):
            for metric in ('clean','six_practical'):
                deltas=[]
                for seed in SEEDS:
                    lookup={v['method']:v['accuracy'] for v in rows if (v['model_seed'],v['readout'],v['population'],v['metric'])==(seed,route,population,metric)}
                    delta=lookup['SIG1-PL']-lookup['SIG1-Sup'];deltas.append(delta)
                    paired.append(dict(model_seed=seed,readout=route,population=population,metric=metric,PL_minus_Sup_pp=delta))
                paired_summary.append(dict(readout=route,population=population,metric=metric,seeds=3,mean_delta_pp=statistics.mean(deltas),sample_sd_delta_pp=statistics.stdev(deltas)))
    out=Path(rt['run_root'])/'analysis';out.mkdir()
    for name,values in [('per_seed.csv',rows),('summary.csv',summary),('paired_deltas.csv',paired),('paired_summary.csv',paired_summary)]:
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(out/'complete.json',dict(status='SIG1_MATCHED_BASELINES_ANALYZED',run_id=RUN,models=6,seeds=list(SEEDS),results=summary,
        paired_PL_minus_Sup=paired_summary,formal_population_per_seed=168000,readouts=READOUTS,new_G=0,new_restored_cache=0,truth_last=True,target_feedback_used=False,
        scope='PAMF SIG_TIME single-view supervised and ordinary PL; supervised F plus G-LU is not an entirely L-only system'))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed six-model matching baselines required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop=stop or not good;del active[key]
        pending=processes()
        for job in sorted(jobs,key=lambda v:dict(smoke=0,score=1,predict=2,train=3)[v['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v[0].pid for v in active.values()}):continue
                gpu=None
            else:
                available=[]
                for gpu in job['gpu_pool']:
                    pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
                    if load<2 and free>=10000:available.append((load,-free,gpu))
                if not available:continue
                gpu=min(available)[2]
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:
        analyze(rt,jobs);write(root/'complete.json',read(root/'analysis/complete.json'))
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'SIG1_MATCHED_BASELINES_ANALYZED';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    # Truth-side scoring runs independently on CPU, without loading F/source IQ.
    if kind=='score':return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='smoke':return smoke([read(p) for p in c['train_configs']],c['output'])
    if kind=='train':return train(c)
    if kind=='predict':return predict(c)
    raise ValueError('Unknown matched baseline worker')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
