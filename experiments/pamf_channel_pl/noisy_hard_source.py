"""Matched source-only NoSIG noisy-hard-label CE/GCE and raw/H/G pilot."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,snapshot,unchanged,VIEWS
from .nosig_rgpl_base import RUN as BASE_RUN,SEED,prefix,check_state,source_evaluate
from .restoration_labeler import RUN as LABELER_RUN
from .rgpl_source_graph import RUN as GRAPH_RUN,teacher_folder,load_teacher
from .noisy_hard_objective import METHODS,RECIPES,installed

RUN='20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-m6-r01'
RELEASE='20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-release-r01'


def raw_labeler(project):return project+'/runs/'+LABELER_RUN+'/RawLabeler/final.pt'


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
            or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
            or rt['allowed_gpus']!=list(range(8))):raise ValueError('One fixed noisy-hard source pilot required')


def config(ctx,commit,method):
    validate_context(ctx)
    if method not in METHODS:raise ValueError('Fixed six-method source matrix required')
    c=deepcopy(read(Path(teacher_folder(ctx['project']))/'complete.json')['config']);rid='F0-'+method+'-s'+str(SEED)
    c.update(run_id=RUN,row_id=rid,paper_method=method,algorithm_id='pamf_noisy_hard_pl_v1',code_commit=commit,
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',epochs=50,optimizer_updates=11100,
        segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=teacher_folder(ctx['project']),
        raw_labeler=raw_labeler(ctx['project']),noisy_U_cache=ctx['project']+'/runs/'+RUN+'/noisy_U_cache',noisy_recipe=deepcopy(RECIPES[method]),
        repl_row=dict(row_id=rid,protocol_id='F0',model_seed=SEED,method=method,group='noisy_hard_source',L_A=True,L_B=True,U=True,structure='NO_SIGNATURE',
            blocks=dict(style=False,joint=False,guard=False,extrapolation=False)))
    return validate_config(c)


def validate_config(c):
    method=c.get('paper_method');rid='F0-'+str(method)+'-s'+str(SEED);row=c.get('repl_row',{})
    root=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if (method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid or c.get('model_seed')!=SEED or c.get('protocol_id')!='F0'
            or c.get('algorithm_id')!='pamf_noisy_hard_pl_v1' or c.get('method_id')!='pamf_t1_e50' or c.get('num_classes')!=6
            or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3] or c.get('no_sigtime') is not True
            or c.get('regime')!='matched_sat_plain_pl' or c.get('ablation')!='F21' or c.get('U_use') is not True or c.get('U_start_epoch')!=14
            or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1. or c.get('labeled_batch')!=128 or c.get('unlabeled_batch')!=256
            or c.get('raw_labeler')!=raw_labeler(c['project']) or c.get('noisy_U_cache')!=c['project']+'/runs/'+RUN+'/noisy_U_cache'
            or c.get('noisy_recipe')!=RECIPES[method] or c.get('shared_NoSIG_E20')!=prefix(c['project'])
            or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('segment_epochs')!=20 or c.get('epoch_offset')!=30
            or c.get('step_offset')!=6660 or c.get('parent_source')!=teacher_folder(c['project']) or c.get('output_root') not in (root,root+'/segment50')
            or not c.get('default_head_joint') or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))
            or row.get('structure')!='NO_SIGNATURE' or row.get('U') is not True or row.get('L_A') is not True or row.get('L_B') is not True or any(row.get('blocks',{}).values())):
        raise ValueError('Matched NoSIG E30+20/default-head/hidden-U source recipe required')
    if any(k.startswith('hard_pl') or k in ('truth','target_truth','target_inputs','tests','views_root','dataset_path','checkpoint','teacher_checkpoint','core_mechanisms','supervised_architecture') for k in c):
        raise ValueError('Source noisy-hard pilot rejects target or foreign objectives')
    return c


def cache(rt):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_repl_completion.source import reader_structure
    from .u_realization_cache import KEYS
    _,g,c,contract=load_teacher(rt,'cuda:0');del g
    cp=torch.load(raw_labeler(rt['project']),map_location='cpu',weights_only=False);check_state(cp)
    if (cp['schema']!='pamf_restoration_labeler_v1' or cp['run_id']!=LABELER_RUN or cp['method']!='RawLabeler' or cp['model_seed']!=SEED
            or cp['source_contract']!=contract or cp['parent']!=teacher_folder(rt['project']) or cp['parent_config']!=c or cp['new_epochs']!=10
            or cp['new_updates']!=2220 or cp['cumulative_epochs']!=40 or cp['checkpoint_selection']!='final_only' or cp['U_training_reads']
            or cp['target_reads'] or cp['head_fit_updates'] or not cp['default_head_joint']):raise ValueError('Known source-only RawLabeler default-head lineage required')
    with reader_structure(c):model=reader_from_payload(c,cp,'cuda:0')
    del cp;root=Path(rt['run_root'])/'noisy_U_cache';root.mkdir(exist_ok=False)
    L=SourceDataset(c['source_cache'],'L_s',contract);U=SourceDataset(c['source_cache'],'U_s',contract)
    graph=Path(rt['project'])/'runs'/GRAPH_RUN
    feature_header=read(graph/'features_complete.json')
    if feature_header['source_contract']!=contract or feature_header['model_seed']!=SEED or feature_header['teacher_epoch']!=30 or 'y' in U.metadata:
        raise ValueError('Own original NoSIG E30 hidden source U roles required')
    with np.load(graph/'U_s/metadata.npz',allow_pickle=False) as a:
        if 'y' in a.files or list(a['ids'])!=contract['role_ids']['U_s']:raise ValueError('No U truth in original confidence cache')
        original=a['raw_confidence'].copy()
    with torch.inference_mode():
        smoke=model(L.x[:2].to('cuda:0'))
        if smoke.shape!=(2,6) or not bool(torch.isfinite(smoke).all()):raise ValueError('Actual RawLabeler checkpoint source L2 smoke failed')
    before=snapshot(model);labels=np.empty(56700,np.int64);confidence=np.empty(56700,np.float32);offset=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(U,batch_size=256,shuffle=False,num_workers=0):
            if not bool((batch[1]==-1).all()) or bool(batch[3]['tx_label_visible'].any()):raise ValueError('U true labels exposed')
            p=model(batch[0].to('cuda:0')).softmax(1);n=len(batch[0])
            if not bool(torch.isfinite(p).all()):raise ValueError('Finite raw pseudo-label probabilities required')
            labels[offset:offset+n]=np.asarray(p.argmax(1).cpu().tolist(),np.int64);confidence[offset:offset+n]=fp32_copy(p.max(1).values);offset+=n
            write(root/'progress.json',dict(seen=offset,total=56700,U_truth_read=False,target_reads=0),exclusive=False)
    if offset!=56700 or not unchanged(model,before):raise ValueError('Complete frozen raw U inference required')
    physical=np.stack([U.metadata[k] for k in KEYS],axis=1);candidate=(confidence>=.95)&(original<.95)&(U.metadata['rx_i']!=1)
    if not bool(candidate.any()):raise ValueError('No noisy candidates; do not launch no-op students')
    np.savez(root/'noisy_U_targets.npz',physical=physical,ids=U.metadata['ids'],label=labels,candidate=candidate,teacher_confidence=confidence,original_confidence=original)
    write(root/'complete.json',dict(status='NOISY_SOURCE_U_CACHE_COMPLETE',source_contract=contract,teacher=raw_labeler(rt['project']),model_seed=SEED,
        unique_noisy_candidates=int(candidate.sum()),U_records=56700,F_logical_packets=56702,G_logical_packets=0,teacher_parameters_and_registered_buffers_unchanged=True,
        U_truth_read=False,target_reads=0,optimizer_updates=0,label_quality='UNKNOWN_U; source V proxy93.829947%, explicitly noisy,not certified'))


def smoke(rt,configs):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    parent=torch.load(Path(teacher_folder(rt['project']))/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(parent);rows=[]
    for c in configs:
        holder={}
        with installed(c,holder):
            ctx=f_train.context(c)
            with f_train.class_dimension(c),native_installed(f_train.original_config(SEED)) as native:
                a=f_train.arguments(c,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
                model=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
            model.load_state_dict(parent['model'],strict=True);model._cd_U_loader=ctx['unlabeled_loader'];pool=model._npl.candidate
            indices=list(np.flatnonzero(pool)[:32]);indices+=list(np.flatnonzero(~pool)[:256-len(indices)])
            if len(indices)!=256:raise ValueError('Actual noisy/non-noisy U256 smoke batch required')
            ubatch=torch.utils.data.default_collate([ctx['unlabeled_loader'].dataset[int(i)] for i in indices]);model._cd_U_iter=iter([ubatch])
            batch=next(iter(ctx['train_loader']));x,y,d=[v.to('cuda:0') for v in batch[:3]];model.train();model._feature_l_steps=0
            with model._phase1_labeled_feature_context():out=model(x,return_aux=True,domain_labels=d)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,extra=(None,batch[3]),domains=d,
                optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all()) or float(head.grad.norm())<=0
                    or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or model._cd_U_seen!=256 or model._npl.noisy_seen<=0
                    or model._npl.extra_seen!=(model._npl.noisy_seen if RECIPES[c['paper_method']]['rho'] else 0)
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters())):raise ValueError('Actual noisy-hard L/U joint backward smoke failed')
            rows.append(dict(method=c['paper_method'],status='PASS',L_packets=768,U_seen=256,noisy_seen=model._npl.noisy_seen,extra_packets=model._npl.extra_seen,
                default_head_gradient_norm=float(head.grad.norm()),optimizer_updates=0,U_truth_read=False,target_reads=0))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',rows=rows,actual_NoSIG_E30_defaulthead=True,optimizer_updates=0,U_truth_read=False,target_reads=0))


def train(c):
    import shutil
    import torch
    from experiments.cric_dura_comparison.f_train import train as segment
    validate_config(c);root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    cfg=dict(c,output_root=str(root/'segment50'));holder={}
    with installed(cfg,holder):segment(cfg)
    done=read(root/'segment50/complete.json');cp=torch.load(root/'segment50/final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
    obj=holder['model']._npl;count=read(Path(c['noisy_U_cache'])/'complete.json')['unique_noisy_candidates'];recipe=RECIPES[c['paper_method']]
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=20 or done['updates']!=4440 or done['target_access'] or done['U_truth_read']
            or cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440 or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt']
            or cp['args']['baseline_ckpt']!=c['parent_source']+'/final_ssdg.pth' or obj.seen!=1134000 or obj.noisy_seen!=count*20
            or obj.extra_seen!=(count*20 if recipe['rho'] else 0)):raise ValueError('Complete own NoSIG E30+20/noisy-mask exposure required')
    first=Path(c['shared_NoSIG_E20']);second=Path(c['parent_source']);chain=[dict(folder=str(p),**read(p/'complete.json')) for p in (first,second,root/'segment50')]
    if sum(x['updates'] for x in chain)!=11100:raise ValueError('Full E50 trajectory required')
    shutil.copyfile(root/'segment50/final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'objective_state.json',dict(method=c['paper_method'],recipe=recipe,U_seen=obj.seen,noisy_exposures=obj.noisy_seen,self_used=obj.self_used,
        disagreement=obj.disagreement,extra_packets=obj.extra_seen,view_counts=obj.view_counts,unique_noisy_candidates=count,
        U_truth_read=False,target_reads=0,teacher_updates=0,G_updates=0,new_head=False))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=chain,source_contract=c['source_contract'],
        initialization='same_contract_scratch20_own10_own20',ancestors=[x['folder'] for x in chain],target_access=False,U_truth_read=False,head_fit_updates=0,signature_parameters=0,source_diagnostic_only=True))


def analyze(rt):
    import numpy as np
    root=Path(rt['run_root']);ctx=read(rt['context']);base=Path(ctx['project'])/'runs'/BASE_RUN/'F/F0-NoSIG-PL-s2026092701/source_validation'
    with np.load(base/'source_V_predictions.npz',allow_pickle=False) as a:baseline=a['logits'].argmax(-1);y=a['y'];ids=a['ids']
    with np.load(Path(ctx['project'])/'runs'/GRAPH_RUN/'V/metadata.npz',allow_pickle=False) as a:hard=a['raw_confidence']<.95;rx=a['rx_i']
    held=rx==1;results=[]
    for method in METHODS:
        c=read(root/'configs'/(method+'.json'));out=Path(c['output_root']).parent/'source_validation'
        if read(out/'complete.json')['status']!='NOSIG_SOURCE_CONTROL_EVALUATED':raise ValueError('Every fixed source row must complete')
        with np.load(out/'source_V_predictions.npz',allow_pickle=False) as a:
            if not np.array_equal(a['ids'],ids) or not np.array_equal(a['y'],y) or list(a['views'])!=list(VIEWS):raise ValueError('Identical source comparison required')
            pred=a['logits'].argmax(-1)
        raw=100*float((pred[0]==y).mean());delta_raw=100*float(((pred[0]==y).astype(float)-(baseline[0]==y)).mean())
        delta_G=sum(100*float(((pred[k,held]==y[held]).astype(float)-(baseline[k,held]==y[held])).mean()) for k in (3,5,7))/3
        hard_net=int((pred[0,hard]==y[hard]).sum()-(baseline[0,hard]==y[hard]).sum())
        results.append(dict(method=method,source_raw=raw,delta_source_raw=delta_raw,delta_held_G_three=delta_G,hard_raw_net=hard_net,
            source_pass=delta_raw>=-.1-1e-12 and delta_G>=.3-1e-12 and hard_net>0))
    passed=[x for x in results if x['source_pass']];chosen=max(passed,key=lambda x:(x['source_raw'],x['delta_held_G_three'],-METHODS.index(x['method'])))['method'] if passed else None
    write(root/'complete.json',dict(status='NOISY_HARD_SOURCE_PILOT_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,results=results,
        source_selected=chosen,U_truth_read=False,target_reads=0,scope='Source-only single student seed;not target success or automatic three-seed launch'))


def prepare(rt):
    root=Path(rt['run_root']);ctx=read(rt['context']);jobs=[]
    for method in METHODS:
        c=config(ctx,rt['code_commit'],method);path=root/'configs'/(method+'.json');write(path,c)
        jobs.append(dict(id=method,kind='train_eval',config=str(path),completion=str(Path(c['output_root']).parent/'source_validation/complete.json'),successful_statuses=['NOSIG_SOURCE_CONTROL_EVALUATED'],dependencies=[str(root/'smoke.json')]))
    shared=dict(config=rt['runtime_file']);initial=[dict(shared,id='cache',kind='cache',completion=str(root/'noisy_U_cache/complete.json'),successful_statuses=['NOISY_SOURCE_U_CACHE_COMPLETE'],dependencies=[]),
        dict(shared,id='smoke',kind='smoke',completion=str(root/'smoke.json'),successful_statuses=['PASS'],dependencies=[str(root/'noisy_U_cache/complete.json')])]
    for j in initial+jobs:j.update(gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.noisy_hard_source')
    jobs=initial+jobs;write(root/'jobs.json',jobs);return jobs


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
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
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:analyze(rt)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'NOISY_HARD_SOURCE_PILOT_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in list(map(str,range(8))):raise ValueError('One admitted physical GPU required')
    if kind in ('cache','smoke'):validate_runtime(c)
    elif kind=='train_eval':validate_config(c)
    else:raise ValueError('Fixed cache/smoke/train_eval actions only')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='cache':return cache(c)
    if kind=='smoke':return smoke(c,[read(Path(c['run_root'])/'configs'/(m+'.json')) for m in METHODS])
    if kind=='train_eval':train(c);return source_evaluate(c,validator=validate_config)
    raise ValueError('Fixed cache/smoke/train_eval actions only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
