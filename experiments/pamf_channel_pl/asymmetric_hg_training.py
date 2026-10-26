"""Asymmetric H/G task and identity-relation source controls."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .feature_task_training import PairedCache
from .asymmetric_hg import extra
from .restored_source import validate_context,validate_prefix_headers,validate_e30,SCENES,VIEWS,snapshot,unchanged

RUN='20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01'
CACHE_RUN='20261025-phase1-PAMF-RestoredPL-m9-r01'
PAIR_CACHE_RUN='20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01'
BASELINE_RUN='20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01'
SEED=2026092701
VARIANTS=('HJoint','HAnchor','GJoint','BothJoint')
ROUTES={m:('H','G') for m in VARIANTS}
TASKS=dict(HJoint='H',HAnchor='H',GJoint='G',BothJoint='mean_H_G')
GRADIENT_RULE='common min(.25,.10*median(primary_norm)/max_method_median_CEplusCL_norm);relativeCE1_CL1;12source_batches'


def recipe(method):
    if method not in VARIANTS:raise ValueError('Fixed four asymmetric H/G candidates required')
    return dict(method=method,views=['H','G'],task_view=TASKS[method],CE_coefficient=1.,CL_coefficient=1.,
        restored_feature_detached_in_CL=method=='HAnchor',temperature=.2,
        pseudo_target='argmax_frozen_three_E30_raw_consensus95',held_extra_rx=1,ramp_steps=444,
        main='unchanged_ordinary_self_PL',extra_forwards=2,
        normalization='full_U_task_CE_mean_and_relation_sum_over_2_times_full_U',
        positive='same_pseudo_class_and_different_RX',negative='different_pseudo_class',
        feature_scale='joint_two_view_accepted_detached_mean_std_floor1e-4_then_l2eps1e-4',
        gradient_rule=GRADIENT_RULE,new_head=False,G_updates=0,teacher_updates=0,reference_updates=0)


def config(case,project,commit,method):
    from .training_plan import make_config
    if method not in VARIANTS or case['model_seed']!=SEED:raise ValueError('Fixed source seed01 required')
    case=deepcopy(case);case['source_config']=read(Path(case['parent_source'])/'complete.json')['config']
    c=make_config(case,project,commit);c.pop('channel_pl');rid='F0-'+method+'-sourceHold-s'+str(SEED)
    c.update(run_id=RUN,row_id=rid,algorithm_id='pamf_asymmetric_hg_source_v1',paper_method=method,
        source_case=case,shared_parent_E30=case['parent_source'].rsplit('/',1)[0]+'/segment30',
        parent_source=case['parent_source'].rsplit('/',1)[0]+'/segment30',segment_epochs=20,epoch_offset=30,step_offset=6660,
        output_root=project+'/runs/'+RUN+'/F/'+rid+'/source',source_cache_output=project+'/runs/'+CACHE_RUN+'/source_U_cache',
        bridge_output=project+'/runs/'+RUN+'/source_preparation',
        feature_task_output=project+'/runs/'+PAIR_CACHE_RUN+'/source_preparation',source_G=None,asymmetric_hg=recipe(method))
    c['repl_row'].update(row_id=rid,method=method)
    return c


def validate_config(c):
    method=c.get('paper_method')
    if (method not in VARIANTS or c.get('asymmetric_hg')!=recipe(method)
            or c.get('run_id')!=RUN or c.get('model_seed')!=SEED or c.get('protocol_id')!='F0'
            or c.get('row_id')!='F0-'+str(method)+'-sourceHold-s'+str(SEED)
            or c.get('algorithm_id')!='pamf_asymmetric_hg_source_v1' or c.get('method_id')!='pamf_t1_e50'
            or c.get('num_classes')!=6 or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
            or c.get('segment_epochs')!=20 or c.get('epoch_offset')!=30 or c.get('step_offset')!=6660
            or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
            or c.get('parent_source')!=c.get('shared_parent_E30') or not c.get('parent_source','').endswith('/segment30')
            or c.get('regime')!='matched_sat_plain_pl' or c.get('ablation')!='F21'
            or c.get('unlabeled_batch')!=256 or c.get('labeled_batch')!=128 or c.get('U_start_epoch')!=14
            or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1. or not c.get('U_use')
            or not c.get('default_head_joint') or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop',
                'satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))):
        raise ValueError('Fixed original default-head E30+20 source-only restoration relations required')
    if any(k in c for k in ('tests','target_inputs','target_truth','truth','dataset_path','views_root','checkpoint','teacher_checkpoint',
                            'channel_pl','restored_pl','receiver_reference','feature_task','residual_mix','restoration_bridge')):
        raise ValueError('Source restoration relations reject target/head-replacement/foreign objectives')
    row=c['repl_row']
    if row['structure']!='full' or not all(row[k] for k in ('L_A','L_B','U')) or any(row['blocks'].values()):
        raise ValueError('Original full supervised skeleton required')
    return c


def inputs(cache,metadata,k,device,x,method):
    if method not in VARIANTS:raise ValueError('Declared asymmetric source method required')
    q,mask,h,g=cache.take(metadata,k,device)
    return q,mask,[h,g]


class Objective:
    def __init__(self,c):
        validate_config(c);self.c=c;self.cache=PairedCache(c);self.metrics={}
        proof=read(Path(c['bridge_output'])/'complete.json')
        if (proof['status']!='SOURCE_ASYMMETRIC_HG_PREPARED' or proof['gradient_rule']!=GRADIENT_RULE
                or proof['model_seed']!=SEED or any(proof[k] for k in ('optimizer_updates','V_reads','target_reads','U_truth_read'))
                or not proof['student_teacher_G_parameters_and_registered_buffers_unchanged']
                or not proof['all_auxiliary_classifier_weight_gradients_zero']):
            raise ValueError('Complete no-update restoration-relation calibration required')
        self.weight=proof['decision']['weight']
        if not 0<self.weight<=.25:raise ValueError('Finite bounded common source-derived dose required')
        self.calls=self.seen=self.raw_accepted=self.extra_accepted=self.held_seen=0
        self.raw_forwards=self.H_forwards=self.G_forwards=self.valid_anchors=self.positive_pairs=self.negative_pairs=0
        self.CE_H_accepted=self.CE_G_accepted=0
        self.expected_extra_per_epoch=sum(int(self.cache.base.accepted[i]) for key,i in self.cache.base.index.items() if key[1]!=1)

    def loss(self,student,device):
        import torch
        from torch.nn import functional as F
        try:batch=next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        if (not bool((batch[1]==-1).all()) or bool(torch.as_tensor(batch[3]['tx_label_visible']).any())
                or set(torch.as_tensor(batch[3]['rx_i']).tolist())-set(self.c['source_rxs'])):
            raise ValueError('Hidden source U labels and original RX only')
        x=batch[0].to(device);domain=batch[2].to(device);rx=batch[3]['rx_i'].to(device)
        student.eval()
        with torch.no_grad():confidence,pseudo=student(x).softmax(1).max(1)
        student.train();accepted=confidence>=.95
        logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
        main=(F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean()
        self.calls+=1;k=(self.calls+6659)%3;method=self.c['paper_method']
        q,mask,views=inputs(self.cache,batch[3],k,device,x,method);mask=mask&(rx!=1)
        auxiliary,detail=extra(student,views,domain,q,mask,rx,method);ramp=min(self.calls/444.,1.)
        n=len(x);self.seen+=n;self.raw_accepted+=int(accepted.sum());self.extra_accepted+=int(mask.sum());self.held_seen+=int((rx==1).sum())
        self.raw_forwards+=ROUTES[method].count('raw')*n;self.H_forwards+=ROUTES[method].count('H')*n
        self.G_forwards+=sum(v.startswith('G') for v in ROUTES[method])*n
        self.valid_anchors+=detail['valid_anchors'];self.positive_pairs+=detail['positive_pairs'];self.negative_pairs+=detail['negative_pairs']
        self.CE_H_accepted+=detail['CE_H_accepted'];self.CE_G_accepted+=detail['CE_G_accepted']
        student._cd_U_seen+=n;student._cd_U_accepted+=int(accepted.sum())
        self.metrics=dict(AH_main_loss=main.detach(),AH_extra_loss=auxiliary.detach(),AH_CE_loss=detail['CE_loss'],AH_CL_loss=detail['relation_loss'],AH_extra_weight=self.weight,AH_ramp=ramp,
            AH_CE_H_accepted=self.CE_H_accepted,AH_CE_G_accepted=self.CE_G_accepted,
            AH_U_seen=self.seen,AH_raw_accepted=self.raw_accepted,AH_extra_accepted=self.extra_accepted,AH_held_seen=self.held_seen,
            AH_raw_forwards=self.raw_forwards,AH_H_forwards=self.H_forwards,AH_G_forwards=self.G_forwards,
            AH_valid_anchors=self.valid_anchors,AH_positive_pairs=self.positive_pairs,AH_negative_pairs=self.negative_pairs,
            AH_G_updates=0,AH_teacher_updates=0,AH_U_truth=0)
        return main+self.weight*ramp*auxiliary

    def state_dict(self):
        return dict(calls=self.calls,U_seen=self.seen,raw_accepted=self.raw_accepted,extra_accepted=self.extra_accepted,held_seen=self.held_seen,
            extra_student_forwards=2*self.seen,expected_extra_per_epoch=self.expected_extra_per_epoch,raw_forwards=self.raw_forwards,
            H_forwards=self.H_forwards,G_forwards=self.G_forwards,valid_anchors=self.valid_anchors,
            positive_pairs=self.positive_pairs,negative_pairs=self.negative_pairs,
            CE_H_accepted=self.CE_H_accepted,CE_G_accepted=self.CE_G_accepted,feature_task_gradient_calls=self.calls*(2 if self.c['paper_method']=='BothJoint' else 1),
            reference_updates=0,G_updates=0,teacher_updates=0,U_truth_read=False,target_reads=0,objective_V_reads=0,V_state_updates=0,new_head=False,
            extra_weight=self.weight,auxiliary_classifier_weight_shield=True,views=list(ROUTES[self.c['paper_method']]),
            task_view=TASKS[self.c['paper_method']],restored_feature_detached_in_CL=self.c['paper_method']=='HAnchor',pseudo_target='hard_argmax_frozen_three_E30_raw_consensus95',new_inference_parameters=0)


def parents(c):
    import torch
    validate_config(c);case=c['source_case'];p=Path(case['parent_source']);d20=read(p/'complete.json');contract=read(c['source_contract'])
    cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_prefix_headers(case,d20,cp,read(d20['config']['source_contract']),contract);del cp
    q=Path(c['shared_parent_E30']);d30=read(q/'complete.json');cp=torch.load(q/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_e30(case,d30,cp,read(d30['config']['source_contract']),contract)
    if q!=p.parent/'segment30' or d30['config']['source_cache']!=c['source_cache']:raise ValueError('Own E30 input contract differs')
    return [dict(folder=str(p),**d20),dict(folder=str(q),**d30)],cp


def smoke(c,output):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.pamf_repl_completion.objectives import loader_scope
    from experiments.pamf_restored_pl.objective import installed
    _,payload=parents(c);obj=Objective(c)
    with installed(c,obj):
        ctx=f_train.context(c)
        with f_train.class_dimension(c),native_installed(f_train.original_config(SEED)) as native:
            a=f_train.arguments(c,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
            if a.use_unlabeled or a.use_ema_teacher:raise ValueError('Native U/EMA off')
            student=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
        student.load_state_dict(payload['model'],strict=True);student.eval()
        with torch.no_grad():logits=student(ctx['train_loader'].dataset.x[:2].to('cuda:0'))
        if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Actual own E30 default-head source L2 smoke failed')
        student.train()
        student._cd_U_loader=ctx['unlabeled_loader'];student._cd_U_iter=iter(student._cd_U_loader)
        batch=next(iter(ctx['train_loader']));x,y,d=[v.to('cuda:0') for v in batch[:3]]
        with student._phase1_labeled_feature_context():out=student(x,return_aux=True,domain_labels=d)
        ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=out,labels=y,
            extra=(None,batch[3]),domains=d,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
        total=ce+correction;total.backward();head=student.id_backbone.encoder.core.id_backbone.cls_head.weight
        if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all()) or float(head.grad.norm())<=0
                or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or obj.calls!=1 or obj.seen!=256 or obj.extra_accepted==0
                or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in student.parameters())):
            raise ValueError('Actual restoration-bridge joint backward smoke failed')
    write(output,dict(status='PASS',model_seed=SEED,L_forward_packets=768,U_seen=256,extra_accepted=obj.extra_accepted,
        head_finite_nonzero=True,optimizer_updates=0,query_reads=0,U_truth_read=False,method=c['paper_method']))


def train(c):
    import shutil
    from experiments.cric_dura_comparison.f_train import train as segment
    from experiments.pamf_restored_pl.objective import installed
    chain,payload=parents(c);del payload;root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();folder=root/'segment50';cfg=dict(c,output_root=str(folder));obj=Objective(cfg)
    write(root/'resolved_config.json',c)
    with installed(cfg,obj):segment(cfg)
    state=obj.state_dict()
    routes=ROUTES[c['paper_method']]
    if (state['calls']!=4440 or state['U_seen']!=1134000 or state['held_seen']!=226800
            or state['extra_accepted']!=20*state['expected_extra_per_epoch'] or state['reference_updates'] or state['G_updates']
            or state['U_truth_read'] or state['target_reads'] or state['objective_V_reads'] or state['V_state_updates']
            or state['teacher_updates'] or state['new_head'] or state['extra_student_forwards']!=2268000
            or state['raw_forwards']!=1134000*routes.count('raw') or state['H_forwards']!=1134000*routes.count('H')
            or state['G_forwards']!=1134000*sum(v.startswith('G') for v in routes)
            or not 0<state['valid_anchors']<=2*state['extra_accepted'] or state['positive_pairs']<=0 or state['negative_pairs']<=0
            or state['CE_H_accepted']!=(state['extra_accepted'] if c['paper_method'] in ('HJoint','HAnchor','BothJoint') else 0)
            or state['CE_G_accepted']!=(state['extra_accepted'] if c['paper_method'] in ('GJoint','BothJoint') else 0)):
        raise ValueError('Source restoration relation execution/budget differs')
    write(folder/'asymmetric_hg_state.json',state);done=read(folder/'complete.json');chain.append(dict(folder=str(folder),**done))
    shutil.copyfile(folder/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',epoch=50,updates=11100,method_id='pamf_asymmetric_hg_source_v1',config=c,
        segments=chain,source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in chain],target_access=False,U_truth_read=False,head_fit_updates=0,
        actual_asymmetric_hg=state,seconds=time.perf_counter()-start,source_diagnostic_only=True))


def source_evaluate(c):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload,load_G
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .restored_source import source_channel,paired_counts
    validate_config(c);source=Path(c['output_root']);done=read(source/'complete.json');contract=read(c['source_contract'])
    if done['config']!=c or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read']:
        raise ValueError('Complete fixed source student required')
    cp=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (cp['run_id']!=RUN or cp['candidate_id']!=c['row_id'] or cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440
            or cp['checkpoint_selection']!='final_only' or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt']):
        raise ValueError('Actual default-head own E30+20 checkpoint differs')
    model=reader_from_payload(c,cp,'cuda:0');del cp
    g=load_G(c['source_G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    data=SourceDataset(c['source_cache'],'V',contract);n=len(data)
    if n!=27000:raise ValueError('Complete original V required')
    old=Path(c['project'])/'runs'/BASELINE_RUN/'source_V_predictions.npz'
    with np.load(old,allow_pickle=False) as a:
        if not np.array_equal(a['ids'],data.metadata['ids']) or not np.array_equal(a['y'],data.metadata['y']):raise ValueError('Source A physical order differs')
        j=next(i for i,(m,s) in enumerate(zip(a['methods'],a['seeds'])) if m=='PLControl' and int(s)==SEED)
        if tuple(a['views'].tolist())!=VIEWS:raise ValueError('Exact original eight source views required')
        control=a['logits'][j].argmax(-1).copy()
    out=source.parent/'source_validation';out.mkdir(exist_ok=False);logits=np.empty((8,n,6),np.float32);offset=0
    # Initialize graph before the frozen registered-buffer snapshot.
    with torch.no_grad():engine(data.x[:2].to('cuda:0'))
    before=snapshot(model);gbefore=snapshot(g)
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');views=[x,engine(x)[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract);views.extend((h,engine(h)[0].clone()))
            for k,view in enumerate(views):
                q=model(view)
                if q.shape!=(len(x),6) or not bool(torch.isfinite(q).all()):raise ValueError('Finite default source head required')
                logits[k,offset:offset+len(x)]=fp32_copy(q)
            offset+=len(x)
            write(out/'progress.json',dict(status='SOURCE_VALIDATING',seen=offset,total=n,target_reads=0),exclusive=False)
    if offset!=n or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Source inference mutated state or incomplete')
    y=data.metadata['y'];pred=logits.argmax(-1);rx=data.metadata['rx_i'];records=[]
    strata=[('overall','ALL',np.ones(n,bool)),('extra_RX_holdout','held',rx==1),('extra_RX_holdout','trained',rx!=1)]
    for dimension,values in [('TX',y),('RX',rx),('day',data.metadata['day_i'])]:
        strata.extend((dimension,str(value),values==value) for value in np.unique(values))
    for k,view in enumerate(VIEWS):
        for dimension,stratum,mask in strata:
            counts=paired_counts(control[k],pred[k],y,mask)
            counts['control_accuracy']=counts.pop('raw_accuracy');counts['candidate_accuracy']=counts.pop('restored_accuracy')
            records.append(dict(method=c['paper_method'],model_seed=SEED,view=view,dimension=dimension,stratum=stratum,**counts))
    clean=next(r for r in records if r['view']=='raw' and r['dimension']=='overall')
    recovery=[r for r in records if r['view'] in ['G_'+scene for scene in SCENES] and r['dimension']=='extra_RX_holdout' and r['stratum']=='held']
    gain=100*sum(r['net_correct'] for r in recovery)/sum(r['n'] for r in recovery);clean_delta=100*clean['net_correct']/clean['n']
    np.savez(out/'source_V_predictions.npz',logits=logits,y=y,ids=data.metadata['ids'],views=np.asarray(VIEWS))
    result=dict(status='SOURCE_HOLDOUT_EVALUATED',method=c['paper_method'],model_seed=SEED,source_V_seen=n,results=records,
        held_RX=1,held_RX_G_three_scene_gain_pp=gain,overall_raw_clean_delta_pp=clean_delta,
        source_scientific_pass=bool(100*sum(r['net_correct'] for r in recovery)>=.3*sum(r['n'] for r in recovery)
            and 100*clean['net_correct']>=-.1*clean['n']),default_head=True,
        parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,target_reads=0,U_truth_read=False,
        scope='Single student seed;source-only extra-U-branch RX holdout,not unseen-base RX,not target performance')
    write(out/'complete.json',result)


def prepare(rt):
    c=read(rt['context']);validate_context(c);case=next(x for x in c['cases'] if x['model_seed']==SEED)
    root=Path(rt['run_root']);jobs=[]
    for method in VARIANTS:
        cfg=config(case,rt['project'],rt['code_commit'],method)
        cfg.update(frozen_sources=rt['frozen_sources'],project=rt['project'],source_G=c['G']);validate_config(cfg)
        path=root/'configs'/(method+'.json');write(path,cfg)
        jobs.append(dict(id=method,kind='train_eval',config=str(path),completion=str(Path(cfg['output_root']).parent/'source_validation/complete.json'),
            successful_statuses=['SOURCE_HOLDOUT_EVALUATED'],dependencies=[str(root/'smoke.json')],gpu_pool=list(range(8)),
            execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.asymmetric_hg_training',row_id=cfg['row_id']))
    calibration=dict(run_id=RUN,source_context=c,output_root=str(root/'source_preparation'),frozen_sources=rt['frozen_sources'],code_commit=rt['code_commit'])
    path=root/'configs/calibration.json';write(path,calibration)
    prep=dict(id='source_calibration',kind='calibration',config=str(path),completion=str(root/'source_preparation/complete.json'),
        successful_statuses=['SOURCE_ASYMMETRIC_HG_PREPARED'],dependencies=[],gpu_pool=list(range(8)),
        execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.asymmetric_hg_training')
    path=root/'configs/smoke.json';write(path,dict(train_config=str(root/'configs/BothJoint.json'),output=str(root/'smoke.json'),frozen_sources=rt['frozen_sources']))
    smoke_job=dict(id='smoke',kind='smoke',config=str(path),completion=str(root/'smoke.json'),successful_statuses=['PASS'],dependencies=[prep['completion']],
        gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.asymmetric_hg_training')
    jobs=[prep,smoke_job]+jobs
    write(root/'jobs.json',jobs);write(root/'frozen_source_matrix.json',dict(run_id=RUN,methods=list(VARIANTS),model_seed=SEED,
        routes={k:list(v) for k,v in ROUTES.items()},gradient_rule=GRADIENT_RULE,epochs=[20,10,20],held_extra_RX=1,
        source_thresholds=dict(held_G_mean_gain_pp=.3,clean_min_delta_pp=-.1),
        selection='Among passes,max held_RX_G_mean_gain;exactties alphabetical method;no target access or expansion within this controller'))
    return jobs


def summarize(rt,jobs):
    results=[read(job['completion']) for job in jobs if job['kind']=='train_eval']
    if len(results)!=4 or any(r['status']!='SOURCE_HOLDOUT_EVALUATED' for r in results):raise ValueError('Complete source matrix required')
    eligible=[r for r in results if r['source_scientific_pass']]
    chosen=sorted(eligible,key=lambda r:(-r['held_RX_G_three_scene_gain_pp'],r['method']))
    write(Path(rt['run_root'])/'complete.json',dict(status='SOURCE_HOLDOUT_ANALYZED',results=results,
        selected_for_future_three_seed=chosen[0]['method'] if chosen else None,source_thresholds=dict(held_G_mean_gain_pp=.3,clean_min_delta_pp=-.1),
        target_reads=0,scope='Source diagnostic only;three-seed target training/evaluation not launched by this run'))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed independent source holdout matrix required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs},started_at=time.time())
    active={};stop=False;atomic(root/'pipeline_state.json',state)
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)
            if not good:stop=True
            del active[key]
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            available=[]
            for candidate in job['gpu_pool']:
                pids,free=capacity(candidate);load=len(pids|{pid for pid,g in pending.items() if g==candidate})
                if load<2 and free>=10000:available.append((load,-free,candidate))
            if not available:continue
            gpu=min(available)[2]
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop)
        atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:summarize(rt,jobs)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'SOURCE_HOLDOUT_ANALYZED';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2)
    if kind=='calibration':
        from .asymmetric_hg_calibration import prepare_source
        return prepare_source(c)
    if kind=='smoke':return smoke(read(c['train_config']),c['output'])
    if kind=='train_eval':
        train(c);return source_evaluate(c)
    raise ValueError('Only source calibration/smoke/train_eval stages allowed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
