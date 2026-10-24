"""Source-only gradient decomposition on disposable original-E20 model copies."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch
from .hard_pl_objective import masked_pair
from .hard_pl_native_as_objective import GuardedNativeAs
from .u_gradprotect_objective import parameter_group, projected_gradients

GROUPS=('t2','cls_head','other')
BRANCHES=('L_clean','L_As','L_aux','U_clean','U_As','U_aux')
_PAIRS=ContextVar('source_diagnostic_labeled_pairs',default=None)


@contextmanager
def labeled_pairs():
    """Reuse the four original L pair calls; always restore the process-local helper."""
    from experiments.cric_dura_comparison import t1_model
    if _PAIRS.get() is not None:raise ValueError('One disposable source pair capture at a time')
    rows=[];token=_PAIRS.set(rows);original=t1_model.pair
    def pair(*args,**kwargs):
        value=original(*args,**kwargs);rows.append(value);return value
    t1_model.pair=pair
    try:yield rows
    finally:
        t1_model.pair=original;_PAIRS.reset(token)


def labeled_components(pairs,total):
    if pairs is None or len(pairs)!=4:raise ValueError('Exactly four original L pairs required')
    clean=.5*pairs[0][0]+.25*pairs[1][0]+.25*pairs[3][0]
    physical=.25*pairs[2][0]
    return dict(L_clean=clean,L_As=physical,L_aux=total-clean-physical)


def unlabeled_components(logits,pseudo,accepted,physical,total):
    if len(logits)!=6:raise ValueError('Exactly six existing U student logits required')
    if (physical.dtype!=torch.bool or physical.shape!=pseudo.shape or not len(pseudo)
            or len(pseudo)%2 or int(physical.sum())*2!=len(pseudo)):
        raise ValueError('Exact original even physical half required')
    ordinary=masked_pair(logits[0],logits[1],pseudo,accepted,1.)[0]
    mixed=masked_pair(logits[2][~physical],logits[3][~physical],pseudo[~physical],accepted[~physical],1.)[0]
    channel=masked_pair(logits[2][physical],logits[3][physical],pseudo[physical],accepted[physical],0.)[0]
    raw=masked_pair(logits[4],logits[5],pseudo,accepted,1.)[0]
    clean=6./1.25*(.5*ordinary+.25*mixed+.25*raw)
    channel=6./1.25*.25*channel
    return dict(U_clean=clean,U_As=channel,U_aux=total-clean-channel)


def _sum(named,*vectors):
    output=[]
    for index,(_,parameter) in enumerate(named):
        values=[v[index] for v in vectors if v[index] is not None]
        output.append(sum(values[1:],values[0].clone()) if values else None)
    return tuple(output)


def _relation(named,left,right,group):
    ll=rr=dot=0.
    for (name,parameter),a,b in zip(named,left,right):
        if parameter_group(name)!=group:continue
        if a is not None:ll+=float(a.double().square().sum())
        if b is not None:rr+=float(b.double().square().sum())
        if a is not None and b is not None:dot+=float((a.double()*b.double()).sum())
    return dict(left_norm=math.sqrt(ll),right_norm=math.sqrt(rr),dot=dot,
                cosine=dot/math.sqrt(ll*rr) if ll>0 and rr>0 else None)


def gradient_facts(named,losses):
    """Six component derivatives; no .backward(), optimizer, parameter or buffer writes."""
    if set(losses)!=set(BRANCHES) or not named:raise ValueError('Fixed six source loss components required')
    parameters=[p for _,p in named];gradients={}
    for name in BRANCHES:
        loss=losses[name]
        if loss.ndim or not bool(torch.isfinite(loss)) or not loss.requires_grad:
            raise ValueError('Finite differentiable scalar source loss required')
        values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        if any(v is not None and not bool(torch.isfinite(v).all()) for v in values):
            raise ValueError('Finite source component gradients required')
        gradients[name]=tuple(None if v is None else v.detach() for v in values)
    gradients['L_full']=_sum(named,*(gradients[k] for k in BRANCHES[:3]))
    gradients['U_full']=_sum(named,*(gradients[k] for k in BRANCHES[3:]))
    expected,guard=projected_gradients(named,gradients['L_full'],gradients['U_full'])
    applied=[]
    for value,labeled in zip(expected,gradients['L_full']):
        applied.append(None if value is None else value.detach() if labeled is None else value.detach()-labeled)
    gradients['applied_U']=tuple(applied)
    groups={}
    for group in GROUPS:
        anchors={}
        for anchor in ('L_full','L_clean','L_As','L_aux'):
            anchors[anchor]=dict(before=_relation(named,gradients[anchor],gradients['U_full'],group),
                                 after=_relation(named,gradients[anchor],gradients['applied_U'],group))
        groups[group]=dict(
            norms={k:_relation(named,v,v,group)['left_norm'] for k,v in gradients.items()},
            anchors=anchors,original_cap1_guard=guard[group],
            L_clean_vs_L_As=_relation(named,gradients['L_clean'],gradients['L_As'],group),
            U_clean_vs_U_As=_relation(named,gradients['U_clean'],gradients['U_As'],group),
            L_clean_vs_U_clean=_relation(named,gradients['L_clean'],gradients['U_clean'],group),
            L_As_vs_U_As=_relation(named,gradients['L_As'],gradients['U_As'],group))
    return dict(groups=groups,losses={k:float(v.detach()) for k,v in losses.items()},
                component_autograd_calls=6,projection='existing full-L cap1;diagnostic only',
                measured_stage='Parameter gradients before native globalclip5/AdamW/decay'),expected


class BranchDiagnostic(GuardedNativeAs):
    """The frozen NativeAs recipe, adding only disposable derivative observations."""
    def __init__(self,c):
        super().__init__(c);self.branch_logits=[]

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        value=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self.branch_logits.append((value,pseudo,accepted));return value

    def guide(self,labeled,student):
        unlabeled=self.pending_gradient_loss
        if unlabeled is None or len(self.branch_logits)!=6:raise ValueError('One original complete source L/U step required')
        logits=[v[0] for v in self.branch_logits];pseudo=self.branch_logits[0][1];accepted=self.branch_logits[0][2]
        if any(not torch.equal(p,pseudo) or not torch.equal(a,accepted) for _,p,a in self.branch_logits):
            raise ValueError('All original U views must share unchanged raw pseudo labels and acceptance')
        step=student._feature_l_steps+self.c.get('step_offset',0)
        physical=(torch.arange(len(pseudo),device=pseudo.device)+step).remainder(2).bool()
        losses=labeled_components(_PAIRS.get(),labeled)
        losses.update(unlabeled_components(logits,pseudo,accepted,physical,unlabeled))
        named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
        facts,expected=gradient_facts(named,losses)
        result=super().guide(labeled,student)
        actual=self.smoke_guard_expected
        for (name,_),value in zip(named,expected):
            observed=actual[name]
            if ((value is None)!=(observed is None) or value is not None
                    and not torch.allclose(value,observed,rtol=3e-4,atol=2e-6)):
                diagnostic=dict(parameter=name,component_sum_is_none=value is None,original_is_none=observed is None)
                if value is not None and observed is not None:
                    diagnostic.update(max_absolute_difference=float((value-observed).abs().max()),
                        component_sum_max_absolute=float(value.abs().max()),original_max_absolute=float(observed.abs().max()))
                raise ValueError('Component gradient sum differs from original complete-L/U guard: '+repr(diagnostic))
        facts.update(original_full_gradient_bridge_matches=True,all_six_U_logits_reused=True,
                     L_pair_calls=len(_PAIRS.get()),U_logits_captured=len(logits),raw_accepted=int(accepted.sum()))
        self.source_branch_facts=facts;self.branch_logits=[]
        return result


def source_loaders(c,contract):
    """Open L/U only; the ordinary training context also opens V and is not used here."""
    from experiments.cric_dura_comparison.f_data import SourceDataset,FixedEpochLoader
    labeled=SourceDataset(c['source_cache'],'L_s',contract)
    unlabeled=SourceDataset(c['source_cache'],'U_s',contract)
    return dict(train_loader=FixedEpochLoader(labeled,128,c['model_seed'],drop_last=True),
                unlabeled_loader=FixedEpochLoader(unlabeled,256,c['model_seed'],drop_last=False))


def collect(rt,plan,analysis_commit):
    import gc,time
    from pathlib import Path
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.f_train import arguments,original_config,class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison import t1_model
    from .hard_pl_train import installed
    from .hard_pl_early_train import source_parents
    from .hard_pl_native_as import validate
    activate(rt);numerical_policy();torch.set_num_threads(2)
    if (plan['seeds']!=[2026092701,2026092702,2026092703] or plan['source_epoch']!=20
            or plan['steps_per_seed']!=1 or plan['L_batch']!=128 or plan['U_batch']!=256):
        raise ValueError('Only the fixed small original-E20 L/U branch diagnostic is allowed')
    start=time.monotonic();rows=[]
    for seed in plan['seeds']:
        row=next(r for r in rt['rows'] if r['model_seed']==seed);c=read(row['train_config'])
        parents,contract=source_parents(c,validate_config=validate);parent=parents[-1];parent_cfg=parent['config']
        payload=torch.load(Path(parent['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (parent['epoch']!=20 or parent['updates']!=4440 or payload['run_id']!=parent_cfg['run_id']
                or payload['candidate_id']!=parent_cfg['row_id'] or payload['epoch']!=20
                or payload['a1_ema_successful_updates']!=4440 or payload['args']['teacher_ckpt']
                or payload['args']['use_ema_teacher'] or payload['args']['use_unlabeled']):
            raise ValueError('Matching original source-only E20 checkpoint/default head required')
        cfg=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=parent['folder'],hard_pl_smoke=True)
        ctx=source_loaders(cfg,contract);holder={};torch.manual_seed(seed);device='cpu'
        with installed(cfg,holder,objective_cls=BranchDiagnostic),class_dimension(cfg),native_installed(original_config(seed)) as native:
            args=arguments(cfg,device);args.input_len=256;args.baseline_ckpt=args.teacher_ckpt=''
            args.from_scratch=args.a1_scratch_only=True
            if args.use_unlabeled or args.use_ema_teacher:raise ValueError('Native hidden U/EMA loop must stay disabled')
            model=t1_model.install(native.build_baseline_model(args,device),cfg).to(device)
            model.load_state_dict(payload['model'],strict=True);model.eval()
            parameter_before={n:p.detach().clone() for n,p in model.named_parameters()}
            buffer_before={n:b.detach().clone() for n,b in model.named_buffers()}
            with torch.no_grad():z=model(ctx['train_loader'].dataset.x[:2])
            if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real original-E20 no-query source smoke failed')
            model.train();model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,domain=batch[:3]
            if x.shape!=(128,2,256) or y.shape!=(128,) or y.min()<0 or y.max()>=6:
                raise ValueError('Fixed true L128 source batch required')
            metadata=batch[3];actual_grad_calls=0;original_grad=torch.autograd.grad
            def counted_grad(*a,**kw):
                nonlocal actual_grad_calls
                actual_grad_calls+=1;return original_grad(*a,**kw)
            torch.autograd.grad=counted_grad
            try:
                with labeled_pairs():
                    with model._phase1_labeled_feature_context():output=model(x,return_aux=True,domain_labels=domain)
                    ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
                    correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=output,labels=y,
                        extra=(None,metadata),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
                    if not bool(torch.isfinite(ce+correction)):raise ValueError('Finite original diagnostic objective required')
            finally:torch.autograd.grad=original_grad
            obj=holder['objective'];state=obj.state_dict();facts=obj.source_branch_facts
            if (actual_grad_calls!=9 or state['calls']!=1 or state['seen']!=256 or state['student_views']!=1536
                    or state['physical_seen']!=128 or state['physical_student_views']!=256 or state['raw_As_seen']
                    or state['EMA_updates'] or state['U_truth_read'] or metrics['T1_L_packets']!=768
                    or sum(model._cd_practical.coverage.values())!=192 or not 0<state['accepted']<=256
                    or obj.pending_gradient_loss is not None or obj.branch_logits):
                raise ValueError('Actual native source packets/permission/derivative/capture counts differ')
            if any(p.grad is not None or not torch.equal(p.detach(),parameter_before[n]) for n,p in model.named_parameters()):
                raise ValueError('Disposable diagnostic must not write parameter gradients or update weights')
            changed_buffers=[n for n,b in model.named_buffers() if not torch.equal(b,buffer_before[n])]
            rows.append(dict(model_seed=seed,source_run_id=parent_cfg['run_id'],source_row_id=parent_cfg['row_id'],
                source_epoch=20,L_samples=128,U_samples=256,L_packet_views=768,U_student_packet_views=1536,
                U_teacher_raw_packet_views=256,source_Practical_generated=192,real_checkpoint_smoke_L_packets=2,
                actual_autograd_calls=actual_grad_calls,source_branch_facts=facts,
                parameter_updates=0,parameter_grad_writes=0,disposable_train_mode_changed_buffer_names=changed_buffers,
                discarded_model_and_buffers=True,U_truth_reads=0,V_reads=0,target_reads=0))
        del model,obj,holder,ctx,payload,parameter_before,buffer_before,output,ce,correction;gc.collect()
    return dict(status='VERIFIED_FIXED_SOURCE_BRANCH_GRADIENT_DIAGNOSTIC',analysis_commit=analysis_commit,
        reference_recipe_execution_commit=rt['code_commit'],seconds=time.monotonic()-start,rows=rows,
        source_epoch=20,steps_per_seed=1,model_seeds=plan['seeds'],source_packet_views=7686,
        GPU_workers=0,CPU_threads=2,actual_autograd_calls=sum(r['actual_autograd_calls'] for r in rows),
        optimizer_steps=0,parameter_updates=0,parameter_grad_writes=0,persisted_buffer_updates=0,
        U_truth_reads=0,V_reads=0,V_gradients=0,target_reads=0,G_training_reads=0,new_training=False,
        active_jobs_changed=False,NONBLOCKING=True,goal_complete=False,
        interpretation='One original-E20 source L/U step per seed on disposable CPU train-mode copies;not a complete trajectory or optimizer update. Full-L protection need not protect each component. Norms of overlapping component gradients are not additive shares. No scientific promotion threshold or automatic new candidate.')
