"""One preregistered original-E20 source comparison using existing six U forwards."""
from contextlib import contextmanager
import math
import torch
from . import source_branch_gradients as source
from . import source_optimizer_shadow as shadow
from .source_optimizer_counterfactual import _first_step_options
from .hard_pl_native_as_objective import GuardedNativeAs
from .branch_gradient_projection import branch_projected_gradients
from .parameter_step_projection import project_parameters
from .rejected_rdrop import RECIPE,rejected_rdrop

KIND='fixed_original_E20_source_rejected_rdrop_v1'
RULES=dict(minimum_rejected_U_per_seed=1,minimum_nonzero_group_increment_norm_retention=.5,
    added_descent_aligns_with_rejected_RDrop_in_some_group_each_seed=True,
    all_actual_parameter_constraints_and_nonexpansion_required=True,all_three_fixed_seeds_required=True)


def validate_plan(plan):
    if (plan.get('kind')!=KIND or plan.get('source_rules')!=RULES or plan.get('recipe')!=RECIPE
            or plan.get('seeds')!=[2026092701,2026092702,2026092703] or plan.get('source_epoch')!=20
            or plan.get('steps_per_seed')!=1 or plan.get('L_batch')!=128 or plan.get('U_batch')!=256
            or plan.get('optimizer_state')!='fresh_moments_first_new_native10_step'
            or any(plan.get(k)!=0 for k in ('V_reads','target_reads','U_truth_reads','G_training_reads'))):
        raise ValueError('Only the fixed original-E20 L/U rejected-RDrop source comparison allowed')


@contextmanager
def parameter_snapshots():
    original=shadow.build;values=[]
    def build(*args,**kwargs):
        optimizer=original(*args,**kwargs);parameters=list(args[1])
        def after_step(*unused):values.append(tuple(p.detach().clone() for p in parameters))
        optimizer.register_step_post_hook(after_step);return optimizer
    shadow.build=build
    try:yield values
    finally:shadow.build=original;values.clear()


def compare_steps(model,named,full_l,clean_l,physical_l,old_u,new_u,extra_gradient,options):
    old_joint,old_raw=branch_projected_gradients(named,full_l,old_u,clean_l,physical_l,max_u_ratio=None)
    new_joint,new_raw=branch_projected_gradients(named,full_l,new_u,clean_l,physical_l,max_u_ratio=None)
    metadata=[]
    with parameter_snapshots() as snapshots:
        for gradients in (full_l,old_joint,new_joint):
            _,detail=shadow.shadow_step(model,named,gradients,**options);metadata.append(detail)
        if len(snapshots)!=3:raise ValueError('Exactly three actual disposable optimizer copies required')
        old_protected,old_details=project_parameters(named,snapshots[0],snapshots[1],full_l,clean_l,physical_l)
        new_protected,new_details=project_parameters(named,snapshots[0],snapshots[2],full_l,clean_l,physical_l)
        # Extra descent relative to the matched protected StepNoCap parameter values.
        extra_descent=tuple(a.double()-b.double() for a,b in zip(old_protected,new_protected))
        changes={}
        for group in source.GROUPS:
            relation=source._relation(named,extra_gradient,extra_descent,group)
            changes[group]=dict(added_descent_norm=relation['right_norm'],rejected_RDrop_gradient_norm=relation['left_norm'],
                dot=relation['dot'],cosine=relation['cosine'],
                source_rule_aligns=relation['right_norm']>0 and relation['dot']>0)
    for details in (old_details,new_details):
        if any(not d['actual_parameter_check'] or not d['norm_nonexpansion'] for d in details.values()):
            raise ValueError('All actual FP32 supervised constraints and increment nonexpansion required')
    return dict(raw_gradient_baseline=old_raw,raw_gradient_rejected_RDrop=new_raw,
        parameter_baseline=old_details,parameter_rejected_RDrop=new_details,added_descent=changes,
        actual_optimizer_copies=3,optimizer_copies=metadata,
        original_parameter_updates=0,shadow_parameter_updates_discarded=True,
        interpretation='Only reject-mask RDrop is added to the otherwise identical uncapped gradient and actual-step operators.Independent fresh optimizer copies,one source batch,not live-moment trajectory or a measured loss reduction.')


class RejectedRDropDiagnostic(GuardedNativeAs):
    def __init__(self,c):
        super().__init__(c);self.branch_logits=[]

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        value=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self.branch_logits.append((value,pseudo,accepted));return value

    def guide(self,labeled,student):
        unlabeled=self.pending_gradient_loss
        if unlabeled is None or len(self.branch_logits)!=6:raise ValueError('Exactly six existing U forward results required')
        logits=[v[0] for v in self.branch_logits];pseudo=self.branch_logits[0][1];accepted=self.branch_logits[0][2]
        if any(not torch.equal(y,pseudo) or not torch.equal(a,accepted) for _,y,a in self.branch_logits):
            raise ValueError('Original raw pseudo labels and acceptance must remain identical for every U view')
        step=student._feature_l_steps+self.c.get('step_offset',0)
        physical=(torch.arange(len(accepted),device=accepted.device)+step).remainder(2).bool()
        extra,coverage=rejected_rdrop(logits,accepted,physical);extra=self.settings['weight']*extra
        components=source.labeled_components(source._PAIRS.get(),labeled)
        named=[(n,p) for n,p in student.named_parameters() if p.requires_grad];parameters=[p for _,p in named]
        with source.native_reference_gradients() as reference:
            original_bridge=super().guide(labeled,student)
        if len(reference)!=2:raise ValueError('Original exact full L/U derivative vectors required')
        clean_l,physical_l,extra_gradient,new_u=[tuple(None if g is None else g.detach() for g in
            torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True))
            for loss in (components['L_clean'],components['L_As'],extra,unlabeled+extra)]
        if any(g is not None and not bool(torch.isfinite(g).all()) for values in (clean_l,physical_l,extra_gradient,new_u) for g in values):
            raise ValueError('Finite component and proposed total U derivatives required')
        closure=source.component_closure(named,source._sum(named,reference[1],extra_gradient),new_u)
        comparison=compare_steps(student,named,reference[0],clean_l,physical_l,reference[1],new_u,extra_gradient,_first_step_options(self.c))
        self.source_rejected_facts=dict(coverage=coverage,external_U_weight=self.settings['weight'],extra_loss=float(extra.detach()),
            component_sum_closure=closure,comparison=comparison,L_pair_calls=len(source._PAIRS.get()),U_logits_reused=6,
            actual_derivative_calls=7,hard_pseudo_or_acceptance_changed=False,extra_model_forwards=0,
            extra_classification_head_forwards=0,physical_As_KL_weight=0.,U_truth_reads=0,target_reads=0)
        self.branch_logits=[];return original_bridge


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
    validate_plan(plan);activate(rt);numerical_policy();torch.set_num_threads(2)
    start=time.monotonic();rows=[]
    for seed in plan['seeds']:
        row=next(r for r in rt['rows'] if r['model_seed']==seed);c=read(row['train_config'])
        parents,contract=source_parents(c,validate_config=validate);parent=parents[-1];parent_cfg=parent['config']
        payload=torch.load(Path(parent['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (parent['epoch']!=20 or parent['updates']!=4440 or payload['run_id']!=parent_cfg['run_id']
                or payload['candidate_id']!=parent_cfg['row_id'] or payload['epoch']!=20
                or payload['a1_ema_successful_updates']!=4440 or payload['args']['teacher_ckpt']
                or payload['args']['use_ema_teacher'] or payload['args']['use_unlabeled']):
            raise ValueError('Original source-only same-contract E20 checkpoint/default head required')
        cfg=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=parent['folder'],hard_pl_smoke=True)
        ctx=source.source_loaders(cfg,contract);holder={};torch.manual_seed(seed);device='cpu'
        with installed(cfg,holder,objective_cls=RejectedRDropDiagnostic),class_dimension(cfg),native_installed(original_config(seed)) as native:
            args=arguments(cfg,device);args.input_len=256;args.baseline_ckpt=args.teacher_ckpt='';args.from_scratch=args.a1_scratch_only=True
            if args.use_unlabeled or args.use_ema_teacher:raise ValueError('Native hidden U/EMA loop must be disabled')
            model=t1_model.install(native.build_baseline_model(args,device),cfg).to(device);model.load_state_dict(payload['model'],strict=True);model.eval()
            parameter_before={n:p.detach().clone() for n,p in model.named_parameters()};buffer_before={n:b.detach().clone() for n,b in model.named_buffers()}
            with torch.no_grad():z=model(ctx['train_loader'].dataset.x[:2])
            if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Real original-E20 no-query source smoke failed')
            model.train();model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,domain=batch[:3];metadata=batch[3]
            if x.shape!=(128,2,256) or y.shape!=(128,) or y.min()<0 or y.max()>=6:raise ValueError('Fixed true L128 batch required')
            actual_grad_calls=0;original_grad=torch.autograd.grad
            def counted_grad(*a,**kw):
                nonlocal actual_grad_calls
                actual_grad_calls+=1;return original_grad(*a,**kw)
            torch.autograd.grad=counted_grad
            try:
                with source.labeled_pairs():
                    with model._phase1_labeled_feature_context():output=model(x,return_aux=True,domain_labels=domain)
                    ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
                    correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=output,labels=y,
                        extra=(None,metadata),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
                    if not bool(torch.isfinite(ce+correction)):raise ValueError('Finite unchanged native source objective required')
            finally:torch.autograd.grad=original_grad
            obj=holder['objective'];state=obj.state_dict();facts=obj.source_rejected_facts
            if (actual_grad_calls!=7 or state['calls']!=1 or state['seen']!=256 or state['student_views']!=1536
                    or state['physical_seen']!=128 or state['physical_student_views']!=256 or state['raw_As_seen']
                    or state['EMA_updates'] or state['U_truth_read'] or metrics['T1_L_packets']!=768
                    or sum(model._cd_practical.coverage.values())!=192 or not 0<state['accepted']<=256
                    or facts['coverage']['rejected_U']!=256-state['accepted'] or obj.pending_gradient_loss is not None or obj.branch_logits):
                raise ValueError('Actual source packet,derivative,original acceptance or permission counts differ')
            if any(p.grad is not None or not torch.equal(p.detach(),parameter_before[n]) for n,p in model.named_parameters()):
                raise ValueError('Diagnostic must not write original gradients or parameters')
            changed_buffers=[n for n,b in model.named_buffers() if not torch.equal(b,buffer_before[n])]
            rows.append(dict(model_seed=seed,source_run_id=parent_cfg['run_id'],source_row_id=parent_cfg['row_id'],source_epoch=20,
                L_samples=128,U_samples=256,L_packet_views=768,U_student_packet_views=1536,U_teacher_raw_packet_views=256,
                source_Practical_generated=192,real_checkpoint_smoke_L_packets=2,actual_autograd_calls=actual_grad_calls,
                source_rejected_facts=facts,parameter_updates=0,parameter_grad_writes=0,
                disposable_train_mode_changed_buffer_names=changed_buffers,discarded_model_and_buffers=True,U_truth_reads=0,V_reads=0,target_reads=0))
        del model,obj,holder,ctx,payload,parameter_before,buffer_before,output,ce,correction;gc.collect()
    comparisons=[r['source_rejected_facts']['comparison'] for r in rows]
    retentions=[v['retention'] for c in comparisons for v in c['parameter_rejected_RDrop'].values() if v['raw_increment_norm']>0]
    aligned=[any(v['source_rule_aligns'] for v in c['added_descent'].values()) for c in comparisons]
    rejected=[r['source_rejected_facts']['coverage']['rejected_U'] for r in rows]
    passed=bool(retentions) and min(retentions)>=.5 and min(rejected)>=1 and all(aligned)
    return dict(status='VERIFIED_FIXED_SOURCE_REJECTED_RDROP',kind=KIND,analysis_commit=analysis_commit,
        reference_recipe_execution_commit=rt['code_commit'],source_rules=RULES,recipe=RECIPE,source_rule_pass=passed,
        scientific_status='SOURCE_COUNTERFACTUAL_PASSED' if passed else 'SCIENTIFIC_FALSIFIED_SOURCE',rows=rows,
        minimum_nonzero_group_increment_norm_retention=min(retentions) if retentions else None,
        each_seed_added_descent_alignment_rule_pass=aligned,rejected_U_per_seed=rejected,
        source_epoch=20,steps_per_seed=1,model_seeds=plan['seeds'],source_packet_views=7686,actual_autograd_calls=21,
        optimizer_steps=9,shadow_optimizer_steps=9,parameter_updates=9,
        parameter_grad_writes=sum(m['shadow_parameter_gradient_tensors'] for c in comparisons for m in c['optimizer_copies']),
        original_model_update_counts=dict(optimizer_steps=0,parameter_updates=0,parameter_grad_writes=0,persisted_buffer_updates=0),
        GPU_workers=0,CPU_threads=2,persisted_buffer_updates=0,U_truth_reads=0,V_reads=0,V_gradients=0,target_reads=0,G_training_reads=0,
        shadow_parameter_updates_discarded=True,new_formal_training_updates=0,automatic_candidate_launch=False,
        seconds=time.monotonic()-start,goal_complete=False,
        interpretation='Original fixed E20,one L128/U256 batch perseed,21derivatives/7686source packet views/nine discarded actual physical-AdamW copies.Fresh moments only;no original model/grad/buffer updates,V/Utruth/target/G reads or healthyjob changes.No target-performance or monotonic-loss claim.')
