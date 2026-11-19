"""One owner: source continuation, frozen clean prediction and blind-U scoring."""
import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_component_guard import plan as base_plan, runner as base
from experiments.pamf_repl_step_guard import runner as scored
from . import plan
from .objective import installed


def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train, t1_model, f_optimizer
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    ctx=read(rt['context']); checks=[]
    for row in plan.candidate_rows():
        c=base_plan.configuration(ctx,row,rt['project'],rt['code_commit'])
        prefix=base_plan.parent(c)
        cp=torch.load(Path(prefix['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        cfg=dict(c,epoch_offset=20,step_offset=4440,segment_epochs=10,hard_pl_smoke=True,
                 output_root=str(Path(c['output_root'])/'segment30'))
        seed = row['model_seed']
        holder={}; torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
        with installed(cfg,holder):
            loaders=f_train.context(cfg)
            with f_train.class_dimension(cfg),native_scope(f_train.original_config(seed)) as native:
                a=f_train.arguments(cfg,'cuda:0'); a.input_len=256
                net=t1_model.install(native.build_baseline_model(a,'cuda:0'),cfg).cuda()
            net.load_state_dict(cp['model'],strict=True); graph=base_plan.check_graph(net,cfg)
            net.train();net._cd_U_loader=loaders['unlabeled_loader'];net._cd_U_iter=iter(net._cd_U_loader)
            batch=next(iter(loaders['train_loader']));x,y,domain=[v.cuda() for v in batch[:3]]
            metadata={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch[3].items()}
            params=[p for p in net.parameters() if p.requires_grad]
            opt=f_optimizer.build(net,params,'CENTER_UNITS',lr=float(a.lr),weight_decay=float(a.weight_decay))
            with net._phase1_labeled_feature_context():output=net(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,_=net._source_labeled_feature_hook(model=net,labeled_output=output,labels=y,
                extra=(None,metadata),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            loss=ce+correction
            if not bool(torch.isfinite(loss)):raise ValueError('Finite source loss required')
            loss.backward();f_optimizer.clip_physical(opt,params,5.);opt.step()
            obj=holder['objective']; state=obj.state_dict()
            if (state['calls']!=1 or not state['accepted'] or state['U_truth_read']
                    or state['U_gradient_guard']['calls']!=1 or state['sampled_reference_adamw']['calls']!=1
                    or state['sampled_reference_adamw']['pending'] or not obj.observer.last_details
                    or obj.parts[cfg['tuning_recipe']['reference']].ndim != 0):
                raise ValueError('Exact reference and sampled actual AdamW update required')
            checks.append(dict(row=row['row_id'],graph=graph,accepted=state['accepted'],
                optimizer_observer=state['sampled_reference_adamw'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del net,opt,loaders,cp,loss
        torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=checks,real_checkpoint=True,
        optimizer_updates=len(checks),all_updates_discarded=True,query_reads=0,U_truth_reads=0))


def worker(c,kind):
    if kind=='score' and c.get('diagnostic_type')=='U_truth_after_freeze':
        from experiments.pamf_repl_step_guard.u_scoring import score
        return score(c)
    if kind=='smoke':
        from experiments.pamf_repl_completion.runner import activate
        activate(c);return smoke(c)
    return base.worker(c,kind)


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch
    def admits(gpu,active,mib):
        if any(item['gpu']==gpu for _,item in active.values()):return False
        return physical_admits(gpu,active,max(mib,12288),max_total=5)
    old=preparation.prepare,queue.admits,queue.launch,queue.analyze
    preparation.prepare,queue.admits=scored.prepare,admits
    queue.launch=partial(launch,module='experiments.pamf_repl_priority.runner')
    queue.analyze=base.analyze
    try:return queue.dispatch(dict(rt,prioritize_prediction=True))
    finally:preparation.prepare,queue.admits,queue.launch,queue.analyze=old


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True)
    args=p.parse_args();c=read(args.config)
    with plan.scoped():
        old=base.installed,scored.plan
        base.installed,scored.plan=installed,base_plan
        try:dispatch(c) if args.action=='dispatch' else worker(c,args.kind)
        finally:base.installed,scored.plan=old
