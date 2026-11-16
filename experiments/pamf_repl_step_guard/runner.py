"""Fixed original-REPL continuation with actual AdamW extra-step protection."""
import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_repl_weight import runner as reference
from . import plan
from .objective import installed


def prepare(rt):
    previous=reference.plan;reference.plan=plan
    try:jobs=reference.prepare(rt)
    finally:reference.plan=previous
    root=Path(rt['run_root']);sources=read(root/'sources.json')
    for row in plan.candidate_rows():
        rid=row['row_id'];source=Path(sources[rid]['source_output']);output=root/'readouts'/rid/'U_diagnostic'
        jid='u_score__'+rid;path=root/'configs'/f'{jid}.json'
        write(path,dict(diagnostic_type='U_truth_after_freeze',row_id=rid,protocol_id=row['protocol_id'],
                        source_output=str(source),output_root=str(output)))
        jobs.append(dict(id=jid,kind='score',config=str(path),completion=str(output/'complete.json'),
                         successful_statuses=['SCORED_U_DIAGNOSTIC'],dependencies=[str(source/'complete.json')]))
    write(root/'jobs.json',jobs,exclusive=False)
    return jobs


def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model,f_optimizer
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    context=read(rt['context']);values=[]
    for row in plan.candidate_rows():
        c=plan.configuration(context,row,rt['project'],rt['code_commit'])
        prefix=plan.parent(c)
        cp=torch.load(Path(prefix['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        cfg=dict(c,epoch_offset=20,step_offset=4440,segment_epochs=10,hard_pl_smoke=True,
                 output_root=str(Path(c['output_root'])/'segment30'))
        holder={};torch.manual_seed(plan.SEED);torch.cuda.reset_peak_memory_stats()
        with installed(cfg,holder):
            loaders=f_train.context(cfg)
            with f_train.class_dimension(cfg),native_scope(f_train.original_config(plan.SEED)) as native:
                a=f_train.arguments(cfg,'cuda:0');a.input_len=256
                net=t1_model.install(native.build_baseline_model(a,'cuda:0'),cfg).cuda()
            net.load_state_dict(cp['model'],strict=True);graph=plan.check_graph(net,cfg)
            net.train();net._cd_U_loader=loaders['unlabeled_loader'];net._cd_U_iter=iter(net._cd_U_loader)
            batch=next(iter(loaders['train_loader']));x,y,domain=[v.cuda() for v in batch[:3]]
            metadata={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch[3].items()}
            opt=f_optimizer.build(net,[p for p in net.parameters() if p.requires_grad],
                                  'CENTER_UNITS',lr=float(a.lr),weight_decay=float(a.weight_decay))
            with net._phase1_labeled_feature_context():output=net(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,_=net._source_labeled_feature_hook(model=net,labeled_output=output,labels=y,
                extra=(None,metadata),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            loss=ce+correction
            if not bool(torch.isfinite(loss)):raise ValueError('Finite source loss required')
            loss.backward();f_optimizer.clip_physical(opt,[p for p in net.parameters() if p.requires_grad],5.);opt.step()
            state=holder['objective'].state_dict();step=state['U_parameter_step_guard']
            details=holder['objective'].parameter_guard.last_details
            if (state['calls']!=1 or not state['accepted'] or state['U_truth_read']
                    or state['U_gradient_guard']['calls']!=1 or step['calls']!=1
                    or step['pending_anchors'] or not all(d['actual_parameter_check'] and d['norm_nonexpansion'] for d in details.values())):
                raise ValueError('Actual source-checkpoint step protection did not complete')
            values.append(dict(row=row['row_id'],graph=graph,accepted=state['accepted'],step_guard=step,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del net,opt,loaders,cp,loss
        torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=values,real_checkpoint=True,
        optimizer_updates=2,all_updates_discarded=True,query_reads=0,U_truth_reads=0))


def worker(c,kind):
    if kind=='score':
        if c.get('diagnostic_type')=='U_truth_after_freeze':
            from .u_scoring import score
            return score(c)
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from experiments.pamf_physics_tri.source import train_trajectory
        train_trajectory(c,plan.parent(c),installed,ssl=True,physical_L=False)
        for epoch,steps in ((30,2220),(50,4440)):
            folder=Path(c['output_root'])/f'segment{epoch}'
            state=read(folder/'repl_state.json')
            if (state['U_parameter_step_guard']['calls']!=steps
                    or not (folder/'U_pseudo_observations.jsonl.gz').is_file()):
                raise ValueError('Every registered U step and offline observation required')
        return
    if kind=='predict':
        from experiments.pamf_physics_tri.development_evaluation import predict
        return predict(c)
    raise ValueError('Unknown registered stage')


def analyze(rt):
    previous=reference.plan;reference.plan=plan
    try:return reference.analyze(rt)
    finally:reference.plan=previous


def dispatch(rt):
    from experiments.pamf_branch_training import runner as queue,prepare as preparation
    from experiments.pamf_physics_branch.runner import admits
    from experiments.pamf_physics_route.runner import launch
    old=preparation.prepare,queue.admits,queue.launch,queue.analyze
    preparation.prepare=prepare
    queue.admits=lambda gpu,active,mib:admits(gpu,active,mib,max_total=5)
    queue.launch=partial(launch,module='experiments.pamf_repl_step_guard.runner')
    queue.analyze=analyze
    try:return queue.dispatch(dict(rt,prioritize_prediction=True))
    finally:preparation.prepare,queue.admits,queue.launch,queue.analyze=old


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config)
    dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
