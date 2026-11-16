"""Scratch labeled-only training, automatic clean prediction and scoring."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_weight import runner as reference
from . import plan


def prepare(rt):
    previous=reference.plan
    reference.plan=plan
    try:
        return reference.prepare(rt)
    finally:
        reference.plan=previous


def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_branch_training.source import checked_e20_parent
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph

    context=read(rt['context']);checks=[]
    for row in plan.candidate_rows():
        c=plan.configuration(context,row,rt['project'],rt['code_commit'])
        # The checkpoint is used for a disposable source-only smoke, never as
        # initialization or ancestry of the scratch supervised experiment.
        original=f"{row['data_setting']}-factorial-ssl-full-permuted-s{plan.SEED}"
        root=Path(rt['project'])/'runs'/'20261027-phase1-pamf-paper-completion-m119-r03'/'F'/original/'source'
        prior=read(root/'complete.json')['config']
        prefix=checked_e20_parent(dict(prior,shared_prefix=str(root/'segment20')))
        if read(prior['source_contract'])!=read(c['source_contract']):
            raise ValueError('Smoke and scratch source contracts differ')
        cp=torch.load(Path(prefix['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        cfg=dict(c,segment_epochs=20,epoch_offset=0,step_offset=0,parent_source=None)
        holder={};torch.manual_seed(plan.SEED);torch.cuda.reset_peak_memory_stats()
        with plan.installed(),paper_scope(cfg,holder,ssl_enabled=False):
            with f_train.class_dimension(cfg),native_scope(f_train.original_config(plan.SEED)) as native:
                args=f_train.arguments(cfg,'cuda:0')
                net=t1_model.install(native.build_baseline_model(args,'cuda:0'),cfg).cuda()
            net.load_state_dict(cp['model'],strict=True)
            graph=check_graph(net,cfg)
            source=SourceDataset(c['source_cache'],'L_s',read(c['source_contract']))
            x,y,domain,metadata=next(iter(torch.utils.data.DataLoader(source,batch_size=128,shuffle=False)))
            x,y,domain=x.cuda(),y.cuda(),domain.cuda()
            metadata={k:v.cuda() if torch.is_tensor(v) else v for k,v in metadata.items()}
            net.train()
            with net._phase1_labeled_feature_context():
                output=net(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,_=net._source_labeled_feature_hook(model=net,labeled_output=output,labels=y,
                extra=(None,metadata),domains=domain,optimizer_step=1,identity_loss=ce,
                labeled_count=128,labeled_inputs=x)
            loss=ce+correction;loss.backward()
            if (args.use_unlabeled or args.use_ema_teacher or net._cd_U_seen
                    or not bool(torch.isfinite(loss))
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                raise ValueError('Finite labeled-only checkpoint smoke required')
            checks.append(dict(row=row['row_id'],graph=graph,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               source_records=128,U_seen=0,optimizer_updates=0,checkpoint_used_for_training=False))
        del net,source,loss,cp
        torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=checks,real_checkpoint=True,
          query_reads=0,U_reads=0,optimizer_updates=0))


def worker(c,kind):
    if kind=='score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from experiments.pamf_paper_completion.source import train
        plan.validate(c)
        with plan.installed():train(c)
        done=read(Path(c['output_root'])/'complete.json')
        for segment in done['segments']:
            progress=read(Path(segment['folder'])/'progress.json')
            if progress['U_actual'] or progress['PL_seen'] or progress['PL_accepted']:
                raise ValueError('Labeled-only control used unlabeled inputs')
        return
    if kind=='predict':
        from experiments.pamf_physics_tri.development_evaluation import predict
        return predict(c)
    raise ValueError('Unknown supervised control stage')


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
    queue.launch=partial(launch,module='experiments.pamf_tri_label_only.runner')
    queue.analyze=analyze
    try:return queue.dispatch(dict(rt,prioritize_prediction=True))
    finally:preparation.prepare,queue.admits,queue.launch,queue.analyze=old


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config)
    dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
