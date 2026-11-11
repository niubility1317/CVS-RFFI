import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_physics_branch import runner as evaluation
from experiments.pamf_physics_route import runner as dispatching
from . import objective as phase_objective
from .plan import candidate_rows,configuration
from .prepare import prepare
from .source import parent,installed,train


def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    ctx,checks=read(rt['context']),[]
    previous=phase_objective.masked_loss;captured=[]
    def capture(logits,pseudo,accepted):
        result=previous(logits,pseudo,accepted);captured.append(result);return result
    phase_objective.masked_loss=capture
    try:
        for row in candidate_rows():
            c=configuration(ctx,row,rt['project'],rt['code_commit']);seed=c['model_seed'];prefix=parent(c)
            cp=torch.load(Path(prefix['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
            # Cross-RX-specific positive-gradient assertions cannot hold with one source RX.
            cfg=dict(c,epoch_offset=20,step_offset=4440,segment_epochs=10,hard_pl_smoke=False)
            holder={};captured.clear();torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats()
            with installed(cfg,holder):
                loaders=f_train.context(cfg)
                with f_train.class_dimension(cfg),native_scope(f_train.original_config(seed)) as native:
                    a=f_train.arguments(cfg,'cuda:0');a.input_len=256
                    net=t1_model.install(native.build_baseline_model(a,'cuda:0'),cfg).cuda()
                ref=net.id_backbone.encoder;views=ref.sequence_pool.views;order=[0,5,3,6,2,1,4,7]
                if len(views)!=3 or any(v.window_order.tolist()!=order for v in views):raise ValueError('Wrong Tri order')
                if sum(p.numel() for p in ref.parameters())!=232993:raise ValueError('Wrong full network')
                net.load_state_dict(cp['model'],strict=True);net.train()
                net._cd_U_loader=loaders['unlabeled_loader'];net._cd_U_iter=iter(net._cd_U_loader)
                batch=next(iter(loaders['train_loader']));x,y,domain=[v.cuda() for v in batch[:3]]
                metadata={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch[3].items()}
                with net._phase1_labeled_feature_context():output=net(x,return_aux=True,domain_labels=domain)
                ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
                correction,metrics=net._source_labeled_feature_hook(model=net,labeled_output=output,labels=y,
                    extra=(None,metadata),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
                if len(captured)!=1 or not bool(torch.isfinite(captured[0])):
                    raise ValueError('Actual coherence pseudo-label loss required')
                named=[p for p in ref.parameters() if p.requires_grad]
                gradients=torch.autograd.grad(captured[0],named,retain_graph=True,allow_unused=True)
                norm=sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5
                if norm<=0 or any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
                    raise ValueError('Coherence branch loss must reach the deployed backbone')
                loss=ce+correction;loss.backward();state=holder['objective'].state_dict();u=state['U_phase_relation']
                if (a.seed!=seed or a.use_unlabeled or a.use_ema_teacher or not bool(torch.isfinite(loss))
                    or state['calls']!=1 or state['seen']!=256 or state['student_views']!=1536
                    or state['U_gradient_guard']['calls']!=1 or state['U_truth_read'] or state['EMA_updates']
                    or u['calls']!=1 or u['selected']!=128 or u['accepted']<=0
                    or state['U_joint_contrast']['anchors']!=0
                    or state['U_receiver_extrapolation']['eligible_packet_views']!=0
                    or not bool(torch.isfinite(metrics['HPL_phase_U_loss']))
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                    raise ValueError('Single-source-RX Tri or source-U channel smoke differs')
                checks.append(dict(row=row['row_id'],order=order,parameters=232993,finite_backward=True,
                    branch_gradient_norm=norm,U_seen=state['seen'],U_views=state['student_views'],
                    phase_U=u,cross_RX_anchors=0,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                del net,loss,correction,ce,output,loaders,cp,holder,gradients
            torch.cuda.empty_cache()
    finally:phase_objective.masked_loss=previous
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=checks,real_checkpoint=True,
          optimizer_updates=0,query_reads=0,U_truth_reads=0))


def worker(c,kind):
    if kind in ('predict','score'):return evaluation.worker(c,kind)
    if kind not in ('train','smoke'):raise ValueError('Only registered stages allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c);return train(c) if kind=='train' else smoke(c)


def admits_shared(gpu,active,required_mib):
    from experiments.pamf_repl_completion.runner import capacity,processes
    cuda,free=capacity(gpu)
    pending={pid for pid,card in processes().items() if card==gpu}
    own=[process for process,item in active.values() if item['gpu']==gpu]
    initializing=sum(process.pid not in cuda for process in own)
    return len(cuda|pending)<4 and len(own)<4 and free>=required_mib*(1+initializing)


def dispatch(rt):
    launch=partial(dispatching.launch,module='experiments.pamf_tri_u_coherence_cal_m1.runner')
    previous=evaluation.admits
    evaluation.admits=admits_shared
    try:dispatching.dispatch(rt,prepare_run=prepare,launch_worker=launch)
    finally:evaluation.admits=previous


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
