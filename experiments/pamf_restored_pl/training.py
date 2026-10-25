"""Own E30 plus20 joint default-head epochs; all parent costs retained."""
from pathlib import Path
import shutil
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_channel_pl.restored_source import validate_prefix_headers,validate_e30
from .plan import validate_config
from .objective import RestoredObjective,installed


def parents(c):
    import torch
    validate_config(c);case=c['source_case'];p=Path(case['parent_source']);d20=read(p/'complete.json')
    contract=read(c['source_contract']);cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_prefix_headers(case,d20,cp,read(d20['config']['source_contract']),contract);del cp
    q=Path(c['shared_parent_E30']);d30=read(q/'complete.json');cp=torch.load(q/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_e30(case,d30,cp,read(d30['config']['source_contract']),contract)
    if (q!=p.parent/'segment30' or d30['config']['source_cache']!=c['source_cache']
            or d30['config']['source_rxs']!=c['source_rxs'] or d30['config']['source_days']!=c['source_days']):
        raise ValueError('Own same-contract source E20+10 required')
    return [dict(folder=str(p),**d20),dict(folder=str(q),**d30)],cp


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    chain,payload=parents(c);del payload
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    folder=root/'segment50';cfg=dict(c,output_root=str(folder));objective=RestoredObjective(cfg)
    write(root/'resolved_config.json',c)
    with installed(cfg,objective):segment(cfg)
    state=objective.state_dict();extra=c['restored_pl']['extra']!='none'
    if (state['calls']!=4440 or state['U_seen']!=1134000
            or state['extra_seen']!=(1134000 if extra else 0)
            or state['extra_student_forwards']!=(2268000 if extra else 0)
            or state['restored_student_forwards']!=(2268000 if c['restored_pl']['extra']=='restored' else 0)
            or state['teacher_updates'] or state['G_updates'] or state['U_truth_read'] or state['target_reads']):
        raise ValueError('Complete fixed-budget restored source mechanism differs')
    write(folder/'restored_pl_state.json',state);done=read(folder/'complete.json');chain.append(dict(folder=str(folder),**done))
    if sum(s['updates'] for s in chain)!=11100:raise ValueError('Full own E50 source trajectory required')
    shutil.copyfile(folder/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',epoch=50,updates=11100,method_id='pamf_restored_pl_v1',
        config=c,segments=chain,source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in chain],target_access=False,U_truth_read=False,head_fit_updates=0,
        actual_restored_pl=state,seconds=time.perf_counter()-start,cumulative_training_seconds=sum(s['seconds'] for s in chain),
        peer_prefix_cost='other2 shared E30 teachers,90unique prefixepochs across3seeds;offlinecachecost separate',
        optimizer_schedule='original native segmented E30+20 entry;strict parent weights,original segment optimizer construction',
        historical_exposed_benchmark=True))


def smoke(c,output):
    import torch
    import torch.nn.functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.pamf_repl_completion.objectives import loader_scope
    chain,payload=parents(c);objective=RestoredObjective(c)
    with installed(c,objective):
        ctx=f_train.context(c)
        with f_train.class_dimension(c),native_installed(f_train.original_config(c['model_seed'])) as native:
            a=f_train.arguments(c,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
            if a.use_unlabeled or a.use_ema_teacher:raise ValueError('No native U/EMA permitted')
            student=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
            student.load_state_dict(payload['model'],strict=True);student.eval()
            with torch.no_grad():q=student(ctx['train_loader'].dataset.x[:2].to('cuda:0'))
            if q.shape!=(2,6) or not bool(torch.isfinite(q).all()):raise ValueError('Real own E30 default-head source smoke failed')
            student.train();student._cd_U_loader=ctx['unlabeled_loader'];student._cd_U_iter=iter(ctx['unlabeled_loader'])
            batch=next(iter(ctx['train_loader']));x,y,domain=[v.to('cuda:0') for v in batch[:3]]
            with student._phase1_labeled_feature_context():out=student(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=out,labels=y,
                extra=(None,batch[3]),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();head=student.id_backbone.encoder.core.id_backbone.cls_head.weight
            state=objective.state_dict()
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                    or float(head.grad.norm())<=0 or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0
                    or state['calls']!=1 or state['U_seen']!=256 or state['restored_student_forwards']!=512
                    or state['extra_accepted']==0 or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in student.parameters())):
                raise ValueError('Real native L/U restored backward smoke failed')
    write(output,dict(status='PASS',source_packets=2,L_forward_packets=768,U_seen=256,extra_accepted=state['extra_accepted'],
        restored_student_forwards=512,parent_epoch=30,default_head_classes=6,finite_head_gradient=True,
        actual_extra_weight=objective.metrics['RPL_extra_weight'],optimizer_updates=0,query_reads=0,U_truth_read=False))
