"""Own E20+10+20 trajectory with a source-channel CE reallocation after E20."""
from pathlib import Path
import shutil
import time
from experiments.cric_dura_comparison.io import read,write
from .objective import installed
from .training_plan import validate_config
from .source_quality import validate_prefix_headers


def source_context(c):
    # Resolve after installed() patches the loader; an early imported alias
    # would bypass the native 256-U scope and silently construct 128-U batches.
    from experiments.cric_dura_comparison import f_data
    return f_data.context(c)


def prefix(c):
    import torch
    validate_config(c);folder=Path(c['shared_prefix']);done=read(folder/'complete.json')
    cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
    case=dict(model_seed=c['model_seed'],source_config=c['source_E20_reference_config'])
    actual=read(done['config']['source_contract']);expected=read(c['source_contract'])
    validate_prefix_headers(case,done,cp,actual,expected)
    if (read(c['expected_source_contract'])!=expected or c['source_rxs']!=actual['source_rxs']
            or c['source_days']!=actual['source_days'] or done['config']['source_contract']!=c['source_contract']):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:continuation role binding')
    return dict(folder=str(folder),**done),cp


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    validate_config(c);first,cp=prefix(c);del cp
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    segments=[first];states=[];previous=c['shared_prefix'];start=time.perf_counter()
    for epochs,offset in ((10,20),(20,30)):
        folder=root/('segment%d'%(offset+epochs));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,
            epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        holder={}
        with installed(cfg,holder):segment(cfg)
        state=holder['objective'].state_dict()
        if (state['calls']!=epochs*222 or state['student_views']!=state['seen']+state['channel_accepted']
                or state['teacher_views']!=state['seen']+state['generated'] or state['EMA_updates']
                or state['U_truth_read'] or state['target_reads'] or state['source_V_training_reads']
                or state['G_training_reads'] or state['new_head'] or state['per_accepted_sample_CE_dose']!=1.):
            raise ValueError('Actual source-channel dose,exposure or permissions differ')
        write(folder/'channel_pl_state.json',state);states.append(state)
        done=read(folder/'complete.json');segments.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,
            updates=(offset+epochs)*222,seconds=time.perf_counter()-start),exclusive=False)
    if len(segments)!=3 or sum(s['updates'] for s in segments)!=11100:raise ValueError('Complete E50 trajectory required')
    totals={k:sum(s[k] for s in states) for k in ('calls','seen','accepted','generated','channel_accepted','teacher_views','student_views')}
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',method_id='pamf_channel_pl_v1',scaffold_id='pamf_t1_e50',
        epoch=50,updates=11100,segments=segments,config=c,source_contract=c['source_contract'],
        initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in segments],
        target_access=False,U_truth_read=False,head_fit_updates=0,actual_channel_pl=totals,
        channel_states=[str(root/('segment%d'%n)/'channel_pl_state.json') for n in (30,50)],
        seconds=time.perf_counter()-start,cumulative_training_seconds=sum(v['seconds'] for v in segments),
        reused_prefix_seconds=segments[0]['seconds'],historical_posthoc_selection_disclosed=True))


def smoke(rt,c):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_train import arguments,original_config,class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison import t1_model
    validate_config(c);first,payload=prefix(c);device='cuda:0'
    cfg=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_prefix'])
    torch.manual_seed(c['model_seed']);holder={}
    with installed(cfg,holder):
        ctx=source_context(cfg)
        with class_dimension(cfg),native_installed(original_config(c['model_seed'])) as native:
            args=arguments(cfg,device);args.input_len=256;args.baseline_ckpt=args.teacher_ckpt=''
            args.from_scratch=args.a1_scratch_only=True
            if args.use_unlabeled or args.use_ema_teacher:raise ValueError('Native U/EMA must remain disabled')
            model=t1_model.install(native.build_baseline_model(args,device),cfg).to(device)
            model.load_state_dict(payload['model'],strict=True);model.eval()
            with torch.no_grad():logits=model(ctx['train_loader'].dataset.x[:2].to(device))
            if logits.shape!=(2,c['num_classes']) or not bool(torch.isfinite(logits).all()):raise ValueError('Real E20 no-query smoke failed')
            model.train();model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,domain=[v.to(device) for v in batch[:3]]
            meta={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
            with model._phase1_labeled_feature_context():output=model(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=output,labels=y,
                extra=(None,meta),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();state=holder['objective'].state_dict()
            head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                or float(head.grad.norm())<=0 or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or state['calls']!=1
                or state['seen']!=256 or not state['channel_accepted'] or state['EMA_updates'] or state['U_truth_read']
                or sum(model._cd_practical.coverage.values())!=64
                or sum(state['source_channel_coverage'].values())!=state['generated']
                or state['student_views']!=256+state['channel_accepted'] or state['per_accepted_sample_CE_dose']!=1.):
                raise ValueError('Actual ChannelPL,L backbone/default head or finite backward failed')
            write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',parent_epoch=20,parent_updates=4440,
                source_samples=2,L_forward_packets=768,L_As_generated=64,U=state,
                default_head_gradient=True,finite_backward=True,optimizer_updates=0,query_reads=0,
                original_checkpoint_updates=0,default_head_classes=c['num_classes']))
