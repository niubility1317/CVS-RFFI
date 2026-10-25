"""Frozen peers are loaded before native trainer patches and RNG initialization."""
from pathlib import Path
import shutil
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_channel_pl.source_quality import validate_prefix_headers
from experiments.pamf_channel_pl.training import source_context
from .plan import validate_config
from .objective import PeerObjective,installed


def prefix(c):
    import torch
    validate_config(c);folder=Path(c['shared_prefix']);done=read(folder/'complete.json')
    cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
    expected=read(c['expected_source_contract']);actual=read(done['config']['source_contract'])
    case=dict(model_seed=c['model_seed'],source_config=c['source_E20_reference_config'])
    validate_prefix_headers(case,done,cp,actual,expected)
    if (read(c['source_contract'])!=expected or c['source_rxs']!=actual['source_rxs']
        or c['source_days']!=actual['source_days'] or done['config']['source_cache']!=c['source_cache']):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:own source prefix')
    return dict(folder=str(folder),**done),cp


def load_peers(c,device):
    import torch
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    validate_config(c);expected=read(c['source_contract']);models=[]
    for case in c['peers']:
        parent=Path(case['parent_source']);done=read(parent/'complete.json');pc=done['config']
        payload=torch.load(parent/'final_ssdg.pth',map_location='cpu',weights_only=False)
        actual=read(pc['source_contract'])
        validate_prefix_headers(case,done,payload,actual,read(case['expected_source_contract']))
        if actual!=expected or pc['source_rxs']!=c['source_rxs'] or pc['source_days']!=c['source_days']:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:peer source roles')
        models.append(reader_from_payload(pc,payload,device));del payload
    return models


def train(c,device='cuda:0'):
    from experiments.cric_dura_comparison.f_train import train as segment
    first,payload=prefix(c);del payload
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    segments=[first];states=[];previous=c['shared_prefix'];start=time.perf_counter()
    for epochs,offset in ((10,20),(20,30)):
        folder=root/('segment%d'%(offset+epochs));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,
            epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        peers=load_peers(cfg,device);objective=PeerObjective(cfg,peers);holder={}
        # f_train.train resets the student RNG after peer model construction.
        with installed(cfg,holder,objective):segment(cfg,device=device)
        state=objective.state_dict();full=cfg['hard_pl']['method']=='HPLT1U'
        if (state['calls']!=epochs*222 or state['student_views']!=state['seen']*(6 if full else 1)
            or state['peer_views']!=2*state['seen'] or state['physical_seen']!=(state['seen']//2 if full else 0)
            or state['physical_student_views']!=(state['seen'] if full else 0) or state['raw_As_seen']
            or state['EMA_updates'] or state['U_truth_read'] or state['peer_updates']
            or state['source_V_training_reads'] or state['target_reads'] or state['G_training_reads']
            or state['new_head'] or objective.capture_active):
            raise ValueError('Actual peer training exposure/mechanisms or permissions differ')
        write(folder/'peer_supervised_state.json',state);states.append(state)
        done=read(folder/'complete.json');segments.append(dict(folder=str(folder),**done));previous=str(folder)
        del objective,holder,peers
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,
            updates=(offset+epochs)*222,seconds=time.perf_counter()-start),exclusive=False)
    if sum(s['updates'] for s in segments)!=11100:raise ValueError('Complete E50 trajectory required')
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    totals={k:sum(s[k] for s in states) for k in ('calls','seen','accepted','raw_accepted','peer_views','student_views','physical_seen','physical_student_views')}
    write(root/'complete.json',dict(status='SOURCE_FROZEN',method_id='pamf_peer_supervised_v1',scaffold_id='pamf_t1_e50',
        epoch=50,updates=11100,segments=segments,config=c,source_contract=c['source_contract'],
        initialization='same_contract_scratch20_own10_own20',ancestors=[v['folder'] for v in segments],
        peer_sources=[p['parent_source'] for p in c['peers']],peer_epoch=20,
        target_access=False,U_truth_read=False,head_fit_updates=0,actual_peer_supervised=totals,
        peer_states=[str(root/('segment%d'%n)/'peer_supervised_state.json') for n in (30,50)],
        seconds=time.perf_counter()-start,cumulative_training_seconds=sum(v['seconds'] for v in segments),
        reused_prefix_seconds=segments[0]['seconds'],peer_prefix_cost='two additional shared E20 prefixes; training and inference-forward cost reported separately',
        historical_posthoc_selection_disclosed=True))


def smoke(rt,c,device='cuda:0'):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_train import arguments,original_config,class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison import t1_model
    first,payload=prefix(c)
    cfg=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_prefix'])
    peers=load_peers(cfg,device);buffers=[{k:v.detach().clone() for k,v in p.named_buffers()} for p in peers]
    objective=PeerObjective(cfg,peers);holder={};torch.manual_seed(c['model_seed'])
    with installed(cfg,holder,objective):
        ctx=source_context(cfg)
        with class_dimension(cfg),native_installed(original_config(c['model_seed'])) as native:
            args=arguments(cfg,device);args.input_len=256;args.baseline_ckpt=args.teacher_ckpt=''
            args.from_scratch=args.a1_scratch_only=True
            if args.use_unlabeled or args.use_ema_teacher:raise ValueError('Native U/EMA must remain disabled')
            model=t1_model.install(native.build_baseline_model(args,device),cfg).to(device)
            model.load_state_dict(payload['model'],strict=True);model.eval()
            with torch.no_grad():logits=model(ctx['train_loader'].dataset.x[:2].to(device))
            if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real own E20 smoke failed')
            model.train();model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,domain=[v.to(device) for v in batch[:3]]
            meta={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
            with model._phase1_labeled_feature_context():output=model(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=output,labels=y,
                extra=(None,meta),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();state=objective.state_dict();full=c['hard_pl']['method']=='HPLT1U'
            head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                or float(head.grad.norm())<=0 or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                or metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or state['calls']!=1
                or state['seen']!=256 or not state['accepted'] or state['peer_views']!=512
                or state['student_views']!=256*(6 if full else 1)
                or sum(model._cd_practical.coverage.values())!=64+(128 if full else 0)
                or any(p.grad is not None for peer in peers for p in peer.parameters())
                or any(not torch.equal(v,b[k]) for peer,b in zip(peers,buffers) for k,v in peer.named_buffers())
                or any(id(p) in {id(v) for peer in peers for v in peer.parameters()} for p in model.parameters())):
                raise ValueError('Actual peer/L/U mechanism or default head finite backward failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',parent_epoch=20,parent_updates=4440,
        source_samples=2,L_forward_packets=768,L_As_generated=64,U=state,device=device,
        default_head_gradient=True,finite_backward=True,optimizer_updates=0,query_reads=0,
        peer_parameters_frozen=True,peer_buffers_unchanged=True,peer_parameters_in_student=False,
        original_checkpoint_updates=0,default_head_classes=6))
