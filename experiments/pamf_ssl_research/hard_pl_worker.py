"""Real F21 parent smoke, two hard-PL forks, automatic dual-readout truth-last."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write


def smoke(rt):
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import context
    from experiments.cric_dura_comparison.f_train import arguments,original_config,class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison import t1_model
    from .hard_pl_train import source_parents,installed
    checked=[]
    for row in rt['rows'][:2]:
        c=read(row['train_config']);parents,_=source_parents(c);parent=parents[-1]['folder']
        cfg=dict(c,segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=parent,hard_pl_smoke=True)
        payload=torch.load(Path(parent)/'final_ssdg.pth',map_location='cpu',weights_only=False)
        ctx=context(cfg);holder={};device='cuda:0';torch.manual_seed(c['model_seed'])
        with installed(cfg,holder),class_dimension(cfg),native_installed(original_config(c['model_seed'])) as native:
            args=arguments(cfg,device);args.input_len=256;args.baseline_ckpt=args.teacher_ckpt=''
            args.from_scratch=args.a1_scratch_only=True
            if args.use_unlabeled or args.use_ema_teacher:raise ValueError('Native U/EMA must be disabled')
            model=t1_model.install(native.build_baseline_model(args,device),cfg).to(device)
            model.load_state_dict(payload['model'],strict=True);model.eval()
            with torch.no_grad():logits=model(ctx['train_loader'].dataset.x[:2].to(device))
            if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real F21 checkpoint inference failed')
            model.train();model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,domain=[v.to(device) for v in batch[:3]]
            meta={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
            with model._phase1_labeled_feature_context():output=model(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=output,labels=y,
                extra=(None,meta),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();obj=holder['objective'];state=obj.state_dict()
            head=model.id_backbone.encoder.core.id_backbone.cls_head.weight;mechanisms=c['hard_pl']['method']=='HPLT1U'
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                or float(head.grad.norm())<=0 or metrics['T1_L_packets']!=768 or state['calls']!=1 or state['seen']!=256
                or not state['accepted'] or state['student_views']!=(1536 if mechanisms else 256)
                or state['physical_seen']!=(128 if mechanisms else 0) or state['EMA_updates']!=0 or obj.capture_active
                or not torch.allclose(metrics['HPL_loss'],2*metrics['HPL_base_loss'])):
                raise ValueError('Actual hard-PL weight/default-head/L/U execution failed')
            if mechanisms and (any(v<=0 for v in state['contrast_anchors'].values())
                or not all(h.grad is not None and bool(torch.isfinite(h.grad).all()) and float(h.grad.norm())>0
                           for maps in obj.smoke_maps for h in maps.values())
                or metrics['HPL_As_KL_weight']!=0 or metrics['HPL_ordinary_KL']<=0 or metrics['HPL_raw_KL']<=0):
                raise ValueError('Actual shallow/middle contrast or paired-dropout mechanism absent')
            checked.append(dict(method=state['method'],parent_epoch=30,parent_updates=6660,L_forward_packets=768,
                U_seen=256,U_accepted=state['accepted'],student_U_views=state['student_views'],
                physical_U_generated=state['physical_seen'],physical_student_views=state['physical_student_views'],
                contrast_anchors=state['contrast_anchors'],shallow_middle_gradient=mechanisms,
                ordinary_raw_RDrop=mechanisms,As_KL_weight=0,weight=2.,threshold=.95,
                EMA_updates=0,finite_backward=True,default_head_gradient=True,feature_hooks_removed=True))
        del model,obj,holder,payload,ctx;gc.collect();torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',methods=checked,target_reads=0,U_truth_reads=0,
        probe_updates_discarded=True,formal_reload_original_F21_E30=True,immediate_continuation=True))


def row(c):
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    cfg=read(c['train_config']);activate(cfg);numerical_policy()
    from .hard_pl_train import train
    train(cfg);gc.collect()
    import torch
    torch.cuda.empty_cache()
    from .source_diagnostics import run as diagnostic
    diagnostic(cfg,Path(cfg['output_root'])/'source_V_diagnostic.json')
    gc.collect();torch.cuda.empty_cache()
    from experiments.cric_dura_comparison.evaluation import predict
    predict(read(c['prediction_config']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','row','score'));parser.add_argument('--config',required=True)
    args=parser.parse_args();value=read(args.config)
    if args.action=='smoke':smoke(value)
    elif args.action=='row':row(value)
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
