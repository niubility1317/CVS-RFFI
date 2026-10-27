"""One no-query smoke: real frozen checkpoint and meaningful new code paths."""
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .plan import make_plan,configuration

def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from .model import installed,load_reader,ce_build
    from .plan import CE_VARIANT
    ctx=read(rt['context']);rows=make_plan()['rows'];checks=[];device='cuda:0'
    row=next(r for r in rows if r['reuse_key']=='current/3/2026092701')
    old=ctx['current_views']['3/2026092701'];h=ctx['handles']['F0'];contract=read(h['source_contract'])
    reader,_=load_reader(old['source_output'],contract,row,device)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    x=SourceDataset(h['source_cache'],'L_s',contract).x[:2].to(device)
    with torch.inference_mode():logits=reader.classify_features(reader.features(x))
    if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real frozen checkpoint failed')
    del reader,logits,x
    chosen=[]
    for r in rows:
        if not r['new_training'] or r['model_seed']!=2026092701 or r['family']=='architecture':continue
        if (r['family'] in ('structure','U_permission') or
            r['family']=='factorial' and (r['data_setting']=='main_split' or r['method']=='ssl' and r['data_setting'] in ('ten_class','source_rx_1'))):chosen.append(r)
    for row in chosen:
        c=configuration(ctx,row,rt['project'],rt['code_commit']);torch.manual_seed(c['model_seed']);torch.cuda.manual_seed_all(c['model_seed'])
        holder={};ssl=row['learning_regime']=='ssl'
        with installed(c,holder,ssl):
            loaders=f_train.context(c)
            if (loaders['unlabeled_loader'] is not None)!=c['U_use']:raise ValueError('Actual U loader permissions differ')
            with f_train.class_dimension(c),native_scope(f_train.original_config(c['model_seed'])) as native:
                net=t1_model.install(native.build_baseline_model(f_train.arguments(c,device),device),c).to(device)
            net._cd_U_loader=loaders['unlabeled_loader'];net._cd_U_iter=iter(net._cd_U_loader) if net._cd_U_loader else None
            b=next(iter(loaders['train_loader']));x,y,domain=[a.to(device) for a in b[:3]]
            net.train();net._feature_l_steps=4440
            with net._phase1_labeled_feature_context():out=net(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
            correction,metrics=net._source_labeled_feature_hook(model=net,labeled_output=out,labels=y,extra=(None,b[3]),domains=domain,
                optimizer_step=4441,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            loss=ce+correction;loss.backward();head=net.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.norm())<=0 or metrics['T1_L_packets']!=768 or
                any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):raise ValueError('Source gradient/objective differs')
            if ssl:
                st=holder['objective'].state_dict()
                if st['calls']!=1 or st['seen']!=256 or st['student_views']!=1536 or st['EMA_updates'] or st['U_truth_read']:raise ValueError('Complete SSL path not executed')
            elif net._cd_U_seen!=(256 if c['U_use'] else 0):raise ValueError('Actual ordinary PL use differs')
            if not row['channel_augmentation'] and net._cd_As_seen!=0:raise ValueError('No-augmentation row used As')
            checks.append(dict(row=row['row_id'],finite_joint_backward=True,default_head_classes=head.shape[0],U_loader=c['U_use']))
            del net,head,loss,ce,correction,out,x,y,domain,loaders,holder
            torch.cuda.empty_cache()
    ce=ce_build(CE_VARIANT,2026092701).to(device);ce.train()
    loss=F.cross_entropy(ce(torch.zeros(2,2,256,device=device)),torch.tensor([0,1],device=device));loss.backward()
    if not bool(torch.isfinite(loss)):raise ValueError('Three-view E200 CE graph failed')
    checks.append(dict(row='E200_three_view_CE',parameters=sum(p.numel() for p in ce.parameters()),finite_backward=True))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',real_checkpoint=True,query_reads=0,U_truth_reads=0,optimizer_updates=0,checks=checks))
