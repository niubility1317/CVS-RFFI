"""Real-checkpoint and source-only graph smoke before the dispatch queue."""
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from .plan import rows, configuration, CE_VARIANT


def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack import runtime
    from experiments.pamf_paper_completion import model as old
    from experiments.pamf_repl_completion.objectives import installed as structure_objectives
    from experiments.pamf_no_reference_completion.model import seed_scope
    from experiments.pamf_branch_factorial.method import load_reader as historical_reader
    from .model import paper_scope, main_graph_scope, structural_graph_scope, ce_build
    ctx = read(rt['context']); sources = read(Path(rt['run_root'])/'sources.json')
    prior = next(v for v in sources.values() if v['row']['reuse_key'] == 'strict_no_frequency-s2026092701')
    contract = read(ctx['handles']['F0']['source_contract'])
    reader, _ = historical_reader(prior['source_output'], contract, 'cuda:0')
    from experiments.cric_dura_comparison.f_data import SourceDataset
    x = SourceDataset(ctx['handles']['F0']['source_cache'],'L_s',contract).x[:2].to('cuda:0')
    with torch.inference_mode():
        pred = reader.classify_features(reader.features(x))
    if pred.shape != (2,6) or not bool(torch.isfinite(pred).all()):
        raise ValueError('Actual source checkpoint smoke failed')
    del reader, pred, x
    wanted = []
    for row in rows():
        if not row['new_training'] or row['model_seed'] != 2026092701:
            continue
        if (row['family'] in ('branch','structure_ce') or
            row['family'] == 'factorial' and (row['data_setting'] == 'main_split' or
            row['method'] == 'ssl' and row['data_setting'] in ('ten_class','source_rx_1'))):
            wanted.append(row)
    checks = []
    for row in wanted:
        c = configuration(ctx,row,rt['project'],rt['code_commit'])
        torch.manual_seed(c['model_seed']); torch.cuda.manual_seed_all(c['model_seed'])
        holder = {}; ssl = row['learning_regime'] == 'ssl'; strict = row['family'] in ('branch','structure_ce')
        scope = structure_objectives(c,holder,False) if strict else old.installed(c,holder,ssl)
        graphscope = structural_graph_scope() if strict else main_graph_scope()
        with seed_scope(c['model_seed']), paper_scope(), graphscope, scope:
            loaders = f_train.context(c)
            if (loaders['unlabeled_loader'] is not None) != c['U_use']:
                raise ValueError('U loader permission differs')
            with f_train.class_dimension(c), runtime.installed(f_train.original_config(c['model_seed'])) as native:
                args = f_train.arguments(c,'cuda:0'); args.input_len = 256
                net = t1_model.install(native.build_baseline_model(args,'cuda:0'),c).to('cuda:0')
            net._cd_U_loader=loaders['unlabeled_loader']; net._cd_U_iter=iter(net._cd_U_loader) if net._cd_U_loader else None
            b=next(iter(loaders['train_loader'])); x,y,domain=[a.to('cuda:0') for a in b[:3]]
            net.train(); net._feature_l_steps=4440
            with net._phase1_labeled_feature_context():
                out=net(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=0. if strict else .01)
            correction,metrics=net._source_labeled_feature_hook(model=net,labeled_output=out,labels=y,extra=(None,b[3]),
                domains=domain,optimizer_step=4441,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            loss=ce+correction; loss.backward(); head=net.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.norm())<=0 or
                    metrics['T1_L_packets'] != (384 if strict else 768) or
                    any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                raise ValueError('Source graph or labeled objective differs')
            if ssl:
                st=holder['objective'].state_dict()
                if st['calls']!=1 or st['seen']!=256 or st['student_views']!=1536 or st['EMA_updates'] or st['U_truth_read']:
                    raise ValueError('Full SSL path not executed')
            elif net._cd_U_seen != (256 if c['U_use'] else 0):
                raise ValueError('Ordinary PL exposure differs')
            checks.append(dict(row=row['row_id'],finite_gradient=True))
            del net,head,loss,ce,correction,out,x,y,domain,loaders,holder
            torch.cuda.empty_cache()
    net=ce_build(CE_VARIANT,2026092701).to('cuda:0')
    loss=F.cross_entropy(net(torch.zeros(2,2,256,device='cuda:0')),torch.tensor([0,1],device='cuda:0')); loss.backward()
    if not bool(torch.isfinite(loss)):
        raise ValueError('E200 graph failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',real_checkpoint=True,query_reads=0,
          U_truth_reads=0,optimizer_updates=0,checks=checks))
