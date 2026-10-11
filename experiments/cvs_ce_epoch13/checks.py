import argparse,copy,json,math
import torch
from . import design as d


def check():
    from .source import require_budget
    from experiments.cvs_phase1_repair.design import cosine_lr
    from experiments.cvs_feature_disentangle.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    configs=[d.config(r) for r in d.rows()];records=[]
    for c in configs:
        d.validate(c);a=configure(d.make_args(c,'cpu'))
        assert a.epochs==a.label_epochs==c['epochs'] and a.pseudo_epochs==0
        assert not a.use_unlabeled and not a.use_ema_teacher and not a.use_concat_sat_channel_aug and a.label_smoothing==0
        assert not any(getattr(a,k) for k in vars(a) if k.startswith('lambda_'))
        require_budget(c['updates'],c['updates'],c['epochs']*222)
        try:require_budget(44400,44400,c['updates'])
        except ValueError:pass
        else:raise AssertionError('Wrong E200 budget accepted')
        lr=[cosine_lr(a.lr,e,c['epochs'],a.phase1_lr_min) for e in range(1,c['epochs']+1)]
        assert math.isclose(lr[0],a.lr) and math.isclose(lr[-1],1e-6) and all(x>=y for x,y in zip(lr,lr[1:]))
        records.append(dict(row_id=c['row_id'],epochs=a.epochs,updates=c['updates'],batch_size=a.batch_size,peak_lr=a.lr,min_lr=lr[-1],weight_decay=a.weight_decay))
    # Actual native routing: compressed budgets preserve the original scratch model.
    baseline=d.original.config(dict(stage='feature_disentangle',arm='native',model_seed=d.SEEDS[0],row_id=f'native-s{d.SEEDS[0]}'))
    with installed(baseline) as n:
        base_args=d.original.make_args(baseline,'cpu');base_args.num_domains,base_args.input_len=5,256
        torch.manual_seed(3344);m=n.build_baseline_model(base_args,torch.device('cpu'));reference={k:v.clone() for k,v in m.state_dict().items()}
    # One epoch budget: compare only the matched seed used for the reference.
    for c in configs[:1]:
        with installed(c) as n:
            a=d.make_args(c,'cpu');a.num_domains,a.input_len=5,256;n._validate_a1_scratch_only(a);n._validate_daot_config(a)
            assert n._resolve_epoch_schedule(a)==c['epochs']
            torch.manual_seed(3344);m=n.build_baseline_model(a,torch.device('cpu'))
            assert all(torch.equal(v,reference[k]) for k,v in m.state_dict().items())
            state=n._feature_training_state();assert state.optimizer is None and state.reference is None and state.style is None and state.fishr is None and state.relation is None
            pairs=list(n._muse_epoch_pairs([1,2,3],None,use_muse=False));assert len(pairs)==222 and all(u is None for _,u in pairs)
            m.train();loss=torch.nn.functional.cross_entropy(m(torch.randn(4,2,256)),torch.tensor([0,1,2,3]));loss.backward()
            assert torch.isfinite(loss) and any(p.grad is not None and float(p.grad.norm())>0 for p in m.id_backbone.parameters())
    return dict(status='PASS',rows=records,original_scratch_state_identical=True,pure_CE_no_auxiliary_objects=True,cosine_compressed_to_each_budget=True,native_pair_count=222,query_access=False)


def remote_preflight():
    from experiments.cvs_stage2_decouple import design as stage2
    from experiments.cvs_stage2_decouple.common import verify_parent
    result=check();refs=[]
    for seed in d.SEEDS:
        pc=stage2.config(next(r for r in stage2.rows() if r['arm']=='C0' and r['model_seed']==seed));refs.append(verify_parent(pc))
        actual=d.read(str(__import__('pathlib').Path(pc['parent_checkpoint']).parent/'resolved_native_args.json'))
        c=d.config(next(r for r in d.rows() if r['model_seed']==seed));a=d.make_args(c,'cpu')
        keys=('batch_size','lr','weight_decay','label_smoothing','use_unlabeled','use_ema_teacher','use_concat_sat_channel_aug','phase1_lr_schedule','phase1_lr_min')
        if any(actual[k]!=getattr(a,k) for k in keys):raise ValueError('Original E200 matched recipe differs')
    result.update(reference_E200_provenance=refs,reference_role='comparison only; no weights loaded into new training',identity_updates=0)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
    value=remote_preflight() if a.remote_preflight else check();d.write(a.output,value);print(json.dumps(value))
