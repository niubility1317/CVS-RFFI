import argparse,copy,json
from types import SimpleNamespace
from . import design as d

def check():
    import torch
    from .data import historical_split,check_roles,historical_native
    from .evaluate import metric
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=d.config(d.rows()[0]);a=configure(d.make_args(c,'cpu'))
    # Registration runs on Windows; the identical config runs on Linux.
    for path in (d.SOURCE,d.CAPSULE,d.TRUTH,c['output_root'],c['source_contract']):
        if not path.startswith(d.PROJECT+'/') or '\\' in path:raise ValueError('Remote path must use portable POSIX text')
    for key,value in [('model_seed',2026092701),('checkpoint_sources',['current_E20']),('test_views',['clean','practical_high']),('target_access',True)]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed config accepted '+key)
    assert not any(getattr(a,k) for k in vars(a) if k.startswith('lambda_'))
    assert a.seed==392002 and a.epochs==a.label_epochs==20 and a.pseudo_epochs==0
    assert not a.use_unlabeled and not a.use_ema_teacher and not a.use_concat_sat_channel_aug
    assert not a.allow_source_target_day_overlap and a.a1_source_screen_only and a.label_smoothing==0
    items=[SimpleNamespace(tx_i=t,rx_i=r,day_i=b,eq_i=1,sig_i=s) for t in range(6) for r in range(7) for b in range(2) for s in range(1000)]
    roles=historical_split(SimpleNamespace(index=items))
    assert [len(r) for r in roles]==[8400,58800,16800]
    assert all(items[i].sig_i<100 for i in roles[0]) and all(100<=items[i].sig_i<800 for i in roles[1]) and all(items[i].sig_i>=800 for i in roles[2])
    check_roles(dict(zip(['L_s','U_s','V'],[[str(i) for i in r] for r in roles])))
    bad=dict(zip(['L_s','U_s','V'],[[str(i) for i in r] for r in roles]));bad['V'][0]=bad['L_s'][0]
    try:check_roles(bad)
    except ValueError:pass
    else:raise AssertionError('Source physical overlap accepted')
    with historical_native(c) as native:
        a.input_len=256;a.num_domains=14
        native._validate_a1_scratch_only(a)
        torch.manual_seed(392002);m=native.build_baseline_model(a,torch.device('cpu'))
        m.train();x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3])
        loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.id_backbone.parameters())
        state=native._feature_training_state()
        assert state.optimizer is None and state.reference is None and state.style is None and state.fishr is None and state.relation is None
        assert len(list(native._muse_epoch_pairs([1,2],None,use_muse=False)))==222
    import numpy as np
    test=metric(np.asarray([0,1,2,3,4,5]),np.asarray([0,1,2,3,4,5]));assert test['accuracy']==test['Macro_F1']==1
    return dict(status='PASS',seed=392002,epochs=20,updates=4440,historical_roles=[8400,58800,16800],clean_count=d.COUNT,query_access=False,pure_CE=True,negative_checks=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(2);result=check()
    if a.remote_preflight:
        from .data import source_contract
        result['actual_source']=source_contract()
    d.write(a.output,result);print(json.dumps(result))
