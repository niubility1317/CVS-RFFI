import argparse,copy,json
from types import SimpleNamespace
from . import design as d

def check():
    import torch
    from .data import supervised_split,check_roles,oracle_native
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu')
    for key,value in [('checkpoint_sources',['old_E20']),('target_access',True),('epochs',20),('updates',6660),('test_views',['clean'])]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed config accepted '+key)
    assert a.epochs==a.label_epochs==30 and a.pseudo_epochs==0
    assert not a.use_unlabeled and not a.use_ema_teacher and not a.use_concat_sat_channel_aug and not a.u_unlabeled_shuffle
    assert a.label_smoothing==0 and not any(getattr(a,k) for k in vars(a) if k.startswith('lambda_'))
    items=[SimpleNamespace(tx_i=t,rx_i=r,day_i=b,eq_i=1,sig_i=s) for t in range(6) for r in (1,3,4,6,8) for b in (1,2,3) for s in range(1000)]
    roles=supervised_split(SimpleNamespace(index=items));assert [len(r) for r in roles]==[72000,0,18000]
    assert all(items[i].sig_i<800 for i in roles[0]) and all(items[i].sig_i>=800 for i in roles[2])
    mapped=dict(zip(['L_s','U_s','V'],[[str(i) for i in r] for r in roles]));check_roles(mapped)
    mapped['V'][0]=mapped['L_s'][0]
    try:check_roles(mapped)
    except ValueError:pass
    else:raise AssertionError('Overlap accepted')
    with oracle_native(c) as n:
        # Actual loader has a short final batch and visits every packet once.
        from torch.utils.data import TensorDataset
        ds=TensorDataset(torch.arange(72000));ds.split_source='ssdg_labeled_tx_visible'
        loader=n.make_loader(ds,128,False,0,torch.device('cpu'),True,2)
        assert len(loader)==563 and loader.drop_last is False
        pairs=list(n._muse_epoch_pairs(loader,None,use_muse=False))
        assert len(pairs)==563 and sum(len(b[0]) for b,u in pairs)==72000 and all(u is None for b,u in pairs)
        a.num_domains=15;a.input_len=256;torch.manual_seed(c['model_seed']);m=n.build_baseline_model(a,torch.device('cpu'))
        m.train();loss=torch.nn.functional.cross_entropy(m(torch.randn(4,2,256)),torch.tensor([0,1,2,3]));loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.id_backbone.parameters())
        state=n._feature_training_state();assert state.optimizer is None and state.reference is None and state.style is None and state.fishr is None
    return dict(status='PASS',roles=[72000,0,18000],epochs=30,updates=16890,query_read=False,negative_checks=True,full_train_pass=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(2);value=check()
    if a.remote_preflight:
        from .data import source_contract
        from .evaluate import view_manifest
        value['actual_source']=source_contract();value['view_manifest_status']=view_manifest()['status']
    d.write(a.output,value);print(json.dumps(value))
