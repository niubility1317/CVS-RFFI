import argparse,copy,json
from . import design as d

def check():
    import torch
    from .data import oracle_native
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu')
    for key,value in [('checkpoint_sources',['old_E30']),('target_access',True),('epochs',20),('training_augmentation',{}),('test_views',['clean'])]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed config accepted '+key)
    assert not a.use_unlabeled and not a.use_ema_teacher and not a.use_sat_consistency
    assert a.epochs==30 and a.label_epochs==30 and a.pseudo_epochs==0 and a.label_smoothing==0
    assert a.use_concat_sat_channel_aug and a.concat_sat_ce_only and a.concat_sat_fused_ce_only
    assert {k:v for k,v in vars(a).items() if k.startswith('lambda_') and v}=={'lambda_sat_cls':.68}
    with oracle_native(c) as native:
        assert native._loss_weights(a,{})['sat_cls']==.68 and native._loss_weights(a,{})['sat_cons']==0
        assert a.concat_sat_start_epoch==a.sat_cons_start_epoch==1
        from torch.utils.data import TensorDataset
        ds=TensorDataset(torch.arange(72000));ds.split_source='ssdg_labeled_tx_visible'
        loader=native.make_loader(ds,128,False,0,torch.device('cpu'),True,2)
        pairs=list(native._muse_epoch_pairs(loader,None,use_muse=False))
        assert len(pairs)==563 and sum(len(b[0]) for b,u in pairs)==72000 and all(u is None for b,u in pairs)
        from concat_sat_channel_aug import ConcatSatChannelAugment
        from cvsrffi.original_leo import FusedCleanSatelliteForward
        augment=ConcatSatChannelAugment(scenarios=d.TRAIN_SCENES,p=1.,seed=2027,schedule=d.SCHEDULE,apply_fn=native.apply_sat_channel_for_scenario)
        stages=[augment.stage_for_epoch(e)[1] for e in (1,6,7,14,15,30)]
        assert [s.start_epoch for s in stages]==[1,1,7,7,15,15]
        assert [s.view_prob for s in stages]==[.3,.3,.6,.6,.8,.8]
        a.num_domains=15;a.input_len=256;torch.manual_seed(c['model_seed']);m=native.build_baseline_model(a,torch.device('cpu'))
        x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3]);domain=torch.zeros(4,dtype=torch.long)
        executed=set()
        # Real simulator and fused forward: verify all registered channel paths,
        # including a satellite CE gradient from the first training stage.
        for epoch in (1,7,15):
            for batch in range(1,80):
                clean,labels,domains,view,info=native._prepare_concat_sat_batch_for_training(augment,x,y,domain,args=a,epoch=epoch,batch_idx=batch)
                assert clean.shape==x.shape and torch.equal(labels,y) and view is not None
                if view.applied:
                    assert view.scenario in d.TRAIN_SCENES and torch.isfinite(view.x).all() and not torch.equal(x,view.x)
                    executed.add(view.scenario)
                    if epoch==1:
                        model=FusedCleanSatelliteForward(m,view.x);out=model(clean,y_tx=labels,domain_labels=domains,return_aux=True)
                        loss=torch.nn.functional.cross_entropy(out['tx_logits'],y)+.68*torch.nn.functional.cross_entropy(model.satellite_output['tx_logits'],y)
                        loss.backward();assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in m.id_backbone.parameters())
                    if all(s in executed for s in d.TRAIN_SCENES):break
        assert executed==set(d.TRAIN_SCENES)
        state=native._feature_training_state();assert state.optimizer is None and state.reference is None and state.style is None and state.fishr is None
    return dict(status='PASS',epochs=30,updates=16890,actual_channel_paths=sorted(executed),satellite_CE_E1=True,query_read=False,negative_checks=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(2);value=check()
    if a.remote_preflight:
        from .data import source_contract
        from .evaluate import view_manifest
        value['actual_source']=source_contract();value['view_manifest_status']=view_manifest()['status']
        # Metadata only: preserve the completed control and ensure paired
        # scoring can consume its fixed predictions after this run finishes.
        from pathlib import Path
        pred=Path(d.config(d.rows()[0])['control_prediction']);prior=d.read(pred.parent/'complete.json')
        if not pred.is_file() or prior['status']!='PREDICTIONS_COMPLETE' or prior['count']!=d.COUNT or prior['views']!=list(d.VIEWS) or prior['truth_read'] or prior['query_fit']:raise ValueError('Fixed control prediction unavailable')
        value['control_predictions_available']=True
    d.write(a.output,value);print(json.dumps(value))
