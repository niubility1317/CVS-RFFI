"""Focused actual F execution and protocol-negative checks, synthetic IQ only."""
import copy
import sys
from pathlib import Path
import torch

def run():
    root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
    from SSDG import train_ssdg as actual_native
    from . import design as d,model as api,supcon
    from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
    from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
    torch.set_num_threads(2);reports=[];negative=0
    for key,value in (('epochs',50),('model_initialization','pretrained'),('test_views',['clean'])):
        c=copy.deepcopy(d.config(d.rows()[0]));c[key]=value
        try:d.validate(c)
        except ValueError:negative+=1
        else:raise AssertionError('Illegal source recipe accepted')
    # Source labels and physically distinct RX positives; sameTX/sameRX excluded.
    yy=torch.tensor([0,0,0,1]);rr=torch.tensor([3,3,4,4]);positive,den,valid=supcon.masks(yy,rr,'CROSS_RX')
    assert positive[0,2] and not den[0,1] and den[0,3]
    for bad_y,bad_r in ((torch.full((4,),-1),rr),(yy,torch.full((4,),2))):
        try:supcon.masks(bad_y,bad_r,'CROSS_RX')
        except ValueError:negative+=1
        else:raise AssertionError('U label/target RX accepted')
    for row in d.rows():
        c=d.config(row);m=api.build(c,'cpu');replica=copy.deepcopy(m).eval();assert replica._joint_channel is not m._joint_channel
        del replica
        m.train();m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
        torch.manual_seed(20261016109);raw=torch.randn(128,2,256);ordinary=raw.roll(2,-1)*.9
        labels=torch.arange(128)%6;domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains]
        meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
        xx,yy,extra=actual_native.move_batch((raw,labels,domains,meta),torch.device('cpu'))
        assert torch.equal(xx,raw) and torch.equal(yy,labels) and len(extra)==2 and torch.equal(extra[1]['rx_i'],rx)
        args=d.make_args(c,'cpu');ordinary_config=actual_native.build_aug_base_cfg(args);assert args.use_aug and isinstance(ordinary_config,dict)
        seen=[];original=m.id_backbone.encoder._view_features
        def traced(x):seen.append(x.detach().clone());return original(x)
        m.id_backbone.encoder._view_features=traced
        W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone()
        opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001)
        with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
        ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=raw)
        total=ce+correction
        assert len(seen)==6 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and torch.equal(seen[2],raw) and torch.equal(seen[2],seen[3]) and torch.equal(seen[4],seen[5])
        assert torch.equal(seen[4],raw)==(row['arm']=='B00')
        assert m._supcon_applied==1 and len(m._joint_details[0]['supcon_details'])==4 and not m._supcon_handles and m._supcon_pending is None
        total.backward();assert m.get_parameter(W).grad is not None and torch.isfinite(m.get_parameter(W).grad).all()
        clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W))
        m.id_backbone.encoder._view_features=original;m.eval()
        with torch.no_grad(),fresh_parameter_caches(m):
            f=m.id_backbone.encoder.features(raw[:2]).clone();export=api.export(m).eval();ef=export(raw[:2]);assert f.shape==(2,160) and torch.equal(f,ef)
        channel=m._joint_channel;audit=channel.audit()
        a,_=channel.apply(raw[:2],{k:v[:2] for k,v in meta.items()},namespace='source_eval/check',scene='practical_low_urban')
        b,_=channel.apply(raw[:2],{k:v[:2] for k,v in meta.items()},namespace='source_eval/check',scene='practical_low_urban')
        assert torch.equal(a,b)
        for kwargs in (dict(namespace='train/test',scene='practical_low_urban'),dict(namespace='source_eval/check',scene='bad')):
            try:channel.apply(raw[:2],{k:v[:2] for k,v in meta.items()},**kwargs)
            except ValueError:negative+=1
            else:raise AssertionError('Unregistered training selection accepted')
        assert all(v['processing_route']=='residual' and v['mode']=='post_sync' and not v['equalization_enabled'] for v in audit['actual_configs'].values())
        reports.append(dict(arm=row['arm'],parameters=sum(p.numel() for p in m.parameters()),L_forwards=6,same_IQ_pairs=True,raw_fixed_weight=.5,ordinary_weight=.25,s_slot_weight=.25,unchanged_aux_schedule_weight=m._joint_details[0]['supcon_weight'],supcon_layer_forward_terms=4,extra_views_have_no_supcon=True,default_W_joint_update=True,export160_exact=True,As_calls=m._joint_As_packets,total=float(total.detach()),audit=audit))
    return dict(status='PASS',negative_cases=negative,rows=reports,source_IQ_reads=0,target_access=False,head_fit_updates=0,synthetic_optimizer_steps=2)

if __name__=='__main__':
    import json
    result=run()
    from . import design as d
    d.write(Path(__file__).resolve().parents[2]/'local_artifacts/F_crossrx_satmix20_20261016/local_checks.json',result)
    print('LOCAL_RESULT '+json.dumps({k:v for k,v in result.items() if k!='rows'}))
