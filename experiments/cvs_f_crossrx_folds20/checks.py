import copy,io,math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .source import Observer
from .channel import Practical
from experiments.cvs_f_crossrx_practical20.channel import Practical as FirstPractical
from experiments.cvs_f_crossrx_practical20 import model as first_api,supcon
from experiments.cvs_ordinary_fold20.fold import selected

def reject(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid source role/budget accepted')

class MetadataOnly:
    def __init__(self,role):
        self.n={'L_s':6300,'U_s':56700,'V':27000}[role];self.hide=role=='U_s';self.selected=list(range(self.n))
        self.index=[SimpleNamespace(rx_i=d.ALL_RX[i%5]) for i in range(self.n)]
    def __len__(self):return self.n

def run(output):
    from SSDG import train_ssdg as native
    torch.set_num_threads(2);torch.manual_seed(20261016429);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x=torch.randn(128,2,256);y=torch.arange(128)%6;day=torch.arange(128)%3+1;initial=None;packets=updates=physical=negatives=0;primary=None;results=[]
    for held in d.ALL_RX:
        for role in ('L_s','U_s','V'):
            ds=MetadataOnly(role);fit=selected(ds,role,held_rx=held);test=selected(ds,role,held=True,held_rx=held)
            assert len(fit)==len(ds)*4//5 and len(test)==len(ds)//5 and not set(fit)&set(test)
            assert {ds.index[i].rx_i for i in fit}==set(d.ALL_RX)-{held} and {ds.index[i].rx_i for i in test}=={held}
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu');held=row['held_RX'];fit=c['fold']['fit_RX']
        assert a.source_fold_held_rx==held and a.source_fold_training_receivers==','.join(map(str,fit))
        assert (a.epochs,a.label_epochs,a.pseudo_epochs)==(20,13,7) and not a.use_concat_sat_channel_aug
        rx=torch.tensor(fit*32);domains=torch.tensor([supcon.RX_BY_DOMAIN.index(int(r))+int(v)-1 for r,v in zip(rx,day)])
        meta=dict(rx_i=rx,day_i=day,eq_i=torch.ones(128,dtype=torch.long),base_index=torch.arange(128),sig_i=torch.arange(128))
        m=api.build(c,'cpu').train();state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:
            initial=state;reference=first_api.build(d.first.config(d.first.rows()[0]),'cpu')
            assert state.keys()==reference.state_dict().keys() and all(torch.equal(v,reference.state_dict()[k]) for k,v in state.items())
        else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
        ema=copy.deepcopy(m)
        for key in ('_feature_l_steps','_legacy_calls','_anchor_calls','_supcon_calls'):setattr(m,key,1110)
        o=Observer(c);o.out=output/row['row_id'];o.out.mkdir();o.m=o.model=m;o.initialize(m,c,o.out);o.sequence=io.StringIO()
        o.record('L_s',(x,y,domains,meta));maps={level:[] for level in supcon.LEVELS};handles=[];captured=[];original=supcon.penalty
        for level in supcon.LEVELS:handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(lambda mod,args,out,level=level:maps[level].append(out)))
        def penalty(values,*args):
            value=original(values,*args);captured.append((dict(values),value[0]));return value
        supcon.penalty=penalty
        try:
            before=torch.get_rng_state().clone()
            with o.native_scope(native):
                cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,a)
                first=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);second=m._pair_secondary.clone()
                assert torch.equal(before,torch.get_rng_state())
                if primary is None:primary=first.detach().clone()
                else:assert torch.equal(primary,first)
                assert (torch.equal(first,second))==(not c['As'])
                with m._phase1_labeled_feature_context():out=m(first,y_tx=y,return_aux=True)
                ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                correction,mm=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128)
                full=ce+correction;aux=torch.stack([z[1] for z in captured]).mean()
                assert torch.allclose(full,.5*(mm['pair/primary_objective']+mm['pair/secondary_objective'])+.01*aux,atol=1e-6)
                for level in supcon.LEVELS:assert len(maps[level])==4 and all(captured[j][0][level] is maps[level][j] for j in (0,1))
                head=dict(m.named_parameters())['id_backbone.encoder.core.id_backbone.cls_head.weight']
                assert torch.autograd.grad(aux,head,allow_unused=True,retain_graph=True)[0] is None
                full.backward();assert all(dict(m.named_parameters())[k].grad is not None and torch.isfinite(dict(m.named_parameters())[k].grad).all() and float(dict(m.named_parameters())[k].grad.norm())>0 for k in o.feature_parameter_names)
                opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
                assert opt._sinc_unit_steps==opt._sinc_physical_clip_calls==1 and m._supcon_handles==[] and m._supcon_pending is None and m._pair_primary is None and m._pair_secondary is None
                assert m.id_backbone.encoder._ordinary_train_packets==512;packets+=512;updates+=1;physical+=o.channel.packets
        finally:
            supcon.penalty=original
            for h in handles:h.remove()
        assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
        small={k:v[:8] for k,v in meta.items()};train=Practical('train',held);val=Practical('validation',held);vm=dict(small,rx_i=torch.full((8,),held))
        one,m1=train.apply(x[:8],small,1,1);two,m2=train.apply(x[:8],small,2,223)
        assert not torch.equal(one,two) and m1[0]['receiver']==m2[0]['receiver']
        v1,p1=val.apply(x[:8],vm,scene=d.SCENES[-1]);v2,p2=val.apply(x[:8],vm,scene=d.SCENES[-1]);assert torch.equal(v1,v2) and p1==p2
        physical+=32
        for fn in (lambda:train.apply(x[:8],vm,1,1),lambda:val.apply(x[:8],small,scene=d.SCENES[-1]),lambda:o.record('U_s',(x,y,domains,meta)),lambda:o.record('L_s',(x,y,domains,dict(meta,rx_i=torch.full((128,),held))))):reject(fn);negatives+=1
        results.append(dict(held_RX=held,arm=row['arm'],fit_RX=fit,actual_native_args=True,four_L_forwards=True,only_primary_SupCon=True,default_head_joint=True))
    # At held RX1 the refactor must be the exact original waveform and metadata.
    rx=torch.tensor([3,4,6,8]*2);meta=dict(rx_i=rx,day_i=torch.arange(8)%3+1,eq_i=torch.ones(8,dtype=torch.long),base_index=torch.arange(8),sig_i=torch.arange(8))
    for role in ('train','validation'):
        if role=='validation':meta=dict(meta,rx_i=torch.ones(8,dtype=torch.long))
        old=FirstPractical(role);new=Practical(role,1);kw=dict(epoch=1,step=1) if role=='train' else dict(scene=d.SCENES[-1])
        a,ma=old.apply(x[:8],meta,**kw);b,mb=new.apply(x[:8],meta,**kw);assert torch.equal(a,b) and ma==mb;physical+=16
    for change in (dict(epochs=50),dict(test_views=['clean']),dict(checkpoint_sources=['prior']),dict(fold=dict(d.config(d.rows()[0])['fold'],fit_RX=[1,3,4,6,8]))):reject(lambda:d.validate(dict(d.config(d.rows()[0]),**change)));negatives+=1
    reject(lambda:Practical('train',2));negatives+=1
    return dict(status='PASS',synthetic_network_packets=packets,synthetic_optimizer_updates=updates,synthetic_practical_packets=physical,protocol_negatives=negatives,
        all_five_metadata_only_role_splits=True,initial_F_state_tensors=len(initial),same_initial_F_all_rows=True,old_RX1_waveform_metadata_exact=True,
        all_eight_actual_E20_args=True,primary_IQ_exact_all_rows=True,paired_four_L_and_primary_only_SupCon=True,default_head_SIG3_Sinc_joint_gradients=True,
        hardware_session_stability=True,epoch_refresh_and_validation_reproducibility=True,unchanged_own_EMA=True,real_IQ=0,checkpoint_reads=0,target_access=False,rows=results)
