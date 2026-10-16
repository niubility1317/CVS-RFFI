import copy,io,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .pairing import Bank,FIT,KEYS,key
from .source import Observer
from experiments.cvs_f_crossrx_practical20 import model as parent_model,design as parent
from experiments.cvs_ordinary_fold20.streams import stream

def reject(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError('Invalid real-pair protocol accepted')
def fixture():
    x=torch.randn(5040,2,256);entries=[]
    for t in range(6):
        for r in FIT:
            for day in (1,2,3):
                for sig in range(70):
                    i=len(entries);entries.append((x[i],t,3+FIT.index(r)*3+day-1,dict(eq_i=1,rx_i=r,day_i=day,sig_i=sig,base_index=i)))
    return entries
class Dataset:
    hide=False;transform=None
    def __init__(self,entries):self.entries=entries
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        x,y,compact,meta=self.entries[i]
        return x,y,int(meta['rx_i'])*4+int(meta['day_i']),meta
DOMAIN_MAP={r*4+day:3*i+day-1 for i,r in enumerate((1,3,4,6,8)) for day in (1,2,3)}
def run(output):
    from SSDG import train_ssdg as native
    torch.set_num_threads(2);torch.manual_seed(20261016719);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    entries=fixture();indices=[i*37%5040 for i in range(128)];batch=[entries[i] for i in indices]
    x=torch.stack([e[0] for e in batch]);y=torch.tensor([e[1] for e in batch]);domains=torch.tensor([e[2] for e in batch]);meta={k:torch.tensor([e[3][k] for e in batch]) for k in KEYS}
    negative=0;initial=None;primary={};secondary={};packets=0;updates=0
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu');m=api.build(c).train();state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:
            initial=state;old=parent_model.build(parent.config(parent.rows()[0]));assert state.keys()==old.state_dict().keys() and all(torch.equal(v,old.state_dict()[k]) for k,v in state.items())
        else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
        ema=copy.deepcopy(m);assert ema._source_labeled_feature_hook.__self__ is ema and ema._supcon_original_hook.__self__ is ema
        for name in ('_feature_l_steps','_legacy_calls','_anchor_calls','_supcon_calls'):setattr(m,name,1110)
        o=Observer(c);o.out=out/row['arm'];o.out.mkdir();o._bank_dataset=Dataset(entries);o._bank_domain_map=DOMAIN_MAP;o.m=o.model=m;o.initialize(m,c,o.out)
        o.bank.calls=1110;o.bank.packets=1110*128;o._current_L_meta=meta;o._current_L_labels=y.tolist()
        # Ordered E6 interface injection only, not an E6-trained synthetic model.
        seen=[];original_forward=m.forward_identity_only
        def observed(value,*args,**kwargs):seen.append((value.detach().clone(),kwargs.get('domain_labels')));return original_forward(value,*args,**kwargs)
        m.forward_identity_only=observed
        rng=torch.get_rng_state().clone();np_state=np.random.get_state()[1].tolist()
        with o.native_scope(native):
            cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,a)
            first=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);other=m._pair_secondary.clone();other_domains=m._pair_secondary_domains.clone()
            assert torch.equal(rng,torch.get_rng_state()) and np_state==np.random.get_state()[1].tolist()
            assert torch.equal(other_domains,domains) if row['arm']=='SAME_RX' else bool((other_domains//3!=domains//3).all())
            primary[row['arm']]=[first.detach().clone()];secondary[row['arm']]=other
            with m._phase1_labeled_feature_context():result=m(first,y_tx=y,return_aux=True)
            ce=F.cross_entropy(result['tx_logits'],y,label_smoothing=.01)
            correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=result,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128)
            full=ce+correction;full.backward()
            assert all(dict(m.named_parameters())[name].grad is not None and torch.isfinite(dict(m.named_parameters())[name].grad).all() and float(dict(m.named_parameters())[name].grad.norm())>0 for name in o.feature_parameter_names)
            # The first native call uses model.forward, so only the following
            # three identity_only calls are intercepted here.
            assert len(seen)==3 and torch.equal(seen[0][0],first) and torch.equal(seen[1][0],other) and torch.equal(seen[2][0],other)
            assert torch.equal(seen[1][1],other_domains) and torch.equal(seen[2][1],other_domains)
            optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);optimizer.step();updates+=1
            assert optimizer._sinc_unit_steps==optimizer._sinc_physical_clip_calls==1 and m._supcon_handles==[] and m._supcon_pending is None
            assert m.id_backbone.encoder._ordinary_train_packets==512;packets+=512
            o._current_L_meta=meta;o._current_L_labels=y.tolist();next_first=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);primary[row['arm']].append(next_first.detach().clone())
            assert not torch.equal(other,m._pair_secondary)
            m._pair_primary=m._pair_secondary=m._pair_secondary_domains=None
        assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
    assert all(torch.equal(a,b) for a,b in zip(primary['SAME_RX'],primary['CROSS_RX'])) and not torch.equal(secondary['SAME_RX'],secondary['CROSS_RX'])
    # Compare primary actual ordinary draws directly to the unchanged private
    # stream, proving the secondary draw cannot advance its RNG or schedule.
    with stream(20261016051,0):raw=native.make_augmentor(native.build_aug_base_cfg(a))
    native.configure_augmentor_for_epoch(raw,native.build_aug_base_cfg(a),6,a)
    for call in (1,2):
        with stream(20261016051,call,x.device):expected=raw(x,labels=y,no_pa=not a.aug_enable_pa_normal)
        assert torch.equal(expected,primary['CROSS_RX'][call-1])
    # Real pair sampling: intact waveform equality, independent RNG refresh,
    # and exhaustive legal cells, with no model forward or real data.
    for mode in d.ARMS:
        bank=Bank(entries)
        for step in (1,2):
            partner,pd,records=bank.select(y,meta,step,mode,x)
            for j,r in enumerate(records):
                k=bank.index[tuple(r['partner'])];assert torch.equal(partner[j],entries[k][0]) and r['TX']==int(y[j]) and r['primary']!=r['partner']
                assert r['primary'][2]==r['partner'][2] and ((r['primary'][1]==r['partner'][1])==(mode=='SAME_RX'))
        assert bank.calls==2 and bank.packets==256
    for change in (dict(epochs=50),dict(checkpoint_sources=['old']),dict(As=True),dict(real_pair=dict(d.config(d.rows()[0])['real_pair'],V_used=True))):reject(lambda:d.validate(dict(d.config(d.rows()[0]),**change)));negative+=1
    for mutate in ('held','duplicate','label','domain','short'):
        broken=list(entries)
        if mutate=='short':broken=broken[:-1]
        else:
            z=list(broken[0]);z[3]=dict(z[3])
            if mutate=='held':z[3]['rx_i']=1
            elif mutate=='duplicate':z=list(broken[1])
            elif mutate=='label':z[1]=-1
            else:z[2]=0
            broken[0]=tuple(z)
        reject(lambda:Bank(broken));negative+=1
    for call in (lambda:Bank(entries).select(torch.full_like(y,-1),meta,1,'CROSS_RX',x),lambda:Bank(entries).select(y,meta,2,'CROSS_RX',x),lambda:Bank(entries).select(y,meta,1,'invalid',x),lambda:Bank(entries).select(y,meta,1,'CROSS_RX',x+.1)):
        reject(call);negative+=1
    q=dict(target_access=False,U_truth_read=False,head_fit_updates=0,optimizer_updates=4440,held_RX=1,fit_RX=list(FIT),complete_health=True,held_V=dict(accuracy=.8,macro_f1=.79),seen_V=dict(source_val_accuracy=.95))
    same=copy.deepcopy(q);same['held_V'].update(accuracy=.81,macro_f1=.8);cross=copy.deepcopy(q);cross['held_V'].update(accuracy=.82,macro_f1=.81)
    assert d.qualify(dict(SAME_RX=same,CROSS_RX=cross),q)['status']=='PASS'
    assert d.qualify(dict(SAME_RX=cross,CROSS_RX=cross),q)['status']=='FAIL'
    illegal=dict(cross,U_truth_read=True);reject(lambda:d.qualify(dict(SAME_RX=same,CROSS_RX=illegal),q));negative+=1
    return dict(status='PASS',protocol_negatives=negative,synthetic_network_packets=packets,synthetic_optimizer_updates=updates,
        synthetic_partner_packets=1024,synthetic_ordinary_IQ_packets=1280,initial_state_tensors=len(initial),matched_C0_initialization=True,original_primary_IQ_two_steps_exact=True,
        actual_partner_RX_day_domains=True,intact_sameTX_day_distinct_packets=True,four_forward_supervised_pairs=True,
        only_original_primary_SupCon=True,unchanged_own_EMA=True,default_head_SIG3_Sinc_joint_gradients=True,
        independent_observation_control_required=True,real_IQ_packets=0,checkpoint_loads=0,target_access=False)
