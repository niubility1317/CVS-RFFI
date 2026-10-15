"""Fixed TX/RX/day-balanced legal L panel; validation draws never train draws."""
from collections import defaultdict
import torch
from . import design as d
from .channel import PairedChannel
_PANEL=None

def load():
    global _PANEL
    if _PANEL is None:
        from experiments.cvs_nocfo_foundation.source import load_source
        from experiments.cvs_nocfo_foundation import design as nd
        L,U,V,contract=load_source(nd.config(nd.rows()[0]))
        if U is not None:raise ValueError('Read-only L panel unexpectedly opened U')
        order=torch.randperm(len(L['ids']),generator=torch.Generator().manual_seed(2026101499));indices=[]
        for tx in range(6):
            for rx in (1,3,4,6,8):
                for day in (1,2,3):
                    ix=order[(L['y'][order]==tx)&(L['rx'][order]==rx)&(L['day'][order]==day)]
                    if len(ix)!=70:raise ValueError('Source panel cell differs')
                    indices.extend(ix[:8].tolist())
        ix=torch.tensor(indices);_PANEL={k:L[k][ix] for k in ('x','y','rx','day')}
        _PANEL['ids']=[L['ids'][i] for i in indices]
        if len(set(_PANEL['ids']))!=720 or not set(_PANEL['ids'])<=set(contract['role_ids']['L_s']):
            raise ValueError('Illegal source panel')
    return _PANEL

def metadata(panel,start,stop):
    return dict(rx_i=panel['rx'][start:stop],day_i=panel['day'][start:stop],
        eq_i=torch.ones(stop-start,dtype=torch.long),base_index=torch.arange(start,stop),
        physical_id=panel['ids'][start:stop])

def evaluate(model,device):
    panel=load();modes={m:m.training for m in model.modules()};model.eval()
    channel=PairedChannel(d.config(d.rows()[0]));summaries=[];hard_margin=[];hits=0;groups=defaultdict(list)
    try:
        with torch.no_grad():
            for start in range(0,720,90):
                stop=start+90;x=panel['x'][start:stop].to(device);y=panel['y'][start:stop].to(device)
                logits=model.forward_identity_only(x)['tx_logits'];hits+=int((logits.argmax(1)==y).sum())
                # Fixed scene-by-scene nominal draws, no curriculum selection.
                for scene in d.TRAIN_SCENES:
                    old=d.TRAIN_SCENES
                    # Selection from a one-element local scene list, no global mutation.
                    satellite,records=apply_fixed_scene(channel,x,metadata(panel,start,stop),scene)
                    output=model.forward_identity_only(satellite)['tx_logits']
                    other=output.masked_fill(torch.nn.functional.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margins=(output.gather(1,y[:,None]).squeeze(1)-other)/30
                    for i,r in enumerate(records):
                        item=dict(correct=bool(output[i].argmax()==y[i]),margin=float(margins[i]),
                            clean_correct=bool(logits[i].argmax()==y[i]))
                        groups[(scene,r['state_end'],r['receiver_lock_status'])].append(item)
                        if scene in d.URBAN and 3<=r['quality_snr_db']<=18 and r['receiver_lock_status']!='locked':
                            hard_margin.append(item['margin'])
    finally:
        for m,mode in modes.items():m.training=mode
    for key,records in sorted(groups.items()):
        correct=[r for r in records if r['clean_correct']]
        summaries.append(dict(scene=key[0],state=key[1],lock=key[2],count=len(records),
            accuracy=sum(r['correct'] for r in records)/len(records),
            margin=sum(r['margin'] for r in records)/len(records),
            clean_correct_retention=sum(r['correct'] for r in correct)/len(correct) if correct else None))
    return dict(clean_accuracy=hits/720,hard_margin=sum(hard_margin)/len(hard_margin) if hard_margin else 0.,
        hard_count=len(hard_margin),strata=summaries,source_count=720,source_role='L_s',query_read=False,
        U_truth_read=False,head_fit_updates=0)

def apply_fixed_scene(channel,x,meta,scene):
    import numpy as np
    from .channel import identities,numpy_iq,tensor_iq,usability
    ch,r=channel.ch,channel.residual;cfg=channel.configs[scene];ids,sessions=identities(meta)
    raw=numpy_iq(x);values=[];records=[]
    for i,pid in enumerate(ids):
        seed=ch.stable_seed(2026101402,'source_validation/fixed',scene,pid)
        record,_=ch.normalize_input(raw[i,0]+1j*raw[i,1])
        y,m=r.ResidualChannel(cfg,seed,ch.receiver_for_session(cfg,2026092701,sessions[i])).process(record)
        m.update(sample_id=pid,session_id=sessions[i],q=usability(m));values.append(np.stack((y.real,y.imag)));records.append(m)
    return tensor_iq(np.stack(values),x.device),records
