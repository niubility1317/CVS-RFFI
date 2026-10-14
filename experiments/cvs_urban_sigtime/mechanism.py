"""Post-freeze source-only branch observations and a fixed clean-correct reference."""
import gzip,json,time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .panel import load,metadata,apply_fixed_scene
from .channel import PairedChannel,imports,numpy_iq,tensor_iq
from .diagnose import frozen_model

@contextmanager
def observe(model):
    """Read actual deployed branch outputs; no alternative forward or fitted readout."""
    encoder=model.id_backbone.encoder;core=encoder.core.id_backbone
    values={};handles=[]
    modules=dict(time=core.t_proj,frequency=core.f_proj,PA=core.pa_proj,
        reference=encoder.response_projection,signature=encoder.sequence_pool)
    for name,module in modules.items():
        def capture(module,args,out,name=name):values[name]=out.detach().flatten(1)
        handles.append(module.register_forward_hook(capture))
    try:yield values
    finally:
        for h in handles:h.remove()

def gates(model):
    encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head
    return dict(PA=head.gain.detach().tanh().tolist(),
        reference=encoder.response_gain.detach().tanh().tolist(),
        signature=(.2*encoder.sequence_pool.gain_raw.detach().tanh()).tolist())

def forward(model,x,values):
    values.clear();out=model.forward_identity_only(x)
    if set(values)!={'time','frequency','PA','reference','signature'}:
        raise ValueError('Existing deployed branch path bypassed')
    return out,{k:v.clone() for k,v in values.items()}

def quality_bin(db):
    return 'unknown' if db is None else '<3' if db<3 else '3-8' if db<8 else '8-15' if db<15 else '>=15'

def evaluate(model,panel,reference_correct,*,reference_controls=False):
    groups=defaultdict(list);count=0;ch,r=imports();channel=PairedChannel(d.config(d.rows()[0]))
    before={k:v.clone() for k,v in model.state_dict().items()};model.eval()
    encoder=model.id_backbone.encoder;core=encoder.core.id_backbone
    modules=dict(time=core.t_proj,frequency=core.f_proj,PA=core.pa_proj,
        reference=encoder.response_projection,signature=encoder.sequence_pool)
    parameters={k:next(m.parameters()) for k,m in modules.items()}
    parameters.update(PA_gain=core.cls_head.gain,reference_gain=encoder.response_gain,
        signature_gain=encoder.sequence_pool.gain_raw,default_head=core.cls_head.weight)
    original_requires={p:p.requires_grad for p in model.parameters()};model.requires_grad_(True);gradients=[]
    with observe(model) as values:
        for start in range(0,len(panel['ids']),90):
            stop=min(start+90,len(panel['ids']));x=panel['x'][start:stop];y=panel['y'][start:stop]
            clean,branches=forward(model,x,values);count+=len(x)
            cg=torch.autograd.grad(F.cross_entropy(clean['tx_logits'],y,label_smoothing=.01),
                tuple(parameters.values()),allow_unused=True)
            for scene in d.SCENES:
                if reference_controls:
                    from .controls import compare
                    batches=defaultdict(list);details=defaultdict(list);raw=numpy_iq(x)
                    for i,pid in enumerate(panel['ids'][start:stop]):
                        record,_=ch.normalize_input(raw[i,0]+1j*raw[i,1])
                        seed=ch.stable_seed(2026101403,'source_diagnostic',scene,pid)
                        session=f"rx{int(panel['rx'][start+i])}:day{int(panel['day'][start+i])}:eq1"
                        outputs,info,_=compare(record,channel.configs[scene],seed,
                            ch.receiver_for_session(channel.configs[scene],2026092701,session))
                        for name,z in outputs.items():batches[name].append(np.stack((z.real,z.imag)));details[name].append(info[name])
                    batches={k:tensor_iq(np.stack(v),'cpu') for k,v in batches.items()}
                else:
                    satellite,records=apply_fixed_scene(channel,x,metadata(panel,start,stop),scene)
                    batches={'full':satellite};details={'full':[dict(v,state=v['state_end']) for v in records]}
                for condition,xx in batches.items():
                    sat,features=forward(model,xx,values);count+=len(xx);logits=sat['tx_logits']
                    sg=torch.autograd.grad(F.cross_entropy(logits,y,label_smoothing=.01),
                        tuple(parameters.values()),allow_unused=True)
                    health={}
                    for (key,p),a,b in zip(parameters.items(),cg,sg):
                        if a is None:a=torch.zeros_like(p)
                        if b is None:b=torch.zeros_like(p)
                        aa=float(a.square().sum());bb=float(b.square().sum())
                        health[key]=dict(clean_norm=aa**.5,satellite_norm=bb**.5,
                            cosine=float((a*b).sum())/(aa*bb)**.5 if aa*bb else None,
                            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
                    gradients.append(dict(scene=scene,condition=condition,batch=start//90,parameters=health))
                    correct=(logits.argmax(1)==y).tolist();clean_correct=(clean['tx_logits'].argmax(1)==y).tolist()
                    other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margins=((logits.gather(1,y[:,None]).squeeze(1)-other)/30).tolist()
                    comparisons={k:dict(cosine=F.cosine_similarity(branches[k],features[k]).tolist(),
                        clean_norm=branches[k].norm(dim=1).tolist(),satellite_norm=features[k].norm(dim=1).tolist()) for k in branches}
                    for i,m in enumerate(details[condition]):
                        base=details['full'][i]
                        item=dict(physical_id=panel['ids'][start+i],correct=correct[i],clean_correct=clean_correct[i],
                            fixed_reference_clean_correct=bool(reference_correct[start+i]),margin=margins[i],
                            branch={k:{metric:v[i] for metric,v in a.items()} for k,a in comparisons.items()})
                        key=(scene,condition,base['state'],base['receiver_lock_status'],
                            quality_bin(base.get('signal_to_added_noise_db')))
                        groups[key].append(item)
    for p,flag in original_requires.items():p.requires_grad_(flag)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):
        raise ValueError('Frozen source mechanism changed model state')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Source mechanism accumulated training gradients')
    summary=[]
    for key,items in sorted(groups.items()):
        retained=[i for i in items if i['fixed_reference_clean_correct']]
        summary.append(dict(scene=key[0],condition=key[1],state=key[2],lock=key[3],quality=key[4],count=len(items),
            accuracy=sum(i['correct'] for i in items)/len(items),margin=sum(i['margin'] for i in items)/len(items),
            fixed_reference_count=len(retained),fixed_reference_retention=sum(i['correct'] for i in retained)/len(retained) if retained else None,
            branches={k:{metric:sum(i['branch'][k][metric] for i in items)/len(items)
                for metric in ('cosine','clean_norm','satellite_norm')} for k in ('time','frequency','PA','reference','signature')}))
    return dict(strata=summary,gradients=gradients,forward_packets=count,gates=gates(model),state_unchanged=True),groups

def run():
    out=d.BASE/'source_mechanism';out.mkdir(exist_ok=False);torch.set_num_threads(2)
    panel=load();reference=frozen_model(device='cpu');started=time.perf_counter()
    with torch.no_grad():fixed=(reference.forward_identity_only(panel['x'])['tx_logits'].argmax(1)==panel['y'])
    eligible=[r for r in d.rows() if (d.BASE/r['row_id']/'source/training_mechanism_complete.json').is_file()]
    if not eligible:raise ValueError('No source-frozen own rows')
    # Keep controls on their original named draws; compare trained rows on a
    # second reference pass using precisely the same validation draws as them.
    all_results={};checkpoints=[('E50_controls',reference,True),('E50_fixed_reference',reference,False)]
    from .model import own_model
    checkpoints.extend((row['row_id'],None,False) for row in eligible)
    with gzip.open(out/'sample_branches.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as f:
        for row,model,controls in checkpoints:
            if model is None:model=own_model(d.config(next(r for r in eligible if r['row_id']==row)),'cpu')
            result,groups=evaluate(model,panel,fixed,reference_controls=controls);all_results[row]=result
            for key,items in groups.items():
                for item in items:f.write(json.dumps(dict(row=row,scene=key[0],condition=key[1],state=key[2],lock=key[3],quality=key[4],**item))+'\n')
            d.write(out/'progress.json',dict(completed=list(all_results),source_only=True,query_read=False))
            del model,groups
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,head_fit_updates=0,
        reference_checkpoint=d.diagnostic_config()[1]['output_root']+'/final_ssdg.pth',
        fixed_reference_ids=[pid for pid,keep in zip(panel['ids'],fixed.tolist()) if keep],
        reference_count=int(fixed.sum()),source_count=720,rows=all_results,seconds=time.perf_counter()-started,
        commit=(d.ROOT/'release_commit.txt').read_text().strip())
    d.write(out/'results.json',result);print(json.dumps(dict(status=result['status'],rows=list(all_results),reference_count=result['reference_count'])))
    return result

if __name__=='__main__':run()
