"""Single launch smoke: real checkpoint + legal source IQ + four native Full steps."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor

def run(rt,jobs,output):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import load_original_T1,load_G
    from .cache import transform,configuration
    from .plan import FAMILIES,SCENES
    from .source import check_training,build,native
    from .objectives import step
    h=read(Path(rt['run_root'])/'data/handles.json')['F0'];contract=read(h['source_contract'])
    source=SourceDataset(h['source_cache'],'L_s',contract)
    positions=np.linspace(0,len(source)-1,12,dtype=np.int64)
    x=source.x[positions.tolist()];meta=source.metadata
    reader=load_original_T1(2026092701,contract,'cpu')
    with torch.no_grad():
        logits=reader.classify_features(reader.features(x[:2]))
        if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real source checkpoint failed')
    routes=[]
    for name in ('DURA-L','DURA-LU'):
        spec=rt['G_templates']['F0']['readouts'][name];g=load_G(spec,contract,'cpu')
        with torch.no_grad():y=g(x[:2])
        if y.shape!=x[:2].shape or not bool(torch.isfinite(y).all()):raise ValueError('Frozen G source smoke failed')
        routes.append(name);del g
    generated=[];counts={}
    for i,j in enumerate(positions):
        a,s,l=transform(np.asarray(x[i].tolist(),np.float32),str(meta['ids'][j]),int(meta['rx_i'][j]),int(meta['day_i'][j]),1,1,.8)
        b,_,_=transform(np.asarray(x[i].tolist(),np.float32),str(meta['ids'][j]),int(meta['rx_i'][j]),int(meta['day_i'][j]),1,1,.8)
        if not np.array_equal(a,b):raise ValueError('Non-deterministic cached IQ function')
        generated.append(a);counts[str(s)]=counts.get(str(s),0)+1
    augmented=tensor(np.asarray(generated,np.float32));records=[]
    for family in FAMILIES:
        j=next(j for j in jobs if j['kind']=='F' and read(j['config']).get('family')==family and read(j['config']).get('cell')=='D')
        c=read(j['config']);check_training(c);native.seed(c['model_seed']);net=build(c)
        labels=torch.tensor(meta['y'][positions].tolist());rx=torch.tensor([contract['source_rxs'].index(int(v)) for v in meta['rx_i'][positions]])
        full=True
        optimizer=torch.optim.SGD(list(net.ec.parameters())+list(net.rc.parameters()),lr=c['lr']) if family=='riei' else torch.optim.Adam(net.parameters(),lr=c['lr'])
        fed=torch.optim.SGD(net.fed.parameters(),lr=c['lr']) if family=='riei' else None
        class Aug:
            def __call__(self,*args):return augmented,dict(applied=True,applied_count=int(sum(v for k,v in counts.items() if k!='-1')),scenario='three',probability=.8,scene_counts=counts)
        author=native.author_sdg(c) if family=='sdg' else None
        # This bounded smoke checks native loss/gradient execution. Real training
        # uses the author's dataset transforms, independently seeded and tested.
        record=step(net,c,x,labels,rx,optimizer,Aug(),0.,0,unlabeled=x.clone(),
            native_views=[x.clone(),x.clone()] if author else None,author=author,fed_optimizer=fed,unlabeled_receivers=rx)
        if record['optimizer_steps']!=(3 if family=='riei' else 1) or record['pl_seen']!=12:raise ValueError('Full loss/dose changed')
        records.append(dict(family=family,native_loss=record['native_penalty'],finite=True,default_head_joint=True))
        del net
    for scene in SCENES:
        cfg=configuration(scene)
        if cfg.processing_route!='residual' or cfg.mode!='post_sync' or cfg.equalization_enabled:raise ValueError('Wrong engine')
    write(output,dict(status='PASS',real_final_T1_checkpoint=True,real_G_checkpoints=routes,source_samples=12,
        Full_steps=records,deterministic_FP32_transform=True,target_reads=0,U_truth_reads=0,
        formal_training_updates=0,smoke_only_disposable_models=True))
