import json,os,time
from pathlib import Path
import numpy as np
import torch
from .models import load_checkpoint
from . import design as d
from experiments.pamf_change_preserve.runner import load_restorer
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference,VIEWS
from experiments.pamf_restoration.runner import array_tensor,tensor_array
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.frozen import classification_statistics

READOUTS=('identity','parent_R03','refined')


@torch.no_grad()
def predict(args):
    ids,_,_=source_reference(args.parent_cache/'validation',args.source_cache)
    row=args.output/args.arm;saved=row/'prediction';saved.mkdir()
    model,item=load_checkpoint(row/'final.pt',args.arm,'cuda:0')
    if (item['epoch'],item['updates'])!=(d.EPOCHS,d.UPDATES):raise ValueError('Fixed final scratch budget required')
    parent=load_restorer(args.parent_restoration/'residual_deterministic/final.pt').to('cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0');elapsed={}
    for view in VIEWS:
        observed=np.load(args.parent_cache/'validation'/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if observed.shape!=(len(ids),2,256):raise ValueError('Fixed source observation layout required')
        arrays={name:np.lib.format.open_memmap(saved/(view+'_'+name+'.npy'),mode='w+',dtype=np.float32,shape=observed.shape)
                for name in ('parent_R03','refined')}
        decisions={k:[] for k in READOUTS};started=time.perf_counter()
        for offset in range(0,len(ids),64):
            y=array_tensor(observed[offset:offset+64],'cuda:0');r=parent(y);value=model(y)
            if not torch.isfinite(r).all() or not torch.isfinite(value).all():raise FloatingPointError('Nonfinite restored IQ')
            for name,x in zip(READOUTS,(y,r,value)):
                scores=reader(x)
                if scores.shape!=(len(y),6) or not torch.isfinite(scores).all():raise ValueError('Invalid original defaulthead output')
                decisions[name].extend(scores.argmax(1).tolist())
            arrays['parent_R03'][offset:offset+len(y)]=tensor_array(r)
            arrays['refined'][offset:offset+len(y)]=tensor_array(value)
        for array in arrays.values():array.flush()
        with (saved/(view+'.npz')).open('xb') as stream:
            np.savez_compressed(stream,ids=ids,**{k:np.asarray(v,np.uint8) for k,v in decisions.items()})
        elapsed[view]=time.perf_counter()-started
        print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE',arm=args.arm,view=view,count=len(ids),seconds=elapsed[view])),flush=True)
    write_json(saved/'complete.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),
        readouts=list(READOUTS),role='source_V',target_reads=0,TX_label_reads=0,state_updates=0,per_view_seconds=elapsed))
    del model,parent,reader;torch.cuda.empty_cache()


def completed_predictions(row,ids):
    saved=row/'prediction';done=json.loads((saved/'complete.json').read_text())
    expected=dict(status='SOURCE_PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),readouts=list(READOUTS),
                  role='source_V',target_reads=0,TX_label_reads=0,state_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('All seven source predictions required before truth')
    decisions={}
    for view in VIEWS:
        with np.load(saved/(view+'.npz'),allow_pickle=False) as packet:
            if set(packet.files)!={'ids',*READOUTS} or not np.array_equal(packet['ids'],ids):raise ValueError('Source prediction IDs differ')
            decisions[view]={}
            for name in READOUTS:
                values=packet[name]
                if values.shape!=(len(ids),) or values.dtype.kind not in 'iu' or (values<0).any() or (values>5).any():
                    raise ValueError('Invalid original defaulthead predictions')
                decisions[view][name]=values.copy()
        for name in ('parent_R03','refined'):
            array=np.load(saved/(view+'_'+name+'.npy'),mmap_mode='r',allow_pickle=False)
            if array.shape!=(len(ids),2,256) or not np.isfinite(array).all():raise ValueError('Incomplete source restored IQ')
    return decisions


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    reference=args.parent_cache/'validation';row=args.output/args.arm
    ids,source,index=source_reference(reference,args.source_cache)
    decisions=completed_predictions(row,ids)
    # No labels or paired-clean scoring until all declared predictions exist.
    labels=np.load(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
    clean=np.load(reference/'clean.npy',allow_pickle=False);energy=np.square(clean).mean((1,2)).clip(1e-4)
    records=[];waveform=[]
    for view in VIEWS:
        arrays=dict(identity=np.load(reference/(view+'.npy'),allow_pickle=False))
        arrays.update({name:np.load(row/'prediction'/(view+'_'+name+'.npy'),allow_pickle=False) for name in ('parent_R03','refined')})
        waveform.append(dict(view=view,NMSE={name:float((np.square(value-clean).mean((1,2))/energy).mean()) for name,value in arrays.items()}))
        for name in READOUTS:
            for baseline in ('identity','parent_R03'):
                groups=[('overall','all',np.ones(len(ids),bool))]
                for dim,values in (('TX',labels),('RX',source.rx[index]),('day',source.day[index])):
                    groups.extend((dim,str(v),values==v) for v in np.unique(values))
                for dimension,value,mask in groups:
                    records.append(dict(view=view,readout=name,baseline=baseline,dimension=dimension,value=value,
                        **classification_statistics(labels[mask],decisions[view][name][mask],decisions[view][baseline][mask])))
    write_json(row/'summary.json',dict(status='ANALYZED',source_only=True,arm=args.arm,count=len(ids),truth_last=True,
        records=records,waveform=waveform,target_reads=0,F_optimizer_updates=0,head_fit_updates=0,
        limitation='Same source development panel and fixed Fseed;target performance unmeasured'))
