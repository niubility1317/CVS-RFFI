"""Fixed blind-gated direct IQ predictions, source-only smoke, truth-last score."""
import argparse
from pathlib import Path
import numpy as np
import torch
from phase1_blind_gate_model import BlindGate
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_axial_spectral_source import source_contract,epoch_data
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.dura_trisig_baseline import DURA as DURA_PREDICTIONS


def load_fixed(out,commit,device):
    d=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
    if d['commit']!=commit or d['selected']['kind']!='gated_direct_iq':raise ValueError('Exact fixed gated model required')
    model=AxialSpectralMask().to(device);model.load_state_dict(d['model'],strict=True)
    gate=BlindGate().to(device);gate.load_state_dict(d['gate'],strict=True)
    return model.eval().requires_grad_(False),gate.eval().requires_grad_(False),float(d['selected']['alpha']),float(d['selected']['threshold'])


def apply(model,gate,alpha,threshold,x):
    enabled=gate(x).sigmoid()>threshold;wave=x+enabled[:,None,None]*alpha*(model(x)-x)
    return wave,enabled


def smoke(root,out,commit,device):
    cache,_,_,training,_,_=source_contract(root);received,_=epoch_data(cache,20)
    model,gate,alpha,threshold=load_fixed(out,commit,device);x=core.tensor(received[training[:16]],device);teacher,_=core.teacher(root,device)
    with torch.no_grad():
        y,active=apply(model,gate,alpha,threshold,x);logits=teacher.classify_features(teacher.features(y))
    if y.shape!=x.shape or logits.shape!=(len(x),6) or not torch.isfinite(y).all() or not torch.isfinite(logits).all():raise ValueError('Source inference smoke failed')
    if not torch.equal(y[~active],x[~active]):raise ValueError('Gate must exactly preserve disabled samples')
    core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,F_frozen=True,gate_input='IQ',DURA_used=False,TX_labels_read=0,test_IQ_reads=0))


@torch.no_grad()
def predict(root,out,commit,device):
    if core.read(out/'prediction_smoke.json').get('status')!='PASS':raise ValueError('Real source smoke required')
    model,gate,alpha,threshold=load_fixed(out,commit,device);teacher,_=core.teacher(root,device);folder,ids=core.reference(root)
    dura=root/DURA_PREDICTIONS;done=core.read(dura/'complete.json');expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=len(ids),views=list(core.VIEWS),TX_truth_reads=0,state_updates=0,head_fit_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()) or not np.array_equal(ids,np.load(dura/'physical_ids.npy',allow_pickle=False)):raise ValueError('Complete matched DURA comparator required')
    dst=out/'flow'/'test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={}
    for view in core.VIEWS:
        received=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(len(ids),2,256) or received.dtype!=np.float32:raise ValueError('Complete Phase1 IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));count=0
        for start in range(0,len(ids),64):
            x=core.tensor(received[start:start+64],device);y,active=apply(model,gate,alpha,threshold,x);r=teacher.classify_features(teacher.features(x));z=teacher.classify_features(teacher.features(y))
            if not torch.isfinite(r).all() or not torch.isfinite(z).all():raise ValueError('Finite decisions required')
            raw[start:start+len(x)]=np.from_dlpack(r.argmax(1).cpu().contiguous());restored[start:start+len(x)]=np.from_dlpack(z.argmax(1).cpu().contiguous());count+=int(active.sum())
        raw.flush();restored.flush();accepted[view]=count;print('PHASE1_BLIND_GATE_VIEW_COMPLETE',view,len(ids),'accepted',count,flush=True)
    core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='smoke':smoke(args.root,args.out,args.commit,args.device)
    elif args.mode=='predict':predict(args.root,args.out,args.commit,args.device)
    else:core.score(args.root,args.out,args.commit)
if __name__=='__main__':main()
