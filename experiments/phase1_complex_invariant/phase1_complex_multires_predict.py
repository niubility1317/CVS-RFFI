"""Truth-blind Phase1 predictions and independent score for a fixed mask."""
import argparse
from pathlib import Path
import numpy as np
import torch

from phase1_complex_multires_mask import ComplexMultiResolutionMask, aligned_observation
from phase1_complex_multires_source import source_inputs
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.dura_trisig_baseline import DURA as DURA_PREDICTIONS


def load_fixed(out,commit,device):
    data=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
    if data['commit']!=commit or data['selected']['kind']!='complex_multires_mask':
        raise ValueError('Exact source-frozen mask checkpoint required')
    model=ComplexMultiResolutionMask().to(device)
    model.load_state_dict(data['model'],strict=True)
    return model.eval().requires_grad_(False),float(data['selected']['alpha'])


@torch.no_grad()
def apply_mask(model,alpha,anchor,received):
    aligned,gate=aligned_observation(anchor,received)
    filtered=model(anchor,aligned,gate)
    output=anchor+alpha*(filtered-anchor)
    if not torch.isfinite(output).all():raise ValueError('Finite restored IQ required')
    return output,gate


def smoke(root,out,commit,device):
    _,_,_,_,_,_,_,_,_,data=source_inputs(root,smoke=True)
    model,alpha=load_fixed(out,commit,device)
    for view in ('clean','practical_low_suburban','practical_mid_urban','practical_low_urban'):
        d=data[view]
        anchor=core.tensor(d['anchor'],device);received=core.tensor(d['observed'],device)
        expected_aligned=core.tensor(d['received'],device)
        expected_gate=core.tensor(d['gate'].astype(np.float32),device)
        with torch.no_grad():
            aligned,gate=aligned_observation(anchor,received)
            torch.testing.assert_close(aligned,expected_aligned,rtol=1e-4,atol=1e-4)
            if not torch.equal(gate,expected_gate.bool()):raise ValueError('Source gate implementation differs')
            output,actual_gate=apply_mask(model,alpha,anchor,received)
            expected=anchor+alpha*(model(anchor,expected_aligned,expected_gate)-anchor)
            torch.testing.assert_close(output,expected,rtol=1e-4,atol=1e-4)
        if not torch.isfinite(output).all():raise ValueError('Nonfinite source smoke')
    f,_=core.teacher(root,device)
    with torch.no_grad():
        logits=f.classify_features(f.features(output))
    if logits.shape[1]!=6 or not torch.isfinite(logits).all():raise ValueError('Invalid frozen ground reader')
    core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,F_frozen=True,DURA_frozen=True,TX_labels_read=0,test_IQ_reads=0))


@torch.no_grad()
def predict(root,out,commit,device):
    smoke_result=core.read(out/'prediction_smoke.json')
    if smoke_result.get('status')!='PASS':raise ValueError('Real source smoke required')
    model,alpha=load_fixed(out,commit,device);f,_=core.teacher(root,device)
    folder,ids=core.reference(root);dura=root/DURA_PREDICTIONS;done=core.read(dura/'complete.json')
    expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=len(ids),views=list(core.VIEWS),TX_truth_reads=0,state_updates=0,head_fit_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete same-record DURA predictions required')
    if not np.array_equal(ids,np.load(dura/'physical_ids.npy',allow_pickle=False)):raise ValueError('DURA physical IDs differ')
    dst=out/'flow'/'test_prediction';dst.mkdir(parents=True,exist_ok=False)
    np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    for view in core.VIEWS:
        received=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        anchor=np.load(dura/(view+'.restored.iq.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(len(ids),2,256) or anchor.shape!=received.shape or received.dtype!=np.float32 or anchor.dtype!=np.float32:
            raise ValueError('Complete aligned Phase1 IQ required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        active=0
        for start in range(0,len(ids),64):
            x=core.tensor(received[start:start+64],device);a=core.tensor(anchor[start:start+64],device)
            output,gate=apply_mask(model,alpha,a,x)
            r=f.classify_features(f.features(x));z=f.classify_features(f.features(output))
            if not torch.isfinite(r).all() or not torch.isfinite(z).all():raise ValueError('Finite ground decisions required')
            raw[start:start+len(x)]=np.from_dlpack(r.argmax(1).cpu().contiguous())
            restored[start:start+len(x)]=np.from_dlpack(z.argmax(1).cpu().contiguous())
            active+=int(gate.sum())
        raw.flush();restored.flush();print('PHASE1_MASK_VIEW_COMPLETE',view,len(ids),'gate',active,flush=True)
    core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('smoke','predict','score'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='smoke':smoke(args.root,args.out,args.commit,args.device)
    elif args.mode=='predict':predict(args.root,args.out,args.commit,args.device)
    else:core.score(args.root,args.out,args.commit)


if __name__=='__main__':main()
