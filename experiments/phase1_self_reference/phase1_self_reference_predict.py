"""Selected self-reference IQ inference; Phase1 predictions before truth."""
import argparse
from pathlib import Path
import numpy as np
import torch
from phase1_self_reference_model import SelfReferenceNet
from self_reference_inference import apply
from self_reference_selection import choose
from phase1_self_reference_source import s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.dura_trisig_baseline import DURA as DURA_PREDICTIONS


def _same_state(expected,actual):
 return set(expected)==set(actual) and all(torch.equal(expected[k].cpu(),actual[k].cpu()) for k in expected)


def load_fixed(root,out,commit,device):
 complete=core.read(out.parent/'completion.json');selection=core.read(out/'source_selection.json');d=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
 if complete.get('status')!='SOURCE_SELECTION_COMPLETE' or complete.get('source_only') is not True or complete.get('target_predictions') is not False or complete.get('selected')!=selection['selected'] or d['selected']!=selection['selected'] or d['commit']!=commit or d['selected']['kind']!='self_reference_net':raise ValueError('Completed and fixed source-selected model required')
 choice=d['selected'];alpha=float(choice['alpha']);threshold=float(choice['threshold'])
 if alpha not in (.5,1.) or threshold not in (.8,.99) or int(choice['epoch']) not in (1,2,3):raise ValueError('Predeclared source-selected parameters required')
 if choose(selection['validation'])!=choice:raise ValueError('Source selection cannot be independently reproduced')
 checkpoint=torch.load(out/('checkpoint_epoch%02d.pt'%int(choice['epoch'])),map_location='cpu',weights_only=False)
 if checkpoint['commit']!=commit or checkpoint['epoch']!=int(choice['epoch']) or not _same_state(checkpoint['model'],d['model']):raise ValueError('Selected source checkpoint mismatch')
 phase,gate=s.initial_models(root,device)
 if not _same_state(d['phase'],phase.phase.state_dict()) or not _same_state(d['gate'],gate.state_dict()):raise ValueError('Source-selected frozen phase and gate required')
 model=SelfReferenceNet().to(device);model.load_state_dict(d['model'],strict=True)
 return model.eval().requires_grad_(False),phase.eval().requires_grad_(False),gate.eval().requires_grad_(False),alpha,threshold


def smoke(root,out,commit,device):
 cache,_,_,training,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,20);clean=training[scene[training]=='clean'][:16];impaired=training[scene[training]!='clean'][:16];ids=np.r_[clean,impaired]
 if len(ids)!=32:raise ValueError('Mixed source smoke required')
 model,phase,gate,alpha,threshold=load_fixed(root,out,commit,device);x=core.tensor(received[ids],device);teacher,_=core.teacher(root,device)
 with torch.no_grad():y,active=apply(model,phase,gate,alpha,threshold,x);logits=teacher.classify_features(teacher.features(y))
 if y.shape!=x.shape or logits.shape!=(len(x),6) or not torch.isfinite(y).all() or not torch.isfinite(logits).all() or not torch.equal(y[~active],x[~active]):raise ValueError('Source inference smoke failed')
 core.write(out/'prediction_smoke.json',dict(status='PASS',records=len(ids),active=int(active.sum()),source_only=True,F_frozen=True,DURA_inference=False,TX_labels_read=0,test_IQ_reads=0,source_selection_recomputed=True))


@torch.no_grad()
def predict(root,out,commit,device):
 if core.read(out/'prediction_smoke.json').get('status')!='PASS':raise ValueError('Real source inference smoke required')
 model,phase,gate,alpha,threshold=load_fixed(root,out,commit,device);teacher,_=core.teacher(root,device);folder,ids=core.reference(root)
 dura=root/DURA_PREDICTIONS;done=core.read(dura/'complete.json');expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=len(ids),views=list(core.VIEWS),TX_truth_reads=0,state_updates=0,head_fit_updates=0)
 if any(done.get(k)!=v for k,v in expected.items()) or not np.array_equal(ids,np.load(dura/'physical_ids.npy',allow_pickle=False)):raise ValueError('Complete matched DURA comparator required')
 dst=out/'flow'/'test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={}
 for view in core.VIEWS:
  values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
  if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 IQ view required')
  raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));count=0
  for start in range(0,len(ids),64):
   x=core.tensor(values[start:start+64],device);y,enabled=apply(model,phase,gate,alpha,threshold,x);a=teacher.classify_features(teacher.features(x));b=teacher.classify_features(teacher.features(y))
   if not torch.isfinite(y).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite predictions required')
   raw[start:start+len(x)]=np.from_dlpack(a.argmax(1).cpu().contiguous());restored[start:start+len(x)]=np.from_dlpack(b.argmax(1).cpu().contiguous());count+=int(enabled.sum())
  raw.flush();restored.flush();accepted[view]=count;print('SELF_REFERENCE_PHASE1_VIEW_COMPLETE',view,len(ids),'accepted',count,flush=True)
 core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted))


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
 if a.mode=='smoke':smoke(a.root,a.out,a.commit,a.device)
 elif a.mode=='predict':predict(a.root,a.out,a.commit,a.device)
 else:core.score(a.root,a.out,a.commit)
if __name__=='__main__':main()
