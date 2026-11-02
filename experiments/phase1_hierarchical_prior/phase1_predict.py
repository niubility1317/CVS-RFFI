"""Fixed Phase1 ground-test predictions and truth-last score for hierarchical prior."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from source_pilot import s,core,BlindActionNet,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,pair
RUN='20261102-phase1-hierarchical-prior-pilot-s2026092701-r01';SUB_SHA='ce2c44b4336245e32f85ed510b7f640f586e6c8d277fe3aff6438fe92547c848'
SELECT='20261102-phase1-hierarchical-prior-validation-s2026092701-r01';SELECT_SHA='74e329da45c5edd29d18deda181d442bdad9e44b41f8326584d34bbfe5fc13e4'
LS='20261102-phase1-hierarchical-prior-labeled-source-s2026092701-r01';LS_PRED_SHA='97c89fe0bf1805af23394743b9bef7297e4f25c145f2408fc490018023266cd5';LS_AUDIT_SHA='208d261b8eaf0770f3c0fe5c1ba3be45eed007c6f40a5a357f8f49971b5c5810';LS_SCORE_SHA='e44b06e73aaa2ac559acca24c00107ce644fbd3cd422b4da65ab9264aba07c76'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load_fixed(root,device):
    select_path=root/'runs'/SELECT/'source_validation.json';lspath=root/'runs'/LS/'predictions.npz';audit=root/'runs'/LS/'independent_audit.json';source_score=root/'runs'/LS/'source_truth_score.json'
    for path,expected in ((select_path,SELECT_SHA),(lspath,LS_PRED_SHA),(audit,LS_AUDIT_SHA),(source_score,LS_SCORE_SHA)):
        if digest(path)!=expected:raise ValueError('Exact selected source and labeled audit evidence required')
    selected=core.read(select_path)['selection']['hierarchical']['primary_parent']
    if selected!=dict(kind='residual',threshold=.8,validation_net=33,reference='parent') or core.read(audit)['status']!='PASS':raise ValueError('Fixed source-selected candidate required')
    subfile=root/'runs'/RUN/'source_hierarchical_prior.pt';blind_ckpt=root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=root/'runs'/PRIOR_RUN/'source_prior.pt'
    for path,expected in ((subfile,SUB_SHA),(blind_ckpt,SHA),(prior_file,PRIOR_SHA)):
        if digest(path)!=expected:raise ValueError('Exact fixed source checkpoint required')
    substate=torch.load(subfile,map_location='cpu',weights_only=False);blindstate=torch.load(blind_ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);blind=BlindActionNet().to(device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);old=(prior['means'].to(device),prior['covariances'].to(device),prior['priors'].to(device));sub=(substate['means'].to(device),substate['covariances'].to(device),substate['priors'].to(device));_,gate=s.initial_models(root,device);gate.eval().requires_grad_(False);teacher,_=core.teacher(root,device);teacher.eval().requires_grad_(False)
    return blind,gate,teacher,old,sub,torch.eye(512,device=device)

@torch.no_grad()
def predict_batch(x,models):
    blind,gate,teacher,old,sub,eye=models;raw_logits=teacher.classify_features(teacher.features(x));raw=raw_logits.argmax(1);active=gate(x).sigmoid()>.8;parent=raw.clone();routed=raw.clone();choice=torch.zeros(len(x),device=x.device,dtype=torch.long)
    if active.any():
        v=x[active];z,candidate=pair(v,blind,old,sub,eye);pl=teacher.classify_features(teacher.features(z));cl=teacher.classify_features(teacher.features(candidate));parent[active]=pl.argmax(1);routed[active]=cl.argmax(1);choice[active]=1
        if not all(torch.isfinite(t).all() for t in (z,candidate,pl,cl)):raise ValueError('Finite fixed output required')
    if not torch.isfinite(raw_logits).all():raise ValueError('Finite input logits required')
    return raw,parent,routed,active,choice

def smoke(root,out,commit,device):
    if out.exists():raise FileExistsError('Immutable experiment output required')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;models=load_fixed(root,device);reference=root/'runs'/LS/'predictions.npz'
    with np.load(reference,allow_pickle=False) as p:data={k:p[k] for k in p.files}
    cache,clean,rx,training,pv,ph=s.source_contract(root);received,scene=s.epoch_data(cache,19);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);ids=data['indices']
    if len(ids)!=1047 or not np.array_equal(physical[ids],data['physical_ids']):raise ValueError('Exact fixed source audit cohort required')
    tested=[];chosen=None
    for batchsize in (32,8):
        mismatches=0;start_time=time.monotonic()
        for start in range(0,len(ids),batchsize):
            ix=ids[start:start+batchsize];raw,parent,routed,active,choice=predict_batch(core.tensor(received[ix],device),models);actual=np.from_dlpack(torch.stack((raw,parent,routed,active.long(),choice),1).cpu().contiguous());expected=np.column_stack((data['predictions'][start:start+len(ix),1],data['predictions'][start:start+len(ix),3],data['predictions'][start:start+len(ix),4],data['active'][start:start+len(ix)],data['choice'][start:start+len(ix)]));mismatches+=int(np.count_nonzero(np.any(actual!=expected,axis=1)))
        tested.append(dict(batch=batchsize,records=len(ids),record_mismatches=mismatches,seconds=round(time.monotonic()-start_time,3)))
        if mismatches==0:chosen=batchsize;break
    if chosen is None:raise ValueError('Fixed source audit predictions or routing not reproduced')
    core.write(out/'prediction_smoke.json',dict(status='PASS',source_only=True,records=len(ids),source_prediction_sha256=LS_PRED_SHA,tests=tested,batch=chosen,F_frozen=True,DURA_inference=False,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_IQ_reads=0,commit=commit));print('HIERARCHICAL_PHASE1_SMOKE PASS',json.dumps(tested),flush=True)

@torch.no_grad()
def predict(root,out,commit,device):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;smoke=core.read(out/'prediction_smoke.json')
    if smoke.get('status')!='PASS' or smoke.get('commit')!=commit or smoke.get('source_prediction_sha256')!=LS_PRED_SHA or smoke.get('batch') not in (8,32) or smoke.get('test_IQ_reads')!=0:raise ValueError('Verified source-only prediction smoke required')
    models=load_fixed(root,device);folder,ids=core.reference(root);dst=out/'flow'/'test_prediction';dst.mkdir(parents=True,exist_ok=False);np.save(dst/'physical_ids.npy',ids,allow_pickle=False);accepted={};selected={};hashes={'physical_ids.npy':digest(dst/'physical_ids.npy')};batch=smoke['batch'];started=time.monotonic()
    for view in core.VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32:raise ValueError('Complete Phase1 ground test IQ view required')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),));count=0;changed=0
        for start in range(0,len(ids),batch):
            x=core.tensor(values[start:start+batch],device);a,_,b,on,choice=predict_batch(x,models);raw[start:start+len(x)]=np.from_dlpack(a.cpu().contiguous());restored[start:start+len(x)]=np.from_dlpack(b.cpu().contiguous());count+=int(on.sum().item());changed+=int((choice>0).sum().item())
            if (start+len(x))%16384==0:print('HIERARCHICAL_PHASE1_PROGRESS',view,start+len(x),'seconds',round(time.monotonic()-started,1),flush=True)
        raw.flush();restored.flush();del raw,restored;accepted[view]=count;selected[view]=changed
        for arm in ('raw','restored'):
            name=view+'.'+arm+'.npy';hashes[name]=digest(dst/name)
        print('HIERARCHICAL_PHASE1_VIEW_COMPLETE',view,len(ids),'active',count,'selected',changed,'seconds',round(time.monotonic()-started,1),flush=True)
    core.write(out/'flow'/'prediction_hashes.json',dict(status='FIXED_PREDICTION_HASHES',hashes=hashes,selected_by_view=selected,batch=batch,source_selection_sha256=SELECT_SHA,source_truth_reads_during_prediction=0,TX_truth_reads=0))
    core.write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(core.VIEWS),F_run=core.F_RUN,F_row=core.F_ROW,commit=commit,TX_truth_reads=0,model_updates=0,accepted_by_view=accepted))
    print('HIERARCHICAL_PHASE1_PREDICTIONS_COMPLETE',len(ids),flush=True)

def score(root,out,commit):
    dst=out/'flow'/'test_prediction';checks=core.read(out/'flow'/'prediction_hashes.json');done=core.read(dst/'complete.json');core.validate_prediction_manifest(done,count=168000,commit=commit)
    if checks.get('status')!='FIXED_PREDICTION_HASHES' or checks.get('TX_truth_reads')!=0 or checks.get('source_selection_sha256')!=SELECT_SHA:raise ValueError('Fixed hashed predictions required')
    expected={'physical_ids.npy'}|{view+'.'+arm+'.npy' for view in core.VIEWS for arm in ('raw','restored')}
    if set(checks['hashes'])!=expected or any(digest(dst/name)!=h for name,h in checks['hashes'].items()):raise ValueError('Full fixed prediction hashes required before truth')
    core.score(root,out,commit);print('HIERARCHICAL_PHASE1_TRUTH_SCORE_COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.mode=='smoke':smoke(a.root,a.out,a.commit,a.device)
    elif a.mode=='predict':predict(a.root,a.out,a.commit,a.device)
    else:score(a.root,a.out,a.commit)
if __name__=='__main__':main()
