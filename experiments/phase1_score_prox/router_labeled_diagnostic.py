"""Truth-last legal L_s source audit of fixed confidence-selected restoration."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
DEP=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-confidence-router-source-validation-s2026092701-r01-release')
if (DEP/'release_commit.txt').read_text().strip()!='66f37276d0b171cb1d6cb396c199ee2c7b296f06':raise ValueError('Exact source screening release required')
sys.path.insert(0,str(DEP/'experiments/phase1_score_prox'))
from router_source_validation import s,core,source_prior,GroundDenoiser,BlindActionNet,action_matrix,posterior,SOURCE,SHA,PRIOR_RUN,PRIOR_SHA,NET_SHA,reconstruct_noise_adaptive,select_confidence,VIEWS
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source
PRIOR='20261102-phase1-score-prox-pilot-s2026092701-r01';SELECT='20261102-phase1-confidence-router-source-validation-s2026092701-r01';SELECT_SHA='2d3cac93b1e177258003891ff203d9b392f57b023cef8fcc387166ae0e017a24'

def predict(root,out,commit,device,smoke):
    if out.exists():raise FileExistsError('Immutable output required')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selpath=root/'runs'/SELECT/'source_validation.json'
    if hashlib.sha256(selpath.read_bytes()).hexdigest()!=SELECT_SHA:raise ValueError('Fixed predeclared source selection required')
    select=core.read(selpath)['selection']['0.05']['primary_parent']
    if select!=dict(kind='confidence_router',threshold=.8,validation_net=19,reference='parent'):raise ValueError('Exact selected candidate required')
    ckpt=root/'runs'/PRIOR/'checkpoint_epoch03.pt';blind_ckpt=root/'runs'/SOURCE/'checkpoint_epoch10.pt';prior_file=root/'runs'/PRIOR_RUN/'source_prior.pt'
    for path,expected in ((ckpt,NET_SHA),(blind_ckpt,SHA),(prior_file,PRIOR_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Exact fixed source checkpoint required')
    netstate=torch.load(ckpt,map_location='cpu',weights_only=False);blindstate=torch.load(blind_ckpt,map_location='cpu',weights_only=False);prior=torch.load(prior_file,map_location='cpu',weights_only=False);net=GroundDenoiser().to(device);net.load_state_dict(netstate['model']);net.eval().requires_grad_(False);model=BlindActionNet().to(device);model.load_state_dict(blindstate['model']);model.eval().requires_grad_(False);means=prior['means'].to(device);covs=prior['covariances'].to(device);priors=prior['priors'].to(device);eye=torch.eye(512,device=device)
    cache,clean,rx,training,pv,ph=s.source_contract(root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);received,scene=s.epoch_data(cache,19)
    if smoke:ids=training[:32];parts=np.zeros(len(ids),dtype=np.int64)
    else:
        val=validation[roles[validation]=='L_s'];hold=heldout[roles[heldout]=='L_s'];ids=np.r_[val,hold];parts=np.r_[np.ones(len(val),dtype=np.int64),np.full(len(hold),2,dtype=np.int64)]
        if not len(val) or not len(hold) or np.any(roles[ids]!='L_s') or np.intersect1d(ids,training).size:raise ValueError('Source L_s receiver holdout required')
    _,gate=s.initial_models(root,device);gate.eval().requires_grad_(False);teacher,_=core.teacher(root,device);teacher.eval().requires_grad_(False);dura=source_prior.frozen_dura(root,device);chunks=[];triggers=[];choices=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),8):
            idx=ids[start:start+8];x=core.tensor(received[idx],device);ground=core.tensor(clean[idx],device);active=gate(x).sigmoid()>.8;parent=x.clone();routed=x.clone();choice=torch.zeros(len(x),dtype=torch.long,device=device)
            if active.any():
                v=x[active];h,lv=model(v);A=action_matrix(h);var=lv.exp();states,w=posterior(v.reshape(len(v),512),A,eye[None]*var[:,None,None],means,covs,priors);z=(states*w.exp().unsqueeze(-1)).sum(1).reshape_as(v);candidates=[z]+[reconstruct_noise_adaptive(net,A,v,z,var,k) for k in (.25,1.,4.)];confidence=torch.stack([F.softmax(teacher.classify_features(teacher.features(c)),1).max(1).values for c in candidates],1);picked=select_confidence(confidence,.05);variants=torch.stack(candidates,1);parent[active]=z;routed[active]=variants[torch.arange(len(v),device=device),picked];choice[active]=picked
            row=torch.stack([teacher.classify_features(teacher.features(t)).argmax(1) for t in (ground,x,dura(x),parent,routed)],1);chunks.append(row.cpu());triggers.append(active.cpu());choices.append(choice.cpu())
            if (start+len(idx))%256==0:print('ROUTER_LS_PREDICT',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    predictions=np.from_dlpack(torch.cat(chunks).contiguous());active=np.from_dlpack(torch.cat(triggers).to(torch.uint8).contiguous());choice=np.from_dlpack(torch.cat(choices).contiguous());path=out/'predictions.npz'
    with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=physical[ids],rx=rx[ids],scene=scene[ids],part=parts,predictions=predictions,active=active,choice=choice)
    manifest=dict(status='SOURCE_PREDICTIONS_FIXED',commit=commit,count=len(ids),source_only=True,role='smoke' if smoke else 'L_s',gate=.8,confidence_margin=.05,source_selection_sha256=SELECT_SHA,model_sha256=NET_SHA,predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
    with (out/'prediction_complete.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print('ROUTER_LS_PREDICTIONS_FIXED',json.dumps(manifest),flush=True)

def score(root,out,commit,device):
    done=core.read(out/'prediction_complete.json');path=out/'predictions.npz'
    if done.get('status')!='SOURCE_PREDICTIONS_FIXED' or done.get('role')!='L_s' or done.get('commit')!=commit or done.get('TX_truth_reads')!=0 or hashlib.sha256(path.read_bytes()).hexdigest()!=done.get('predictions_sha256'):raise ValueError('Fixed source predictions required before source truth')
    with np.load(path,allow_pickle=False) as x:data={name:x[name] for name in x.files}
    cache,clean,rx,training,pv,ph=s.source_contract(root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);idx=data['indices']
    if len(idx)!=done['count'] or not np.array_equal(data['physical_ids'],physical[idx]) or np.any(roles[idx]!='L_s') or np.intersect1d(idx,training).size or not np.array_equal(data['rx'],rx[idx]) or not np.array_equal(idx,np.r_[validation[roles[validation]=='L_s'],heldout[roles[heldout]=='L_s']]):raise ValueError('Exact complete L_s cohorts required')
    teacher,contract=core.teacher(root,device);truth=labels_source.source_labels(root,cache,clean,roles,contract)[idx]
    if np.any(truth<0) or np.any(truth>=6):raise ValueError('Valid L_s truth required')
    result={}
    for part,receivers in ((1,(4,6)),(2,(8,))):
        for view in VIEWS:
            mask=(data['part']==part)&np.isin(data['rx'],receivers)&(data['scene']==view)
            if not np.count_nonzero(mask):raise ValueError('All source cells required')
            pred=data['predictions'];correct=pred==truth[:,None];cell=dict(count=int(np.count_nonzero(mask)),active=int(np.count_nonzero(mask&(data['active']>0))),routed=int(np.count_nonzero(mask&(data['choice']>0))))
            for i,name in enumerate(('clean','raw','dura','parent','candidate')):
                c=correct[:,i];cell[name]=dict(correct=int(np.count_nonzero(mask&c)),repairs_vs_dura=int(np.count_nonzero(mask&c&~correct[:,2])),damages_vs_dura=int(np.count_nonzero(mask&~c&correct[:,2])),repairs_vs_parent=int(np.count_nonzero(mask&c&~correct[:,3])),damages_vs_parent=int(np.count_nonzero(mask&~c&correct[:,3])))
            result[f'{part}:{view}']=cell
    result=dict(status='SOURCE_LS_TRUTH_LAST_COMPLETE',commit=commit,predictions_sha256=done['predictions_sha256'],source_counts=result,source_only=True,gate=.8,confidence_margin=.05,truth_role='L_s',U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Frozen F trained on source IDs; these source IDs and views have been explored; no independent blind or target test.')
    with (out/'source_truth_score.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('ROUTER_LS_TRUTH_LAST_COMPLETE',json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='predict':predict(a.root,a.out,a.commit,a.device,a.smoke)
    else:
        if a.smoke:raise ValueError('No truth access in smoke')
        score(a.root,a.out,a.commit,a.device)
if __name__=='__main__':main()
