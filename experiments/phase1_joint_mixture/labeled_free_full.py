"""Truth-last legal L_s source audit of fixed full-source hard mixture."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from validate_free_full import s,core,source_prior,BlindActionNet,physical,SOURCE,PRIOR_RUN,MixtureController,VIEWS
from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_source

def predict(root,out,commit,device,smoke):
 if out.exists():raise FileExistsError('Immutable output required')
 out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 selection=core.read(root/'runs/20261102-phase1-free-full-hard-validation-s2026092701-r01/source_validation.json')['selection']['0.5']['primary_parent']
 if selection!=dict(kind='residual',threshold=.8,validation_net=31,reference='parent'):raise ValueError('Fixed source selected candidate required')
 bs=torch.load(root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False);state=torch.load(root/'runs/20261102-phase1-free-mixture-full-hard-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)
 if bs.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or bs.get('epoch')!=10 or p.get('source_ids')!=8192 or state.get('commit')!='9d734755b3e443fac7cf278750958b0e741178ee':raise ValueError('Fixed source models required')
 model=BlindActionNet().to(device);model.load_state_dict(bs['model']);model.eval().requires_grad_(False);prior=(p['means'].to(device),p['covariances'].to(device),p['priors'].to(device));net=MixtureController().to(device);net.load_state_dict(state['model']);net.eval().requires_grad_(False);eye=torch.eye(512,device=device)
 cache,clean,rx,training,pv,ph=s.source_contract(root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical_ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);received,scene=s.epoch_data(cache,19)
 if smoke:ids=training[:32];parts=np.zeros(len(ids),dtype=np.int64)
 else:
  val=validation[roles[validation]=='L_s'];hold=heldout[roles[heldout]=='L_s'];ids=np.r_[val,hold];parts=np.r_[np.ones(len(val),dtype=np.int64),np.full(len(hold),2,dtype=np.int64)]
  if not len(val) or not len(hold) or np.any(roles[ids]!='L_s') or np.intersect1d(ids,training).size:raise ValueError('Source L_s receiver holdout required')
 _,gate=s.initial_models(root,device);gate.eval().requires_grad_(False);teacher,_=core.teacher(root,device);teacher.eval().requires_grad_(False);dura=source_prior.frozen_dura(root,device);chunks=[];triggers=[];choices=[];tick=time.monotonic()
 with torch.no_grad():
  for start in range(0,len(ids),8):
   idx=ids[start:start+8];x=core.tensor(received[idx],device);ground=core.tensor(clean[idx],device);active=gate(x).sigmoid()>.8;parent=x.clone();candidate=x.clone();choice=torch.zeros(len(x),dtype=torch.long,device=device)
   if active.any():
    st,lp,v,z=physical(model,teacher,x[active],prior,eye);parent[active]=z.reshape(-1,2,256);candidate[active]=net(v,lp,st,scale=.5)[0];choice[active]=1
   row=torch.stack([teacher.classify_features(teacher.features(t)).argmax(1) for t in (ground,x,dura(x),parent,candidate)],1);chunks.append(row.cpu());triggers.append(active.cpu());choices.append(choice.cpu())
   if (start+len(idx))%256==0:print('FREE_FULL_HARD_LS_PREDICT',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
 predictions=np.from_dlpack(torch.cat(chunks).contiguous());active=np.from_dlpack(torch.cat(triggers).to(torch.uint8).contiguous());choice=np.from_dlpack(torch.cat(choices).contiguous());path=out/'predictions.npz'
 with path.open('xb') as f:np.savez(f,indices=ids,physical_ids=physical_ids[ids],rx=rx[ids],scene=scene[ids],part=parts,predictions=predictions,active=active,choice=choice)
 manifest=dict(status='SOURCE_PREDICTIONS_FIXED',commit=commit,count=len(ids),source_only=True,role='smoke' if smoke else 'L_s',gate=.8,predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0)
 with (out/'prediction_complete.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2);f.write('\n')
 print('FREE_FULL_HARD_LS_PREDICTIONS_FIXED',json.dumps(manifest),flush=True)

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
    result=dict(status='SOURCE_LS_TRUTH_LAST_COMPLETE',commit=commit,predictions_sha256=done['predictions_sha256'],source_counts=result,source_only=True,gate=.8,truth_role='L_s',U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Frozen F trained on source IDs; these source IDs and views have been explored; no independent blind or target test.')
    with (out/'source_truth_score.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('FREE_FULL_HARD_LS_TRUTH_LAST_COMPLETE',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.mode=='predict':predict(a.root,a.out,a.commit,a.device,a.smoke)
 else:
  if a.smoke:raise ValueError('No truth access in smoke')
  score(a.root,a.out,a.commit,a.device)
if __name__=='__main__':main()
