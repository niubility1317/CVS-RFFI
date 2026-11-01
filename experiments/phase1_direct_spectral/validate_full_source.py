"""Predeclared fixed direct-network source validation; never reads Phase1 test."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from direct_spectral_model import DirectSpectralRestorer
from train_full import s,core,source_prior
SEL=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261102-phase1-bayes-residual-source-validation-s2026092701-r01-release')
if (SEL/'release_commit.txt').read_text().strip()!='86acfb414d70c638ef086e318739cc158a9b6601':raise ValueError('Exact selection rules release required')
sys.path.insert(0,str(SEL/'experiments/phase1_bayes_residual_validation'))
from residual_selection import choose,VIEWS
RUN='20261102-phase1-direct-spectral-full-s2026092701-r01'
TRAIN_COMMIT='79053b41ca9d349b267c8c06b7b4655de8dc7769'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable output required')
    a.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    record=core.read(a.root/'runs'/RUN/'result.json');checkpoint=a.root/'runs'/RUN/'checkpoint_epoch03.pt'
    if record.get('status')!='COMPLETE_SOURCE_ONLY_DIRECT_SPECTRAL_FULL' or record.get('commit')!=TRAIN_COMMIT or record.get('fit_ids',0)<40000 or record.get('updates',0)<8000 or not record.get('F_frozen') or record.get('blind_parent_loaded') is not False or any(record.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Completed source-only full direct training required')
    ckhash=hashlib.sha256(checkpoint.read_bytes()).hexdigest();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('commit')!=TRAIN_COMMIT or saved.get('updates')!=record['updates']:raise ValueError('Exact fixed epoch3 checkpoint required')
    cache,clean,rx,training,pv,ph=s.source_contract(a.root);validation,heldout=s.new_splits(cache,rx,training,pv,ph);received,scene=s.epoch_data(cache,19);teacher,_=core.teacher(a.root,a.device);teacher.eval().requires_grad_(False);_,gate=s.initial_models(a.root,a.device);gate.eval().requires_grad_(False);net=DirectSpectralRestorer().to(a.device);net.load_state_dict(saved['model'],strict=True);net.eval().requires_grad_(False)
    if a.smoke:
        idx=training[:32];x=core.tensor(received[idx],a.device);v=net(x);prob=gate(x).sigmoid();logits=teacher.classify_features(teacher.features(v))
        if not all(torch.isfinite(z).all() for z in (v,prob,logits)):raise ValueError('Finite source output required')
        with (a.out/'result.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',records=len(idx),active=int((prob>.8).sum().item()),checkpoint_sha256=ckhash,source_only=True,test_reads=0),f,indent=2)
        print('DIRECT_VALIDATION_SMOKE PASS',len(idx),flush=True);return
    dura=source_prior.frozen_dura(a.root,a.device);ids=np.r_[validation,heldout];split=np.r_[np.ones(len(validation),dtype=np.int64),np.full(len(heldout),2,dtype=np.int64)];out=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),32):
            idx=ids[start:start+32];x=core.tensor(received[idx],a.device);y=core.tensor(clean[idx],a.device);conf,target=F.softmax(teacher.classify_features(teacher.features(y)),1).max(1);raw=teacher.classify_features(teacher.features(x)).argmax(1);d=dura(x);base=teacher.classify_features(teacher.features(d)).argmax(1);prob=gate(x).sigmoid();candidate=net(x);pred=teacher.classify_features(teacher.features(candidate)).argmax(1);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);_,de=s.align(d,y);_,re=s.align(x,y);_,ce=s.align(candidate,y);out.append(torch.stack((target.float(),conf,base.float(),prob,de/den,raw.float(),re/den,pred.float(),ce/den),1).cpu())
            if (start+len(idx))%1024==0:print('DIRECT_VALIDATION_PROGRESS',start+len(idx),'seconds',round(time.monotonic()-tick,1),flush=True)
    values=np.from_dlpack(torch.cat(out).contiguous());result={}
    for threshold in (.8,.99):
        cells={}
        for part,receivers in ((1,(4,6)),(2,(8,))):
            for receiver in receivers:
                for view in VIEWS:
                    mask=(split==part)&(rx[ids]==receiver)&(scene[ids]==view);v=values[mask]
                    if not len(v):raise ValueError('All source receiver/scene cells required')
                    trusted=v[:,1]>=.95;active=v[:,3]>threshold;pred=np.where(active,v[:,7],v[:,5]);iq=np.where(active,v[:,8],v[:,6]);base=int(np.count_nonzero(trusted&(v[:,2]==v[:,0])));base_iq=float(np.mean(v[:,4]))
                    cells[f'{part}:{receiver}:{view}']=dict(count=len(v),valid_count=int(np.count_nonzero(trusted)),active=int(np.count_nonzero(active)),baseline_correct=base,parent_correct=base,candidate_correct=int(np.count_nonzero(trusted&(pred==v[:,0]))),baseline_iq=base_iq,parent_iq=base_iq,candidate_iq=float(np.mean(iq)))
        result[str(threshold)]=cells
    selection={'primary_dura':choose(result,True,'baseline'),'accuracy_dura':choose(result,False,'baseline')}
    summary=dict(status='SOURCE_VALIDATION_COMPLETE',commit=a.commit,checkpoint_sha256=ckhash,validation_ids=len(validation),heldout_ids=len(heldout),validation=result,selection=selection,source_only=True,F_frozen=True,DURA_inference=False,TX_labels_read=0,U_s_truth_reads=0,V_reads=0,test_reads=0,caveat='Source IDs observed in other views, pseudo targets not truth; no automatic target prediction.')
    with (a.out/'source_validation.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    print('DIRECT_VALIDATION_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()
