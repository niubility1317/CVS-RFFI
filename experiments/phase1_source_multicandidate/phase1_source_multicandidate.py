"""Sealed Phase1 source-only paired predictions for fixed IQ experts."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

GATE_RUN='20261102-phase1-blind-gate-s2026092701-r01';GATE_COMMIT='c905f3ab3d7f0d121e2c44c3b0bfb26188fa5825'
CURR_RUN='20261102-phase1-curriculum-blind-gate-s2026092701-r01';CURR_COMMIT='3a7f72e591a90bc3f4cc14f031bf76e7fbe8b2db'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def metrics(predictions,target,mask):
    correct={k:(v==target) for k,v in predictions.items()};base=correct['dura'];union=np.logical_or.reduce([correct[k] for k in ('dura','semantic','curriculum')]);out={'count':int(mask.sum()),'oracle_any_correct':int((mask&union).sum())}
    for name,success in correct.items():out[name]=dict(correct=int((mask&success).sum()),repairs_vs_dura=int((mask&~base&success).sum()),damages_vs_dura=int((mask&base&~success).sum()))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable output exists')
    release=Path(__file__).resolve().parents[2]
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Exact diagnostic release required')
    gate_release=args.root/'releases'/(GATE_RUN+'-release')
    if (gate_release/'release_commit.txt').read_text(encoding='ascii').strip()!=GATE_COMMIT:raise ValueError('Exact Phase1 gate release required')
    sys.path.insert(0,str(gate_release/'experiments/phase1_blind_gate'))
    from phase1_blind_gate_source import source_contract,epoch_data,new_splits,VIEWS as SOURCE_VIEWS
    from phase1_blind_gate_predict import load_fixed,apply
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow import anchor_source_smoke as prior
    from experiments.pamf_phase1_flow import feature_conditioned_experiment as reader
    if VIEWS!=SOURCE_VIEWS:raise ValueError('Exact source scenes required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache,clean,rx,train,prior_v,prior_h=source_contract(args.root);validation,ids=new_splits(cache,rx,train,prior_v,prior_h)
    if len(ids)!=3921 or np.intersect1d(ids,np.r_[train,validation]).size:raise ValueError('Exact disjoint student-heldout identities required')
    received,scene=epoch_data(cache,18);roles=np.load(cache/'train/roles.npy',allow_pickle=False);physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    sem,gate_a,alpha_a,threshold_a=load_fixed(args.root/'runs'/GATE_RUN/'source',GATE_COMMIT,args.device)
    curr,gate_b,alpha_b,threshold_b=load_fixed(args.root/'runs'/CURR_RUN/'source',CURR_COMMIT,args.device)
    dura=prior.frozen_dura(args.root,args.device);teacher,contract=core.teacher(args.root,args.device)
    with torch.no_grad():
        smoke=core.tensor(received[ids[:8]],args.device)
        for wave in (smoke,dura(smoke),apply(sem,gate_a,alpha_a,threshold_a,smoke)[0],apply(curr,gate_b,alpha_b,threshold_b,smoke)[0]):
            logits=teacher.classify_features(teacher.features(wave))
            if logits.shape!=(len(smoke),6) or not torch.isfinite(wave).all() or not torch.isfinite(logits).all():raise ValueError('Fixed source inference smoke failed')
    args.out.mkdir(parents=True);buffers={k:[] for k in ('clean_prob','raw','dura','semantic','curriculum')}
    with torch.no_grad():
        for start in range(0,len(ids),32):
            index=ids[start:start+32];x=core.tensor(received[index],args.device);y=core.tensor(clean[index],args.device)
            clean_prob=F.softmax(teacher.classify_features(teacher.features(y)),1);buffers['clean_prob'].append(np.from_dlpack(clean_prob.cpu().contiguous()))
            for key,wave in (('raw',x),('dura',dura(x)),('semantic',apply(sem,gate_a,alpha_a,threshold_a,x)[0]),('curriculum',apply(curr,gate_b,alpha_b,threshold_b,x)[0])):
                logits=teacher.classify_features(teacher.features(wave))
                if not torch.isfinite(logits).all():raise ValueError('Finite source predictions required')
                buffers[key].append(np.from_dlpack(logits.argmax(1).cpu().contiguous()))
    fixed={k:np.concatenate(v) for k,v in buffers.items()}
    if any(len(v)!=len(ids) for v in fixed.values()):raise ValueError('Incomplete source predictions')
    with (args.out/'predictions.npz').open('xb') as f:np.savez_compressed(f,physical_ids=physical[ids],scene=scene[ids],**fixed)
    core.write(args.out/'predictions_manifest.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',count=len(ids),commit=args.commit,source_only=True,TX_labels_read=0,U_s_labels_read=0,V_reads=0,test_reads=0,model_updates=0,source_cache_epoch=18))
    print('SOURCE_MULTICANDIDATE_PREDICTIONS_FIXED',len(ids),flush=True)
    # Only after predictions are sealed, read legitimate L_s source labels.
    labels=reader.source_labels(args.root,cache,clean,roles,contract);labeled=roles[ids]=='L_s'
    if np.any(labels[ids[~labeled]]!=-1):raise ValueError('U_s true labels forbidden')
    prob=fixed['clean_prob'];target=np.where(labeled,labels[ids],prob.argmax(1));valid=labeled|(prob.max(1)>=.95);pred={k:fixed[k] for k in ('raw','dura','semantic','curriculum')};summary={}
    for view in VIEWS:
        mask=(scene[ids]==view)&valid;summary[view]=dict(all_valid=metrics(pred,target,mask),L_s_only=metrics(pred,target,mask&labeled))
    core.write(args.out/'result.json',dict(status='SOURCE_MULTICANDIDATE_COMPLETE',count=len(ids),valid_count=int(valid.sum()),L_s_count=int(labeled.sum()),views=summary,predictions_before_source_labels=True,U_s_truth_reads=0,V_reads=0,test_reads=0,source_cache_epoch=18,source_caveat='physical IDs and other channel views previously inspected; no independent blind claim'))
    print('SOURCE_MULTICANDIDATE_COMPLETE',flush=True)
if __name__=='__main__':main()
