"""Post-freeze source heldout IQ and classification comparison; no target reads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT_RUN='20261102-phase1-direct-axial-iq-s2026092701-r01'
CODE_COMMIT='bc9ac03f23271614f332fd06bd86e77a0cd2cca5'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def paired_counts(base, candidate, target, valid):
    before=base==target;after=candidate==target
    return dict(count=int(valid.sum()),baseline=int((valid & before).sum()),candidate=int((valid & after).sum()),repairs=int((valid & ~before & after).sum()),damages=int((valid & before & ~after).sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable diagnostic output exists')
    release=Path(__file__).resolve().parents[2]
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Exact diagnostic release required')
    original=args.root/'releases'/(ROOT_RUN+'-release')
    if (original/'release_commit.txt').read_text(encoding='ascii').strip()!=CODE_COMMIT:raise ValueError('Exact frozen direct model release required')
    sys.path.insert(0,str(original/'experiments/phase1_axial_spectral'))
    from phase1_axial_spectral_source import source_contract,epoch_data,align
    from phase1_axial_spectral_predict import load_fixed
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow import anchor_source_smoke as prior
    from experiments.pamf_phase1_flow import feature_conditioned_experiment as labels_reader
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache,clean,rx,training,validation,ids=source_contract(args.root)
    if len(ids)!=279 or (rx[ids]!=8).any():raise ValueError('Exact receiver 8 heldout set required')
    if np.intersect1d(ids,training).size or np.intersect1d(ids,validation).size:raise ValueError('Source split overlap')
    received,scene=epoch_data(cache,20);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    model,alpha=load_fixed(args.root/'runs'/ROOT_RUN/'source',CODE_COMMIT,'cpu')
    if abs(alpha-.25)>1e-12:raise ValueError('Exact source-selected alpha required')
    dura=prior.frozen_dura(args.root,'cpu');teacher,contract=core.teacher(args.root,'cpu')
    # The first real source batch checks all three frozen paths before full prediction.
    with torch.no_grad():
        x=core.tensor(received[ids[:8]],'cpu');a=dura(x);z=x+alpha*(model(x)-x)
        for wave in (x,a,z):
            logits=teacher.classify_features(teacher.features(wave))
            if wave.shape!=x.shape or logits.shape!=(len(x),6) or not torch.isfinite(wave).all() or not torch.isfinite(logits).all():raise ValueError('Source path smoke failed')
    args.out.mkdir(parents=True);results={k:[] for k in ('clean_prob','raw','dura','direct','raw_iq','dura_iq','direct_iq')}
    with torch.no_grad():
        for start in range(0,len(ids),16):
            ix=ids[start:start+16];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');a=dura(x);z=x+alpha*(model(x)-x)
            p=F.softmax(teacher.classify_features(teacher.features(y)),1)
            results['clean_prob'].append(np.from_dlpack(p.contiguous()))
            den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12)
            for name,wave in (('raw',x),('dura',a),('direct',z)):
                logits=teacher.classify_features(teacher.features(wave));_,error=align(wave,y)
                if not torch.isfinite(logits).all() or not torch.isfinite(error).all():raise ValueError('Finite source predictions required')
                results[name].append(np.from_dlpack(logits.argmax(1).contiguous()))
                results[name+'_iq'].append(np.from_dlpack((error/den).contiguous()))
    fixed={k:np.concatenate(v) for k,v in results.items()}
    if any(len(v)!=len(ids) for v in fixed.values()):raise ValueError('All source predictions must be complete')
    with (args.out/'predictions.npz').open('xb') as f:np.savez_compressed(f,physical_ids=physical[ids],scene=scene[ids],**fixed)
    core.write(args.out/'predictions_manifest.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',count=len(ids),commit=args.commit,model_commit=CODE_COMMIT,source_only=True,source_labels_read=0,U_s_labels_read=0,V_reads=0,test_reads=0,model_updates=0))
    print('SOURCE_PREDICTIONS_FIXED',len(ids),flush=True)
    # Read only legitimate L_s labels after all predictions have been fixed.
    truth=labels_reader.source_labels(args.root,cache,clean,roles,contract)
    labeled=roles[ids]=='L_s'
    if (truth[ids[~labeled]]!=-1).any():raise ValueError('U_s truth must not be materialized')
    prob=fixed['clean_prob'];target=np.where(labeled,truth[ids],prob.argmax(1));valid=labeled | (prob.max(1)>=.95)
    summary={}
    for view in VIEWS:
        group=scene[ids]==view;mask=group&valid;lab=mask&labeled
        comparison=paired_counts(fixed['dura'],fixed['direct'],target,mask)
        real=paired_counts(fixed['dura'],fixed['direct'],target,lab)
        raw=paired_counts(fixed['raw'],fixed['direct'],target,mask)
        quality={k:float(fixed[k+'_iq'][group].mean()) for k in ('raw','dura','direct')}
        summary[view]=dict(iq_count=int(group.sum()),iq_normalized_mse=quality,classification=comparison,L_s_classification=real,raw_vs_direct=raw)
    core.write(args.out/'result.json',dict(status='SOURCE_PAIRED_COMPLETE',count=len(ids),commit=args.commit,model_commit=CODE_COMMIT,valid_count=int(valid.sum()),L_s_count=int(labeled.sum()),views=summary,predictions_before_source_labels=True,U_s_labels_read=0,V_reads=0,test_reads=0,ground_reader='Tri-SIG-TIME SSL',DURA_used_only_for_comparison=True,nonblind_development=True))
    print('SOURCE_PAIRED_COMPLETE',flush=True)

if __name__=='__main__':main()
