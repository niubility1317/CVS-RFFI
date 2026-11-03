"""Replay two frozen source models on the established Phase1 source cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.phase1_dura_denoiser_fullsource.logic import match_ids
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_posterior_denoiser.model import PosteriorDenoiser
from experiments.phase1_dura_volterra.pipeline import load_components

SOURCES=(
 ('20261103-phase1-dura-context-source-s2026092701-r01','2d37010cec9a298cd7e80057e9116c7d1ddd3a15','phase1_dura_context_v1'),
 ('20261103-phase1-dura-posterior-denoiser-source-s2026092701-r01','0907dd9fed520aeb9750558c9f4bfe96ffc73fb6','phase1_dura_posterior_denoiser_v1'))
PRIOR='20261102-phase1-unroll-source-replay-s2026092701-r01'

def model_for_schema(schema):
    if schema == 'phase1_dura_context_v1':
        return ContextRecovery()
    if schema == 'phase1_dura_posterior_denoiser_v1':
        return PosteriorDenoiser()
    raise ValueError('Only the two registered source models are allowed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gate_numpy(active):
    return np.from_dlpack(active.to(dtype=torch.int64).contiguous())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--commit',required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Unique source replay output required')
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Exact release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
    import source_pilot as deps
    prior=args.root/'runs'/PRIOR
    pm=json.loads((prior/'prediction_complete.json').read_text())
    if (pm.get('status')!='SOURCE_PREDICTIONS_FIXED' or pm.get('count')!=11761
            or not pm.get('source_only') or not pm.get('DURA_comparator_only')
            or any(pm.get(k)!=0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads'))
            or sha(prior/'source_predictions.npz')!=pm.get('sha256')):
        raise ValueError('The fixed source comparator is required')
    with np.load(prior/'source_predictions.npz',allow_pickle=False) as z:
        ids,split,rx_saved,scene_saved,old=[z[k].copy() for k in ('indices','split','rx','scene','values')]
    if ids.shape!=(11761,) or len(np.unique(ids))!=len(ids) or old.shape!=(len(ids),15):
        raise ValueError('Complete source comparator rows required')
    cache,clean,rx,training,pv,ph=deps.s.source_contract(args.root)
    validation,heldout=deps.s.new_splits(cache,rx,training,pv,ph)
    received,scene=deps.s.epoch_data(cache,19)
    if (not np.array_equal(ids,np.r_[validation,heldout]) or np.intersect1d(ids,training).size
            or not np.array_equal(rx_saved,rx[ids]) or not np.array_equal(scene_saved,scene[ids])):
        raise ValueError('Matching disjoint source cohort required')
    models=[];fit_ids=None;pilot=np.full((len(ids),2),-1,dtype=np.int64)
    for column,(name,commit,schema) in enumerate(SOURCES):
        p=args.root/'runs'/name;meta=json.loads((p/'prediction_complete.json').read_text())
        expected=dict(status='SOURCE_PREDICTIONS_FIXED',commit=commit,count=672,fit_ids=8192,
                      F_frozen=True,DURA_loaded=True,DURA_frozen=True,TX_labels_read=0,
                      U_s_truth_reads=0,V_reads=0,test_reads=0,model_updates_after_prediction=0)
        if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen source model required')
        state=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
        if (state.get('commit')!=commit or state.get('schema')!=schema or state.get('epochs')!=5
                or len(state.get('fit_ids',[]))!=8192 or np.intersect1d(ids,state['fit_ids']).size):
            raise ValueError('Disjoint exact source checkpoint required')
        if fit_ids is None: fit_ids=np.asarray(state['fit_ids']).copy()
        elif not np.array_equal(fit_ids,state['fit_ids']): raise ValueError('Matched models must use identical ordered fit IDs')
        model=model_for_schema(schema);model.load_state_dict(state['model'],strict=True)
        models.append(model.eval().requires_grad_(False))
        with np.load(p/'source_predictions.npz',allow_pickle=False) as z:
            pi=z['indices'].copy();pv=z['values'].copy()
        if pi.shape!=(672,) or pv.shape!=(672,9):raise ValueError('Complete original source pilot required')
        positions=match_ids(ids,pi);pilot[positions,column]=pv[:,7].astype(np.int64)
    if not np.array_equal(pilot[:,0]>=0,pilot[:,1]>=0) or int((pilot[:,0]>=0).sum())!=672:
        raise ValueError('Both source pilots must use identical physical records')
    blind,gate,dura,eye=load_components(args.root,'cpu',deps)
    reader,_=deps.core.teacher(args.root,'cpu');reader.eval().requires_grad_(False)
    total=64 if args.smoke else len(ids);rows=[];matched=0;tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,total,8):
            stop=min(start+8,total);x=deps.core.tensor(received[ids[start:stop]],'cpu')
            anchor=dura(x);active=gate(x).reshape(-1).sigmoid()>.8
            base=reader.classify_features(reader.features(anchor)).argmax(1)
            raw=reader.classify_features(reader.features(x)).argmax(1)
            v=old[start:stop]
            if (not np.array_equal(np.from_dlpack(base.contiguous()),v[:,2].astype(np.int64))
                    or not np.array_equal(np.from_dlpack(raw.contiguous()),v[:,5].astype(np.int64))
                    or not np.array_equal(gate_numpy(active),v[:,3]>.8)):
                raise ValueError('DURA, raw and gate must match fixed source comparator')
            outputs=[anchor.clone(),anchor.clone()]
            if active.any():
                _,lv=blind(x[active])
                for j,model in enumerate(models):outputs[j][active]=model(x[active],anchor[active],lv)
            predictions=torch.stack([reader.classify_features(reader.features(y)).argmax(1) for y in outputs],dim=1)
            got=np.from_dlpack(predictions.contiguous()).copy();expected=pilot[start:stop];has=expected[:,0]>=0
            if has.any() and not np.array_equal(got[has],expected[has]):
                raise ValueError('Fixed original pilot decisions must replay exactly')
            matched+=int(has.sum());rows.append(got)
            if stop%512==0:print('FULLSOURCE_PAIR_PROGRESS',stop,'seconds',round(time.monotonic()-tick,1),flush=True)
    if not args.smoke and matched!=672:raise ValueError('All 672 fixed pilot decisions must replay')
    args.out.mkdir(parents=True)
    if args.smoke:
        (args.out/'smoke.json').write_text(json.dumps(dict(status='PASS',count=total,pilot_matches=matched,
            TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,test_reads=0)),encoding='utf8')
        return
    with (args.out/'predictions.npz').open('xb') as stream:
        np.savez(stream,indices=ids,split=split,rx=rx_saved,scene=scene_saved,predictions=np.concatenate(rows))
    out=dict(status='SOURCE_PREDICTIONS_FIXED',commit=args.commit,count=len(ids),pilot_replayed=matched,
             source_models=[x[0] for x in SOURCES],comparator_sha256=pm['sha256'],
             prediction_sha256=sha(args.out/'predictions.npz'),TX_truth_reads=0,U_s_truth_reads=0,
             V_reads=0,test_reads=0,model_updates=0)
    (args.out/'prediction_complete.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('FULLSOURCE_PAIR_PREDICTIONS_FIXED',len(ids),flush=True)

if __name__=='__main__':main()
