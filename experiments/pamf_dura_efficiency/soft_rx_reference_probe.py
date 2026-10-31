"""Check a source soft-posterior, leave-RX feature reference before any G training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .head_features import head_feature_basis
from .soft_rx_reference import fit_references,make_reference,head_geometry,eligible,BETA,RX,DAYS

RUN='20261031-diagnostic-DURA-source-softRXreference-s2026092701-r01'
RULE=('Fit62488sourcecleanfeatures excludingfixed512(128L384U);actualfrozenFsoft6D posterior only,nohardTX. '
 'EqualRX/daycellmeans,eachreference excludesownRX;target=normalize(.75sourceunitfeature+.25softotherRXreference). '
 'All90fitRX/day/soft-coordinate effectiveN>=32,all512originalheadchoices preserved, '
 'heldoutprobe originalhead-visibleRXscatter<=.8original andclassspread>=.9original. '
 'Failure doesnotrelax beta/threshold;passonlysupports separatelyregisteredsourceIQgradient/feasibilitycalibration, '
 'not Gtraining, headreplacement, targetprediction, actualTXefficacy ornewunseenRXclaim.')


def original_head_constants(reader):
    head=reader.core.id_backbone.cls_head
    normalized=torch.nn.functional.normalize(head.weight.detach().float(),dim=1,eps=1e-4)
    return numpy_values(normalized).astype(np.float64),float(head.scale)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source reference diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);probe=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    if rx.shape!=(63000,) or day.shape!=(63000,) or set(rx.tolist())!=set(RX) or set(day.tolist())!=set(DAYS):raise ValueError('Original sourceRX/day handles required')
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';meta=base.read_json(cache/'complete.json')
    expected=dict(status='SOURCE_F_FEATURES_COMPLETE',count=63000,dimension=160,TX_labels_read=0,V_reads=0,target_reads=0,F_updates=0,F_teacher=f_guided.teacher_reference())
    if any(meta.get(k)!=value for k,value in expected.items()):raise ValueError('Original authorized source F clean cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original source F physical order required')
    cached=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if cached.shape!=(63000,160) or cached.dtype!=np.float32 or not np.isfinite(cached).all():raise ValueError('Complete finite source FP32 features required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={key:value.clone() for key,value in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CPU_threads=2,GPU_workers=0,rule=RULE))
    with torch.no_grad():
        real=reader.features(tensor(clean[probe[:2]],'cpu'));scores=reader.classify_features(real)
    cache_error=float(np.max(abs(numpy_values(real)-cached[probe[:2]])))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or cache_error>3e-5:raise ValueError('Actual source2Ls F/defaulthead/cache smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_feature_inputs=2,cache_max_error=cache_error,TX_labels_read=0,target_reads=0))
    print(json.dumps(dict(event='SOURCE_SOFT_RX_REFERENCE_SMOKE_PASS',real_Ls=2,cache_max_error=cache_error)),flush=True)
    started=time.perf_counter();logits=[]
    with torch.no_grad():
        for offset in range(0,63000,1024):
            value=reader.classify_features(tensor(cached[offset:offset+1024],'cpu'))
            if value.shape!=(min(1024,63000-offset),6) or not torch.isfinite(value).all():raise ValueError('Original frozen source soft coordinates required')
            logits.append(numpy_values(value))
    scores=np.concatenate(logits).astype(np.float64);exp=np.exp(scores-scores.max(1,keepdims=True));probabilities=exp/exp.sum(1,keepdims=True)
    features=np.asarray(cached,dtype=np.float64);features=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-4)
    fit_mask=np.ones(63000,bool);fit_mask[probe]=False
    if int(fit_mask.sum())!=62488:raise ValueError('Fixed512 excluded from reference fitting required')
    fitted=fit_references(features,probabilities,rx,day,fit_mask)
    target=make_reference(features[probe],probabilities[probe],rx[probe],fitted['leave_RX_centroids'])
    with torch.no_grad():reference_logits=numpy_values(reader.classify_features(tensor(target.astype(np.float32),'cpu')))
    if reference_logits.shape!=(512,6) or not np.isfinite(reference_logits).all():raise ValueError('Original default head reference scores required')
    agreement=reference_logits.argmax(1)==scores[probe].argmax(1)
    basis=np.asarray(head_feature_basis(reader).tolist(),np.float64)
    if basis.shape!=(160,5):raise ValueError('Original six-class centeredhead rank5 required')
    original=head_geometry(features[probe],probabilities[probe],rx[probe],basis)
    changed=head_geometry(target,probabilities[probe],rx[probe],basis)
    minimum=float(fitted['cell_effective_n'].min());decision=eligible(minimum,int(agreement.sum()),original,changed)
    head_weight,head_scale=original_head_constants(reader)
    phi_offset=4*((target-features[probe])@basis)
    summaries={}
    groups={'all':np.ones(512,bool)}
    for key,values in dict(role=roles[probe],rx=rx[probe],day=day[probe]).items():
        for value in np.unique(values):groups[f'{key}:{value}']=values==value
    original_margin=np.sort(scores[probe],axis=1)[:,-1]-np.sort(scores[probe],axis=1)[:,-2]
    reference_margin=np.sort(reference_logits,axis=1)[:,-1]-np.sort(reference_logits,axis=1)[:,-2]
    for key,mask in groups.items():
        summaries[key]=dict(count=int(mask.sum()),choices_preserved=int(agreement[mask].sum()),
            reference_change_unit_feature_MSE=float(np.square(target[mask]-features[probe][mask]).sum(1).mean()),
            head_phi_offset_norm_mean=float(np.linalg.norm(phi_offset[mask],axis=1).mean()),
            original_head_margin_mean=float(original_margin[mask].mean()),reference_head_margin_mean=float(reference_margin[mask].mean()))
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('Frozen original F changed')
    np.savez(args.output/'reference_geometry.npz',source_positions=probe,physical_ids=ids[probe],roles=roles[probe],rx=rx[probe],day=day[probe],
        fit_mask=fit_mask,full_probabilities=probabilities,probe_features=features[probe],reference_features=target,head_basis=basis,
        original_logits=scores[probe],reference_logits=reference_logits,head_weight=head_weight,head_scale=np.asarray(head_scale),
        phi_offset=phi_offset,choices_preserved=agreement,**fitted)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_rule=RULE,source_records=63000,fit_records=62488,
        probe_records=512,probe_L=128,probe_U=384,beta=BETA,RX=list(RX),days=list(DAYS),soft_coordinates=6,hard_pseudo_TX_labels=False,
        fit_cell_min_effective_n=minimum,fit_cell_effective_n=fitted['cell_effective_n'].tolist(),reference_choices_preserved=int(agreement.sum()),
        original_probe_head_geometry=original,reference_probe_head_geometry=changed,
        RX_scatter_ratio=changed['RX_scatter']/original['RX_scatter'],class_spread_ratio=changed['class_spread']/original['class_spread'],
        groups=summaries,source_reference_geometry_eligible=decision,F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        F_new_feature_inputs=2,F_head_only_inputs=63512,F_smoke_head_inputs=2,source_F_cache_metadata=meta,
        G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='SourceF-informed soft reference geometry only;prototype fit excludes512 andownRX butF trainedonallsource;notnewFhead/actualTX/unknownRXgeneralization/reachableIQ/targetefficacy;noGtraining'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',source_reference_geometry_eligible=decision,optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
