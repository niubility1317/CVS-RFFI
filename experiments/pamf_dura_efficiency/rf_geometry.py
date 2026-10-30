"""Source-only output-gradient calibration of fixed analytic RF transforms."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_restoration.views import PAMFViews, sample_nmse
from experiments.pamf_dura_direct_train import runner as training, no_variance, f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices, tensor, numpy_values

RUN = '20261030-diagnostic-DURA-source-analytic-RF-geometry-s2026092701-r01'
SOURCE_RUN = '20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
SOURCE_COMMIT = 'cf40edea90667d9645a724b1e8f333027cdf7869'
BRANCHES = ('frequency', 'pa', 'sinc', 'highfreq', 'reference')
RULE = ('Alpha=.25/median32batch(||gRF||/||gIQ||), where gradients are with respect to outputIQ, '
        'RF=sum5 normalized analytic branch errors. Both fixed source cached01/20 require '
        'median alpha*ratio in [.125,.5], mean gradient cosine>=0 and at most8/16 negative cosines. '
        'Finite nonzero gradients required. No weight grid, F output, TX, V or target calibration. '
        'Passing permits only a separately registered matched scratch E20 training, not an efficacy claim.')


def analytic_terms(restored, clean, extractor):
    with torch.no_grad():
        reference = extractor(clean.detach())
    predicted = extractor(restored)
    return {key: sample_nmse(predicted[key], reference[key]) for key in BRANCHES}


def gradient_pair(restored, clean, extractor):
    variable = restored.detach().requires_grad_(True)
    iq = sample_nmse(variable, clean.detach())
    terms = analytic_terms(variable, clean, extractor)
    rf = sum(terms.values())
    g_iq = torch.autograd.grad(iq.mean(), variable)[0]
    g_rf = torch.autograd.grad(rf.mean(), variable)[0]
    norms = (g_iq.norm(), g_rf.norm())
    if any(not torch.isfinite(v).all() for v in (iq, rf, g_iq, g_rf)) or any(v <= 0 for v in norms):
        raise FloatingPointError('Finite nonzero analytic/IQ output gradients required')
    return iq, terms, g_iq, g_rf


def calibration_decision(batches):
    if len(batches) != 32 or any(sum(r['view'] == view for r in batches) != 16 for view in (1, 20)):
        raise ValueError('Exactly sixteen B32 batches in each fixed source view required')
    ratios = np.asarray([r['ratio'] for r in batches], dtype=np.float64)
    cosines = np.asarray([r['cosine'] for r in batches], dtype=np.float64)
    if not np.isfinite(ratios).all() or np.any(ratios <= 0) or not np.isfinite(cosines).all():
        raise ValueError('Invalid output-gradient geometry')
    alpha = .25 / float(np.median(ratios))
    views = []
    for view in (1, 20):
        take = np.asarray([r['view'] == view for r in batches])
        pressure = float(np.median(alpha * ratios[take]))
        cosine = float(np.mean(cosines[take])); negative = int((cosines[take] < 0).sum())
        views.append(dict(view=view,median_pressure=pressure,mean_cosine=cosine,negative_batches=negative,
                          passed=bool(.125 <= pressure <= .5 and cosine >= 0 and negative <= 8)))
    return dict(alpha=alpha,views=views,analytic_RF_study_eligible=all(r['passed'] for r in views))


def source_outputs(path, indices, ids, roles, clean):
    # npz members are lazy. In particular, never open the saved F_features,
    # F_logits, teacher-choice arrays, aligned IQ or other model/view files.
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],clean=clean[indices]).items():
            if not np.array_equal(saved[key], expected):
                raise ValueError('Saved source positions/physical roles/clean mismatch: '+key)
        restored = saved['native']
    if restored.shape != (512,2,256) or restored.dtype != np.float32 or not np.isfinite(restored).all():
        raise ValueError('Fixed complete FP32 source noVar output required')
    return restored


def run(args):
    if args.output.name != RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('New immutable CPU source-only diagnostic root required')
    no_variance.configure();p = training.paths(args.root)
    done = training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected = dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,
                    training_F_access=False,V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
    if any(done.get(k) != v for k,v in expected.items()):
        raise ValueError('Fixed source-final noVar provenance required')
    clean,roles = training.source_training(p);indices = source_indices(roles);ids = f_guided.source_ids(p,roles)
    saved_root = args.root/'runs'/SOURCE_RUN
    outputs = {view: source_outputs(saved_root/f'noVar-{view}.npz',indices,ids,roles,clean) for view in (1,20)}
    model = training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT).eval().requires_grad_(False)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader = load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before = [{k:v.clone() for k,v in m.state_dict().items()} for m in (model,reader)]
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0))
    with torch.no_grad():
        smoke = reader(model(tensor(clean[indices[:2]],'cpu')))
        if smoke.shape != (2,6) or not torch.isfinite(smoke).all():
            raise ValueError('Actual source2Ls checkpoint smoke failed')
    for m,state in zip((model,reader),before):
        if any(not torch.equal(v,state[k]) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()):
            raise ValueError('Smoke changed fixed G/F state')
    del model,reader,before,smoke
    print(json.dumps(dict(event='REAL_SOURCE_G_F_CHECKPOINT_SMOKE_PASS',Ls=2,target_reads=0)),flush=True)
    extractor = PAMFViews().eval().requires_grad_(False)
    fixed = {k:v.clone() for k,v in extractor.state_dict().items()}
    batches = [];vectors = {key:[] for key in ('g_iq','g_rf','iq_nmse','branch_nmse','view','source_positions')}
    started = time.perf_counter()
    for view in (1,20):
        for offset in range(0,512,32):
            x = tensor(clean[indices[offset:offset+32]],'cpu');r = tensor(outputs[view][offset:offset+32],'cpu')
            begin = time.perf_counter();iq,terms,g_iq,g_rf = gradient_pair(r,x,extractor)
            seconds = time.perf_counter()-begin
            norm_iq,norm_rf = float(g_iq.norm()),float(g_rf.norm())
            cosine = float((g_iq*g_rf).sum()/(g_iq.norm()*g_rf.norm()))
            row = dict(view=view,offset=offset,batch=32,iq_nmse=float(iq.detach().mean()),
                       branches={k:float(v.detach().mean()) for k,v in terms.items()},
                       iq_gradient_norm=norm_iq,rf_gradient_norm=norm_rf,ratio=norm_rf/norm_iq,
                       cosine=cosine,seconds=seconds)
            batches.append(row)
            for key,value in dict(g_iq=g_iq,g_rf=g_rf,iq_nmse=iq,branch_nmse=torch.stack(list(terms.values()),1)).items():
                vectors[key].append(numpy_values(value.detach()))
            vectors['view'].append(np.full(32,view,dtype=np.int64));vectors['source_positions'].append(indices[offset:offset+32])
            print(json.dumps(dict(event='SOURCE_RF_GEOMETRY_BATCH',**row)),flush=True)
    elapsed = time.perf_counter()-started
    if any(not torch.equal(v,fixed[k]) for k,v in extractor.state_dict().items()) or any(p.grad is not None for p in extractor.parameters()):
        raise ValueError('Fixed analytic transforms changed or accumulated gradients')
    decision = calibration_decision(batches)
    np.savez(args.output/'gradients.npz',**{key:np.concatenate(values) for key,values in vectors.items()})
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_rule=RULE,decision=decision,batches=batches,
        analytic_branches=BRANCHES,source_records=512,source_visits=1024,L_s=128,U_s=384,views=[1,20],
        G_analysis_inputs=0,F_analysis_inputs=0,G_smoke_inputs=2,F_smoke_inputs=2,G_unchanged=True,F_unchanged=True,
        analytic_transform_inputs=2048,analytic_parameters_frozen=True,trained_F_weights_used_in_calibration=False,
        saved_F_members_opened=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=elapsed,
        scope='OutputIQ gradients,not G parameter gradients;source-only feasibility,not target efficacy;broader research F-informed'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',G_unchanged=True,F_unchanged=True,
        optimizer_updates=0,target_reads=0,goal_complete=False,**decision))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('root','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
