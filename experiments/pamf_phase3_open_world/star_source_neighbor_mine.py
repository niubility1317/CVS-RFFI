"""Freeze source-L pair rules before mining label-free source OE neighbors."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from . import runner
from . import star_source_neighbor_export as exported
from . import star_source_neighbors as algorithm

SCHEMA='pamf_phase3_source_neighbor_mining_v1'
STATUS='SOURCE_NEIGHBOR_MINING_COMPLETE'
ARMS=('P1','PAIR')


def _load(root,role):
    required={'ids','rx_i','day_i','P0','P1'}|({'y'} if role=='L' else set())
    with np.load(Path(root)/(role+'.npz'),allow_pickle=False) as saved:
        if set(saved.files)!=required:raise ValueError('Exact source feature fields required; OE TX/y/K forbidden')
        values={k:saved[k].copy() for k in saved.files}
    n=exported.COUNTS[role]
    if (values['ids'].shape!=(n,) or values['ids'].dtype.kind!='U' or len(set(values['ids'].tolist()))!=n
            or any(values[k].shape!=(n,) or values[k].dtype!=np.int64 for k in ('rx_i','day_i'))
            or not np.isin(values['rx_i'],(1,3,4,6,8)).all() or not np.isin(values['day_i'],(1,2,3)).all()
            or any(values[k].shape!=(n,160) or values[k].dtype!=np.float32 or not np.isfinite(values[k]).all()
                or not np.allclose(np.linalg.norm(values[k].astype(np.float64),axis=1),1.,rtol=0.,atol=2e-6) for k in ('P0','P1'))):
        raise ValueError('Matching source role IDs, public domains and unit FP32 paired features required')
    if role=='L' and (values['y'].shape!=(n,) or values['y'].dtype!=np.int64 or set(values['y'].tolist())!=set(range(6))):
        raise ValueError('Only original source-L known-six labels are legal for rule calibration')
    return values


def _candidates(values,arm):
    return algorithm.candidates(values['ids'],values['P1'],values['rx_i'],values['day_i'],
        z0=values['P0'] if arm=='PAIR' else None,block_size=128)


def mine(feature_root,output,code_commit):
    start=time.perf_counter();feature_root=Path(feature_root);output=Path(output)
    if output.exists():raise FileExistsError('Use one new immutable source mining output')
    marker=runner._read(feature_root/'complete.json');exported.validate_config(marker['config'])
    expected=dict(schema=exported.SCHEMA,status=exported.STATUS,run_id=exported.RUN_ID,code_commit=code_commit,
        counts=exported.COUNTS,source_only=True,model_states_unchanged=True,model_updates=0,optimizer_updates=0,
        target_reads=0,private_truth_reads=0,source_V_reads=0,source_novel_reads=0,original_U_s_reads=0)
    if (any(marker.get(k)!=v for k,v in expected.items()) or marker['config']['code_commit']!=code_commit
            or Path(marker['config']['output_root'])!=feature_root):
        raise ValueError('Complete original-F0 source L/OE features must precede all mining')
    L=_load(feature_root,'L')
    output.mkdir(parents=True,exist_ok=False)
    candidates={a:_candidates(L,a) for a in ARMS}
    calibrations={a:algorithm.calibrate(candidates[a],L['y'],L['rx_i']) for a in ARMS}
    runner._write(output/'calibration.json',dict(schema=SCHEMA,run_id=exported.RUN_ID,code_commit=code_commit,
        source_role='L_only',OE_values_used=False,target_reads=0,private_truth_reads=0,arms=calibrations))
    # No OE feature values are loaded until both source-known thresholds are fixed.
    OE=_load(feature_root,'OE')
    if set(L['ids'].tolist()).intersection(OE['ids'].tolist()):raise ValueError('L and OE physical IDs must be disjoint')
    results={}
    for arm in ARMS:
        chosen_L=algorithm.apply(candidates[arm],calibrations[arm]['threshold'])
        source_candidate=_candidates(OE,arm)
        chosen_OE=algorithm.apply(source_candidate,calibrations[arm]['threshold'])
        l_summary=algorithm.summary(L['ids'],L['rx_i'],L['day_i'],chosen_L)
        oe_summary=algorithm.summary(OE['ids'],OE['rx_i'],OE['day_i'],chosen_OE)
        for role,bundle,pairs in (('L',L,chosen_L),('OE',OE,chosen_OE)):
            with (output/(arm+'_'+role+'_pairs.npz')).open('xb') as stream:
                np.savez(stream,ids=bundle['ids'],pair_indices=pairs)
        def covered(summary):
            return (summary['covered_node_fraction']>=.20 and len(summary['by_RX'])==5
                and all(v['covered_node_fraction']>=.10 for v in summary['by_RX'].values()))
        eligible=bool(calibrations[arm]['constraints_passed'] and covered(l_summary) and covered(oe_summary)
            and oe_summary['edge_count']>=256)
        results[arm]=dict(calibration=calibrations[arm],L=l_summary,OE=oe_summary,
            candidate_edges=dict(L=len(candidates[arm]['pair_indices']),OE=len(source_candidate['pair_indices'])),
            usable_source_pairs=eligible,OE_true_pair_precision=None,OE_true_TX_count=None)
        print('SOURCE_NEIGHBOR_ARM_COMPLETE',arm,'L',l_summary['edge_count'],'OE',oe_summary['edge_count'],flush=True)
    qualified=results['PAIR']['usable_source_pairs']
    done=dict(schema=SCHEMA,status=STATUS,run_id=exported.RUN_ID,code_commit=code_commit,primary='PAIR',arms=results,
        counts=exported.COUNTS,features_identity=marker,source_only=True,known_rule_frozen_before_OE=True,
        source_L_labels_used_for_rule=True,OE_labels_used=False,OE_true_pair_precision=None,
        connected_components_are_TX_labels=False,transitive_pseudo_labels_created=False,
        next_step='SOURCE_PAIRS_ELIGIBLE_FOR_SEPARATE_TRAINING' if qualified else 'NO_RELIABLE_SOURCE_NEIGHBORS',
        eligible_for_separate_source_training=qualified,controls_cannot_replace_primary=True,
        target_reads=0,private_truth_reads=0,source_V_reads=0,source_novel_reads=0,original_U_s_reads=0,
        neural_updates=0,optimizer_updates=0,new_F_forward_rows=0,new_G_forward_rows=0,
        A_K=None,C_K=None,R_U=None,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_status='ACTIVE',
        seconds=time.perf_counter()-start)
    runner._write(output/'results.json',done);runner._write(output/'complete.json',done)
    print(STATUS,done['next_step'],flush=True);return done


def main():
    p=argparse.ArgumentParser();p.add_argument('--feature-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--code-commit',required=True);a=p.parse_args();mine(a.feature_root,a.output,a.code_commit)


if __name__=='__main__':main()
