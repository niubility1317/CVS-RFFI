"""Truth-last source-prototype identity scoring with the original F0 baseline."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_score as fresh
from . import star_dura_best_score as previous
from . import star_consensus_score as common

RUN_ID='20261031-phase3-prototype-identity-s2026092701-r01'
SCHEMA='pamf_phase3_prototype_identity_scores_v1'
STATUS='PROTOTYPE_IDENTITY_SCORES_COMPLETE'
ARMS=('OLD_PCD','PROTO')
METHODS=tuple(arm+'_'+method for arm in ARMS for method in ('pooled_ratio','predicted_ratio'))
PRIMARY=METHODS[-1]
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
_read,_write,_source_truth=fresh._read,fresh._write,fresh._source_truth
_future_truth=fresh._future_truth


def _prediction(root):
    # This artifact-only gate reconstructs prototype identity, density routing,
    # thresholds and exact parent controls; it performs no forward or fitting.
    from . import prototype_identity_run as producer
    done,_,_,_,cal,rows=producer._prediction_gate(root)
    if (done.get('run_id')!=RUN_ID or done.get('method_names')!=list(METHODS)
            or done.get('primary_method')!=PRIMARY or set(rows)!={'source_V','source_NV','Q_clean_DURA','Q_star_DURA'}
            or any(set(value)!=set(ARMS) for value in rows.values()) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('All four frozen readouts and three complete DURA conditions required before truth')
    return done,cal,rows


def _pair(new,old,nn,on,index,km=None,nm=None):
    return previous.paired_counts(new,old,nn,on,index,km,nm)


def _control_gate(main,control):
    gain=main['A_K'] is not None and control['A_K'] is not None and main['A_K']>control['A_K']+1e-12
    kept=main['R_U'] is not None and control['R_U'] is not None and main['R_U']>=control['R_U']-1e-12
    return bool(gain),bool(kept)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable prototype-identity score root required')
    done,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_NV']['OLD_PCD'])
    truths={base:_future_truth(path,rows['Q_'+base+'_DURA']['OLD_PCD']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Private future condition/scenario mismatch')
    if any(not np.array_equal(truths['clean'][key],truths['star'][key]) for key in ('ids','y','tx','receiver','day')):
        raise ValueError('Clean/star private physical labels and dates must be paired')
    methods={}
    for j,name in enumerate(METHODS):
        arm=ARMS[j//2];index=j%2
        source=dict(known=rows['source_V'][arm],novel=rows['source_NV'][arm])
        paths={'source_DURA':previous._source(source,nt,index)}
        paths.update({condition:fresh._target(rows['Q_'+condition][arm],truths[condition.split('_')[0]],condition,index) for condition in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,classification_model=arm,diagnostic_only=name!=PRIMARY,source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    main=methods[PRIMARY]['paths'];paired={}
    for condition in CONDITIONS:
        if condition=='source_DURA':
            new,old=rows['source_V']['PROTO'],rows['source_V']['OLD_PCD'];nn,on=rows['source_NV']['PROTO'],rows['source_NV']['OLD_PCD']
            km=dict(common._masks(new['rx_i'],new['scenario']));nm=dict(common._masks(nn['rx_i'],nt['scenario']));uy=nt['tx'];unknowns=common.UNKNOWN_TX
            rx_masks={str(r):(new['rx_i']==r,nn['rx_i']==r) for r in common.RECEIVERS}
            day_masks={str(d):(new['day_i']==d,nn['day_i']==d) for d in (1,2,3)}
        else:
            t=truths[condition.split('_')[0]];y=t['y'];k=y<6;u=~k
            a,b=rows['Q_'+condition]['PROTO'],rows['Q_'+condition]['OLD_PCD']
            def known(v):return dict(ids=v['ids'][k],y=y[k],closed=v['closed'][k],scores=v['scores'][k])
            def novel(v):return dict(ids=v['ids'][u],scores=v['scores'][u])
            new,old,nn,on=known(a),known(b),novel(a),novel(b)
            scenes=('clean',) if condition=='clean_DURA' else common.SCENES
            km={s:t['scenario'][k]==s for s in scenes};nm={s:t['scenario'][u]==s for s in scenes};uy=y[u];unknowns=fresh.TARGET_UNKNOWN
            rx_masks={str(r):(t['receiver'][k]==r,t['receiver'][u]==r) for r in np.unique(t['receiver'])}
            day_masks={d:(t['day'][k]==d,t['day'][u]==d) for d in fresh.DAYS}
        paired[condition]=dict(counts=_pair(new,old,nn,on,1),groups={g:_pair(new,old,nn,on,1,mask,nm[g]) for g,mask in km.items()},
            by_RX={g:_pair(new,old,nn,on,1,*masks) for g,masks in rx_masks.items()},by_day={g:_pair(new,old,nn,on,1,*masks) for g,masks in day_masks.items()},
            by_known_TX={str(c):_pair(new,old,nn,on,1,new['y']==c) for c in range(6)},by_unknown_TX={str(c):_pair(new,old,nn,on,1,nm=uy==c) for c in unknowns},
            delta=previous._delta(main[condition]['overall'],methods['OLD_PCD_predicted_ratio']['paths'][condition]['overall']))
    positive,kept=_control_gate(main['star_DURA']['overall'],methods['OLD_PCD_predicted_ratio']['paths']['star_DURA']['overall'])
    known=bool(main['source_DURA']['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(known and positive and kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,arm_order=list(ARMS),conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired_primary_vs_OLD_PCD_predicted_ratio=paired,
        star_A_K_gain_vs_OLD_PCD=paired['star_DURA']['delta']['A_K'],star_A_K_strictly_higher_than_OLD_PCD=positive,star_R_U_not_lower_than_OLD_PCD=kept,
        source_known_12_passed=known,target_protection_passed=bool(protection),primary_composite_success=success,development_gate_passed=success,eligible_primary=success,adaptation_success=None,
        next_step='DEVELOPMENT_PROTOTYPE_IDENTITY_PASSED_REQUIRES_SEPARATE_CONFIRMATION' if success else 'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,
        classification_path='source PCD unit-feature nearest arithmetic class mean',identity_baseline_path='same_DURA_output_original_F0',original_F0_unchanged=True,
        original_default_head_unchanged=True,source_control_readouts_reused_exactly=True,prototype_centers_renormalized=False,
        new_neural_updates=0,new_GMM_fits=0,new_source_prototype_fits=1,new_source_threshold_sets=2,new_target_U_updates=0,
        prototype_values=960,prototype_bytes=7680,threshold_values=24,threshold_bytes=192,new_threshold_values=12,new_threshold_bytes=96,
        prediction_costs={k:v for k,v in done.items() if k.endswith('_reads') or k.endswith('_forward_rows') or k.endswith('_parameters')},
        frozen_prediction_identity={k:v for k,v in done.items() if k in ('parent_F0_identity','dura_identity','model_identity','source_geometry_identity','reference_prediction_identity')},
        PCD_has_inherited_target_U_ancestor=True,inherited_target_U_count=576,this_run_target_U_reads=0,this_run_target_DA=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,
        source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        original_F0_parameters=232993,PCD_parameters=334721,existing_restorer_parameters=266951,primary_deployed_neural_parameters=601672,
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Source-only prototype/known-V calibration over frozen U-adapted PCD. Source V is calibration, four future unknowns stay unregistered; no new UDA or blind confirmation claim')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_OLD_PCD_predicted_ratio')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
