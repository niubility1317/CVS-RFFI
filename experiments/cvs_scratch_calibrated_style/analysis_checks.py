"""Synthetic complete source-log checks. No models, remote data or target access."""
import argparse
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from . import analyze_source as a
from . import design as d
from .provenance import student_initialization,teacher_path

LEVELS=('time_down','t2.pool')


def fixture(policy):
    c=d.config(next(r for r in d.rows() if r['arm']==policy+'_M_BOTH'))
    when=c['multilevel']['calibration_after_updates'];eps=c['multilevel']['fishr_eps'];coefficient=.2/(1+eps)
    calibration=dict(status='FIXED_CALIBRATION_COMPLETE',after_identity_updates=when,scheduled_after_identity_updates=when,
        U_access=False,V_access=False,parameters_buffers_unchanged=True,modes_preserved=True,python_numpy_RNG_preserved=True,
        state_unchanged=True,pairs=12,extra_source_readouts=1440,model_forwards=24,RNG_preserved=True,source_role='L_s only',
        target_access=False,identity_updates=0,head_updates=0,all_layers_calibrated=True,calibration_reference='own_margin_term',
        auxiliary_heads=0,fixed_coefficients={k:coefficient for k in LEVELS},
        levels={k:dict(mean_CE_gradient_norm=1.,mean_term_gradient_norm=1.,uncapped_coefficient=coefficient,coefficient=coefficient,cap_active=False) for k in LEVELS},
        records=[dict(pair=i,day=i%3+1,cross_batch_PID_overlap=1,
            levels={k:dict(main_ce_gradient_norm=1.,term_gradient_norm=1.) for k in LEVELS}) for i in range(12)])
    warmup=dict(status='NOT_APPLICABLE',optimizer_steps=0,base_state_unchanged=True,backbone_updates=0,auxiliary_inheritance=False,
        source_L_cache_packets=0,head_packet_occurrences=0,cache_forward_calls=0)
    teachers={str(s):dict(status='VERIFIED',seed=s,checkpoint=teacher_path(s),epoch=20,scratch_only=True,
        ancestors=[],source_roles='EXACT_MATCH',training_target_access=False) for s in c['teacher']['seeds']}
    init=dict(student_initialization(c),parent_provenance=student_initialization(c),incremental_epochs=20,total_epochs=20,
        checkpoint_sources=c['checkpoint_sources'],ancestors=c['checkpoint_sources'],teacher_provenance=teachers,
        target_contact=False,source_probability_cache_used=True)
    costs=dict(parent_training_epochs=0,incremental_epochs=20,total_epochs=20,parent_training_cost_reused=False,
        teacher_training_epochs_reused=40,historical_model_epochs_including_teachers=60,head_warmup=warmup,
        fishr_calibration=calibration,auxiliary_trainable_parameters=0)
    resolved=dict(c,head_warmup=warmup,fishr_calibration=calibration,commit='synthetic-source-only',trainable_parameters=3,execution=dict(
        EMA=False,satellite_generator=False,U_truth_hidden=True,U_iterator=True,auxiliary_CE_to_identity=False,
        Fishr_levels=list(LEVELS),auxiliary_neural_networks=False,scratch=True,E20_weights_loaded=False,
        E200_weights_loaded=False,sensitivity_probes_trainable=False))
    initial=deepcopy(resolved)
    if when:initial['fishr_calibration']=dict(status='PENDING_CALIBRATION',scheduled_after_identity_updates=when)
    ready=dict(after_identity_updates=when,calibration=calibration,resolved=resolved)
    fixed=a._expected_counts(c);counts={k:0 for k in fixed};counts.update(mixed_U_packets=0,mixed_L_packets=0)
    steps=[];epochs=[]
    for step in range(1,4441):
        active=step>444;aux=active and step%4==0;ramp=max(0.,min(1.,(step-444)/888))
        detail={key:0 for key in counts if key not in ('identity_updates','L_packets','EMA_updates','satellite_calls')}
        detail.update(U_packets=128 if active else 0,weak_forwards=int(active),teacher_source='peer',teacher_selection='all',teacher_forwards=0,
            accepted_U=128 if active else 0,accepted_agreement=64 if active else 0,accepted_disagreement=64 if active else 0,
            KL_full_batch_denominator=128,U_accuracy=None,ramp=ramp,
            auxiliary_steps=int(aux),auxiliary_L=120 if aux else 0,auxiliary_forwards=2 if aux else 0,fishr_steps=int(aux))
        raw=coefficient*ramp if aux else 0.;scalar=.2*2/(1+1e-12)
        cap_enabled=aux and c['multilevel']['joint_gradient_cap']['enabled']
        cap=dict(applied=bool(cap_enabled),enabled=bool(cap_enabled),scalar=scalar if cap_enabled else 1.,
            gradient_evaluations=2 if cap_enabled else 0,seconds=.01 if cap_enabled else 0.,groups={})
        if cap_enabled:
            cap.update(scalar_detached=True,complete_disjoint_parameter_coverage=True,rho=.2,eps=1e-12,
                raw_weighted_fishr=raw,effective_weighted_fishr=raw*scalar,
                groups={g:dict(norm_id=2.,raw_R=1.,capped_R=scalar,norm_bound=.4,scalar_bound=scalar,parameter_scalars=1) for g in ('shallow','middle','rest')})
        detail['joint_gradient_cap']=cap
        metrics={key:0. for key in ('ce','KL','feature','margin','style_ce','aux_ce','fishr')}
        metrics.update(ce=1.,KL=.1 if active else 0.,fishr=1. if aux else 0.)
        metrics.update({'weighted_'+key:value*(1. if key=='ce' else ramp) for key,value in metrics.copy().items()})
        metrics.update(weighted_fishr=raw*cap['scalar'],weighted_KL_same=.04*ramp if active else 0.,
            weighted_KL_disagree=.06*ramp if active else 0.)
        metrics['total_loss']=sum(metrics['weighted_'+key] for key in ('ce','KL','feature','margin','style_ce','aux_ce','fishr'))
        if aux:
            sensitivity=dict(score_requires_grad=True,sensitivity_eps=1e-6,low_norm_rate=0.,raw_sensitivity_rms_min=1.,coordinates=8,shape=[60,4,8])
            estimator='finite_L_corrected_cross_AB_margin_sensitivity'
            detail['multilevel']=dict(aux_ce_to_backbone=False,fishr_levels=list(LEVELS),auxiliary_packets=120,aux_forwards=2,
                pair_A_distinct_PID=60,pair_B_distinct_PID=60,cross_batch_PID_overlap=1,distinct_PID=119,
                source_RX=[1,3,4,6,8],day=(step//4-1)%3+1,independent_AB=True,calibration_sampling=False,
                estimator=estimator,head_Fishr_gradient=False,EMA=False,auxiliary_heads=0,statistic='margin',aux_ce=0.,
                sensitivity_eps=1e-6,native_CPU_CUDA_RNG_preserved=True,
                levels={key:dict(fishr_computed=True,estimator=estimator,sensitivity_A=deepcopy(sensitivity),sensitivity_B=deepcopy(sensitivity)) for key in LEVELS})
            detail['fishr_budget']=dict(dynamic_rescaling=False,fixed_denominator=2,EMA=False,scope='raw_pre_joint_cap',
                weighted_loss=raw,effective_weighted_loss=metrics['weighted_fishr'],joint_cap_scalar=cap['scalar'],
                levels={key:dict(fixed_calibrated_coefficient=coefficient,coefficient=coefficient*ramp,coefficient_detached=True,
                    cap_active=False,reason=None,raw_loss=1.) for key in LEVELS},combined_group_gradient_ratios={})
        for key in counts:
            counts[key]+=1 if key=='identity_updates' else 128 if key=='L_packets' else detail.get(key,0)
        audit=dict(weighted_dense_gradient_norms={'ce':dict(encoder=1.),'aux_ce':dict(encoder=0.)}) if step%444==0 else None
        steps.append(dict(step=step,epoch=(step-1)//222+1,metrics=metrics,details=detail,gradient_audit=audit))
        if step%222==0:
            epoch=step//222
            validation=dict(V_count=27000,V_RX_accuracy={str(rx):.5 for rx in (1,3,4,6,8)}) if epoch in c['validation']['epochs'] else None
            epochs.append(dict(epoch=epoch,total_epoch=epoch,optimizer_steps=step,counts=deepcopy(counts),validation=validation,weight_movement={}))
    done=dict(status='SOURCE_TRAINED',config=c,target_access=False,target_evaluated=False,epoch=20,total_epochs=20,
        initialization=init,parent_checkpoint=None,checkpoint_sources=c['checkpoint_sources'],teacher_provenance=teachers,
        resources=costs,optimizer_steps=4440,logged_steps=4440,mechanism_execution=counts,distinct_U_visited=56700)
    return dict(c=c,calibration=calibration,warmup=warmup,initial=initial,resolved=resolved,ready=ready,initialization=init,
        final_resolved=deepcopy(resolved),steps=steps,epochs=epochs,done=done,
        frozen=dict(status='SOURCE_FROZEN',config=c,target_access=False,target_scores_consumed=False,total_epochs=20,parent_checkpoint=None))


def write_fixture(base,f):
    source=base/f['c']['row_id']/'source';source.mkdir(parents=True,exist_ok=True)
    def write(path,value):path.write_text(json.dumps(value,allow_nan=False),encoding='utf8')
    for filename,key in [('completion.json','done'),('resolved_config.json','resolved'),('initialization.json','initialization'),
        ('initial_resolved_config.json','initial'),('resolved_config_final.json','final_resolved'),('calibration_ready.json','ready'),
        ('calibration.json','calibration'),('head_warmup.json','warmup')]:write(source/filename,f[key])
    write(source.parent/'source_frozen.json',f['frozen']);(source/'head_warmup.jsonl').write_text('',encoding='utf8')
    with gzip.open(source/'step_metrics.jsonl.gz','wt',encoding='utf8') as stream:
        for record in f['steps']:stream.write(json.dumps(record,allow_nan=False)+'\n')
    for epoch in f['epochs']:
        records=f['steps'][(epoch['epoch']-1)*222:epoch['epoch']*222]
        for key in records[0]['metrics']:epoch[key]=sum(r['metrics'][key] for r in records)/222
    (source/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in f['epochs']),encoding='utf8')
    flat=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in f['epochs']]
    with (source/'epoch_metrics.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    logs=base/'logs';logs.mkdir(exist_ok=True)
    lines=['RESOLVED '+json.dumps(f['initial'])]
    if not f['c']['multilevel']['calibration_after_updates']:lines.append('CALIBRATION_READY '+json.dumps(f['ready']))
    for epoch in f['epochs']:
        lines.append('EPOCH '+json.dumps(epoch))
        if epoch['optimizer_steps']==f['c']['multilevel']['calibration_after_updates']:
            lines.append('CALIBRATION_READY '+json.dumps(f['ready']))
    lines.append('SOURCE_TRAINED '+json.dumps(f['done']))
    # A sentinel after the closed source segment must never be parsed.
    lines.append('FORBIDDEN_POST_SOURCE_SENTINEL {not-json}')
    (logs/(f['c']['row_id']+'.log')).write_text('\n'.join(lines)+'\n',encoding='utf8')


def analyze_fixture(base,f):
    write_fixture(base,f)
    row={key:f['c'][key] for key in ('row_id','arm','model_seed')}
    return a.analyze_row(base,row,f['c'])


def run():
    passed=[];rejections={};missed=[]
    with TemporaryDirectory() as directory:
        base=Path(directory)
        f=fixture('DELAYCAP');row={key:f['c'][key] for key in ('row_id','arm','model_seed')}
        assert a.analyze_row(base,row,f['c'])['status']=='PARTIAL'
        for policy in ('DELAY','CAP','DELAYCAP'):
            candidate=fixture(policy);result=analyze_fixture(base,candidate)
            assert result['status']=='COMPLETE' and result['steps']==4440
            assert len(result['joint_cap_records'])==(999 if policy!='DELAY' else 0)
            assert not result['source_stdout']['target_segment_read']
            passed.append(policy)
        pristine=fixture('DELAYCAP')
        def wrong_time(f):
            f['calibration']['after_identity_updates']=440
            f['final_resolved']['fishr_calibration']['after_identity_updates']=440
        def bad_pending(f):
            f['initial']['fishr_calibration']['status']='FIXED_CALIBRATION_COMPLETE'
        def unrelated_resolved(f):
            f['initial']['commit']='different-initial-commit'
        def final_resolved(f):
            f['final_resolved']['commit']='different-final-commit'
        def cap_over_bound(f):
            cap=f['steps'][447]['details']['joint_gradient_cap'];cap['groups']['shallow']['capped_R']=.5
        def weighted_mismatch(f):
            metrics=f['steps'][447]['metrics'];metrics['weighted_fishr']+=.1;metrics['total_loss']+=.1
        def forged_bound(f):
            item=f['steps'][447];cap=item['details']['joint_gradient_cap'];cap['scalar']=.8
            for group in cap['groups'].values():group.update(capped_R=.8,norm_bound=1.,scalar_bound=.8)
            cap['effective_weighted_fishr']=cap['raw_weighted_fishr']*.8
            old=item['metrics']['weighted_fishr'];item['metrics']['weighted_fishr']=cap['effective_weighted_fishr']
            item['metrics']['total_loss']+=item['metrics']['weighted_fishr']-old
            item['details']['fishr_budget'].update(effective_weighted_loss=item['metrics']['weighted_fishr'],joint_cap_scalar=.8)
        def effective_metadata(f):f['steps'][447]['details']['joint_gradient_cap']['effective_weighted_fishr']+=.1
        def raw_budget_mismatch(f):f['steps'][447]['details']['fishr_budget']['weighted_loss']+=.1
        def incomplete_coverage(f):f['steps'][447]['details']['joint_gradient_cap']['groups']['rest']['parameter_scalars']=0
        cases=dict(wrong_calibration_time=wrong_time,wrong_pending_status=bad_pending,unrelated_initial_resolved=unrelated_resolved,
            inconsistent_final_resolved=final_resolved,cap_over_bound=cap_over_bound,actual_weighted_mismatch=weighted_mismatch,
            forged_cap_bound=forged_bound,effective_cap_metadata=effective_metadata,
            raw_budget_mismatch=raw_budget_mismatch,incomplete_parameter_coverage=incomplete_coverage)
        for name,mutate in cases.items():
            bad=deepcopy(pristine);mutate(bad)
            try:analyze_fixture(base,bad)
            except ValueError as exc:rejections[name]=str(exc)
            else:missed.append(name)
    return dict(status='PASS' if not missed else 'FAILED',synthetic_only=True,remote_access=False,target_access=False,GPU_used=False,
        complete_4440_step_policies=passed,total_positive_steps=4440*len(passed),absent_source_partial=True,
        closed_source_sentinel_ignored=True,rejected_cases=rejections,missed_cases=missed)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('local_artifacts/calibrated_analysis_checks.json'))
    args=parser.parse_args();result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
    if result['status']!='PASS':raise SystemExit(1)
