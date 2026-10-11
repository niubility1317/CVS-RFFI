"""Full closed source-log analysis; prediction, scoring and truth are never opened."""
import argparse
import csv
import gzip
import io
import json
import math
import re
import statistics
import time
from pathlib import Path


class Moments:
    def __init__(self):
        self.values=[]
    def add(self,value):
        if value is None:return
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Nonfinite/non-numeric observation')
        self.values.append(float(value))
    def result(self):
        values=self.values;n=len(values)
        if not n:return dict(observations=0,mean=None,min=None,max=None,first=None,last=None)
        ordered=sorted(values)
        def quantile(p):
            position=(n-1)*p;lo=int(position);hi=math.ceil(position)
            return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
        return dict(observations=n,mean=math.fsum(values)/n,min=ordered[0],max=ordered[-1],
                    first=values[0],last=values[-1],positive=sum(v>0 for v in values),
                    zero=sum(v==0 for v in values),negative=sum(v<0 for v in values),
                    q10=quantile(.1),median=quantile(.5),q90=quantile(.9))


def _add(mapping,key,value):
    mapping.setdefault(key,Moments()).add(value)


def _finish(mapping):
    return {key:value.result() for key,value in mapping.items()}


def _finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(_finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(_finite(v) for v in value)
    return True


def _read_json(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    if not _finite(value):raise ValueError('Nonfinite JSON: '+str(path))
    return value


def _read_jsonl(path):
    rows=[]
    with path.open(encoding='utf-8') as stream:
        for index,line in enumerate(stream,1):
            if not line.strip():raise ValueError('Blank structured record '+str(path)+':'+str(index))
            item=json.loads(line)
            if not _finite(item):raise ValueError('Nonfinite record '+str(path)+':'+str(index))
            rows.append(item)
    return rows


def _expected_counts(c):
    n=c['updates'];active=n-c['semi']['warmup_updates'];mc=c['multilevel']
    aux=n//mc['interval']-c['semi']['warmup_updates']//mc['interval'] if mc['enabled'] else 0
    return dict(identity_updates=n,L_packets=n*c['batch_size'],U_packets=active*c['batch_size'] if c['unlabeled'] else 0,
                weak_forwards=active if c['unlabeled'] else 0,physical_L=0,physical_U=0,feature_U=0,margin_U=0,
                EMA_updates=0,satellite_calls=0,auxiliary_steps=aux,auxiliary_L=aux*mc['batch_size'],
                auxiliary_forwards=aux*2,
                fishr_steps=aux if mc['fishr_mode']!='none' else 0,mixstyle_steps=active if mc['mixstyle_l'] else 0,
                style_L_packets=active*c['batch_size'] if mc['mixstyle_l'] else 0,
                style_search_forwards=active*5 if mc['style_selection'] in ('random','hard') else 0,
                style_search_L=active*c['batch_size']*5 if mc['style_selection'] in ('random','hard') else 0,
                mixstyle_U_steps=active if mc['mixstyle_u'] else 0)


def _source_stdout(path):
    """The worker reuses stdout for target evaluation: STOP at source-end marker.

    This reads every source-phase line, but deliberately never reads the later
    prediction/scoring segment of the shared log, even if it already exists.
    """
    result=dict(path=str(path),source_lines=0,source_utf8_bytes=0,closed_source_segment=False,
                errors=[],warnings=[],nonfinite_markers=[],resolved=[],calibration_ready=[],epochs=[],warmup=[],source_trained=None,
                target_segment_read=False)
    if not path.is_file():return result
    with path.open(encoding='utf-8',errors='strict') as stream:
        for line in stream:
            result['source_lines']+=1;result['source_utf8_bytes']+=len(line.encode('utf-8'))
            if re.search(r'Traceback|CUDA out of memory|FloatingPointError|RuntimeError|ValueError|\bKilled\b',line):
                result['errors'].append(dict(line=result['source_lines'],text=line.rstrip()))
            if re.search(r'\b(?:Warning|warning|UserWarning|RuntimeWarning)\b',line):
                result['warnings'].append(dict(line=result['source_lines'],text=line.rstrip()))
            if re.search(r'(?<![A-Za-z_])(?:NaN|nan|Infinity|inf)(?![A-Za-z_])',line):
                result['nonfinite_markers'].append(dict(line=result['source_lines'],text=line.rstrip()))
            for prefix,key in [('RESOLVED ','resolved'),('CALIBRATION_READY ','calibration_ready'),('EPOCH ','epochs'),('HEAD_WARMUP ','warmup')]:
                if line.startswith(prefix):result[key].append(json.loads(line[len(prefix):]))
            if line.startswith('SOURCE_TRAINED '):
                result['source_trained']=json.loads(line[len('SOURCE_TRAINED '):])
                result['closed_source_segment']=True
                break
    return result


def _style_add(target,info,packets):
    target.setdefault('calls',0);target['calls']+=1
    target.setdefault('enabled_calls',0);target['enabled_calls']+=int(info['enabled'])
    target.setdefault('packet_layer_occurrences',0);target['packet_layer_occurrences']+=packets
    for key in ('eligible_packets','changed_packets'):
        target.setdefault(key,0);target[key]+=int(info.get(key,0))
    target.setdefault('stats',{})
    for key,value in info.items():
        if isinstance(value,(int,float)):_add(target['stats'],key,value)
    if not 0<=info.get('changed_packets',0)<=info.get('eligible_packets',0)<=packets:
        raise ValueError('Invalid actual style coverage')
    donors=info.get('donors',[])
    if donors and (len(donors)!=packets or any(not 0<=j<packets for j in donors)):
        raise ValueError('Invalid logged donor occurrence index')


def _style_finish(value):
    result={k:v for k,v in value.items() if k!='stats'}
    result['metrics']=_finish(value.get('stats',{}))
    count=result.get('packet_layer_occurrences',0)
    result['eligible_fraction']=result.get('eligible_packets',0)/count if count else None
    result['changed_fraction']=result.get('changed_packets',0)/count if count else None
    result['conditional_changed_fraction']=result.get('changed_packets',0)/result['eligible_packets'] if result.get('eligible_packets') else None
    return result


def _check_warmup(source,c,stdout):
    summary=_read_json(source/'head_warmup.json');records=_read_jsonl(source/'head_warmup.jsonl')
    if c['multilevel']['statistic'] not in ('none','margin'):
        raise ValueError('Scratch matrix contains no learned probe heads')
    expected_steps=0
    if len(records)!=expected_steps or summary['optimizer_steps']!=expected_steps:
        raise ValueError('Incomplete head warmup')
    if not summary['base_state_unchanged'] or summary['backbone_updates']!=0 or summary['auxiliary_inheritance']:
        raise ValueError('Warmup changed backbone or inherited auxiliary state')
    if summary['source_L_cache_packets']!=(6300 if expected_steps else 0) or summary['head_packet_occurrences']!=expected_steps*60:
        raise ValueError('Warmup packet budget differs')
    if summary['cache_forward_calls']!=(50 if expected_steps else 0):raise ValueError('Warmup cache coverage differs')
    stats={}
    for i,row in enumerate(records,1):
        if row['step']!=i or row['source_L_packets']!=60 or row['backbone_updates'] or row['lr']!=.003 or row['weight_decay']!=.0001:
            raise ValueError('Warmup step/optimizer protocol differs')
        for key,value in row.items():
            if isinstance(value,(int,float)):_add(stats,key,value)
    printed=[records[i-1] for i in (1,50,100,150,200)] if expected_steps else []
    if stdout['warmup']!=printed:raise ValueError('Warmup stdout/structured record mismatch')
    return dict(summary=summary,records=records,statistics=_finish(stats),
                backbone_unchanged_evidence='warmup runtime exact state_dict comparison; retained summary true; no extra checkpoint load')


def _check_calibration(source,c):
    """Read the one initial L-only calibration; never infer it from training."""
    value=_read_json(source/'calibration.json');mc=c['multilevel']
    if (value.get('after_identity_updates')!=mc['calibration_after_updates']
            or value.get('scheduled_after_identity_updates')!=mc['calibration_after_updates']
            or value.get('U_access') is not False or value.get('V_access') is not False
            or not value.get('parameters_buffers_unchanged') or not value.get('modes_preserved')
            or not value.get('python_numpy_RNG_preserved')):
        raise ValueError('Calibration timing or state preservation differs')
    if value.get('state_unchanged') is not True:
        raise ValueError('Calibration changed model state')
    if not mc['enabled']:
        if (value.get('status')!='NOT_APPLICABLE' or value.get('pairs')!=0
                or value.get('extra_source_readouts')!=0 or value.get('fixed_coefficients')!={}):
            raise ValueError('Inactive arm performed Fishr calibration')
        return value
    expected=dict(status='FIXED_CALIBRATION_COMPLETE',pairs=12,extra_source_readouts=1440,
                  model_forwards=24,RNG_preserved=True,source_role='L_s only',target_access=False,
                  identity_updates=0,head_updates=0,all_layers_calibrated=True)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Fixed source calibration budget/permission differs')
    margin=mc['statistic']=='margin'
    if not margin:raise ValueError('Only own-margin calibration is registered for scratch students')
    if margin and (value.get('calibration_reference')!='own_margin_term' or value.get('auxiliary_heads')!=0):
        raise ValueError('Margin calibration must use its own term and zero auxiliary heads')
    levels=('time_down','t2.pool');records=value['records']
    if (len(records)!=12 or [r['pair'] for r in records]!=list(range(12))
            or [r['day'] for r in records]!=[1,2,3]*4
            or set(value['fixed_coefficients'])!=set(levels) or set(value['levels'])!=set(levels)):
        raise ValueError('Incomplete fixed twelve-pair/two-layer calibration')
    for row in records:
        if set(row['levels'])!=set(levels) or not 0<=row['cross_batch_PID_overlap']<=60:
            raise ValueError('Calibration paired source support differs')
    for level in levels:
        summary=value['levels'][level]
        ce=math.fsum(r['levels'][level]['main_ce_gradient_norm'] for r in records)/12
        reference_key='term_gradient_norm' if margin else 'old_pair_gradient_norm'
        summary_key='mean_term_gradient_norm' if margin else 'mean_old_pair_gradient_norm'
        old=math.fsum(r['levels'][level][reference_key] for r in records)/12
        if min(ce,old)<=mc['fishr_eps']:
            raise ValueError('Invalid calibration reference gradient norm')
        raw=mc['fishr_ratio']*ce/(old+mc['fishr_eps']);coefficient=min(mc['fishr_cap'],raw)
        for key,want in (('mean_CE_gradient_norm',ce),(summary_key,old),
                         ('uncapped_coefficient',raw),('coefficient',coefficient)):
            if not math.isclose(summary[key],want,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Calibration coefficient/recount differs: '+level+'/'+key)
        if (summary['cap_active']!=(raw>mc['fishr_cap'])
                or not math.isclose(value['fixed_coefficients'][level],coefficient,rel_tol=1e-10,abs_tol=1e-12)):
            raise ValueError('Fixed calibration cap differs')
    return value


def _style_search(style,c,step,ramp):
    """Validate one compact actual search/replay record using source scalars only."""
    mc=c['multilevel'];selection=mc['style_selection'];search=selection in ('hard','random')
    expected=dict(role='L_style',source_role='L_s',style_selection=selection,style_input_valid=True,
                  target_access=False,V_access=False,U_access=False,U_truth=False,candidates=4,
                  batch_occurrences=c['batch_size'],p=1.,alpha=.1,
                  eval_readouts=5 if search else 0,candidate_eval_forwards=4 if search else 0,
                  eval_forwards=5 if search else 0,train_forwards=1,total_extra_forwards=6 if search else 1,
                  candidate_state_unchanged=True,candidate_rng_unchanged=True,
                  original_training_flags_restored=True,main_rng_restored=True,
                  dropout_same_mask_requested=True,normal_training_dropout=True,
                  selection_dropout_free=True,tie_rule='smallest style index; style wins equality with clean',
                  clean_candidate_included=True,candidate_pool_size=5,
                  train_diff_is_not_eval_selection_causal_effect=True)
    if any(style.get(k)!=v for k,v in expected.items()):raise ValueError('L search permission/state/RNG/cost differs')
    directions=[1,1,1,1] if mc['style_pool']=='POS' else [1,-1,1,-1]
    if (style.get('style_transform')!='bounded_logstats_v1' or style.get('style_pool')!=mc['style_pool']
            or style.get('candidate_directions')!=directions):
        raise ValueError('Bounded style pool or directions differ')
    if not math.isclose(style['eta'],ramp,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Style search strength differs')
    levels=['time_down','t2.pool'] if search and not style['identity_selected'] else []
    if style['selected_levels']!=levels or style['actual_hook_order']!=levels or set(style['levels'])!=set(levels):
        raise ValueError('Selected candidate native hook order differs')
    if not 1<=style['distinct_physical_ids']<=c['batch_size'] or style['duplicate_occurrences']!=c['batch_size']-style['distinct_physical_ids']:
        raise ValueError('Style physical occurrence count differs')
    stats={}
    if search:
        losses=style['candidate_losses'];chosen=style['selected_index'];seeds=style['candidate_seeds']
        expected_seeds=[(c['model_seed']+step*1000003+i*104729)%2**63 for i in range(4)]
        if len(losses)!=4 or seeds!=expected_seeds or type(chosen) is not int or not -1<=chosen<4:
            raise ValueError('Candidate pool/private seed/selection differs')
        clean=style['eval_clean'];selected=style['eval_selected']
        winner=max(range(4),key=lambda i:losses[i])
        identity=clean['CE']>losses[winner]
        expected_chosen=-1 if identity else winner
        if (chosen!=expected_chosen or style['identity_selected'] is not identity
                or style['style_winner_index']!=winner
                or style['selected_seed']!=(None if identity else seeds[winner])):
            raise ValueError('Clean-inclusive maximum/tie/seed selection differs')
        if identity and selected!=clean:raise ValueError('Identity candidate readout differs from clean')
        if selected['CE']!=max(clean['CE'],losses[winner]):raise ValueError('Selected CE below five-candidate maximum')
        stats['identity_selected']=float(identity)
        direction=None if identity else directions[chosen]
        if style.get('selected_direction')!=direction:raise ValueError('Selected direction differs')
        stats['negative_selected']=float(direction==-1)
        stats['positive_selected']=float(direction==1)
        for record in (clean,selected):
            if set(record)!={'CE','margin','accuracy'} or record['CE']<0 or not 0<=record['accuracy']<=1:
                raise ValueError('Invalid actual L candidate readout')
        stats.update(eval_clean_CE=clean['CE'],eval_selected_CE=selected['CE'],
                     eval_CE_delta=selected['CE']-clean['CE'],
                     eval_clean_margin=clean['margin'],eval_selected_margin=selected['margin'],
                     eval_margin_delta=selected['margin']-clean['margin'],
                     eval_clean_accuracy=clean['accuracy'],eval_selected_accuracy=selected['accuracy'],
                     eval_accuracy_delta=selected['accuracy']-clean['accuracy'],
                     candidate_CE_mean=statistics.mean(losses),candidate_CE_min=min(losses),
                     candidate_CE_max=max(losses),candidate_CE_spread=max(losses)-min(losses),
                     selected_minus_candidate_mean=selected['CE']-statistics.mean(losses))
        for i,value in enumerate(losses):stats['candidate_'+str(i)+'_CE']=value
        for i,level in enumerate(levels):
            info=style['levels'][level]
            if (info.get('transform')!='bounded_logstats_v1' or info.get('direction')!=direction
                    or info.get('moments_detached') is not True
                    or not 0<=info['mean_shift_in_sigma_max']<=ramp+1e-6
                    or not math.exp(-ramp*math.log(2.))-1e-6<=info['scale_ratio_min']<=info['scale_ratio_max']<=math.exp(ramp*math.log(2.))+1e-6
                    or not 0<=info['mean_clipped_fraction']<=1
                    or not 0<=info['log_scale_clipped_fraction']<=1):
                raise ValueError('Bounded direction or realized statistics violate registration')
            if (not info['donor_rules_verified'] or info['seed']!=(style['selected_seed']+i*1709)%2**63
                    or not info['input_requires_grad'] or not info['output_requires_grad']
                    or not math.isclose(info['eta'],ramp,rel_tol=1e-12,abs_tol=1e-12)
                    or not info['enabled']):
                raise ValueError('Selected differentiable candidate donor/seed/strength differs')
    elif (style['candidate_losses'] or style['candidate_seeds'] or style['selected_index'] is not None
          or style['selected_seed'] is not None or style['eval_clean'] is not None or style['eval_selected'] is not None):
        raise ValueError('Replay unexpectedly evaluated or selected style candidates')
    if style['changed_packets']!=sum(v['changed_packets'] for v in style['levels'].values()):
        raise ValueError('Actual selected style packet-layer sum differs')
    return stats


def analyze_row(base,registered,c):
    rid=registered['row_id'];source=base/rid/'source'
    base_result=dict(row_id=rid,arm=registered['arm'],seed=registered['model_seed'])
    if not (source/'completion.json').is_file():
        return dict(base_result,status='PARTIAL',reason='source completion absent; live gzip not opened')
    stdout=_source_stdout(base/'logs'/(rid+'.log'))
    if not stdout['closed_source_segment']:
        return dict(base_result,status='PARTIAL',reason='source stdout not closed by SOURCE_TRAINED; gzip not opened',
                    source_stdout_lines_seen=stdout['source_lines'])
    done=_read_json(source/'completion.json');resolved=_read_json(source/'resolved_config.json')
    initialization=_read_json(source/'initialization.json');frozen=_read_json(base/rid/'source_frozen.json')
    if done['config']!=c or frozen['config']!=c or any(resolved.get(k)!=v for k,v in c.items()):
        raise ValueError('Actual configuration differs from fixed preregistration')
    if done['status']!='SOURCE_TRAINED' or done['target_access'] or done['target_evaluated'] or done['epoch']!=20 or done['total_epochs']!=20:
        raise ValueError('Source completion or budget contract differs')
    if (frozen['status']!='SOURCE_FROZEN' or frozen['target_access'] or frozen['target_scores_consumed']
            or frozen['total_epochs']!=20 or frozen['parent_checkpoint'] is not None):
        raise ValueError('Source freeze differs')
    if (c['model_initialization']!='scratch' or c['parent_checkpoint'] is not None or c['initialization_checkpoint_sources']
            or initialization!=done['initialization'] or initialization['scratch_only'] is not True
            or initialization['scratch_scope']!='student_identity_only' or initialization['reference_E20_loaded']
            or initialization['initialization_checkpoint_sources'] or initialization['parent_checkpoint'] is not None or initialization['parent_training_epochs']!=0
            or initialization['incremental_epochs']!=20 or initialization['total_epochs']!=20):
        raise ValueError('Student has inherited weights or non-scratch budget/provenance')
    proof=initialization['parent_provenance']
    expected_student=dict(status='SCRATCH_INITIALIZATION',seed=c['model_seed'],checkpoint=None,parent_checkpoint=None,
                          initialization_checkpoint_sources=[],ancestors=[],scratch_only=True,scratch_scope='student_identity_only',
                          source_roles='EXACT_MATCH',training_target_access=False,reference_E20_loaded=False,
                          parent_training_epochs=0,auxiliary_inheritance=False)
    if any(proof.get(k)!=v for k,v in expected_student.items()):raise ValueError('Random student initialization proof differs')
    if initialization['source_roles']!='EXACT_MATCH' or initialization['target_contact'] or initialization['auxiliary_inheritance']:
        raise ValueError('Invalid inherited source provenance')
    teachers=initialization.get('teacher_provenance',{})
    if set(teachers)!={str(s) for s in c['teacher']['seeds']}:raise ValueError('Teacher provenance keys differ')
    expected_sources=[]
    for seed in c['teacher']['seeds']:
        proof=teachers[str(seed)];path=c['reference_checkpoint'].rsplit('/',3)[0]+f'/E20-s{seed}/source/final_ssdg.pth'
        expected_sources.append(path)
        expected=dict(status='VERIFIED',seed=seed,checkpoint=path,epoch=20,scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',training_target_access=False)
        if any(proof.get(k)!=v for k,v in expected.items()) or seed==c['model_seed']:raise ValueError('Invalid teacher source proof')
    if (c['checkpoint_sources']!=expected_sources or initialization['checkpoint_sources']!=expected_sources or initialization['ancestors']!=expected_sources
            or done['checkpoint_sources']!=expected_sources or done['teacher_provenance']!=teachers
            or done['parent_checkpoint'] is not None
            or initialization['source_probability_cache_used']!=bool(c['teacher']['seeds'])):
        raise ValueError('Pretrained source-U teacher ancestry differs from the no-parent student')
    costs=done['resources']
    if (costs['parent_training_epochs']!=0 or costs['incremental_epochs']!=20 or costs['total_epochs']!=20
            or costs['parent_training_cost_reused'] or costs['teacher_training_epochs_reused']!=20*len(teachers)
            or costs['historical_model_epochs_including_teachers']!=20+20*len(teachers)):
        raise ValueError('Scratch student/teacher historical training cost differs')
    initial_resolved=_read_json(source/'initial_resolved_config.json')
    ready=_read_json(source/'calibration_ready.json')
    final_resolved=_read_json(source/'resolved_config_final.json')
    if (stdout['calibration_ready']!=[ready] or ready['resolved']!=resolved or final_resolved!=resolved
            or ready['after_identity_updates']!=c['multilevel']['calibration_after_updates']
            or ready['calibration']!=resolved['fishr_calibration']):
        raise ValueError('Calibration-ready and final resolved evidence differ')
    initial_status='PENDING_CALIBRATION' if c['multilevel']['calibration_after_updates'] else 'FIXED_CALIBRATION_COMPLETE'
    if initial_resolved['fishr_calibration']['status']!=initial_status:
        raise ValueError('Initial calibration lifecycle differs')
    if {k:v for k,v in initial_resolved.items() if k!='fishr_calibration'}!={k:v for k,v in resolved.items() if k!='fishr_calibration'}:
        raise ValueError('Calibration changed unrelated resolved configuration')
    if stdout['resolved']!=[initial_resolved] or stdout['source_trained']!=done:
        raise ValueError('Full source stdout differs from committed artifacts')
    warmup=_check_warmup(source,c,stdout)
    if resolved['head_warmup']!=warmup['summary'] or done['resources']['head_warmup']!=warmup['summary']:
        raise ValueError('Warmup cost/summary mismatch')
    calibration=_check_calibration(source,c)
    if resolved['fishr_calibration']!=calibration or done['resources']['fishr_calibration']!=calibration:
        raise ValueError('Fixed calibration artifact/resolved/resource mismatch')
    execution=resolved['execution']
    if (execution['EMA'] or execution['satellite_generator'] or not execution['U_truth_hidden']
            or execution['U_iterator']!=c['unlabeled'] or execution['auxiliary_CE_to_identity']
            or execution['Fishr_levels']!=c['multilevel']['fishr_levels']
            or execution['auxiliary_neural_networks'] or not execution['scratch'] or execution['E20_weights_loaded']
            or execution['E200_weights_loaded']
            or execution['sensitivity_probes_trainable']):
        raise ValueError('Actual execution permissions/gradient route differ')
    if done['resources']['auxiliary_trainable_parameters']!=0:
        raise ValueError('Head-free arm unexpectedly trained auxiliary parameters')
    fixed=_expected_counts(c);counts={k:0 for k in fixed};counts.update(mixed_U_packets=0,mixed_L_packets=0)
    per_epoch={};metric_stats={};detail_stats={};first={};gradients={};active_gradients={};gradient_ratios={}
    audit_records=[];layers={};layer_records=[];l_style={};style_identity={};style_search={};style_records=[]
    selected_indices={str(i):0 for i in range(-1,4)};budgets={};combined={};budget_snapshots=[];cap_records=[];cap_stats={};epochs_running=[]
    count=0;step_file=source/'step_metrics.jsonl.gz'
    stable_before=(step_file.stat().st_size,step_file.stat().st_mtime_ns)
    # Completion and full source-end markers prove the gzip writer has closed.
    # Reading to EOF also checks the gzip footer/CRC; truncated files are errors.
    with gzip.open(step_file,'rt',encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line);count+=1
            if item['step']!=count or item['epoch']!=(count-1)//222+1 or not _finite(item):
                raise ValueError('Nonconsecutive/nonfinite full source step '+str(count))
            ep=item['epoch'];metrics=item['metrics'];detail=item['details']
            aggregate=per_epoch.setdefault(ep,dict(steps=0,sums={},metric_counts={},style_L={},style_identity={},style_search={},budget={}))
            aggregate['steps']+=1
            for key,value in metrics.items():
                if isinstance(value,(int,float)):
                    _add(metric_stats,key,value);aggregate['sums'][key]=aggregate['sums'].get(key,0.)+value
                    aggregate['metric_counts'][key]=aggregate['metric_counts'].get(key,0)+1
            counts['identity_updates']+=1;counts['L_packets']+=c['batch_size']
            for key in counts:
                if key in ('identity_updates','L_packets','EMA_updates','satellite_calls'):continue
                counts[key]+=detail[key]
                if detail[key] and key not in first:first[key]=count
            for key,value in detail.items():
                if isinstance(value,(int,float)):_add(detail_stats,key,value)
            active=count>c['semi']['warmup_updates']
            if detail['U_packets']!=(c['batch_size'] if active and c['unlabeled'] else 0):raise ValueError('U execution schedule differs')
            if detail.get('teacher_source')!=c['teacher']['source'] or detail.get('teacher_selection')!=c['teacher']['selection'] or detail.get('teacher_forwards')!=0:
                raise ValueError('Teacher source/selection/forward execution differs')
            if not math.isclose(metrics['weighted_KL'],metrics['weighted_KL_same']+metrics['weighted_KL_disagree'],rel_tol=1e-5,abs_tol=1e-7):
                raise ValueError('Agreement/disagreement KL decomposition differs')
            components=[metrics['weighted_'+k] for k in ('ce','KL','feature','margin','style_ce','aux_ce','fishr')]
            expected_total=sum(components)
            # Signed cross terms can cancel CE; bound float32 summation error
            # by component magnitudes rather than the near-zero final loss.
            if not math.isclose(metrics['total_loss'],expected_total,rel_tol=1e-5,
                                abs_tol=max(1e-7,2e-6*sum(abs(v) for v in components))):
                raise ValueError('Loss double counts audit-only KL')
            if detail['U_packets']:
                if detail['KL_full_batch_denominator']!=c['batch_size']:raise ValueError('KL denominator differs')
                if c['teacher']['selection']=='all' and detail['accepted_U']!=c['batch_size']:raise ValueError('ALL drops source U samples')
                if detail['accepted_agreement']+detail['accepted_disagreement']!=detail['accepted_U']:raise ValueError('Accepted U decomposition differs')
            expected_aux=int(active and c['multilevel']['enabled'] and count%c['multilevel']['interval']==0)
            if detail['auxiliary_steps']!=expected_aux:raise ValueError('Auxiliary execution schedule differs')
            if detail['U_accuracy'] is not None:raise ValueError('U identity accuracy must remain hidden')
            if (detail['mixstyle_U_steps'] or detail['mixed_U_packets'] or detail.get('U_style')):
                raise ValueError('This matrix forbids U style execution')
            expected_style=int(active and c['multilevel']['mixstyle_l'])
            if detail['mixstyle_steps']!=expected_style or detail['style_L_packets']!=expected_style*c['batch_size']:
                raise ValueError('Main labeled style schedule differs')
            search_forwards=5*expected_style if c['multilevel']['style_selection'] in ('random','hard') else 0
            if detail['style_search_forwards']!=search_forwards or detail['style_search_L']!=search_forwards*c['batch_size']:
                raise ValueError('Actual clean+4-candidate source search cost differs')
            audit=item.get('gradient_audit')
            if bool(audit)!=(count%444==0):raise ValueError('Missing/unexpected scheduled gradient audit')
            if audit:
                if not c['multilevel']['aux_ce_to_backbone'] and any(v!=0. for k,v in audit['weighted_dense_gradient_norms']['aux_ce'].items() if k!='multilevel_heads'):
                    raise ValueError('Detached auxiliary CE gradient reached identity')
                audit_records.append(dict(step=count,epoch=ep,audit=audit))
                for loss,groups in audit['weighted_dense_gradient_norms'].items():
                    for group,value in groups.items():
                        _add(gradients,loss+'/'+group,value)
                        if active:_add(active_gradients,loss+'/'+group,value)
                        reference=audit['weighted_dense_gradient_norms']['ce'].get(group)
                        if reference is not None and reference>0:
                            _add(gradient_ratios,loss+'/'+group,value/reference)
            multi=detail.get('multilevel')
            if bool(multi)!=bool(expected_aux):raise ValueError('Missing/unexpected paired auxiliary details')
            if multi:
                margin=c['multilevel']['statistic']=='margin'
                expected_estimator='finite_L_corrected_cross_AB_margin_sensitivity' if margin else c['multilevel']['fishr_estimator']
                if multi['aux_ce_to_backbone']!=c['multilevel']['aux_ce_to_backbone'] or multi['fishr_levels']!=c['multilevel']['fishr_levels']:
                    raise ValueError('Actual multilevel gradient route differs')
                if (multi['auxiliary_packets']!=120 or multi['aux_forwards']!=2
                        or multi['pair_A_distinct_PID']!=60 or multi['pair_B_distinct_PID']!=60
                        or not 0<=multi['cross_batch_PID_overlap']<=60
                        or multi['distinct_PID']!=120-multi['cross_batch_PID_overlap']
                        or set(multi['source_RX'])!={1,3,4,6,8} or multi['day'] not in (1,2,3)
                        or not multi['independent_AB'] or multi['calibration_sampling']
                        or multi['estimator']!=expected_estimator
                        or multi['head_Fishr_gradient'] or multi['EMA']):
                    raise ValueError('Auxiliary source support differs')
                if margin and (multi['auxiliary_heads']!=0 or multi['statistic']!='margin'
                               or multi['aux_ce']!=0 or metrics['aux_ce']!=0 or metrics['weighted_aux_ce']!=0
                               or multi['sensitivity_eps']!=1e-6 or not multi['native_CPU_CUDA_RNG_preserved']):
                    raise ValueError('Head-free main-margin gradient/normalization route differs')
                layer_records.append(dict(step=count,epoch=ep,statistic=c['multilevel']['statistic'],details=multi))
                for level,info in multi['levels'].items():
                    stats=layers.setdefault(level,{})
                    for key,value in info.items():
                        if isinstance(value,(int,float)):_add(stats,key,value)
                    if info['fishr_computed']!=(level in c['multilevel']['fishr_levels']):
                        raise ValueError('Computed Fishr layer differs')
                    if margin and info['fishr_computed']:
                        if info['estimator']!=expected_estimator:raise ValueError('Layer margin estimator differs')
                        for split in ('A','B'):
                            sensitivity=info['sensitivity_'+split]
                            if (not sensitivity['score_requires_grad'] or sensitivity['sensitivity_eps']!=1e-6
                                    or not 0<=sensitivity['low_norm_rate']<=1
                                    or sensitivity['raw_sensitivity_rms_min']<0
                                    or sensitivity['coordinates']!=2*sensitivity['shape'][1]):
                                raise ValueError('Invalid live normalized margin-sensitivity diagnostic')
                            for key,value in sensitivity.items():
                                if isinstance(value,(int,float)):_add(stats,'sensitivity_'+split+'/'+key,value)
            style=detail.get('L_style')
            if bool(style)!=bool(expected_style):raise ValueError('Missing/unexpected main labeled style details')
            if style:
                search_values=_style_search(style,c,count,detail['ramp'])
                style_records.append(dict(step=count,epoch=ep,details=style))
                for key,value in search_values.items():
                    _add(style_search,key,value);_add(aggregate['style_search'],key,value)
                if style['selected_index'] is not None:selected_indices[str(style['selected_index'])]+=1
                changed=0
                for level,info in style['levels'].items():
                    if not info['donor_rules_verified']:
                        raise ValueError('L style donor audit differs')
                    _style_add(l_style.setdefault(level,{}),info,c['batch_size'])
                    _style_add(aggregate['style_L'].setdefault(level,{}),info,c['batch_size'])
                    changed+=info['changed_packets']
                if changed!=detail['mixed_L_packets'] or changed!=style['changed_packets']:raise ValueError('L packet-layer count mismatch')
                for key in ('L_accuracy','style_L_accuracy','style_prediction_change','clean_identity_margin','styled_identity_margin'):
                    _add(style_identity,key,detail[key]);_add(aggregate['style_identity'],key,detail[key])
                _add(style_identity,'style_accuracy_delta',detail['style_L_accuracy']-detail['L_accuracy'])
                _add(style_identity,'style_margin_delta',detail['styled_identity_margin']-detail['clean_identity_margin'])
                for key in ('style_ce','weighted_style_ce'):
                    _add(style_identity,key,metrics[key]);_add(aggregate['style_identity'],key,metrics[key])
            budget=detail.get('fishr_budget')
            if bool(budget)!=bool(expected_aux):raise ValueError('Missing/unexpected fixed Fishr weights')
            if budget and budget['levels']:
                if budget['dynamic_rescaling'] or budget['fixed_denominator']!=2 or budget['EMA']:
                    raise ValueError('Fishr weights adapted after initial calibration')
                budget_snapshots.append(dict(step=count,epoch=ep,**budget))
                for level,info in budget['levels'].items():
                    calibrated=calibration['fixed_coefficients'][level]
                    coefficient=calibrated*detail['ramp'] if level in c['multilevel']['fishr_levels'] else 0.
                    if (not math.isclose(info['fixed_calibrated_coefficient'],calibrated,rel_tol=1e-12,abs_tol=1e-12)
                            or not math.isclose(info['coefficient'],coefficient,rel_tol=1e-10,abs_tol=1e-12)):
                        raise ValueError('Training Fishr coefficient differs from fixed initial value')
                    record=budgets.setdefault(level,dict(calls=0,cap_count=0,skip_reasons={},stats={}))
                    record['calls']+=1;record['cap_count']+=int(info['cap_active'])
                    if info['reason']:record['skip_reasons'][info['reason']]=record['skip_reasons'].get(info['reason'],0)+1
                    for key,value in info.items():
                        if isinstance(value,(int,float)):
                            _add(record['stats'],key,value)
                            _add(aggregate['budget'].setdefault(level,{}),key,value)
                    if not info['coefficient_detached']:raise ValueError('Fishr coefficient not detached')
                for level,info in budget['combined_group_gradient_ratios'].items():
                    for key,value in info.items():_add(combined,level+'/'+key,value)
            cap=detail['joint_gradient_cap']
            want_cap=bool(expected_aux and c['multilevel']['joint_gradient_cap']['enabled'])
            if bool(cap['applied'])!=want_cap:raise ValueError('Joint cap execution schedule differs')
            if want_cap:
                if (not cap['scalar_detached'] or not cap['complete_disjoint_parameter_coverage']
                        or cap['gradient_evaluations']!=2 or set(cap['groups'])!={'shallow','middle','rest'}
                        or not 0<=cap['scalar']<=1 or cap['rho']!=.2 or cap['eps']!=1e-12):
                    raise ValueError('Joint cap definition or parameter coverage differs')
                if not math.isclose(cap['raw_weighted_fishr']*cap['scalar'],metrics['weighted_fishr'],rel_tol=1e-5,abs_tol=1e-7):
                    raise ValueError('Effective Fishr differs from detached joint scaling')
                if (not math.isclose(cap['effective_weighted_fishr'],metrics['weighted_fishr'],rel_tol=1e-5,abs_tol=1e-7)
                        or not math.isclose(cap['raw_weighted_fishr'],budget['weighted_loss'],rel_tol=1e-5,abs_tol=1e-7)
                        or sum(v['parameter_scalars'] for v in cap['groups'].values())!=resolved['trainable_parameters']):
                    raise ValueError('Joint cap raw/effective loss or parameter coverage differs')
                scale_expected=min([1.]+[.2*v['norm_id']/(v['raw_R']+1e-12) for v in cap['groups'].values() if v['raw_R']>0])
                if not math.isclose(cap['scalar'],scale_expected,rel_tol=1e-10,abs_tol=1e-12):
                    raise ValueError('Joint cap scalar differs from measured group norms')
                for group,values in cap['groups'].items():
                    if min(values['norm_id'],values['raw_R'],values['capped_R'],values['norm_bound'])<0 or not math.isclose(values['norm_bound'],.2*values['norm_id'],rel_tol=1e-10,abs_tol=1e-12):
                        raise ValueError('Joint cap bound differs from rho times identity norm')
                    if values['capped_R']>values['norm_bound']+1e-7*max(1.,values['norm_bound']):
                        raise ValueError('Actual joint auxiliary gradient exceeds group bound')
                    if not math.isclose(values['capped_R'],values['raw_R']*cap['scalar'],rel_tol=1e-6,abs_tol=1e-9):
                        raise ValueError('Capped group norm differs from scalar')
                    for key in ('norm_id','raw_R','capped_R','norm_bound'):_add(cap_stats,group+'/'+key,values[key])
                cap_records.append(dict(step=count,epoch=ep,**cap))
                for key in ('scalar','seconds','raw_weighted_fishr','effective_weighted_fishr'):_add(cap_stats,key,cap[key])
            elif cap['gradient_evaluations'] or cap['seconds'] or cap['groups']:
                raise ValueError('Inactive cap performed extra computation')
            if count%222==0:epochs_running.append(dict(epoch=ep,counts=dict(counts)))
    if stable_before!=(step_file.stat().st_size,step_file.stat().st_mtime_ns):raise ValueError('Supposedly closed source log changed during scan')
    if count!=4440 or done['optimizer_steps']!=4440 or done['logged_steps']!=4440 or len(per_epoch)!=20:
        raise ValueError('Incomplete 4440-step/20-epoch source record')
    if any(aggregate['steps']!=222 for aggregate in per_epoch.values()):raise ValueError('Incomplete epoch')
    if any(counts[k]!=value for k,value in fixed.items()) or counts!=done['mechanism_execution']:
        raise ValueError('Actual mechanism counts differ from fixed schedule/completion')
    if counts['mixed_U_packets'] or not 0<=counts['mixed_L_packets']<=counts['style_L_packets']*len(c['multilevel']['style_levels']):
        raise ValueError('Invalid main L style packet-layer count')
    if done['distinct_U_visited']!=(56700 if c['unlabeled'] else 0):raise ValueError('Incomplete U coverage')
    epochs=_read_jsonl(source/'epoch_metrics.jsonl')
    if [row['epoch'] for row in epochs]!=list(range(1,21)) or stdout['epochs']!=epochs:
        raise ValueError('Epoch JSONL/stdout sequence differs')
    for ep,cumulative in zip(epochs,epochs_running):
        aggregate=per_epoch[ep['epoch']]
        if ep['optimizer_steps']!=ep['epoch']*222 or ep['total_epoch']!=ep['epoch'] or ep['counts']!=cumulative['counts']:
            raise ValueError('Epoch budget/cumulative mechanism count mismatch')
        for key,value in aggregate['sums'].items():
            if key not in ep or not math.isclose(ep[key],value/aggregate['metric_counts'][key],rel_tol=1e-9,abs_tol=1e-10):
                raise ValueError('Full step/epoch mean mismatch: '+key)
        expected_v=ep['epoch'] in c['validation']['epochs']
        if bool(ep['validation'])!=expected_v:raise ValueError('Source V schedule mismatch')
        if expected_v and (ep['validation']['V_count']!=27000 or set(ep['validation']['V_RX_accuracy'])!={'1','3','4','6','8'}):
            raise ValueError('Source V support differs')
    with (source/'epoch_metrics.csv').open(encoding='utf-8',newline='') as stream:csv_rows=list(csv.DictReader(stream))
    if len(csv_rows)!=20:raise ValueError('Incomplete compact CSV')
    for row,ep in zip(csv_rows,epochs):
        if int(row['epoch'])!=ep['epoch']:raise ValueError('CSV epoch differs')
        for key,value in row.items():
            expected=ep[key]
            if isinstance(expected,(int,float)) and not math.isclose(float(value),expected,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('CSV/JSON epoch metric mismatch: '+key)
    for aggregate in per_epoch.values():
        aggregate['style_L']={k:_style_finish(v) for k,v in aggregate['style_L'].items()}
        aggregate['style_identity']=_finish(aggregate['style_identity'])
        aggregate['style_search']=_finish(aggregate['style_search'])
        aggregate['budget']={k:_finish(v) for k,v in aggregate['budget'].items()}
    budget_summary={level:dict(calls=value['calls'],cap_count=value['cap_count'],cap_fraction=value['cap_count']/value['calls'],
                     skip_reasons=value['skip_reasons'],statistics=_finish(value['stats'])) for level,value in budgets.items()}
    return dict(base_result,status='COMPLETE',steps=count,source_only=True,epochs=epochs,
                source_V=[dict(epoch=e['epoch'],**e['validation']) for e in epochs if e['validation']],
                parameter_movement=[dict(epoch=e['epoch'],groups=e['weight_movement']) for e in epochs],
                step_metrics=_finish(metric_stats),detail_summary=_finish(detail_stats),counts=counts,
                fixed_expected_counts=fixed,first_execution=first,per_epoch=per_epoch,
                gradient_audits=audit_records,gradient_summary=_finish(gradients),
                active_gradient_summary=_finish(active_gradients),gradient_ratio_summary=_finish(gradient_ratios),
                layer_summary={level:_finish(value) for level,value in layers.items()},
                layer_records=layer_records,
                L_style={level:_style_finish(value) for level,value in l_style.items()},
                style_identity=_finish(style_identity),style_search_summary=_finish(style_search),
                style_search_selected_indices=selected_indices,style_records=style_records,U_style={},
                fishr_budget=budget_summary,combined_fishr_gradient_ratios=_finish(combined),
                fishr_budget_records=budget_snapshots,joint_cap_records=cap_records,joint_cap_summary=_finish(cap_stats),warmup=warmup,calibration=calibration,initialization=initialization,
                resolved_execution=resolved['execution'],execution_commit=resolved['commit'],resources=done['resources'],
                stdout_errors=stdout['errors'],
                source_stdout=dict(path=stdout['path'],lines=stdout['source_lines'],utf8_bytes=stdout['source_utf8_bytes'],
                    errors=stdout['errors'],warnings=stdout['warnings'],nonfinite_markers=stdout['nonfinite_markers'],
                    closed_source_segment=True,target_segment_read=False),
                files=dict(step_log=str(step_file),epochs=str(source/'epoch_metrics.jsonl'),csv=str(source/'epoch_metrics.csv'),
                    warmup=str(source/'head_warmup.jsonl'),calibration=str(source/'calibration.json'),completion=str(source/'completion.json')))


def analyze(base,configs):
    base=Path(base);rows=[];pending=[];errors=[]
    for c in configs:
        row={key:c[key] for key in ('row_id','arm','model_seed')}
        try:
            result=analyze_row(base,row,c)
            if result['status']=='COMPLETE':rows.append(result)
            else:pending.append(result)
        except (ValueError,KeyError,TypeError,OSError,EOFError) as exc:
            errors.append(dict(row,reason=repr(exc),status='INVALID_SOURCE_EVIDENCE'))
    return dict(status='FAILED' if errors else 'PARTIAL' if pending else 'COMPLETE',run_id=base.name,
                read_at=time.time(),expected_rows=len(configs),rows=rows,pending=pending,errors=errors,
                total_steps=sum(r['steps'] for r in rows),total_head_warmup_steps=sum(r['warmup']['summary']['optimizer_steps'] for r in rows),
                total_calibration_readouts=sum(r['calibration']['extra_source_readouts'] for r in rows),
                source_only=True,full_closed_source_step_epoch_CSV_warmup_and_stdout_read=True,
                stdout_boundary='All source lines through SOURCE_TRAINED; later prediction/scoring segment never read',
                non_closed_gzip_read=False,target_prediction_or_scores_read=False)


# LOCAL CONTROL: remote payload stops above this line; stdlib source reader only.
def collect():
    from . import design as d
    from experiments.cvs_phase1_stack.publish import ssh
    own=Path(__file__).read_text(encoding='utf-8').split('# LOCAL CONTROL:')[0]
    configs=[d.config(row) for row in d.rows()]
    payload=own+'\nprint(json.dumps(analyze('+repr(d.BASE.as_posix())+','+repr(configs)+'),allow_nan=False))\n'
    compile(payload,'source-only-analysis','exec')
    return json.loads(ssh(payload))


def report(value):
    def fmt(number,digits=4):return 'N/A' if number is None else f'{number:.{digits}f}'
    def stat(row,level,key):return row.get('fishr_budget',{}).get(level,{}).get('statistics',{}).get(key,{}).get('mean')
    lines=['# 从零训练 20 轮：主身份 margin 敏感度与困难风格搜索源日志分析','',
        f"状态：{value['status']}。完整核对 {len(value['rows'])}/{value['expected_rows']} 行、{value['total_steps']} 次主更新、{value['total_head_warmup_steps']} 次辅助头预热更新及 {value['total_calibration_readouts']} 次校准 L 模型读出。",
        f"每个完成行读取全部 4,440 条逐步记录、20 轮 JSONL/CSV、完整暖头和指定时点校准记录，以及共享 stdout 中至 SOURCE_TRAINED 的全部源训练内容。后续预测/评分段不读取，未完成行的活动 gzip 不打开。固定矩阵为 {value['expected_rows']} 行、{4440*value['expected_rows']:,} 次主更新、{20*value['expected_rows']} 个 epoch。",'',
        '|方案／种子|主 CE：首轮→末轮|末轮 U 接受数／128|末轮源 V／最差 RX|辅助步数|主训练／暖头／校准秒|',
        '|---|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        first,last=row['epochs'][0],row['epochs'][-1];v=last['validation'];cost=row['resources']
        lines.append(f"|{row['row_id']}|{first['ce']:.4f} → {last['ce']:.4f}|{last['accepted_U']:.2f}|{v['V_accuracy']:.4%}／{v['V_worst_RX']:.4%}|{row['counts']['auxiliary_steps']}|{cost['identity_training_and_source_diagnostics_seconds']:.1f}／{row['warmup']['summary']['seconds']:.1f}／{row['calibration']['seconds']:.1f}|")
    lines+=['','## 三种子源域汇总','','下表只使用固定末轮源 V，不作候选排序或选模。标准差为三个模型种子的样本标准差。','',
        '|方案|完成种子数|源 V 均值 ± 标准差|最差 RX 均值|相对 CE 配对变化（百分点）|',
        '|---|---:|---:|---:|---:|']
    reference={r['seed']:r['source_V'][-1]['V_accuracy'] for r in value['rows'] if r['arm']=='CE'}
    for arm in dict.fromkeys(r['arm'] for r in value['rows']):
        group=[r for r in value['rows'] if r['arm']==arm]
        accuracy=[r['source_V'][-1]['V_accuracy'] for r in group]
        paired=[r['source_V'][-1]['V_accuracy']-reference[r['seed']] for r in group if r['seed'] in reference]
        deviation=statistics.stdev(accuracy) if len(accuracy)>1 else None
        lines.append(f"|{arm}|{len(group)}/3|{statistics.mean(accuracy):.4%} ± {fmt(deviation*100 if deviation is not None else None)}%|{statistics.mean(r['source_V'][-1]['V_worst_RX'] for r in group):.4%}|{fmt(statistics.mean(paired)*100 if paired else None)}|")
    lines+=['','## 辅助 CE 梯度路径核验','', '|方案／种子|设计允许 CE 更新骨干|实际辅助 CE 身份梯度最大范数|辅助头梯度最大范数|','|---|---|---:|---:|']
    for row in value['rows']:
        entries=row['gradient_summary'];keys=[k for k in entries if k.startswith('aux_ce/') and not k.endswith('/multilevel_heads')]
        max_identity=max((entries[k]['max'] for k in keys),default=0.)
        heads=entries.get('aux_ce/multilevel_heads',{}).get('max')
        lines.append(f"|{row['row_id']}|{row['resolved_execution']['auxiliary_CE_to_identity']}|{max_identity:.8g}|{fmt(heads,8)}|")
    lines+=['','## 无标签监督信息与分歧','',
        '学生身份网络全部随机初始化，没有加载原 E20 权重；非 CE 方案仍使用另外两个预训练原 E20 的固定概率均值与 ALL soft KL，因此整个方法仍依赖教师预训练。下表只用 U 预测和梯度，不读取身份真值。教师－学生分歧不等于纠错；KL_same/KL_disagree 只拆分同一个 KL，不重复加入总目标。','',
        '|方案／种子|教师／门控|教师熵|教师－学生分歧比例|平均接受分歧数／128|分歧 KL|两教师分歧比例|',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        stats=row['detail_summary'];s=lambda k:stats.get(k,{}).get('mean')
        ex=row['resolved_execution']
        lines.append(f"|{row['row_id']}|{ex['teacher_source']}／{ex['U_selection']}|{fmt(s('teacher_entropy'))}|{fmt(s('teacher_live_disagreement'))}|{fmt(s('accepted_disagreement'))}|{fmt(s('KL_disagree'),8)}|{fmt(s('peer_disagreement'))}|")
    lines+=['','## 初始固定校准与实际 Fishr 梯度','',
        '全部方案没有辅助头或预热。CAP 在0更新校准；DELAY/DELAYCAP 在444更新后校准，使用自身 margin 敏感度交叉项梯度作为分母。每个启用行校准 12 对独立 L60，共 1,440 次样本读出、24 次前向、零身份更新。系数为 0.2×平均主 CE 梯度范数／平均自身项梯度范数，cap=10,000。以下表格为固定校准和ramp的原始项；CAP/DELAYCAP另有联合限幅，实际生效值见联合限幅表。实测梯度比例不要求始终等于 0.2；选定层之和固定除以 2，未启用层为零。','',
        '|方案／种子|层|固定系数|校准 cap|训练记录数|ramp 后系数均值|均值前实测比例|除以 2 后直接比例|合并后对主 CE 比例|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        for level,summary in row['fishr_budget'].items():
            combined=row['combined_fishr_gradient_ratios'].get(level+'/ratio_to_main_ce',{}).get('mean')
            initial=row['calibration']['levels'][level]
            lines.append(f"|{row['row_id']}|{level}|{fmt(initial['coefficient'],6)}|{initial['cap_active']}|{summary['calls']}|{fmt(stat(row,level,'coefficient'),6)}|{fmt(stat(row,level,'achieved_ratio_pre_level_mean'))}|{fmt(stat(row,level,'achieved_direct_contribution_ratio'))}|{fmt(combined)}|")
    lines+=['','## 有符号 Fishr 与独立批采样','',
        '全部方案使用主身份平滑 margin 对逐样本通道缩放/平移的导数，经逐样本 RMS 归一化后做有限总体修正的 A/B 中心化方差内积。这是 Fishr 式虚拟坐标敏感度方差约束，不等同于标准模型参数经验 Fisher。平滑 margin 为正确 logit 减其余 logits 的 logsumexp。独立批允许 PID 重叠；负内积是合法估计值，不截断，也不据此判定性能优劣。','',
        '|方案／种子|层|原始 Fishr 均值|最小值|负值次数／实际计算次数|',
        '|---|---|---:|---:|---:|']
    for row in value['rows']:
        for level,entries in row['layer_summary'].items():
            raw=entries.get('raw_fishr')
            if raw:lines.append(f"|{row['row_id']}|{level}|{fmt(raw['mean'],8)}|{fmt(raw['min'],8)}|{raw['negative']}/{raw['observations']}|")
    lines+=['','## L 主路径 MixStyle 实际覆盖','',
        '下表是选定候选训练重放的层统计，覆盖分母为该层实际样本出现次数；两层之和是 packet-layer 次数，不是不同物理样本数。p=1、alpha=0.1、eta=ramp，层顺序固定为 time_down→t2.pool，供体使用当前完整候选内的实际中间特征。','',
        '|方案／种子|层|调用数|合法 donor 比例|实际改变比例|相对特征变化均值|',
        '|---|---|---:|---:|---:|---:|']
    for row in value['rows']:
        for level,summary in row['L_style'].items():
            delta=summary['metrics'].get('delta_ratio',{}).get('mean')
            lines.append(f"|{row['row_id']}|{level}|{summary['calls']}|{summary['eligible_fraction']:.2%}|{summary['changed_fraction']:.2%}|{fmt(delta)}|")
    lines+=['','## 风格路径的身份读出','',
        '仅汇总实际额外 L 训练前向。选定正向或反向变换时重放同一候选；选定clean时不装hook。所有重放使用clean之前的随机状态，共用Dropout掩码。候选难度只比较 eval/no_grad 读出，训练模式差值不能冒充该选择的因果效果。此处 margin 为正确 logit 减最大错误 logit，与敏感度中的平滑 margin 不同。供体依据合法 L 真标签；U 没有风格混合。','',
        '|方案／种子|调用数|clean／styled 准确率|准确率变化（百分点）|预测改判比例|clean／styled margin|margin 变化|style CE／加权 CE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        s=row['style_identity']
        if not s:continue
        mean=lambda k:s[k]['mean']
        lines.append(f"|{row['row_id']}|{s['style_ce']['observations']}|{mean('L_accuracy'):.4%}／{mean('style_L_accuracy'):.4%}|{100*mean('style_accuracy_delta'):+.4f}|{mean('style_prediction_change'):.4%}|{mean('clean_identity_margin'):.5f}／{mean('styled_identity_margin'):.5f}|{mean('style_margin_delta'):+.5f}|{mean('style_ce'):.5f}／{mean('weighted_style_ce'):.5f}|")
    lines+=['','## Margin 敏感度的数值尺度','',
        '低范数率按原始逐样本 RMS≤1e-6 统计；A/B 分开保留。归一化低范数样本没有被静默删除。','',
        '|方案／种子|层／批|原始 RMS 均值|低范数率|归一化 RMS 均值|平滑 margin 均值|',
        '|---|---|---:|---:|---:|---:|']
    for row in value['rows']:
        for level,stats in row['layer_summary'].items():
            for split in ('A','B'):
                prefix='sensitivity_'+split+'/'
                if prefix+'low_norm_rate' not in stats:continue
                get=lambda key:stats[prefix+key]['mean']
                lines.append(f"|{row['row_id']}|{level}/{split}|{get('raw_sensitivity_rms_mean'):.8g}|{get('low_norm_rate'):.4%}|{get('normalized_score_rms_mean'):.6f}|{get('main_margin_mean'):.6f}|")
    lines+=['','## 完整候选搜索与额外成本','',
        'POS 与 BIDIR 均读取 clean＋4 个完整候选，并重放一个有梯度候选。按整批CE最大值选择，clean平局保留增强，增强平局取最小索引。POS四正向；BIDIR索引1/3反向，均限制均值位移和log-scale。下表的准确率和 margin 来自 dropout 关闭的选择读出。每次 4 个 CE、完整选定层统计和选择索引保留在 style_records。','',
        '|方案／种子|搜索次数|eval clean→selected CE|候选 CE 均值／范围宽度|eval margin 变化|eval 准确率变化（百分点）|额外 eval 前向／L|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        s=row['style_search_summary']
        if not s:continue
        get=lambda key:s[key]['mean']
        lines.append(f"|{row['row_id']}|{s['eval_CE_delta']['observations']}|{get('eval_clean_CE'):.6f}→{get('eval_selected_CE'):.6f}|{get('candidate_CE_mean'):.6f}／{get('candidate_CE_spread'):.6f}|{get('eval_margin_delta'):+.6f}|{100*get('eval_accuracy_delta'):+.4f}|{row['counts']['style_search_forwards']}/{row['counts']['style_search_L']}|")
    lines+=['','## 证据与限制','',
        '- JSON 保留全部 20 轮的损失、学习率、源 V 各 RX／日期、参数位移，以及所有 10 次分组梯度审计；layer_records 保留每次每层的完整原始数值，包括 A/B 的低范数率和归一化前后敏感度。',
        '- 每次 Fishr 的固定系数、cap、实测直接／合并梯度比例和有符号损失全部保留；额外 2×L60 前向各自批内唯一、批间允许重叠。源 V 仅作健康诊断。',
        '- 本批没有辅助头预热；分析器核验零暖头记录及初始校准状态不变证据，不重新加载 checkpoint，也不访问目标预测或评分。',
        '- 学生从零训练 20 轮，累计 20 轮，没有父训练或暖头。启用统计项的行校准 1,440 次 L 读出，训练辅助 119,880 次 L 使用。风格/replay 行另有 511,488 次有梯度 L 主路径读出。随机/困难搜索各再增加 19,980 次 eval 前向、2,557,440 次 L 样本读出；L_REPLAY 不做这 5 次搜索读出，成本不等同。',
        '- 非 CE 方案复用两份原 E20 教师，历史模型训练成本为学生 20＋教师 40＝60 模型轮；CE 为 20。与 E20＋20 续训的累计学生训练量不同，不能把跨批差异全归因于初始化。前批同预算固定对照只由独立报告读取；源分析不读取其目标评分。P0只作原E20评分参考。','',
        '未完成行：'+(', '.join(r['row_id'] for r in value['pending']) or '无。'),
        '证据异常：'+('; '.join(r['row_id']+': '+r['reason'] for r in value['errors']) or '无。')]
    stdout_errors=[r['row_id'] for r in value['rows'] if r['source_stdout']['errors']]
    lines+=['源训练 stdout 异常标记：'+(', '.join(stdout_errors) or '无。')]
    lines+=['','## 联合辅助梯度上限','', '|方案|辅助限幅步|s均值|s最小|新增求导秒|','|---|---:|---:|---:|---:|']
    for row in value['rows']:
        cap=row.get('joint_cap_summary',{})
        if not cap:continue
        scalar=cap['scalar'];seconds=cap['seconds']
        lines.append(f"|{row['row_id']}|{scalar['observations']}|{scalar['mean']:.6g}|{scalar['min']:.6g}|{seconds['mean']*seconds['observations']:.3f}|")
    lines+=['', '各组全部步骤的norm_id、raw_R、capped_R与0.2范数界逐条核验，参数组互斥且完整覆盖。此处限幅作用于联合R；旧预算表记录限幅前值。边界不代表AdamW位移或性能保证。']
    lines+=['','## clean候选实际选择','','|方案|clean选择次数/3996|','|---|---:|']
    for row in value['rows']:
        count=row['style_search_selected_indices'].get('-1',0)
        lines.append(f"|{row['row_id']}|{count}/3996|")
    lines+=['','clean选择时不安装风格hook，仍重放一个相同dropout状态的可微clean前向；源样本读出与搜索预算不减少。层覆盖统计以实际安装hook的packet-layer次数为分母，clean选择次数单独报告。']
    from .style_report import lines as style_lines
    lines+=style_lines(value['rows'])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    from . import design as d
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=collect()
    d.write(args.output/'evidence/source_full_analysis.json',value)
    (args.output/'source_trajectory.md').write_text(report(value),encoding='utf-8')
    print(json.dumps(dict(status=value['status'],complete_rows=len(value['rows']),expected_rows=value['expected_rows'],
                         steps=value['total_steps'],pending=value['pending'],errors=value['errors'],source_only=True)))
