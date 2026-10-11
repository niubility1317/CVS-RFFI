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
                mixstyle_U_steps=active if mc['mixstyle_u'] else 0)


def _source_stdout(path):
    """The worker reuses stdout for target evaluation: STOP at source-end marker.

    This reads every source-phase line, but deliberately never reads the later
    prediction/scoring segment of the shared log, even if it already exists.
    """
    result=dict(path=str(path),source_lines=0,source_utf8_bytes=0,closed_source_segment=False,
                errors=[],warnings=[],nonfinite_markers=[],resolved=[],epochs=[],warmup=[],source_trained=None,
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
            for prefix,key in [('RESOLVED ','resolved'),('EPOCH ','epochs'),('HEAD_WARMUP ','warmup')]:
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
    expected_steps=200 if c['multilevel']['enabled'] else 0
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
        old=math.fsum(r['levels'][level]['old_pair_gradient_norm'] for r in records)/12
        if min(ce,old)<=mc['fishr_eps']:
            raise ValueError('Invalid calibration reference gradient norm')
        raw=mc['fishr_ratio']*ce/(old+mc['fishr_eps']);coefficient=min(mc['fishr_cap'],raw)
        for key,want in (('mean_CE_gradient_norm',ce),('mean_old_pair_gradient_norm',old),
                         ('uncapped_coefficient',raw),('coefficient',coefficient)):
            if not math.isclose(summary[key],want,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Calibration coefficient/recount differs: '+level+'/'+key)
        if (summary['cap_active']!=(raw>mc['fishr_cap'])
                or not math.isclose(value['fixed_coefficients'][level],coefficient,rel_tol=1e-10,abs_tol=1e-12)):
            raise ValueError('Fixed calibration cap differs')
    return value


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
    if done['status']!='SOURCE_TRAINED' or done['target_access'] or done['target_evaluated'] or done['epoch']!=20 or done['total_epochs']!=40:
        raise ValueError('Source completion or budget contract differs')
    if frozen['status']!='SOURCE_FROZEN' or frozen['target_access'] or frozen['target_scores_consumed']:
        raise ValueError('Source freeze differs')
    if initialization!=done['initialization'] or initialization['scratch_only'] or not initialization['reference_E20_loaded']:
        raise ValueError('Missing E20 provenance')
    if initialization['source_roles']!='EXACT_MATCH' or initialization['target_contact'] or initialization['auxiliary_inheritance']:
        raise ValueError('Invalid inherited source provenance')
    teachers=initialization.get('teacher_provenance',{})
    if set(teachers)!={str(s) for s in c['teacher']['seeds']}:raise ValueError('Teacher provenance keys differ')
    expected_sources=[c['parent_checkpoint']]
    for seed in c['teacher']['seeds']:
        proof=teachers[str(seed)];path=c['parent_checkpoint'].rsplit('/',3)[0]+f'/E20-s{seed}/source/final_ssdg.pth'
        expected_sources.append(path)
        expected=dict(status='VERIFIED',seed=seed,checkpoint=path,epoch=20,scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',training_target_access=False)
        if any(proof.get(k)!=v for k,v in expected.items()) or seed==c['model_seed']:raise ValueError('Invalid teacher source proof')
    if initialization['parent_checkpoint']!=c['parent_checkpoint'] or initialization['checkpoint_sources']!=expected_sources:
        raise ValueError('Wrong fixed parent')
    if stdout['resolved']!=[resolved] or stdout['source_trained']!=done:
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
            or execution['Fishr_levels']!=c['multilevel']['fishr_levels']):
        raise ValueError('Actual execution permissions/gradient route differ')
    fixed=_expected_counts(c);counts={k:0 for k in fixed};counts.update(mixed_U_packets=0,mixed_L_packets=0)
    per_epoch={};metric_stats={};detail_stats={};first={};gradients={};active_gradients={};gradient_ratios={}
    audit_records=[];layers={};l_style={};style_identity={};budgets={};combined={};budget_snapshots=[];epochs_running=[]
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
            aggregate=per_epoch.setdefault(ep,dict(steps=0,sums={},metric_counts={},style_L={},style_identity={},budget={}))
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
                if multi['aux_ce_to_backbone']!=c['multilevel']['aux_ce_to_backbone'] or multi['fishr_levels']!=c['multilevel']['fishr_levels']:
                    raise ValueError('Actual multilevel gradient route differs')
                if (multi['auxiliary_packets']!=120 or multi['aux_forwards']!=2
                        or multi['pair_A_distinct_PID']!=60 or multi['pair_B_distinct_PID']!=60
                        or not 0<=multi['cross_batch_PID_overlap']<=60
                        or multi['distinct_PID']!=120-multi['cross_batch_PID_overlap']
                        or set(multi['source_RX'])!={1,3,4,6,8} or multi['day'] not in (1,2,3)
                        or not multi['independent_AB'] or multi['calibration_sampling']
                        or multi['estimator']!=c['multilevel']['fishr_estimator']
                        or multi['head_Fishr_gradient'] or multi['EMA']):
                    raise ValueError('Auxiliary source support differs')
                for level,info in multi['levels'].items():
                    stats=layers.setdefault(level,{})
                    for key,value in info.items():
                        if isinstance(value,(int,float)):_add(stats,key,value)
                    if info['fishr_computed']!=(level in c['multilevel']['fishr_levels']):
                        raise ValueError('Computed Fishr layer differs')
            style=detail.get('L_style')
            if bool(style)!=bool(expected_style):raise ValueError('Missing/unexpected main labeled style details')
            if style:
                if (style['role']!='L_main_identity' or style['source_role']!='L_s' or style['labels']!='legal L truth'
                        or style['target_access'] or style['U_access'] or style['U_truth']
                        or style['training_flags_modified'] or not style['extra_forward_CPU_CUDA_RNG_restored']
                        or style['styled_forwards']!=int(bool(c['multilevel']['style_levels']))
                        or style['batch_occurrences']!=c['batch_size']):
                    raise ValueError('L style permission/forward contract differs')
                if set(style['levels'])!=set(c['multilevel']['style_levels']):raise ValueError('L style layers differ')
                changed=0
                for level,info in style['levels'].items():
                    if not info['donor_rules_verified'] or info['role']!='L_main_identity':
                        raise ValueError('L style donor audit differs')
                    _style_add(l_style.setdefault(level,{}),info,c['batch_size'])
                    _style_add(aggregate['style_L'].setdefault(level,{}),info,c['batch_size'])
                    changed+=info['changed_packets']
                if changed!=detail['mixed_L_packets']:raise ValueError('L packet-layer count mismatch')
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
        if ep['optimizer_steps']!=ep['epoch']*222 or ep['total_epoch']!=20+ep['epoch'] or ep['counts']!=cumulative['counts']:
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
                L_style={level:_style_finish(value) for level,value in l_style.items()},
                style_identity=_finish(style_identity),U_style={},
                fishr_budget=budget_summary,combined_fishr_gradient_ratios=_finish(combined),
                fishr_budget_records=budget_snapshots,warmup=warmup,calibration=calibration,initialization=initialization,
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
    lines=['# E20＋20：独立批 Fishr 与 L 主路径 MixStyle 源日志分析','',
        f"状态：{value['status']}。完整核对 {len(value['rows'])}/{value['expected_rows']} 行、{value['total_steps']} 次主更新、{value['total_head_warmup_steps']} 次辅助头预热更新及 {value['total_calibration_readouts']} 次校准 L 模型读出。",
        '每个完成行读取全部 4,440 条逐步记录、20 轮 JSONL/CSV、完整暖头和初始校准记录，以及共享 stdout 中至 SOURCE_TRAINED 的全部源训练内容。后续预测/评分段不读取，未完成行的活动 gzip 不打开。固定完整矩阵为 33 行、146,520 次主更新、660 个 epoch。','',
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
        '非 CE 方案都使用另外两个原 E20 的固定概率均值与 ALL soft KL。下表只用 U 预测和梯度，不读取身份真值。教师－学生分歧不等于纠错；KL_same/KL_disagree 只拆分同一个 KL，不重复加入总目标。','',
        '|方案／种子|教师／门控|教师熵|教师－学生分歧比例|平均接受分歧数／128|分歧 KL|两教师分歧比例|',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        stats=row['detail_summary'];s=lambda k:stats.get(k,{}).get('mean')
        ex=row['resolved_execution']
        lines.append(f"|{row['row_id']}|{ex['teacher_source']}／{ex['U_selection']}|{fmt(s('teacher_entropy'))}|{fmt(s('teacher_live_disagreement'))}|{fmt(s('accepted_disagreement'))}|{fmt(s('KL_disagree'),8)}|{fmt(s('peer_disagreement'))}|")
    lines+=['','## 初始固定校准与实际 Fishr 梯度','',
        '每个 Fishr 行在暖头后执行 12 对独立 L60，共 1,440 次样本模型读出、24 次前向、零身份/头更新。两层都按 0.2×平均主 CE 梯度范数／平均旧双批 Fishr 梯度范数校准，cap=10,000。之后仅乘固定 ramp，不按当前批重标定。下面的梯度比例是实测值，不要求始终等于 0.2。选定层之和固定除以 2，未启用层贡献为零。','',
        '|方案／种子|层|固定系数|校准 cap|训练记录数|ramp 后系数均值|均值前实测比例|除以 2 后直接比例|合并后对主 CE 比例|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        for level,summary in row['fishr_budget'].items():
            combined=row['combined_fishr_gradient_ratios'].get(level+'/ratio_to_main_ce',{}).get('mean')
            initial=row['calibration']['levels'][level]
            lines.append(f"|{row['row_id']}|{level}|{fmt(initial['coefficient'],6)}|{initial['cap_active']}|{summary['calls']}|{fmt(stat(row,level,'coefficient'),6)}|{fmt(stat(row,level,'achieved_ratio_pre_level_mean'))}|{fmt(stat(row,level,'achieved_direct_contribution_ratio'))}|{fmt(combined)}|")
    lines+=['','## 有符号 Fishr 与独立批采样','',
        'OLD_PAIR 使用两个 pooled-n 方差差平方的均值；F_* 和 JOINT 使用分层有限总体修正后的 A/B 中心化内积。独立批允许 PID 重叠；负内积是合法估计值，不截断，也不据此判定性能优劣。','',
        '|方案／种子|层|原始 Fishr 均值|最小值|负值次数／实际计算次数|',
        '|---|---|---:|---:|---:|']
    for row in value['rows']:
        for level,entries in row['layer_summary'].items():
            raw=entries.get('raw_fishr')
            if raw:lines.append(f"|{row['row_id']}|{level}|{fmt(raw['mean'],8)}|{fmt(raw['min'],8)}|{raw['negative']}/{raw['observations']}|")
    lines+=['','## L 主路径 MixStyle 实际覆盖','',
        '覆盖分母为该层实际调用中的样本出现次数；两层之和是 packet-layer 次数，不是不同物理样本数。未开启 Bernoulli 调用也计入调用分母；强度等稀疏指标使用各自实际观测次数。','',
        '|方案／种子|层|调用数|合法 donor 比例|实际改变比例|相对特征变化均值|',
        '|---|---|---:|---:|---:|---:|']
    for row in value['rows']:
        for level,summary in row['L_style'].items():
            delta=summary['metrics'].get('delta_ratio',{}).get('mean')
            lines.append(f"|{row['row_id']}|{level}|{summary['calls']}|{summary['eligible_fraction']:.2%}|{summary['changed_fraction']:.2%}|{fmt(delta)}|")
    lines+=['','## 风格路径的身份读出','',
        '仅汇总实际额外 L 前向。L_REPLAY 保留同等额外身份 CE 和 RNG 隔离，但不安装风格 hook，供各 STYLE 对照额外监督与层扰动。clean 与 styled/replay 是训练模式下不同前向，dropout 掩码不同，不能把单次差值当作隔离后的增强因果效应。供体依据 L 真标签；U 没有风格混合。','',
        '|方案／种子|调用数|clean／styled 准确率|准确率变化（百分点）|预测改判比例|clean／styled margin|margin 变化|style CE／加权 CE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        s=row['style_identity']
        if not s:continue
        mean=lambda k:s[k]['mean']
        lines.append(f"|{row['row_id']}|{s['style_ce']['observations']}|{mean('L_accuracy'):.4%}／{mean('style_L_accuracy'):.4%}|{100*mean('style_accuracy_delta'):+.4f}|{mean('style_prediction_change'):.4%}|{mean('clean_identity_margin'):.5f}／{mean('styled_identity_margin'):.5f}|{mean('style_margin_delta'):+.5f}|{mean('style_ce'):.5f}／{mean('weighted_style_ce'):.5f}|")
    lines+=['','## 证据与限制','',
        '- JSON 保留全部 20 轮的损失、学习率、源 V 各 RX／日期、参数位移；保留所有 10 次完整分组梯度审计及其独立观测数。',
        '- 每次 Fishr 的固定系数、cap、实测直接／合并梯度比例和有符号损失全部保留；额外 2×L60 前向各自批内唯一、批间允许重叠。源 V 仅作健康诊断。',
        '- 暖头的骨干不变结论来自训练时逐位 state_dict 比较及日志。分析器不重新加载 checkpoint，也不访问目标预测或评分。',
        '- 主训练成本为追加 20 轮，累计 40 轮。Fishr 行暖头 200 步、校准 1,440 次 L 读出及主训练 119,880 次额外 L 使用单列。STYLE 行额外 511,488 次 L 主路径读出，CE/SSL/STYLE-only 不创建辅助头。','',
        '未完成行：'+(', '.join(r['row_id'] for r in value['pending']) or '无。'),
        '证据异常：'+('; '.join(r['row_id']+': '+r['reason'] for r in value['errors']) or '无。')]
    stdout_errors=[r['row_id'] for r in value['rows'] if r['source_stdout']['errors']]
    lines+=['源训练 stdout 异常标记：'+(', '.join(stdout_errors) or '无。')]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    from . import design as d
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=collect()
    d.write(args.output/'evidence/source_full_analysis.json',value)
    (args.output/'source_trajectory.md').write_text(report(value),encoding='utf-8')
    print(json.dumps(dict(status=value['status'],complete_rows=len(value['rows']),expected_rows=value['expected_rows'],
                         steps=value['total_steps'],pending=value['pending'],errors=value['errors'],source_only=True)))
