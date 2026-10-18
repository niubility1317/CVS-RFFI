"""P68 CPU-only same-parent full-channel-pair readout; no IQ/checkpoint access."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import importlib.util
import json
import time
import traceback
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np

RUN = '20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01'
REF = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SAME = '20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
SEED, STEPS = 2026092701, 222
ARMS = ('NOAS30', 'AS30', 'SAME_CE6', 'INDEP_CE6')
NEW_ARMS = ('INDEP_CE6',)
PAIRS = (('INDEP_CE6', 'SAME_CE6'), ('AS30', 'NOAS30'),
         ('SAME_CE6', 'AS30'), ('SAME_CE6', 'NOAS30'),
         ('INDEP_CE6', 'AS30'), ('INDEP_CE6', 'NOAS30'))
PRIMARY = ('INDEP_CE6', 'SAME_CE6')
BOOT_COMMIT = '735e171fc064094f227fe6807d75836a0b3b0e3a'
PAIR_MEANS = ('original_mixed_KL','ordinary_half_KL','satellite_half_KL','used_mixed_KL',
              'removed_satellite_KL','satellite_CE0','satellite_CE1','As_IQ_pair_MSE')


def load_common():
    path = Path(__file__).with_name('cvs_u_quality66_analysis_20261018.py')
    spec = importlib.util.spec_from_file_location('_p68_cpu_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN, module.NEW_ARMS, module.PAIRS = RUN, NEW_ARMS, PAIRS
    module.training_complete, module.row_specs = training_complete, row_specs
    return module


def row_specs(project):
    p = Path(project) / 'runs'
    values = (('NOAS30', REF, 'B00'), ('AS30', REF, 'B01'),
              ('SAME_CE6', SAME, 'AS_CE'), ('INDEP_CE6', RUN, 'B01'))
    return [dict(key=key, run=run, arm=arm, row_id=arm + '-s' + str(SEED),
                 root=p / run / (arm + '-s' + str(SEED)), reused=key != 'INDEP_CE6',
                 parent=p / (SAME if key == 'INDEP_CE6' else run) /
                 (('AS_CE' if key == 'INDEP_CE6' else arm) + '-s' + str(SEED)) / 'stage20',
                 parent_run=SAME if key == 'INDEP_CE6' else run,
                 parent_row_id=('AS_CE' if key == 'INDEP_CE6' else arm) + '-s' + str(SEED),
                 parent_As_KL_off=key in ('SAME_CE6', 'INDEP_CE6'))
            for key, run, arm in values]


def training_complete(spec, project):
    roles = c.read(Path(project) / 'runs' / c.CONTRACT_RUN / 'source_contract.json')['role_ids']
    for folder, parent, epochs, labels, pseudo, updates in (
            (spec['parent'], True, 20, 13, 7, 4440),
            (spec['root'] / 'source', False, 10, 0, 10, 2220)):
        done = c.read(folder / 'completion.json'); cfg = done['config']
        frozen = c.read(folder / 'source_frozen.json'); init = c.read(folder / 'initialization.json')
        result = c.read(folder / 'result.json'); args = c.read(folder / 'actual_native_args.json')
        args = args.get('args', args)
        expected_sources = [] if parent else [str(spec['parent'] / 'final_ssdg.pth')]
        c.require((done.get('epoch'), done.get('optimizer_steps'), cfg.get('epochs'),
                   cfg.get('label_epochs'), cfg.get('pseudo_epochs'), cfg.get('updates')) ==
                  (epochs, updates, epochs, labels, pseudo, updates), 'Exact E20+10 calendar required')
        c.require(cfg.get('run_id') == (spec['parent_run'] if parent else spec['run']) and
                  cfg.get('row_id') == (spec['parent_row_id'] if parent else spec['row_id']) and
                  cfg.get('model_seed') == SEED and cfg.get('As') is (spec['key'] != 'NOAS30'),
                  'Actual model run/row/seed/As differs')
        c.require(cfg.get('checkpoint_sources') == expected_sources and init.get('checkpoint_sources') == expected_sources and
                  init.get('scratch_only') is parent and init.get('source_roles') == 'EXACT_MATCH' and
                  init.get('target_contact') is False, 'Exact scratch20/own student and ownEMA origin required')
        c.require(frozen.get('status') == 'SOURCE_FROZEN' and frozen.get('config') == cfg and
                  frozen.get('selection') == 'final_only' and frozen.get('target_access') is False,
                  'Fixed final-only source checkpoint required')
        c.require(done.get('target_access') is False and done.get('head_fit_updates') == 0 and
                  done.get('native_fitted_probes_disabled') is True and result.get('config') == cfg and
                  result.get('U_truth_read') is False and result.get('target_access') is False and
                  result.get('head_fit_updates') == 0, 'Joint defaulthead/hidden U/source flags differ')
        c.require(c.read(folder / 'source_contract.json')['role_ids'] == roles and
                  result.get('source_RX') == list(c.RX) and result.get('source_RX_holdout') is None and
                  done['final_source_metrics'].get('source_val_count') == 27000,
                  'Exact full L6300/U56700/V27000/all5RX source roles required')
        c.require(cfg.get('Ag_mode') == 'NONE' and cfg.get('new_Ag') is False and
                  cfg.get('L_NN_per_step') == 768 and cfg.get('total_L_loss_scale') == 1.25,
                  'Original sixL/1.25/no-Ag recipe required')
        c.require(args.get('pseudo_temporal_mode') == 'batch_neighbor' and
                  args.get('pseudo_temporal_gate') is True and args.get('pseudo_temporal_bank_min_streak') == 2,
                  'Original native U temporal recipe required')
        if not parent:
            c.require(cfg.get('parent_source') == str(spec['parent']) and
                      cfg.get('epoch_offset') == 20 and cfg.get('step_offset') == 4440,
                      'Exact fixed own20 source and tail clock required')
        if spec['parent_As_KL_off']:
            c.require(cfg.get('satellite_mode') == 'AS_CE', 'Common parent and tail must both retain As KL off')
    return c.read(spec['root'] / 'source/completion.json')


c = load_common()
_write = c.write


def comparison_context(new, control):
    primary = (new, control) == PRIMARY
    return dict(same_parent=primary, primary_mechanism_comparison=primary,
        first20_As_KL_history_confounded=(new in ARMS[2:] and control == 'AS30'),
        causal_attribution='Second complete As view within common AS_CE20 tail' if primary else 'Whole-recipe descriptive reference',
        matched_noAs_parent=False if new in ARMS[2:] and control == 'NOAS30' else None)


def enrich_formal(obj):
    rows = obj['rows']; obj['comparison_context'] = [dict(new=a, control=b, **comparison_context(a, b)) for a, b in PAIRS]
    obj['primary'] = dict(new=PRIMARY[0], control=PRIMARY[1], same_parent=True,
        parent_run=SAME, parent_row='AS_CE-s' + str(SEED), parent_epoch=20,
        parent_loss_history='Satellite KL already off throughout first20; unlike original P43 As20',
        by_view={v: dict(accuracy_delta_pp=100*(rows[PRIMARY[0]][v]['accuracy']-rows[PRIMARY[1]][v]['accuracy']),
            macro_f1_delta_pp=100*(rows[PRIMARY[0]][v]['macro_f1']-rows[PRIMARY[1]][v]['macro_f1']),
            worst_RX_delta_pp=100*(rows[PRIMARY[0]][v]['worst_RX']-rows[PRIMARY[1]][v]['worst_RX'])) for v in c.VIEWS})
    for arm in NEW_ARMS:
        stage = obj['stage'][arm]
        stage['NoAs30_is_whole_recipe_reference'] = True
        stage['same_parent_NoAs_control_available'] = False
        stage['reproducible_dual_gain'] = False
    obj['primary_independent_sampling_is_not_new_algorithm'] = True
    obj['Ag_claim'] = False


def processed_write(path, obj):
    if Path(path).name == 'formal_summary.json': enrich_formal(obj)
    _write(path, obj)


c.write = processed_write


def source_criteria(clean, noas_clean, pooled_midlow, as_midlow, same_midlow):
    """Source-only absolute-pp rules, fixed before formal access; used for tests/readout."""
    clean_delta = 100*(clean-noas_clean)
    gain = 100*(pooled_midlow-max(as_midlow, same_midlow))
    checks = dict(clean_vs_NoAs_ge_minus_0p1pp=clean_delta >= -.1-1e-10,
                  pooled_midlow_vs_best_control_ge_0p5pp=gain >= .5-1e-10)
    return dict(checks=checks, source_qualified=all(checks.values()),
                clean_delta_pp=clean_delta, pooled_midlow_gain_pp=gain)


def read_source_decision(project):
    value = c.read(Path(project) / 'runs' / RUN / 'source_matrix_decision.json')
    c.require(value.get('status') in c.DECISION_COMPLETE | c.DECISION_PARTIAL and
              value.get('formal_scores_read') is False and value.get('target_feedback') is False and
              value.get('NN') == 0 and value.get('model_seed_count') == 1,
              'Producer source decision must already be fixed before formal access')
    if value['status'] in c.DECISION_COMPLETE:
        checks = value.get('checks')
        c.require(isinstance(checks, dict) and len(checks) == 2 and all(type(v) is bool for v in checks.values()) and
                  type(value.get('source_qualified')) is bool and value['source_qualified'] == all(checks.values()),
                  'Two fixed source checks and saved verdict must agree')
    return value


def parse_tail(folder, is_as, independent=False):
    folder = Path(folder); result = c.read(folder / 'result.json'); cost = c.native_cost(folder)
    c.require(result.get('optimizer_steps') == 2220 and result.get('L_NN') == 1704960 and
              result.get('L_As') == (142080 if is_as else 0) and result.get('Ag_packets') == 0 and
              result.get('extra_U_NN') == 0 and result.get('U_As') == 0 and
              cost['native_NN'] == 3162962 and cost['extra_student_U_NN'] == cost['extra_frozen_teacher_U_NN'] == 0,
              'One original-native tail budget required')
    streams = [c.lines(folder / name) for name in ('reference_steps.jsonl.gz', 'shape_steps.jsonl.gz',
                                                  'step_metrics.jsonl.gz', 'unit_steps.jsonl.gz')]
    sums = defaultdict(Counter); count = Counter(); ordinary = []; max_error = 0.; total_u = accepted = 0
    for step, values in enumerate(zip_longest(*streams), 1):
        c.require(all(x is not None for x in values), 'Complete original reference/shape/native/unit streams required')
        ref, shape, native, unit = values; epoch = (step-1)//STEPS+1; m = native['metrics']
        c.require(all(x.get('step') == step for x in values) and native.get('epoch') == epoch and
                  all(x.get('cumulative_step') == step+4440 for x in (ref, shape, unit)), 'Contiguous complete tail clocks required')
        c.require(ref.get('L_forwards') == 6 and ref.get('As_packets') == (64 if is_as else 0) and
                  ref.get('Ag_packets') == 0 and ref.get('Ag_mode') == 'NONE' and
                  ref.get('ordinary_redraws') == 0 and ref.get('all_satellite_L_supervised') is True,
                  'Original sixL/supervised complete practical positions required')
        c.require(shape.get('mode') == 'NONE' and shape.get('NN') == 256 and shape.get('weight') == .25 and
                  shape.get('same_IQ_each_pair') is True, 'Original raw RDrop pair retained')
        for key in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','total_L','ordinary_raw_MSE'):
            c.finite(ref[key], key); sums[epoch][key] += ref[key]
        for key in ('shape_CE','shape_KL','shape_RD'): c.finite(shape[key], key); sums[epoch][key] += shape[key]
        expected = .5*(ref['common_CE']+ref['common_KL'])+.5*(ref['extra_CE']+ref['extra_KL'])+.1*ref['SupCon']+.25*shape['shape_RD']
        error = abs(expected-ref['total_L']); max_error = max(max_error, error)
        c.require(error < 3e-5 and abs(shape['shape_RD']-shape['shape_CE']-shape['shape_KL']) < 3e-5 and
                  abs(ref['SupCon_weight']-.1) < 1e-12, 'CE/RDrop/SupCon loss decomposition differs')
        n = int(m['train/pseudo_total']); selected = int(m['train/pseudo_selected'])
        c.require(0 <= selected <= n and n > 0, 'Original hidden native U counts required')
        total_u += n; accepted += selected; count[epoch] += 1
        sums[epoch]['native_loss'] += c.finite(m['train/loss'], 'native loss')
        sums[epoch]['native_U_count'] += n; sums[epoch]['native_accepted_count'] += selected
        ordinary.append(dict(step=step+4440, MSE=ref['ordinary_raw_MSE']))
    c.require(sum(count.values()) == 2220 and set(count) == set(range(1,11)) and
              set(count.values()) == {222} and total_u == 567000, 'All ten full native U epochs required')
    epochs = list(c.lines(folder / 'epoch_metrics.jsonl')); raw = list(c.lines(folder / 'metrics_epoch.jsonl'))
    c.require([r['epoch'] for r in epochs] == [r['epoch'] for r in raw] == list(range(1,11)), 'All compact/raw epoch metrics required')
    curves = [dict(epoch=r['epoch']+20, local_epoch=r['epoch'],
        **{k:v/STEPS for k,v in sums[r['epoch']].items()},
        native_accepted_fraction=sums[r['epoch']]['native_accepted_count']/sums[r['epoch']]['native_U_count'],
        native_accepted_total=sums[r['epoch']]['native_accepted_count'], native_U_total=sums[r['epoch']]['native_U_count'],
        source_val_accuracy_pct=c.finite(r['val_tx_acc'],'source accuracy'), lr=c.finite(r['lr'],'lr'),
        grad_backbone=c.finite(r['train_grad_backbone'],'gradient'),epoch_wall_seconds=c.finite(r['epoch_time_s'],'wall')) for r in epochs]
    c.require(abs(curves[0]['lr']-.0002)<1e-15 and abs(curves[-1]['lr']-.000001)<1e-15,
              'Original reset cosine10 schedule required')
    return dict(status='FULL_TAIL_LOGS_VERIFIED',updates=2220,epochs=10,L_NN=1704960,
        U_packets=total_u,native_pseudo_selected=accepted,acceptance_is_not_precision=True,
        maximum_L_decomposition_error=max_error,cost=cost,curve=curves), ordinary


def same_sequence(first, second):
    roles = Counter(); count = 0
    for a,b in zip_longest(c.lines(first),c.lines(second)):
        c.require(a is not None and b is not None and a == b,'Complete physical L/U source sequence differs')
        c.require(a.get('role') in ('L','U') and isinstance(a.get('ids'),list),'Physical sequence schema differs')
        roles[a['role']] += len(a['ids']); count += 1
    c.require(roles['L'] == 2220*128 and roles['U'] == 567000,'Complete native L/U sequence coverage required')
    return dict(records=count,packet_occurrences=dict(roles),exact=True)


def pair_recount(records, physics, expected_steps=2220, packets_per_view=64):
    """Only saved channel metadata is read, never waveform arrays or new draws."""
    counts = Counter(); strata = Counter(); pairs = Counter(); sums = Counter(); steps_out = []
    for step, (r,p) in enumerate(zip_longest(records,physics),1):
        c.require(r is not None and p is not None and r.get('step') == p.get('step') == step and
                  r.get('cumulative_step') == p.get('cumulative_step') == step+4440,'All independent pair steps/physics required')
        c.require(r.get('method') == 'INDEP_CE6' and r.get('L_NN') == 768 and r.get('extra_NN') == 0 and
                  r.get('generated_As0') == r.get('generated_As1') == packets_per_view and
                  r.get('same_hardware') is True and r.get('As_same_IQ') is False and
                  r.get('ordinary_raw_same_IQ') is True and r.get('all_labeled_CE_retained') is True and
                  r.get('two_separate_CE_mean') is True and r.get('cross_channel_KL') is False and
                  r.get('bypassed_elements') == 0,'Same sixL/CE/dropout, distinct full views/no cross-channel KL required')
        mse = c.finite(r['As_IQ_pair_MSE'],'As pair MSE'); c.require(mse >= 0,'Nonnegative independent-view diagnostic required')
        for key in PAIR_MEANS:sums[key] += c.finite(r[key],key)
        c.require(abs(r['used_mixed_KL']-.5*r['ordinary_half_KL']) < 3e-5 and
                  abs(r['removed_satellite_KL']-.5*r['satellite_half_KL']) < 3e-5,
                  'Only ordinary mixed KL may contribute to training')
        for a,b in zip_longest(p.get('view0',[]),p.get('view1',[])):
            c.require(a is not None and b is not None and a.get('sample_id') == b.get('sample_id') and
                      a.get('session_id') == b.get('session_id') and
                      b.get('namespace') == a.get('namespace','')+'/view1-full-independent' and
                      b.get('hardware_object_reused') is True and b.get('from_same_raw') is True,
                      'Same raw physical ID/hardware and independent complete named stream required')
            for view, obs in (('view0',a),('view1',b)):
                pid = obs.get('sample_id')
                c.require(isinstance(pid,list) and len(pid)==5 and pid[1] in c.RX and pid[2] in c.DAYS and
                          obs.get('scene') in c.SCENES and obs.get('state_end') in c.STATES and
                          obs.get('receiver_lock_status') in c.LOCKS and obs.get('channel_equalization_applied') is False,
                          'Complete full residual/EQoff source physics required')
                scene,state,lock=obs['scene'],obs['state_end'],obs['receiver_lock_status']
                counts[view]+=1
                strata[(view,int(pid[1]),int(pid[2]),scene,state,lock)]+=1
            pairs[(a['scene'],b['scene'],a['state_end'],b['state_end'],a['receiver_lock_status'],b['receiver_lock_status'])]+=1
        c.require(len(p.get('view0',[]))==len(p.get('view1',[]))==packets_per_view,'Exactly64 physical observations in each view required')
        steps_out.append(dict(step=step,cumulative_step=step+4440,epoch=(step-1)//STEPS+21,
            As_IQ_pair_MSE=mse,satellite_CE0=c.finite(r['satellite_CE0'],'CE0'),satellite_CE1=c.finite(r['satellite_CE1'],'CE1')))
    c.require(len(steps_out)==expected_steps and counts==dict(view0=expected_steps*packets_per_view,view1=expected_steps*packets_per_view),
              'Complete independent-pair physical support required')
    physical=[dict(view=v,RX=rx,day=d,scene=s,state=st,lock=lk,count=n) for (v,rx,d,s,st,lk),n in sorted(strata.items())]
    transitions=[dict(scene0=s0,scene1=s1,state0=st0,state1=st1,lock0=l0,lock1=l1,count=n) for (s0,s1,st0,st1,l0,l1),n in sorted(pairs.items())]
    return dict(status='SAVED_PAIR_PHYSICS_RECOUNTED',steps=len(steps_out),counts=dict(counts),
                means={key:value/len(steps_out) for key,value in sums.items()},
                physical_strata=physical,pair_transitions=transitions,NN=0,IQ_reads=0,checkpoint_reads=0,
                physical_independence_proven_by_draws=False,total_SNR_known=False),steps_out


def validate_pair_summary(saved,pair):
    n=pair['steps'];first=pair['counts']['view0'];second=pair['counts']['view1']
    expected=dict(method='INDEP_CE6',steps=n,generated_As0=first,generated_As1=second,
        generated_As_total=first+second,As_forward_NN=first+second,L_NN=n*768,
        total_L_CE_scale=1.25,mixed_group_weight=.5,As_fraction=.5,As_CE_coefficient=.25,
        per_pass_As_CE_coefficient=.125,ordinary_raw_KL_unchanged=True,As_KL=False,
        original_dropout=True,dropout_bypassed_elements=0,same_hardware_steps=n,
        full_physics_steps=n,view0_namespace_unchanged=True,second_from_same_raw=True,
        second_from_first_As=False,session_count=15,extra_NN=0,extra_U_NN=0,
        extra_parameters=0,extra_losses=0,U_truth_read=False,target_access=False)
    c.require(isinstance(saved,dict) and all(saved.get(key)==value for key,value in expected.items()),
              'Actual independent-pair summary budget/semantics differs')
    for key,value in pair['means'].items():
        c.require(abs(c.finite(saved.get(key+'_mean'),key+'_mean')-value)<3e-5,
                  'Saved whole-tail pair mean differs: '+key)


def training_analysis(project,specs,out):
    tails={};curves=[];coverage=[];by_key={s['key']:s for s in specs}
    for spec in specs:
        key=spec['key'];tail,ordinary=parse_tail(spec['root']/'source',key!='NOAS30',key=='INDEP_CE6')
        tail['parent_run']=spec['parent_run'];tail['parent_row_id']=spec['parent_row_id'];tail['parent_As_KL_off']=spec['parent_As_KL_off']
        if key=='INDEP_CE6':
            control=by_key['SAME_CE6'];tail['physical_sequence']=same_sequence(spec['root']/'source/training_sequence.jsonl.gz',control['root']/'source/training_sequence.jsonl.gz')
            c.require(ordinary==tails['SAME_CE6']['ordinary'],'Unchanged ordinary augmentation MSE stream required')
            result=c.read(spec['root']/'source/result.json')
            pair,steps=pair_recount(c.lines(spec['root']/'source/full_as_pair_steps.jsonl.gz'),c.lines(spec['root']/'source/full_as_pair_physics.jsonl.gz'))
            validate_pair_summary(result['full_as_pair_summary'],pair)
            tail['pair_physics']=pair;tail['full_as_pair_summary']=result['full_as_pair_summary']
            c.csv_write(out/'pair_steps.csv',steps);c.csv_write(out/'pair_physical_strata.csv',pair['physical_strata']);c.csv_write(out/'pair_transitions.csv',pair['pair_transitions'])
            coverage=coverage_tables(result,pair)
        tail['ordinary']=ordinary
        # New candidate shares the existing SAT18 parent; do not invent its own stage20 folder.
        for r in c.epoch_curve(spec['parent'],0)+tail['curve']:
            curves.append(dict(row_id=key,reused=spec['reused'],model_seed=SEED,parent_reused=True,**r))
        tails[key]=tail
    for t in tails.values():t.pop('ordinary')
    c.csv_write(out/'training_curves.csv',curves);c.csv_write(out/'pair_TX_RX_scene_state_lock.csv',coverage)
    result=dict(status='FULL_AVAILABLE_TRAINING_LOGS_VERIFIED',rows=tails,new_model_epochs=10,new_updates=2220,
        parent_epoch20_reused=True,parent_history_difference=True,extra_NN=0,U_truth_read=False,
        native_U_temporal_mode='batch_neighbor',coverage_is_not_identity_performance=True)
    c.write(out/'training_summary.json',result)
    return result,curves


def coverage_tables(result,pair):
    """Normalize producer's labeled coverage; physical RX totals are independently recounted."""
    output=[]
    tables=dict(view0=result.get('TX_RX_scene_state_lock_view0'),
                view1=result.get('TX_RX_scene_state_lock_view1'),
                combined=result.get('TX_RX_scene_state_lock'))
    c.require(all(isinstance(v,list) for v in tables.values()),'Both physical views and combined labeled coverage required')
    c.require(result.get('L_As_total')==result.get('L_As_forward_NN')==284160,'Both generated/forward As view counts required')
    normalized={}
    for view,rows in tables.items():
        table=Counter()
        for row in rows:
            k=(int(row['TX']),int(row['RX']),row['scene'],row['state'],row['lock']);n=row['count']
            c.require(k[0] in range(6) and k[1] in c.RX and k[2] in c.SCENES and k[3] in c.STATES and k[4] in c.LOCKS and type(n) is int and n>0 and k not in table,'Actual labeled coverage cell differs')
            table[k]=n
        c.require(sum(table.values())==142080*(2 if view=='combined' else 1),'Labeled view packet total differs')
        normalized[view]=table
        for tx in range(6):
            for rx in c.RX:
                for scene in c.SCENES:
                    c.require(sum(n for (t,r,s,_,_),n in table.items() if (t,r,s)==(tx,rx,scene))>0,'Missing TX x RX x scene support')
                    for state in c.STATES:
                        for lock in c.LOCKS:output.append(dict(view=view,TX=tx,RX=rx,scene=scene,state=state,lock=lock,count=table[(tx,rx,scene,state,lock)]))
    c.require(normalized['combined']==normalized['view0']+normalized['view1'],'Combined labeled coverage does not reconcile')
    for view in ('view0','view1'):
        a=Counter();b=Counter()
        for (_,rx,scene,state,lock),n in normalized[view].items():a[(rx,scene,state,lock)]+=n
        for r in pair['physical_strata']:
            if r['view']==view:b[(r['RX'],r['scene'],r['state'],r['lock'])]+=r['count']
        c.require(a==b,'Actual physical and labeled RX/scene/state/lock coverage differ')
    return output


def costs(project,specs,local_nn=0,smoke_nn=772):
    rows={};new={};historical={}
    for spec in specs:
        tail=c.native_cost(spec['root']/'source');q=c.read(spec['root']/'source_qualification/complete.json');p=c.read(spec['root']/'prediction/complete.json')
        c.require(q.get('NN')==12600 and p.get('actual_NN')==1176002,'Actual source/formal forward counters differ')
        record=dict(parent=c.native_cost(spec['parent']),tail=tail,source_NN=q['NN'],formal_NN=p['actual_NN'],
            source_wall_seconds=q.get('wall_seconds'),formal_wall_seconds=p.get('wall_seconds'),parent_run=spec['parent_run'],parent_row_id=spec['parent_row_id'])
        rows[spec['key']]=record;(historical if spec['reused'] else new)[spec['key']]=record
    smoke=c.read(Path(project)/'runs'/RUN/'real_checkpoint_smoke.json')
    c.require(smoke.get('status')=='PASS' and smoke.get('NN')==smoke_nn==772 and local_nn==0,'One actual shared772 NN smoke/local0 required')
    total=sum(r['tail']['training_total_NN']+r['source_NN']+r['formal_NN'] for r in new.values())+smoke_nn
    c.require(len(new)==1 and new['INDEP_CE6']['tail']['native_NN']==3162962 and total==4352336,'One native10E/source/formal/smoke budget differs')
    return dict(rows=rows,new_native_NN=3162962,new_source_NN=12600,new_formal_NN=1176002,shared_remote_smoke_NN=772,
        new_total_remote_NN=total,local_verification_NN=0,new_model_epochs=10,new_updates=2220,
        student_parent_epochs=20,student_parent_updates=4440,student_cumulative_epochs=30,student_cumulative_updates=6660,
        parent_training_reused_not_repeated=True,parent_As_KL_off=True,parent_run=SAME,parent_boot_commit=BOOT_COMMIT,
        historical_reference_model_epochs=90,historical_reference_updates=19980,
        unique_total_model_epochs_including_reused=100,unique_total_updates_including_reused=22200,
        new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),extra_student_NN=0,extra_teacher_NN=0,
        reference_reprediction=0,reference_rescoring=0,analysis_NN=0,budget_exhaustion_is_not_convergence=True)


def plots(out,curves,source_rows,formal,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#64748b','#d97706','#8b5cf6','#2563eb')));files=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=180);files.append(str(path))
        plt.close(fig)
    if curves:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for arm in ARMS:
            rr=[r for r in curves if r['row_id']==arm]
            for ax,key,title in zip(axes,('native_loss','source_val_accuracy_pct','grad_backbone'),('Native training loss','Source V accuracy (%)','Backbone gradient norm')):
                ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],label=arm,color=colors[arm]);ax.set(xlabel='Cumulative epoch',title=title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);save(fig,'training_curves')
    if source_rows:
        fig,axes=plt.subplots(2,3,figsize=(17,8))
        for si,scene in enumerate(c.SCENES[1:]):
            for di,dim in enumerate(('state','lock','signal_to_added_noise_db')):
                ax=axes[si,di];rr=[r for r in source_rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dim];labels=list(dict.fromkeys(r['stratum'] for r in rr));xx=np.arange(len(labels))
                for i,arm in enumerate(ARMS):
                    selected=[next(r for r in rr if r['row_id']==arm and r['stratum']==label) for label in labels]
                    ax.bar(xx+(i-1.5)*.2,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in selected],.2,label=arm,color=colors[arm])
                ax.set(xticks=xx,xticklabels=labels,title=scene+' / '+dim,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=8);save(fig,'source_quality_state_lock_noise')
    fig,axes=plt.subplots(1,2,figsize=(15,5));xx=np.arange(7)
    for arm in ARMS:axes[0].plot(xx,[100*formal['rows'][arm][v]['accuracy'] for v in c.VIEWS],marker='.',label=arm,color=colors[arm])
    for new,old,label in (('INDEP_CE6','SAME_CE6','Independent - same (primary)'),('INDEP_CE6','AS30','Independent - P43 As (history differs)'),('INDEP_CE6','NOAS30','Independent - P43 NoAs (recipe reference)')):
        axes[1].plot(xx,[100*(formal['rows'][new][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS],marker='.',label=label)
    for ax,title,ylabel in zip(axes,('Frozen all7 default-head readout','Paired changes; one seed'),('Accuracy (%)','Absolute percentage points')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title,ylabel=ylabel);ax.tick_params(axis='x',labelrotation=30);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    axes[1].axhline(0,color='grey',lw=.7);save(fig,'formal_paired')
    if training:
        pair=training['rows']['INDEP_CE6']['pair_physics'];fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,dim,allowed in zip(axes,('scene','state','lock'),(c.SCENES,c.STATES,c.LOCKS)):
            xx=np.arange(len(allowed))
            for i,view in enumerate(('view0','view1')):
                hist=Counter()
                for r in pair['physical_strata']:
                    if r['view']==view:hist[r[dim]]+=r['count']
                ax.bar(xx+(i-.5)*.35,[hist[a] for a in allowed],.35,label=view)
            ax.set(xticks=xx,xticklabels=allowed,title='Actual full-channel '+dim,ylabel='Generated observations');ax.tick_params(axis='x',labelrotation=20);ax.legend();ax.grid(axis='y',alpha=.2)
        save(fig,'pair_physical_coverage')
    return files


def report(out,result):
    formal=result['formal'];rows=formal['rows'];cost=result['cost'];stage=formal['stage']['INDEP_CE6']
    text=['# P68：固定六次前向的独立完整 practical 双视图','',
        '唯一新增训练从 SAT18 AS_CE20 的 student 和 own evolving EMA 分叉，追加10轮；同父 SAME_CE6 复用既存 SAT18 AS_CE30。两个有标签 satellite 前向分别接收同一 raw IQ 和 RX hardware 下的独立完整 practical 实现；两次 CE 均保留。普通与 raw 同 IQ RDrop、原 SupCon、native U、默认 cosine30 分类头和六次 L 前向保留。没有新增 Ag、部署模块、教师前向或额外一致性损失。','',
        '主比较是 INDEP_CE6−SAME_CE6。原 P43 As20 的前20轮保留 As KL，SAT18 共同父20轮已关闭 As KL；因此相对 P43 As30 的整配方差异不能只归因于新增独立信道视图。P43 NoAs30 为同预算、同 seed、同数据角色的整配方参照，没有额外训练同 AS_CE20 来源的 NoAs 控制。','',
        '| 环境 | P43 NoAs % | P43 As % | SAME CE6 % | INDEP CE6 % | INDEP−SAME pp |','|---|---:|---:|---:|---:|---:|']
    for v in c.VIEWS:
        text.append('| '+v+' | '+' | '.join(f'{100*rows[a][v]["accuracy"]:.6f}' for a in ARMS)+f' | {formal["primary"]["by_view"][v]["accuracy_delta_pp"]:+.6f} |')
    primary_urban=100*np.mean([rows['INDEP_CE6'][v]['accuracy']-rows['SAME_CE6'][v]['accuracy'] for v in c.VIEWS[-3:]])
    text += ['',f'相对 SAME：urban 三场景均值 {primary_urban:+.6f} pp，clean {formal["primary"]["by_view"]["clean"]["accuracy_delta_pp"]:+.6f} pp。相对 P43 NoAs30：clean {stage["clean_delta_vs_NoAs30_pp"]:+.6f} pp，urban 均值 {stage["urban_mean_vs_NoAs30_pp"]:+.6f} pp；clean 下降不超过0.1绝对百分点：{stage["clean_within_0p1pp"]}。',
        '只有一个模型 seed；源域两个 channel draw 是重复观测，不能证明跨 seed 可重复双提升，小于0.1个百分点的上涨只能作初步观察。源判据在正式读出前已固定，本分析只报告，不按正式分数晋级或追加预算。', '',
        '全部七环境 accuracy、Macro-F1、最差 RX，逐 RX/day/TX、修复与新增错误、margin/30、固定 P43 NoAs30 clean 正确集合保持率见 CSV；这个集合不是 AS_CE20 父模型 clean 正确集合。源面板提供两 draw 的 LOS/PARTIAL/BLOCKED、lock 和 added-noise 分层。新增噪声比不是总 SNR，分层关联不能识别物理因果。完整 V27000 F1 与正式逐包状态未保存，不推断。', '',
        f'本次新增10 model-epochs／2220更新；父20轮已训练成本单列且不重复计新成本。新增 native NN={cost["new_native_NN"]}，source={cost["new_source_NN"]}，formal={cost["new_formal_NN"]}，共享smoke={cost["shared_remote_smoke_NN"]}，远端合计={cost["new_total_remote_NN"]}，本地NN=0。训练墙钟={cost["new_training_wall_seconds_sum"]:.3f}秒；独立完整信道额外CPU工作包含在实际墙钟内。预算耗尽不等于收敛。', '',
        '已存源判定：'+json.dumps(result['stored_source_decision'],ensure_ascii=False),'',
        '未完成／限制：'+json.dumps(result['missing'],ensure_ascii=False),'','实际图表：']
    text += ['- '+p for p in result['figures']]
    text += ['','离线分析 NN、训练更新、IQ/checkpoint 读取、重推理与重新评分均为0。']
    (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def run(project,out,truth=None,code_commit=None,local_nn=0,smoke_nn=772):
    project=Path(project);decision=read_source_decision(project)
    out=Path(out);c.require(not out.exists(),'Immutable new analysis output required');out.mkdir(parents=False)
    c.write(out/'stored_source_decision.json',decision);specs=row_specs(project);started=time.perf_counter()
    missing=[];training=None;curves=[];source=None;source_rows=[]
    if decision['status'] in c.DECISION_PARTIAL:missing.append(dict(component='source_decision',error='Stored producer decision PARTIAL; not recomputed'))
    for name in ('training','source'):
        try:
            if name=='training':training,curves=training_analysis(project,specs,out)
            else:source,source_rows=c.source_analysis(project,specs,out,decision)
        except Exception as error:
            failure=dict(component=name,error=repr(error),traceback=traceback.format_exc(),formal_still_required=True);missing.append(failure);c.write(out/(name+'_failure.json'),failure)
    truth=Path(truth) if truth else project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal=c.formal_analysis(project,specs,out,truth);cost=costs(project,specs,local_nn,smoke_nn);c.write(out/'costs.json',cost)
    try:figures=plots(out,curves,source_rows,formal,training)
    except Exception as error:
        figures=[];failure=dict(component='plots',error=repr(error),traceback=traceback.format_exc());missing.append(failure);c.write(out/'plot_failure.json',failure)
    result=dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',run_id=RUN,
        code_commit=code_commit,stored_source_decision=decision,source=source,formal=formal,training=training,cost=cost,
        figures=figures,missing=missing,model_seed_count=1,crossseed_confirmed=False,reproducible_dual_gain=False,
        source_decision_recomputed=False,target_feedback=False,goal_complete=False,NN=0,IQ_reads=0,checkpoint_reads=0,
        training_updates=0,wall_seconds=time.perf_counter()-started)
    report(out,result);c.write(out/'summary.json',{k:v for k,v in result.items() if k!='training'});c.write(out/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--truth',type=Path)
    parser.add_argument('--code-commit');parser.add_argument('--local-nn',type=int,default=0);parser.add_argument('--smoke-nn',type=int,default=772)
    parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:
        path=Path(__file__).with_name('cvs_full_channel_pair68_analysis_checks_20261018.py')
        spec=importlib.util.spec_from_file_location('_p68_analysis_checks',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.checks();return
    if args.project is None or args.out is None:parser.error('--project and --out required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
