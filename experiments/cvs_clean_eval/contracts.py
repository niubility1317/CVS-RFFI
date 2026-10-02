"""Source selection/provenance checks; never read target truth or IQ here."""
import json
from pathlib import Path
from experiments.cvs_clean_design.model import BASELINES,VARIANTS as FIRST
from experiments.cvs_residual_identity.model import VARIANTS as SECOND
from experiments.cvs_balanced_identity.model import VARIANTS as THIRD
from experiments.cvs_interaction_identity.model import VARIANTS as FOURTH
from experiments.cvs_attentive_identity.model import VARIANTS as FIFTH
from experiments.cvs_stability_identity.model import VARIANTS as SIXTH
from experiments.cvs_coherence_identity.model import VARIANTS as SEVENTH
from experiments.cvs_simplex_identity.model import VARIANTS as EIGHTH
from experiments.cvs_rf_operator_identity.model import VARIANTS as NINTH, operator_contract
from experiments.cvs_observable_identity.model import VARIANTS as TENTH, observable_contract
from experiments.cvs_gauge_identity.model import VARIANTS as ELEVENTH, gauge_contract
from experiments.cvs_reference_identity.model import VARIANTS as TWELFTH, response_contract
from experiments.cvs_equivariant_identity.model import VARIANTS as THIRTEENTH, equivariant_contract
from experiments.cvs_synchronized_identity.model import VARIANTS as FOURTEENTH, synchronized_contract
from experiments.cvs_coordinate_identity.model import VARIANTS as FIFTEENTH, coordinate_contract
from experiments.cvs_additive_identity.model import VARIANTS as SIXTEENTH, additive_contract
from experiments.cvs_fractional_identity.model import VARIANTS as SEVENTEENTH, fractional_contract
from experiments.cvs_residual_identity.dispatch import combine_research_selection

SEEDS={2026092701,2026092702,2026092703,2026092704}
CANDIDATES=set(FIRST)-set(BASELINES)|set(SECOND)
REUSED_BASELINE_RUN='20261001-phase1-clean-baselines-manysig-m16-r01'
REUSED_RESIDUAL_RUN='20261001-phase1-cvs-selected-clean-manysig-m20-r01'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen_selection(path):
    selection=read(path)
    if selection.get('scope')=='fractional_source':
        from experiments.cvs_fractional_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_fractional_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Fractional selection differs from actual twelve source-only records')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in SEVENTEENTH:
            raise ValueError('Existing baseline retained; unselected fractional model cannot receive new query')
        return selection
    if selection.get('scope')=='additive_source':
        from experiments.cvs_additive_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_additive_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Additive selection differs from actual twelve source-only records')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in SIXTEENTH:
            raise ValueError('Existing baseline retained; unselected additive model cannot receive new query')
        return selection
    if selection.get('scope')=='coordinate_source':
        from experiments.cvs_coordinate_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_coordinate_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Coordinate selection differs from actual twelve source-only records')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in FIFTEENTH:
            raise ValueError('Existing baseline retained; unselected coordinate model cannot receive new query')
        return selection
    if selection.get('scope')=='synchronized_source':
        from experiments.cvs_synchronized_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_synchronized_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Synchronized selection differs from actual twelve source-only records')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in FOURTEENTH:
            raise ValueError('Existing baseline retained; unselected synchronized model cannot receive new query')
        return selection
    if selection.get('scope')=='equivariant_source':
        from experiments.cvs_equivariant_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_equivariant_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Equivariant source/control selection differs from actual source-only rule')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in THIRTEENTH:
            raise ValueError('Existing baseline retained; unselected equivariant model cannot receive new query')
        return selection
    if selection.get('scope')=='reference_source':
        from experiments.cvs_reference_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_reference_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Reference source/control selection differs from actual source-only rule')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in TWELFTH:
            raise ValueError('Existing baseline retained; unselected reference model cannot receive new query')
        return selection
    if selection.get('scope')=='gauge_source':
        from experiments.cvs_gauge_identity.dispatch import validate_spec,read_source_record,select_source_candidate,PROJECT
        matrix=validate_spec(read(selection['source_matrix_ref']))
        original=read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
        records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
        records += [read_source_record(r,original,'cvs_gauge_identity') for r in matrix['rows']]
        actual=select_source_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Gauge source/control selection differs from actual source-only rule')
        if not actual['new_candidate_selected'] or actual['selected_variant'] not in ELEVENTH:
            raise ValueError('Existing baseline retained; unselected gauges cannot receive new query')
        return selection
    if selection.get('scope') in ('balanced_source','interaction_source','attentive_source','stability_source','coherence_source','simplex_source','rf_operator_source','observable_source','gauge_source','reference_source','equivariant_source','synchronized_source','coordinate_source','additive_source','fractional_source'):
        if selection['scope']=='balanced_source':
            from experiments.cvs_balanced_identity.freeze import select_performance_candidate
            family=THIRD
        elif selection['scope']=='observable_source':
            from experiments.cvs_observable_identity.dispatch import select_source_candidate as select_performance_candidate
            family=TENTH
        elif selection['scope']=='rf_operator_source':
            from experiments.cvs_rf_operator_identity.dispatch import select_source_candidate as select_performance_candidate
            family=NINTH
        elif selection['scope']=='simplex_source':
            from experiments.cvs_simplex_identity.dispatch import select_source_candidate as select_performance_candidate
            family=EIGHTH
        elif selection['scope']=='coherence_source':
            from experiments.cvs_coherence_identity.dispatch import select_source_candidate as select_performance_candidate
            family=SEVENTH
        elif selection['scope']=='stability_source':
            from experiments.cvs_stability_identity.dispatch import select_source_candidate as select_performance_candidate
            family=SIXTH
        elif selection['scope']=='attentive_source':
            from experiments.cvs_attentive_identity.dispatch import select_source_candidate as select_performance_candidate
            family=FIFTH
        else:
            from experiments.cvs_interaction_identity.dispatch import select_source_candidate as select_performance_candidate
            family=FOURTH
        matrix=read(selection['source_matrix_ref']);records=[]
        expected={(v,s) for v in family for s in SEEDS}
        if len(matrix['rows'])!=8 or {(r['variant'],r['model_seed']) for r in matrix['rows']}!=expected:
            raise ValueError('Balanced source matrix incomplete')
        for row in matrix['rows']:
            q=Path(row['source_output']);done=read(q/'completion.json');profile=read(q/'resource_profile.json')
            if done['status']!='SOURCE_TRAINED' or done['epoch']!=200 or done['steps']!=10000 or done['target_access'] or done['target_evaluated']:
                raise ValueError('Balanced source selection before completion')
            m=done['final_source_metrics']
            records.append(dict(variant=row['variant'],seed=row['model_seed'],accuracy=m['source_val_accuracy'],worst_rx=m['source_val_worst_rx'],parameters=profile['total_parameters'],macs=profile['conv_linear_macs_per_sample']))
        actual=select_performance_candidate(records)
        if any(selection.get(k)!=v for k,v in actual.items()):raise ValueError('Balanced selection differs from actual source-only rule')
        return selection
    if selection.get('scope')=='baseline_only':
        if (selection.get('status')!='FIXED_BASELINES_FROZEN' or selection.get('target_access') is not False or
            selection.get('target_score_used') is not False or selection.get('test_variants')!=list(BASELINES) or
            selection.get('model_seeds')!=sorted(SEEDS)):
            raise ValueError('Fixed baseline plan changed')
        source=read(selection['source_matrix_ref'])
        expected={(v,s) for v in BASELINES for s in SEEDS}
        actual={(r['variant'],r['model_seed']) for r in source['rows']}
        if not expected<=actual:raise ValueError('Baselines absent from original source matrix')
        return selection
    if len(selection.get('source_selection_refs',[]))!=2:
        raise ValueError('Missing completed source selections')
    first,second=[read(p) for p in selection['source_selection_refs']]
    actual=combine_research_selection(first,second)
    if any(selection.get(key)!=value for key,value in actual.items()):
        raise ValueError('Research selection differs from frozen source rule')
    if selection['selected_variant'] not in CANDIDATES:
        raise ValueError('No registered final CVS variant')
    return selection


def evaluation_variants(selection):
    if selection.get('scope') in ('balanced_source','interaction_source','attentive_source','stability_source','coherence_source','simplex_source','rf_operator_source','observable_source','gauge_source','reference_source','equivariant_source','synchronized_source','coordinate_source','additive_source','fractional_source'):return [*BASELINES,'residual_fusion',selection['selected_variant']]
    return list(BASELINES) if selection.get('scope')=='baseline_only' else [*BASELINES,selection['selected_variant']]


def validate_reused_row(row):
    residual=row.get('reuse_from_run')==REUSED_RESIDUAL_RUN and row['variant']=='residual_fusion'
    baseline=row.get('reuse_from_run')==REUSED_BASELINE_RUN and row['variant'] in BASELINES
    if not (residual or baseline):raise ValueError('Only fixed completed baseline/previous residual rows can be reused')
    old_run=REUSED_RESIDUAL_RUN if residual else REUSED_BASELINE_RUN
    out=Path(row['output_root'])
    if out.parts[-3:]!=(old_run,row['row_id'],'prediction'):raise ValueError('Reused output outside original run')
    cfg=read(row['config']);original=frozen_selection(cfg['selection_file'])
    if baseline and original.get('scope')!='baseline_only':raise ValueError('Reused baseline selection changed')
    if residual and (original.get('scope') in ('baseline_only','balanced_source','interaction_source','attentive_source','stability_source','coherence_source','simplex_source','rf_operator_source','observable_source','gauge_source','reference_source','equivariant_source','synchronized_source','coordinate_source','additive_source','fractional_source') or original.get('selected_variant')!='residual_fusion'):
        raise ValueError('Previous residual source selection changed')
    validate_predict_config(cfg,original)
    marker=read(out.parents[1]/'scoring_clean_complete.json');models=5 if residual else 4
    if marker['status']!='SCORED_COMPLETE' or marker['models']!=models or marker['seeds']!=4 or marker['rows']!=models*4:
        raise ValueError('Original reused matrix not complete')
    return cfg


def source_method(variant):
    if variant in SEVENTEENTH:return 'cvs_fractional_identity'
    if variant in SIXTEENTH:return 'cvs_additive_identity'
    if variant in FIFTEENTH:return 'cvs_coordinate_identity'
    if variant in FOURTEENTH:return 'cvs_synchronized_identity'
    if variant in THIRTEENTH:return 'cvs_equivariant_identity'
    if variant in TWELFTH:return 'cvs_reference_identity'
    if variant in ELEVENTH:return 'cvs_gauge_identity'
    if variant in TENTH:return 'cvs_observable_identity'
    if variant in NINTH:return 'cvs_rf_operator_identity'
    if variant in EIGHTH:return 'cvs_simplex_identity'
    if variant in SEVENTH:return 'cvs_coherence_identity'
    if variant in SIXTH:return 'cvs_stability_identity'
    if variant in FIFTH:return 'cvs_attentive_identity'
    if variant in FOURTH:return 'cvs_interaction_identity'
    if variant in THIRD:return 'cvs_balanced_identity'
    if variant in SECOND:return 'cvs_residual_identity'
    if variant in FIRST:return 'cvs_clean_design'
    raise ValueError('Unregistered architecture')


def validate_predict_config(c,selection):
    if any(c.get(k) for k in ('truth','p1_truth','target_truth','resume','teacher','initial_checkpoint')):
        raise ValueError('Frozen predictor rejects truth and inheritance overrides')
    if c.get('views')!=['clean'] or c.get('method')!='cvs_clean_eval' or c.get('model_seed') not in SEEDS:
        raise ValueError('Unregistered clean evaluation contract')
    if c['variant'] not in evaluation_variants(selection):
        raise ValueError('Nonselected CVS cannot receive target predictions')
    expected_parent=c['fractional_source_root'] if c['variant'] in SEVENTEENTH else c['additive_source_root'] if c['variant'] in SIXTEENTH else c['coordinate_source_root'] if c['variant'] in FIFTEENTH else c['synchronized_source_root'] if c['variant'] in FOURTEENTH else c['equivariant_source_root'] if c['variant'] in THIRTEENTH else c['reference_source_root'] if c['variant'] in TWELFTH else c['gauge_source_root'] if c['variant'] in ELEVENTH else c['observable_source_root'] if c['variant'] in TENTH else c['rf_operator_source_root'] if c['variant'] in NINTH else c['simplex_source_root'] if c['variant'] in EIGHTH else c['coherence_source_root'] if c['variant'] in SEVENTH else c['stability_source_root'] if c['variant'] in SIXTH else c['attentive_source_root'] if c['variant'] in FIFTH else c['interaction_source_root'] if c['variant'] in FOURTH else c['balanced_source_root'] if c['variant'] in THIRD else (c['baseline_source_root'] if c['variant'] in FIRST else c['residual_source_root'])
    expected=Path(expected_parent)/(c['variant']+'-s'+str(c['model_seed']))/'source'
    if Path(c['source_output'])!=expected:raise ValueError('Source row/seed path mismatch')
    return c


def checkpoint_contract(c,done,initial,contract,expected,resolved,payload):
    if (done['status']!='SOURCE_TRAINED' or done['epoch']!=200 or done['steps']!=10000 or done['target_access'] or done['target_evaluated'] or
        initial['status']!='SCRATCH' or not initial['scratch_only'] or initial['checkpoint'] is not None or initial['ancestors'] or
        initial['checkpoint_sources'] or initial['target_access'] or initial['target_contact'] or initial['model_seed']!=c['model_seed']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED_OR_TARGET_CONTAMINATED')
    # The original role contract predates runtime input/class-map extensions.
    keys=('role_ids','source_rxs','source_days','ratios','split_seed')
    if any(contract[k]!=expected[k] for k in keys):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if expected['num_classes']!=6:raise ValueError('Expected class count mismatch')
    for key in ('classes','equalized','out_len','normalize'):
        if key in expected and contract[key]!=expected[key]:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)
    if contract['equalized']!=1 or contract['out_len']!=256 or not contract['normalize'] or len(contract['classes'])!=6:
        raise ValueError('Source input/class contract mismatch')
    if (resolved['method']!=source_method(c['variant']) or resolved['variant']!=c['variant'] or resolved['model_seed']!=c['model_seed'] or
        resolved['augmentation'] or resolved['domain_backbone'] or resolved['extra_losses'] or resolved['target_access'] or
        resolved['epochs']!=200 or resolved['steps_per_epoch']!=50 or resolved['selection']!='fixed_last_epoch' or
        resolved['source_counts']!={'L_s':6300,'U_s':56700,'V':27000}):raise ValueError('Source resolved training contract mismatch')
    if c['variant'] in SEVENTEENTH:
        from experiments.cvs_fractional_identity.source import validate_config
        validate_config(resolved)
        if resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']:
            raise ValueError('Frozen fractional full-FP32 source provenance mismatch')
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Fractional scratch provenance differs')
        if resolved.get('fractional_active') is not True or resolved.get('fractional_actual')!=fractional_contract(c['variant']) or resolved.get('fractional')!=fractional_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Fractional waveform correction differs from registered source variant')
    if c['variant'] in SIXTEENTH:
        from experiments.cvs_additive_identity.source import validate_config
        validate_config(resolved)
        if resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']:
            raise ValueError('Frozen additive full-FP32 source provenance mismatch')
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Additive scratch provenance differs')
        if resolved.get('coordinate_active') is not True or resolved.get('coordinate_actual')!=additive_contract(c['variant']) or resolved.get('coordinate')!=additive_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Wholeidentity additive coordinate injection differs from registered source variant')
    if c['variant'] in FIFTEENTH:
        from experiments.cvs_coordinate_identity.source import validate_config
        validate_config(resolved)
        if resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']:
            raise ValueError('Frozen coordinate full-FP32 source provenance mismatch')
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Coordinate scratch provenance differs')
        if resolved.get('coordinate_active') is not True or resolved.get('coordinate_actual')!=coordinate_contract(c['variant']) or resolved.get('coordinate')!=coordinate_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Wholeidentity coordinate retention differs from registered source variant')
    if c['variant'] in FOURTEENTH:
        from experiments.cvs_synchronized_identity.source import validate_config
        validate_config(resolved)
        if resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']:
            raise ValueError('Frozen synchronized full-FP32 source provenance mismatch')
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Synchronized scratch provenance differs')
        if resolved.get('synchronized_active') is not True or resolved.get('synchronized_actual')!=synchronized_contract(c['variant']) or resolved.get('synchronized')!=synchronized_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Wholeidentity synchronization differs from registered source variant')
    if c['variant'] in THIRTEENTH:
        from experiments.cvs_equivariant_identity.source import validate_config
        validate_config(resolved)
        if 'numerical_policy' in resolved and (resolved.get('backend_flags')!=resolved['numerical_policy'] or done.get('backend_flags')!=resolved['numerical_policy']):
            raise ValueError('Frozen equivariant source precision provenance mismatch')
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Equivariant scratch provenance differs')
        if resolved.get('equivariant_active') is not True or resolved.get('equivariant_actual')!=equivariant_contract(c['variant']) or resolved.get('equivariant')!=equivariant_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Wholeidentity equivariant activation differs from registered source variant')
    if c['variant'] in TWELFTH:
        from experiments.cvs_reference_identity.source import validate_config
        validate_config(resolved)
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Reference scratch provenance differs')
        if resolved.get('response_active') is not True or resolved.get('response_actual')!=response_contract(c['variant']) or resolved.get('response')!=response_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Known-excitation response activation differs from registered source variant')
    if c['variant'] in ELEVENTH:
        from experiments.cvs_gauge_identity.source import validate_config
        validate_config(resolved)
        if any(contract.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if initial.get('physical_roles')!='EXACT_MATCH' or initial.get('selection')!='fixed_last_epoch':raise ValueError('Gauge scratch provenance differs')
        if resolved.get('gauge_active') is not True or resolved.get('gauge_actual')!=gauge_contract(c['variant']) or resolved.get('gauge')!=gauge_contract(c['variant']) or resolved.get('classifier_scale')!=30.:
            raise ValueError('Wholeidentity phase gauge activation differs from registered source variant')
    if c['variant'] in TENTH and (resolved.get('observables_active') is not True or resolved.get('observables_actual')!=observable_contract(c['variant']) or resolved.get('observables')!=observable_contract(c['variant']) or resolved.get('classifier_scale')!=30.0):
        raise ValueError('Wholeidentity observable activation differs from registered source variant')
    if c['variant'] in NINTH and (resolved.get('rf_operator_active') is not True or resolved.get('rf_operator_actual')!=operator_contract(c['variant']) or resolved.get('rf_operator')!=operator_contract(c['variant']) or resolved.get('classifier_scale')!=30.0):
        raise ValueError('RF operator activation differs from registered source variant')
    if c['variant'] in EIGHTH and (resolved.get('coherence_phase_active') is not True or resolved.get('dsq_active') is not False or resolved.get('time_stability_channels')!=8 or resolved.get('freq_stability_channels')!=0 or resolved.get('coherence_epsilon')!=1e-6 or resolved.get('classifier_geometry')!=('learned_rotated_simplex' if c['variant']=='simplex_learned' else 'fixed_simplex') or resolved.get('classifier_scale')!=30.0):
        raise ValueError('Simplex classifier activation differs from registered source variant')
    if c['variant'] in SEVENTH and (resolved.get('coherence_phase_active') is not True or resolved.get('dsq_active')!=(c['variant']=='coherence_dsq') or resolved.get('time_stability_channels')!=8 or resolved.get('freq_stability_channels')!=(4 if c['variant']=='coherence_dsq' else 0) or resolved.get('coherence_epsilon')!=1e-6):
        raise ValueError('Complex coherence activation differs from registered source variant')
    if c['variant'] in SIXTH and (resolved.get('phase_delta_active') is not True or resolved.get('dsq_active')!=(c['variant']=='phase_dsq') or resolved.get('time_stability_channels')!=8 or resolved.get('freq_stability_channels')!=(4 if c['variant']=='phase_dsq' else 0)):
        raise ValueError('Physical cue activation differs from registered source variant')
    if (payload['epoch']!=200 or payload['source_contract']!=contract or payload['initialization']!=initial or
        payload['selection']!='fixed_last_epoch' or payload['method']!=source_method(c['variant']) or payload['variant']!=c['variant'] or
        payload['config']!=resolved or payload['classes']!=contract['classes'] or payload['num_classes']!=6):
        raise ValueError('Checkpoint contents disagree with source artifacts')


def build_model(variant):
    if variant in SEVENTEENTH:
        from experiments.cvs_fractional_identity.model import build
    elif variant in SIXTEENTH:
        from experiments.cvs_additive_identity.model import build
    elif variant in FIFTEENTH:
        from experiments.cvs_coordinate_identity.model import build
    elif variant in FOURTEENTH:
        from experiments.cvs_synchronized_identity.model import build
    elif variant in THIRTEENTH:
        from experiments.cvs_equivariant_identity.model import build
    elif variant in TWELFTH:
        from experiments.cvs_reference_identity.model import build
    elif variant in ELEVENTH:
        from experiments.cvs_gauge_identity.model import build
    elif variant in TENTH:
        from experiments.cvs_observable_identity.model import build
    elif variant in NINTH:
        from experiments.cvs_rf_operator_identity.model import build
    elif variant in EIGHTH:
        from experiments.cvs_simplex_identity.model import build
    elif variant in SEVENTH:
        from experiments.cvs_coherence_identity.model import build
    elif variant in SIXTH:
        from experiments.cvs_stability_identity.model import build
    elif variant in FIFTH:
        from experiments.cvs_attentive_identity.model import build
    elif variant in FOURTH:
        from experiments.cvs_interaction_identity.model import build
    elif variant in THIRD:
        from experiments.cvs_balanced_identity.model import build
    elif variant in SECOND:
        from experiments.cvs_residual_identity.model import build
    else:
        from experiments.cvs_clean_design.model import build
    return build(variant)
