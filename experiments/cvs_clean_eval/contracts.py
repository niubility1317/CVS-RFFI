"""Source selection/provenance checks; never read target truth or IQ here."""
import json
from pathlib import Path
from experiments.cvs_clean_design.model import BASELINES,VARIANTS as FIRST
from experiments.cvs_residual_identity.model import VARIANTS as SECOND
from experiments.cvs_balanced_identity.model import VARIANTS as THIRD
from experiments.cvs_interaction_identity.model import VARIANTS as FOURTH
from experiments.cvs_residual_identity.dispatch import combine_research_selection

SEEDS={2026092701,2026092702,2026092703,2026092704}
CANDIDATES=set(FIRST)-set(BASELINES)|set(SECOND)
REUSED_BASELINE_RUN='20261001-phase1-clean-baselines-manysig-m16-r01'
REUSED_RESIDUAL_RUN='20261001-phase1-cvs-selected-clean-manysig-m20-r01'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen_selection(path):
    selection=read(path)
    if selection.get('scope') in ('balanced_source','interaction_source'):
        if selection['scope']=='balanced_source':
            from experiments.cvs_balanced_identity.freeze import select_performance_candidate
            family=THIRD
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
    if selection.get('scope') in ('balanced_source','interaction_source'):return [*BASELINES,'residual_fusion',selection['selected_variant']]
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
    if residual and (original.get('scope') in ('baseline_only','balanced_source','interaction_source') or original.get('selected_variant')!='residual_fusion'):
        raise ValueError('Previous residual source selection changed')
    validate_predict_config(cfg,original)
    marker=read(out.parents[1]/'scoring_clean_complete.json');models=5 if residual else 4
    if marker['status']!='SCORED_COMPLETE' or marker['models']!=models or marker['seeds']!=4 or marker['rows']!=models*4:
        raise ValueError('Original reused matrix not complete')
    return cfg


def source_method(variant):
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
    expected_parent=c['interaction_source_root'] if c['variant'] in FOURTH else c['balanced_source_root'] if c['variant'] in THIRD else (c['baseline_source_root'] if c['variant'] in FIRST else c['residual_source_root'])
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
    if (payload['epoch']!=200 or payload['source_contract']!=contract or payload['initialization']!=initial or
        payload['selection']!='fixed_last_epoch' or payload['method']!=source_method(c['variant']) or payload['variant']!=c['variant'] or
        payload['config']!=resolved or payload['classes']!=contract['classes'] or payload['num_classes']!=6):
        raise ValueError('Checkpoint contents disagree with source artifacts')


def build_model(variant):
    if variant in FOURTH:
        from experiments.cvs_interaction_identity.model import build
    elif variant in THIRD:
        from experiments.cvs_balanced_identity.model import build
    elif variant in SECOND:
        from experiments.cvs_residual_identity.model import build
    else:
        from experiments.cvs_clean_design.model import build
    return build(variant)
