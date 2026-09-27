"""Prepared IR research matrices; no process launch or target data access."""
from copy import deepcopy
from pathlib import Path
import json

BASE='acf0a6407c4a2cd114851e9f60d0cd346e4b2c77'
SEEDS=(392005,392006,392007)


def build_matrices():
    from .response_config import make_row
    profiles=[('SIM',{'method':'SIM'}),('EG',{'method':'EG'}),
              ('IR',{'method':'IR_EG'}),('IR_G0',{'method':'IR_EG','ir_curvature':'zero'}),
              ('OR_EG',{'method':'IR_EG','ir_feature_anchor':'origin'}),
              ('IR_ENCODER_OFF',{'method':'IR_EG','encoder_adversary':False}),
              ('BR_EG',{'method':'BR_IR_EG','ir_kappa':0.}),
              ('BR_IR',{'method':'BR_IR_EG'})]
    rows=[]
    for label,response in profiles:
        for seed in SEEDS:
            row=make_row(f'{label}_s{seed}',response,seed)
            rows.append(dict(label=label,row=row,seeds=dict(model=seed,split=392005,data=392005,
                        augmentation=392005,support=None,evaluation=392005),
                        seed_notes='split/data/augmentation/evaluation fixed by frozen native joint profile; support not applicable in Phase1',
                        source_screen=seed==392005,confirmation_requires_frozen_candidate=True,
                        output=f'runs/ir_eg_v1/{label}_s{seed}',launch=False))
    common=dict(schema='ir_experiment_matrix_v1',launch=False,git_base=BASE,
                status='PREPARED_NOT_LAUNCHED',budget_kind='budget_audit',epochs=200,
                accepted_main_steps=44400,scientific_convergence_claim=False,
                expected_source=dict(receivers=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000),
                source_contract_ref=None,source_contract_status='bind_existing_frozen_manifest_before_execution',
                checkpoint=dict(initialization='scratch',sources=[],selection='source_V_only'),
                target_policy='freeze_candidate_and_checkpoint_then_predictions_then_independent_truth_scorer',
                reference_configs=['core90_recipe_reference.json','a1_native_recipe_reference.json'],
                primary_metrics=['clean_accuracy','leo_mean_accuracy','macro_f1','worst_TX','worst_RX','day0','RX_by_TX'],
                costs=['same_physical_exposure_wall_seconds','peak_device_bytes','same_source_convergence_total_seconds',
                       'data','teacher','field_forward','backward','GGN_CG','snapshot','logging','evaluation'],
                promotion='no_automatic_promotion; report every paired seed, failure and negative contrast')
    performance=dict(deepcopy(common),rows=[r for r in rows if not r['label'].startswith('BR_')])
    efficiency=dict(deepcopy(common),rows=[r for r in rows if r['label'] in ('EG','IR','BR_EG','BR_IR')])
    convergence=dict(schema='ir_convergence_confirmation_v1',launch=False,frozen=False,
        status='PROTOCOL_PREPARATION',budget_audit_separate=True,source_only=True,
        reason='No approved executable continuation manifest was supplied; safety cap and post-E200 learning-rate policy need freezing before a future authorized run.',
        safety_cap_epochs=None,post_E200_learning_rate_policy=None,
        last_discrete_switch_epoch=181,minimum_epochs_after_last_switch=40,
        windows=4,window_epochs=10,source_V_metric_change_pp_max=.2,ce_relative_improvement_max=.005,
        required_scheduled_lr_drop_after_mechanism=True,panel_logit_relative_rms_max=.02,
        panel_logit_rms_denominator_min=1.,gradient_and_displacement_median_growth_max=.20,
        all_zero_windows='record_separately',unexplained_nonfinite_steps_allowed=0,
        safety_cap_status='safety_cap_not_converged',target_access=False)
    for doc in (performance,efficiency):validate_matrix(doc)
    return performance,efficiency,convergence


def validate_matrix(doc):
    if doc['launch'] is not False or doc['budget_kind']!='budget_audit' or doc['scientific_convergence_claim']:
        raise ValueError('prepared historical budgets cannot imply convergence or launch')
    ids=set(); outputs=set()
    for item in doc['rows']:
        row=item['row']; c=row['joint']
        if row['id'] in ids or item['output'] in outputs:raise ValueError('duplicate row/output')
        ids.add(row['id']);outputs.add(item['output'])
        if not c['native_dr'] or c['normalization_scales']!='reuse_origin_used_scales':raise ValueError('native joint contract')
        if (c['labeled_batch'],c['unlabeled_batch'],c['epochs'],c['steps_per_epoch'])!=(128,256,200,222):raise ValueError('unequal exposure')
        if c['disable_adversarial_head_loss'] or not row['response']['unlabeled_head']:raise ValueError('head supervision must remain enabled')
        if item['launch'] is not False or set(item['seeds'])!=set(('model','split','data','augmentation','support','evaluation')):raise ValueError('launch or seed roles')
    return doc


def validate_convergence_for_launch(doc):
    if not doc.get('frozen') or not doc.get('safety_cap_epochs') or not doc.get('post_E200_learning_rate_policy'):
        raise ValueError('confirmation continuation manifest is not completely frozen')
    if doc.get('target_access') or not doc.get('source_only'):
        raise ValueError('source-only convergence required')


def write_configs(root):
    root=Path(root)
    for name,doc in zip(('ir_eg_v1.json','br_ir_eg_v1.json','convergence_confirmation.json'),build_matrices()):
        (root/name).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    unique={item['row']['id']:item for doc in build_matrices()[:2] for item in doc['rows']}
    (root/'ir_rows').mkdir(exist_ok=True)
    for row_id,item in unique.items():
        (root/'ir_rows'/f'{row_id}.json').write_text(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',row=item['row']),indent=2)+'\n',encoding='utf-8')
    return unique
