"""Prepare configs and resolve real runtime args. This program cannot train."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.joint_config import make_row
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.dr_objective import resolved_options

def build():
    rows=[]
    for seed in (392005,392006,392007):
        for dr in (False,True):
            for eg in (False,True):
                rows.append(make_row(f'R0-{int(dr)}{int(eg)}_s{seed}',
                    dict(model_seed=seed,native_dr=dr,solver_mode='full_EG' if eg else 'simultaneous')))
        for eg in (False,True):
            rows.append(make_row(f'ADV0-{int(eg)}_s{seed}',dict(model_seed=seed,
                solver_mode='full_EG' if eg else 'simultaneous',outer_adv_weight=0.,disable_adversarial_head_loss=True)))
    candidate=dict(solver_mode='full_EG',normalization_scales='reuse_origin_used_scales')
    templates=[dict(id='D1_fixed_divisor',parent='R0-11',overrides=candidate)]
    templates += [dict(id=f'D2_kappa_{k}',parent='D1_fixed_divisor',overrides={'predictor_lr_ratio':k}) for k in (.25,.5,1.)]
    templates += [dict(id=f'D3_grl_{g}',parent='SOURCE_SELECTED_D2',overrides={'labeled_encoder_grl_multiplier':g}) for g in (.5,1.)]
    templates += [dict(id=f'D4_logit_{v}',parent='SOURCE_SELECTED_D3',overrides={'daot_logit':v}) for v in (.1,.2)]
    templates += [dict(id='D5_hp_ramp',parent='SOURCE_SELECTED_D4',overrides={'hp_ramp':True})]
    return dict(schema='native_dr_eg_prepared_v1',launch=False,status='PREPARED_NOT_LAUNCHED',
        seed_source='358a087a docs/research/daot_fasttrust_game_20260914/tables/all_scored_row_records.csv',
        dr_off_definition='same E1 U256 domain+self, DAOT/H/P identity off; no ordinary pseudo CE fallback',
        adv_zero_definition='L adv CE and U adv-head CE off; U z_dom CE and self retained',
        target_access=False,rows=rows,r1_templates=templates,
        r1_reference_candidate=make_row('R1_reference_not_selected',dict(candidate,predictor_lr_ratio=.5,
            labeled_encoder_grl_multiplier=.5,daot_logit=.1,hp_ramp=True)),
        deferred=['gradient_budget','dynamic_EG','L_U_scale_statistics_changes','extra_DAOT','negative_route'],
        source_selection=dict(metric='single_V_accuracy',tie_break='lower_V_CE',
            target_feedback=False,checkpoint='fixed_E200',paired_seeds=[392005,392006,392007]),
        evaluation=dict(scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'],
            evaluation_seed=392005,expected_per_scene=168000,expected_per_run=672000,
            prediction_before_truth=True,historical_development_conditions=True,
            independent_confirmation='NOT_AVAILABLE_NOT_CLAIMED'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'configs/native_dr_eg')
    args=p.parse_args()
    spec=build();args.output.mkdir(parents=True,exist_ok=True)
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    for row in spec['rows']+[spec['r1_reference_candidate']]:
        resolved=resolve_args(recipe,row,dataset='ManySig.pkl',output='NOT_LAUNCHED/'+row['id'])
        (args.output/(row['id']+'.json')).write_text(json.dumps(dict(row=row,resolved=vars(resolved),
            resolved_dr=vars(resolved_options(reference,resolved)),
            effective_mechanisms=dict(DAOT=row['joint']['native_dr'],HP_identity=row['joint']['native_dr'],
                U_domain_self=True,ordinary_pseudo_fallback=False,full_EG=resolved.game_solver=='extragradient',
                frozen_teacher_views=True,controller=False,dynamic_curriculum=False,
                extra_DAOT=False,negative=False,frozen_anchor=False)),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (args.output/'matrix.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',rows=len(spec['rows']),r1_templates=len(spec['r1_templates']))))
if __name__=='__main__':main()
