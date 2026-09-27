"""Generate reviewable configuration, never invoke train or a launcher."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.response_config import make_row,stage_table
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.dr_objective import resolved_options

def build():
    candidates=[(m,dict(method=m),'primary') for m in ('SIM','EG','CF_EG','TR_EG','XT_DANN','DRIC')]
    candidates += [(m+'_encoder_off',dict(method=m,encoder_adversary=False),'adversarial_specificity') for m in ('EG','CF_EG','TR_EG','XT_DANN','DRIC')]
    candidates += [('DRIC_no_drift',dict(method='DRIC',drift=False),'ablation'),
        ('DRIC_L_head_only',dict(method='DRIC',unlabeled_head=False),'asymmetry_control'),
        ('EG_L_head_only',dict(method='EG',unlabeled_head=False),'asymmetry_control'),
        ('DRIC_no_identity_budget',dict(method='DRIC',identity_constraint=False),'ablation'),
        ('DRIC_shuffled_drift',dict(method='DRIC',shuffle_drift=True),'negative_control'),
        ('DRIC_task_projection',dict(method='TASK_PROJECT'),'simple_explanation'),
        ('response_tracking',dict(method='FR',identity_constraint=False),'related_local_solver'),
        ('local_competitive',dict(method='CGD',identity_constraint=False),'related_local_solver'),
        ('RK2',dict(method='RK2'),'related_solver'),
        ('EG_weak_encoder',dict(method='EG',encoder_multiplier=.5),'simple_explanation'),
        ('CF_tolerance_02',dict(method='CF_EG',margin_tolerance=.02),'sensitivity'),
        ('CF_sparse5',dict(method='CF_EG',cf_interval=5),'sparse_approximation'),
        ('CF_sparse10',dict(method='CF_EG',cf_interval=10),'sparse_approximation'),
        ('CF_random_beta',dict(method='CF_EG',random_beta=True),'negative_control'),
        ('TR_unmatched',dict(method='TR_EG',unmatched_rotation=True),'negative_control'),
        ('XT_exposure_control',dict(method='XT_DANN',xt_exposure_control=True),'exposure_control')]
    candidates += [('CF_fixed_'+str(b),dict(method='CF_EG',fixed_beta=b),'simple_explanation') for b in (0.,.5,1.)]
    candidates += [('TR_gamma_'+str(g),dict(method='TR_EG',transport_gamma=g),'sensitivity') for g in (0.,.25,1.)]
    candidates += [('XT_mu_'+str(m),dict(method='XT_DANN',xt_mu=m),'sensitivity') for m in (0.,.05,.2)]
    rows=[]
    for name,settings,purpose in candidates:
        for seed in (392005,392006,392007):
            rows.append(dict(row=make_row(name+'_s'+str(seed),settings,seed),purpose=purpose,
                status='PREPARED_NOT_LAUNCHED',release_stage='mechanism_screen_before_source_selected_full_scan'))
    dependencies=[dict(id='matched_compute_'+method,status='DEPENDENCY_TEMPLATE',method=method,
        requires='measured total compute including all branches, probes, monitoring, derivatives, and source-only chosen budget',
        target_feedback=False) for method in ('DRIC','CF_EG','TR_EG','XT_DANN','EG','RK2')]
    dependencies += [dict(id='CF_matched_mean_strength',status='DEPENDENCY_TEMPLATE',requires='source-only pilot mean actual adversarial displacement',target_feedback=False),
        dict(id='symmetric_LU_adversary',status='DIFFERENT_CONTRACT_NOT_ENABLED',requires='explicitly changed U-to-encoder scientific contract',target_feedback=False)]
    return dict(schema='response_games_prepared_v1',status='PREPARED_NOT_LAUNCHED',launch=False,rows=rows,
        dependencies=dependencies,previous_matrix='../native_dr_eg/matrix.json',
        source_selection=dict(metric='single_V_accuracy',tie_break='lower_V_CE',checkpoint='fixed_E200',target_feedback=False),
        execution_order=['same-state bounded mechanism experiments after future launch authorization','source-only candidate selection','paired full scan','fixed predictions then independent scoring'],
        promotion='no automatic selection or launch; no target feedback',
        baseline_distinction='new encoder-off retains L/U head CE; old ADV0 removes head CE',
        seeds=dict(model=[392005,392006,392007],data=392005,augmentation=392005,evaluation=392005),
        metrics=['LEO mean','Macro-F1','weakest TX','weakest RX','RX/day/TX','rescue/harm','paired seed mean and SD','adversarial difference in differences','matched accepted steps and compute'])

def main():
    matrix=build();out=ROOT/'configs/response_games';out.mkdir(parents=True,exist_ok=True)
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    native=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    for item in matrix['rows']:
        row=item['row'];args=resolve_args(recipe,row,dataset='ManySig.pkl',output='NOT_LAUNCHED/'+row['id'])
        doc=dict(item,resolved=vars(args),resolved_dr=vars(resolved_options(native,args)),
            stage_table=stage_table(row['response']))
        (out/(row['id']+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'matrix.json').write_text(json.dumps(matrix,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=matrix['status'],rows=len(matrix['rows']),dependencies=len(matrix['dependencies']))))
if __name__=='__main__':main()
