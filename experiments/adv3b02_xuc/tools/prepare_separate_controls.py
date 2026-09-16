"""Freeze 3 native DAOT/RC4 and 18 pure response-game controls."""
import json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/scripts'))
from run_a1_fast_v2 import v2_matrix
from cvsrffi.xuc_fusion.response_config import make_row,stage_table
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.dr_objective import resolved_options

def main():
    out=ROOT/'configs/separate_controls';out.mkdir(exist_ok=True)
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text())
    reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text())
    original=v2_matrix();native_options={**original['core90_options'],**original['rows'][0]['options']}
    rows=[]
    for seed in (392005,392006,392007):
        rid='DAOT_RC4_NATIVE_s'+str(seed);options=deepcopy(native_options)
        options.update({'--seed':str(seed),'--amp':'true','--a1_source_screen_only':'true',
            '--muse_external_final_eval':'true','--eval_sat_channel':'false','--phase2_export_prototypes':'false',
            '--num_workers':'0','--device':'cuda:0','--baseline_ckpt':'','--teacher_ckpt':''})
        doc=dict(id=rid,options=options,seed=seed,status='PREPARED',reference='F0_FIXED_BASE native recipe; source-only and no export; no game solver')
        (out/(rid+'.json')).write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
        rows.append(dict(family='native_baseline',row=dict(id=rid,joint=dict(model_seed=seed))))
    for seed in (392005,392006,392007):
        for method in ('SIM','EG','CF_EG','TR_EG','XT_DANN','DRIC'):
            row=make_row('PURE_'+method+'_s'+str(seed),dict(method=method),seed,pure_game=True)
            args=resolve_args(recipe,row,dataset='ManySig.pkl',output='NOT_LAUNCHED/'+row['id'])
            dr=resolved_options(reference,args);dr.fasttrust_rc4=False;dr.use_adv3b02_daot_stn=False
            doc=dict(row=row,resolved=vars(args),resolved_dr=vars(dr),stage_table=stage_table(row['response']))
            (out/(row['id']+'.json')).write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
            rows.append(dict(family='pure_game',row=row))
    (out/'matrix.json').write_text(json.dumps(dict(rows=rows,target_feedback=False,scope='native recipe comparison and matched response-only ablations'),indent=2)+'\n',encoding='utf-8')
    print('PREPARED 21 rows')
if __name__=='__main__':main()
