"""Recover only the twelve verified E11 failures; retain scientific argv and GPU."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'configs/phase1_adv3b02_pair24_manifest.json'
OUTPUT=ROOT/'configs/phase1_adv3b02_pair12_recovery_manifest.json'
FAILED={'B_SAFE','TANGENT','ROUTE','TANGENT_ROUTE'}


def build_recovery(parent):
    result=copy.deepcopy(parent)
    old_id=parent['run_id']
    new_id='phase1_adv3b02_pair12_manysig_e200_20260905_r2'
    result['run_id']=new_id
    for key in ('code_root','output_root'):
        result[key]=parent[key].replace(old_id,new_id)
    result['max_active']=12
    result['replaces_failed_rows_from']=old_id
    result['healthy_rows_remain_in']=old_id
    result['rows']=[]
    for original in parent['rows']:
        if original['candidate'] not in FAILED:
            continue
        row=copy.deepcopy(original)
        row['argv']=[value.replace(old_id,new_id) for value in row['argv']]
        row['cwd']=result['code_root']
        row['environment']={k:v.replace(old_id,new_id) for k,v in row['environment'].items()}
        result['rows'].append(row)
    result['smoke_argv']=[value.replace(old_id,new_id) for value in parent['smoke_argv']]
    result['smoke_argv'][-1]='cuda'
    result['smoke_argv']+=['--amp','--training-manifest',result['code_root']+'/configs/'+OUTPUT.name]
    return result


def build_safe_recovery(parent):
    result=copy.deepcopy(parent)
    old_id=parent['run_id']
    new_id='phase1_adv3b02_safe3_manysig_e200_20260905_r3'
    result['run_id']=new_id
    for key in ('code_root','output_root'):
        result[key]=parent[key].replace(old_id,new_id)
    result['max_active']=3
    result['replaces_failed_rows_from']=old_id
    result['healthy_rows_remain_in']=[
        'phase1_adv3b02_pair24_manysig_e200_20260905_r1',old_id]
    result['rows']=[]
    for original in parent['rows']:
        if original['candidate']!='B_SAFE':
            continue
        row=copy.deepcopy(original)
        row['argv']=[v.replace(old_id,new_id) for v in row['argv']]
        row['cwd']=result['code_root']
        row['environment']={k:v.replace(old_id,new_id) for k,v in row['environment'].items()}
        result['rows'].append(row)
    result['smoke_argv']=[v.replace(old_id,new_id).replace(
        'phase1_adv3b02_pair12_recovery_manifest.json','phase1_adv3b02_safe3_recovery_manifest.json')
        for v in parent['smoke_argv']]
    result['selection']['comparison_boundary']='NO_PROMOTION: EMA cache fix changes teacher semantics; mixed-release comparisons are diagnostic only.'
    return result


if __name__=='__main__':
    result=build_recovery(json.loads(PARENT.read_text(encoding='utf-8')))
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({'rows':len(result['rows']),'output':str(OUTPUT)}))
