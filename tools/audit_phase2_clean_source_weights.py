"""Read-only CPU inventory of explicitly selected Phase2 source weights."""
import json
from pathlib import Path
import sys

import torch

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
SEEDS=(2026092701,2026092702,2026092703,2026092704)
GROUP1=('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft',
        'orthogonal','mopc_hr','lora_triplet_knn')
CONTRACT=ROOT/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/source_contract.json'
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code'))


def finite_model(d):
    state=d.get('model')
    return isinstance(state,dict) and len(state)>0 and all(
        not torch.is_tensor(v) or not (v.is_floating_point() or v.is_complex()) or bool(torch.isfinite(v).all())
        for v in state.values())


def main():
    torch.set_num_threads(1)
    expected=json.loads(CONTRACT.read_text());items=[]
    for profile in GROUP1:
        for seed in SEEDS:
            source_run='20261101-phase2-mopc-author-group1-m4-r04' if profile=='mopc_hr' else '20261030-phase2-group1-concat-midPL-threepool-m4-r01'
            path=ROOT/source_run/'source'/f'{profile}-s{seed}'/'source.pt'
            row=dict(profile=profile,seed=seed,path=str(path),status='MISSING')
            if path.is_file():
                try:
                    p=torch.load(path,map_location='cpu',weights_only=False);r=p.get('recipe',{})
                    matched=(p.get('schema')=='group1_A_v1' and p.get('profile')==profile and p.get('seed')==seed
                             and p.get('target_contact') is False and p.get('ancestors')==[]
                             and p.get('source_contract')==expected and finite_model(p))
                    if profile=='orthogonal':matched=matched and r.get('schema')=='group1_source_orthogonal_tempered_PL_v2'
                    if profile=='mopc_hr':matched=matched and isinstance(r.get('architecture'),dict)
                    row.update(status='METADATA_MATCH' if matched else 'NOT_MATCHED',recipe_schema=r.get('schema'),
                               input_contract_match=p.get('source_contract')==expected,target_contact=p.get('target_contact'),
                               finite_parameters=finite_model(p))
                except Exception as exc:row.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
            items.append(row)
    for profile in ('tweak_m1','issl_fixed_v1'):
        for seed in SEEDS:
            path=ROOT/'20261029-phase2-external-threepool-m4-r01'/'source'/f'{profile}-s{seed}'/'source.pt'
            row=dict(profile=profile,seed=seed,path=str(path),status='MISSING')
            if path.is_file():
                try:
                    p=torch.load(path,map_location='cpu',weights_only=False);v=p.get('provenance',{});actual=v.get('source_contract',{})
                    match=all(actual.get(k)==expected.get(k) for k in ('role_ids','source_rxs','source_days','counts','classes','ratios','split_seed'))
                    valid=(v.get('schema')=='phase2_external_additional_source_v1' and v.get('profile')==profile
                           and v.get('model_seed')==seed and v.get('initialization')=='scratch'
                           and v.get('checkpoint_sources')==[] and v.get('target_contact') is False
                           and v.get('U_s_TX_access') is False and match and finite_model(p))
                    row.update(status='METADATA_MATCH' if valid else 'NOT_MATCHED',input_contract_match=match,
                               target_contact=v.get('target_contact'),finite_parameters=finite_model(p),
                               source_spec=v.get('source_spec'),history=v.get('history'))
                except Exception as exc:row.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
            items.append(row)
    for seed in SEEDS:
        path=ROOT/'20261027-phase1-pamf-repl-shuffled-views-m16-r02'/'F'/f'main_split-repl-shuffled_time-3view-s{seed}'/'source'
        row=dict(profile='PAMF',seed=seed,path=str(path),status='MISSING')
        if (path/'complete.json').is_file() and (path/'final_ssdg.pth').is_file():
            try:
                done=json.loads((path/'complete.json').read_text());c=done['config']
                checkpoint=torch.load(path/'final_ssdg.pth',map_location='cpu',weights_only=False)
                valid=(done.get('status')=='SOURCE_FROZEN' and done.get('epoch')==50 and done.get('updates')==11100
                       and done.get('target_access') is False and done.get('U_truth_read') is False
                       and c.get('model_seed')==seed and c.get('view_comparison',{}).get('view_count')==3
                       and json.loads(Path(c['source_contract']).read_text())==expected and finite_model(checkpoint))
                row.update(status='METADATA_MATCH' if valid else 'NOT_MATCHED',finite_parameters=finite_model(checkpoint),
                           target_access=done.get('target_access'),views=c.get('view_comparison',{}).get('view_count'))
            except Exception as exc:row.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
        items.append(row)
    dura=ROOT/'20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'
    row=dict(profile='DURA_v1',seed=2026092701,path=str(dura),status='MISSING')
    if dura.is_file():
        try:
            p=torch.load(dura,map_location='cpu',weights_only=False)
            valid=(p.get('schema')=='R65_four_route_checkpoint_v1' and p.get('arm')=='stage2'
                   and p.get('epoch')==20 and p.get('updates')==4940 and p.get('target_training') is False
                   and p.get('training_F_access') is False and p.get('training_TX_labels_read')==0 and finite_model(p))
            row.update(status='METADATA_MATCH' if valid else 'NOT_MATCHED',finite_parameters=finite_model(p))
        except Exception as exc:row.update(status='ERROR',error=type(exc).__name__+': '+str(exc))
    items.append(row)
    print(json.dumps(dict(status='INVENTORIED_NOT_SMOKED',items=items,query_read=False,training=False)))


if __name__=='__main__':main()
