"""Emit clean-comparison job descriptions without starting experiments."""
import argparse
import itertools
import json
from pathlib import Path

from .clean_pairing import EPISODES, KS, POOLS, RECEIVERS

SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
NEW_CLASSES = (2, 5, 10, 20)


def groups():
    da = ['protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage', 'poster_ft', 'tweak_m1']
    cil = ['mopc_hr_public_code', 'orthogonal_fixed_pseudolabel', 'lora_balanced_adaptive_knn', 'issl_fixed_v1_long']
    cores = ['mopc_hr_public_code', 'orthogonal_fixed_pseudolabel', 'lora_balanced_adaptive_knn']
    return {
        'pamf': {'methods':['pamf_trisigtime_ssl_scalr_r42'], 'paths':['S0Q0', 'S1Q1'],
                 'prerequisites':['bind frozen SCALR R42 to clean input with only requested paths and stage-limited query']},
        'native_da': {'methods':da, 'paths':['S0Q0'],
                      'prerequisites':['verify source checkpoints and clean runner', 'bind tweak M1 per-query inference']},
        'native_cil': {'methods':cil, 'paths':['S0Q0'],
                       'prerequisites':['verify fixed Orthogonal source', 'verify ISSL long-budget binding',
                                        'resolve genuine MoPC nonfinite failures']},
        'hybrid': {'candidates':[{'B':b,'C':c} for b,c in itertools.product(da, cores)], 'paths':['S0Q0'],
                   'prerequisites':['verify B-to-C state adapters', 'bind Tweak M1', 'resolve MoPC nonfinite failures']},
    }


def tasks():
    result=[]
    for pool, episode, receiver, seed in itertools.product(POOLS, EPISODES, RECEIVERS, SEEDS):
        points=list(itertools.product(KS,NEW_CLASSES)) if episode.startswith('MIX') else [(5,20)]
        result.append({'capsule':f'capsules/{pool}/{episode}/{receiver}', 'seed':seed,
                       'points':[list(p) for p in points], 'pilot_point':[5,20]})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,help='New JSON file; no overwrite')
    args=p.parse_args()
    plan={'status':'PLANNED_NOT_LAUNCH_READY', 'input_condition':'clean_ground_pre_satellite_normalized',
          'data_run':'20261112-phase2-cleanpaired-data-r01', 'groups':groups(), 'tasks':tasks(),
          'training_or_prediction_started':False, 'query_truth_forbidden':True,
          'K_and_N_averaged':False, 'group_launchers_not_yet_validated':True}
    output=Path(args.output)
    with output.open('x',encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,separators=(',',':'))
    print(json.dumps({'status':plan['status'], 'capsule_seed_tasks':len(plan['tasks']),
                      'groups':list(plan['groups']), 'output':str(output)}))


if __name__=='__main__':
    main()
