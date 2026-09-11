"""Authorized fresh E600 numerical mitigation and matched clean ECRS pair."""
from copy import deepcopy
from pathlib import Path
from run_a1_fast_v2 import main, v2_command
from run_a1_ecrs_cross_rx import ecrs_matrix
from run_a1_fast_selected_adv3b02 import BASE_RUN
from cvsrffi.a1_budget_schedule import compressed_options


def repair_matrix():
    from SSDG import train_ssdg as train
    source = ecrs_matrix()
    rows = []
    for name, original_id in (
        ('B1_FP32_E600', 'X0_A1_RUNTIME'),
        ('B1_CLEAN_ECRS_FP32_E600', 'X1_CROSS_RX_CLEAN'),
    ):
        original = next(r for r in source['rows'] if r['id'] == original_id)
        args = train.build_arg_parser().parse_args(v2_command(
            source, project_root=Path('/p'), run_root=Path('/r'), row=original)[3:])
        options = compressed_options(args,
            {**deepcopy(source['core90_options']), **original['options']}, 600)
        options.update({'--amp': 'false', '--a1_tail_lr': 'continuous',
            '--a1_source_screen_only': 'true', '--a1_periodic_target_start': '300',
            '--a1_periodic_target_interval': '20',
            '--a1_periodic_target_inputs': '{project_root}/runs/'+BASE_RUN+'/target_inputs',
            '--a1_periodic_target_truth': '{project_root}/runs/'+BASE_RUN+'/target_truth/truth_sidecar.json',
            '--base_candidate': name+'_SOURCE_RANDOM'})
        rows.append({'id': name, 'gpu': 0, 'options': options,
                     'evaluation_epochs': list(range(300, 601, 20))})
    return {'seed': 392005, 'initialization': 'random_no_checkpoint',
        'teacher': 'ema_of_this_run_student', 'core90_options': {}, 'rows': rows,
        'max_gpu_processes': 2, 'gpu_pool': list(range(8)),
        'final_evaluation': 'exploratory_periodic_target', 'target_feedback_allowed': False,
        'budget_policy': 'stretched_200_reference_clock_continuous_cosine',
        'selection_basis': 'source evidence, reproduced FP16 gradient anomaly, matched ECRS ablation',
        'known_limit': 'FP32 removes reproduced first-batch AMP overflow; late collapse causality and E600 stability unproven'}


if __name__ == '__main__':
    main(repair_matrix, check_script='check_a1_e600_repair.py')
