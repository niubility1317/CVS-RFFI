"""Pre-registered source-selected A1 E400/E600 experiments, scratch only."""
from pathlib import Path
from copy import deepcopy
from run_a1_fast_v2 import main, v2_command
from run_a1_ecrs_cross_rx import ecrs_matrix
from run_a1_r3_cross_rx import r3_cross_rx_matrix
from run_a1_fast_selected_adv3b02 import BASE_RUN
from cvsrffi.a1_budget_schedule import compressed_options


def extended_matrix():
    from SSDG import train_ssdg as train
    x2 = ecrs_matrix()
    r3 = r3_cross_rx_matrix()
    rows = []
    for name, total, gpu, source, original_id in (
        ('X2_E400', 400, 6, x2, 'X2_CROSS_RX_VIEWS'),
        ('X2_E600', 600, 3, x2, 'X2_CROSS_RX_VIEWS'),
        ('R3_CLEAN_RX_E400', 400, 0, r3, 'R3_REFERENCE_CLEAN_CROSS_RX'),
    ):
        original = next(r for r in source['rows'] if r['id'] == original_id)
        args = train.build_arg_parser().parse_args(v2_command(
            source, project_root=Path('/p'), run_root=Path('/r'), row=original)[3:])
        base = {**deepcopy(source['core90_options']), **original['options']}
        options = compressed_options(args, base, total)
        options.update({'--a1_source_screen_only': 'true',
            '--a1_periodic_target_start': str(total//2), '--a1_periodic_target_interval': '20',
            '--a1_periodic_target_inputs': '{project_root}/runs/'+BASE_RUN+'/target_inputs',
            '--a1_periodic_target_truth': '{project_root}/runs/'+BASE_RUN+'/target_truth/truth_sidecar.json',
            '--base_candidate': name+'_SOURCE_SELECTED_RANDOM'})
        rows.append({'id': name, 'gpu': gpu, 'options': options,
            'method_source': original_id, 'evaluation_epochs': list(range(total//2,total+1,20))})
    return {'seed': 392005, 'initialization': 'random_no_checkpoint', 'core90_options': {},
        'teacher': 'ema_of_this_run_student', 'rows': rows, 'max_gpu_processes': 2,
        'final_evaluation': 'exploratory_periodic_target', 'target_feedback_allowed': False,
        'budget_policy': 'proportionally_stretched_200_epoch_reference_clock',
        'selection_basis': 'completed scratch E200 source V clean/LEO, resource cost and executed mechanisms'}


if __name__ == '__main__':
    main(extended_matrix, check_script='check_a1_extended_budgets.py')
