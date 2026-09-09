"""R3 reference plus clean cross-receiver identity loss; source-only development."""
from copy import deepcopy
from run_a1_scratch_no_checkpoint import scratch_matrix
from run_a1_fast_v2 import main


def r3_cross_rx_matrix():
    original = scratch_matrix()
    reference = next(row for row in original['rows'] if row['id'] == 'R3_REFERENCE')
    options = deepcopy(original['core90_options'])
    for key in ('daot_efficiency_mode', 'daot_batched_scale_readback',
                'daot_skip_mean_metadata', 'a1_r3_aux_scale'):
        options['--' + key] = reference[key]
    options.update({'--a1_ecrs_cross_rx_weight': '0.05',
                    '--a1_ecrs_cross_rx_margin': '0.2',
                    '--a1_ecrs_cross_rx_scope': 'clean'})
    return {'seed': 392005, 'initialization': 'random_no_checkpoint',
            'teacher': 'ema_of_this_run_student', 'core90_options': options,
            'final_evaluation': 'source_only', 'max_gpu_processes': 2,
            'rows': [{'id': 'R3_REFERENCE_CLEAN_CROSS_RX', 'gpu': 1, 'options': {}}]}


if __name__ == '__main__':
    main(matrix_factory=r3_cross_rx_matrix, check_script='check_a1_r3_cross_rx.py')
