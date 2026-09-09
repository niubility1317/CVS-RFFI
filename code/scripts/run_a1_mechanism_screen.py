"""Source-only single-mechanism screen before any cross-family combination."""
from copy import deepcopy
from run_a1_fast_v2 import v2_matrix, main
from run_a1_fast_selected_adv3b02 import BASE_RUN


def mechanism_matrix():
    base = deepcopy(v2_matrix()['core90_options'])
    base.update({'--a1_runtime_fast': 'true', '--a1_ema_startup_average': 'false',
                 '--a1_logit_coverage_weighting': 'false', '--epochs': '200',
                 '--base_candidate': 'A1_MECHANISM_SCREEN_RANDOM'})
    base['--a1_source_screen_only'] = 'true'
    base.update({'--a1_periodic_target_start': '80',
        '--a1_periodic_target_inputs': '{project_root}/runs/'+BASE_RUN+'/target_inputs',
        '--a1_periodic_target_truth': '{project_root}/runs/'+BASE_RUN+'/target_truth/truth_sidecar.json'})
    definitions = [
        ('B0_FIXED', {}, 'common corrected baseline'),
        ('B1_TAIL_LR', {'--a1_tail_lr': 'continuous'}, 'full-budget cosine without trunk tail freeze'),
        ('B2_RC4_WEIGHT', {'--a1_rc4_reliability_weight': 'calibrated_probability'}, 'same acceptance rules, calibrated weights within same budget'),
        ('D0_NO_ORBIT', {'--daot_ablation': 'A0'}, 'orbit negative control'),
        ('D1_THREE_VIEW', {'--daot_ablation': 'A2'}, 'extra teacher view over B0'),
        ('D2_PHYSICAL_ORBIT', {'--daot_ablation': 'A3'}, 'physical robust aggregation over D1'),
        ('D3_TANGENT', {'--daot_ablation': 'A5'}, 'local covariance nuisance tangent over D2'),
        ('F0_R3_REFERENCE', {'--use_a1_r3': 'true'}, 'reconstruction reference'),
        ('F1_R3_CONTINUOUS', {'--use_a1_r3': 'true', '--a1_r3_schedule': 'continuous'}, 'continuous cross-view curriculum over F0'),
        ('F2_R3_IDENTITY', {'--use_a1_r3': 'true', '--a1_r3_identity_coupling': 'true'}, 'physical response consumes identity over F0'),
        ('F3_R3_SWAP', {'--use_a1_r3': 'true', '--a1_r3_source_fingerprint': 'true'}, 'source content and fingerprint, destination nuisance over F0'),
        ('G0_EQUAL_BRANCH', {'--physical_gate_variant': 'nmfdu_v1', '--a1_fisher_supervision': 'true', '--a1_fisher_equal': 'true'}, 'equal fusion with supervised branches and source evidence'),
        ('G1_FISHER_GATE', {'--physical_gate_variant': 'nmfdu_v1', '--a1_fisher_supervision': 'true'}, 'physical bounded gate over G0'),
        ('E0_RESPONSE_ONLY', {'--a1_response_surface': 'true', '--a1_response_rho': '0'}, 'trained response branch without identity fusion'),
        ('E1_RESPONSE_FUSED', {'--a1_response_surface': 'true'}, 'bounded response fusion over E0'),
        ('E2_RESPONSE_PAIR', {'--a1_response_surface': 'true', '--a1_response_pair': '.03'}, 'source U clean/LEO response consistency over E1'),
    ]
    rows = [{'id': name, 'gpu': i % 8, 'options': options, 'hypothesis': hypothesis}
            for i, (name, options, hypothesis) in enumerate(definitions)]
    return {'seed': 392005, 'initialization': 'random_no_checkpoint', 'core90_options': base,
            'rows': rows, 'max_gpu_processes': 2, 'final_evaluation': 'exploratory_periodic_target',
            'stage': 'individual_mechanism_screen', 'cross_family_combination': 'pending_source_results',
            'target_feedback_allowed': False}


if __name__ == '__main__':
    main(mechanism_matrix, check_script='check_a1_mechanism_screen.py')
