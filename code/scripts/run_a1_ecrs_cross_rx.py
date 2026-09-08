"""A1 identity-path adaptation of ECRS cross-RX discrimination, from scratch."""
from copy import deepcopy
from run_a1_fast_v2 import main, v2_matrix


def ecrs_matrix():
    matrix=deepcopy(v2_matrix())
    matrix['core90_options'].update({'--a1_runtime_fast':'true','--a1_ema_startup_average':'false',
        '--a1_logit_coverage_weighting':'false','--a1_ecrs_cross_rx_margin':'0.2'})
    matrix['max_gpu_processes']=2
    settings=(
        ('X0_A1_RUNTIME',{'--a1_ecrs_cross_rx_weight':'0','--a1_ecrs_cross_rx_scope':'clean'}),
        ('X1_CROSS_RX_CLEAN',{'--a1_ecrs_cross_rx_scope':'clean'}),
        ('X2_CROSS_RX_VIEWS',{}),
        ('X3_WARM_EMA',{'--a1_ema_startup_average':'true'}),
        ('X4_COVERAGE_KL',{'--a1_logit_coverage_weighting':'true'}),
        ('X5_BALANCED_L',{'--use_tx_rx_balanced_sampler':'true'}),
        ('X6_BOUNDED_DOMAIN',{'--identity_domain_objective_mode':'bounded_confusion'}),
        ('X7_COMBINED',{'--a1_ema_startup_average':'true','--a1_logit_coverage_weighting':'true',
            '--use_tx_rx_balanced_sampler':'true','--identity_domain_objective_mode':'bounded_confusion'}))
    matrix['core90_options'].update({'--a1_ecrs_cross_rx_weight':'0.05','--a1_ecrs_cross_rx_scope':'clean_leo',
        '--balanced_sampler_tx_per_batch':'4','--balanced_sampler_domain_per_batch':'4',
        '--balanced_sampler_samples_per_cell':'8','--balanced_sampler_replacement':'true'})
    matrix['rows']=[{'id':name,'gpu':gpu,'options':options} for gpu,(name,options) in enumerate(settings)]
    return matrix


if __name__=='__main__':
    main(matrix_factory=ecrs_matrix,check_script='check_a1_ecrs_cross_rx.py')
