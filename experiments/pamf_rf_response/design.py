from experiments.pamf_unrolled_action_fidelity import design as parent
from experiments.pamf_restoration.data import SCENES

RUN = '20261020-diagnostic-restoration-RF-response-F50-s2026092701-r31'
PARENT = parent.PARENT
PARENT_COMMIT = parent.PARENT_COMMIT
ACTION = parent.ACTION
REFERENCE = parent.REFERENCE
REFERENCE_COMMIT = parent.REFERENCE_COMMIT
CANDIDATE = parent.CANDIDATE
GPUS = parent.GPUS
DIRECTIONS = ('pa_amplitude', 'pa_phase', 'iq_image', 'delay_residual', 'random_control')
AMPLITUDES = (.01, .02)
ROWS = tuple(f'{name}_r{int(scale*100):02d}' for scale in AMPLITUDES for name in DIRECTIONS)
ORACLE_STEPS = 64


def configuration():
    return dict(schema='frozen_RF_response_v1', run=RUN, rows=list(ROWS),
        source_only=True, source_count=1800, views=['clean', *SCENES],
        parent_run=PARENT, parent_commit=PARENT_COMMIT, candidate=CANDIDATE,
        original_contract='original7/63/30;L6300/U56700/V27000;original1800Vpanel',
        checkpoint_choice='Original R17 source-selected unshared;not target-ranking selection',
        input='One current IQ2x256 to unchanged original R17;no paired input at inference',
        experiment='Two source-only counterfactual observations r+-A(delta) with same cached noise and fixed actual AGC',
        directions=list(DIRECTIONS), amplitudes=list(AMPLITUDES),
        projection='Remove complex span of current source clean x;normalize each direction to declared fraction of norm(x)',
        projection_numerics='FP64 construction/two projections;valid projected energy/raw direction energy>1e-8;FP32 output;invalid zero and separately counted',
        random_seed=2026102031, random_rule='Per original source-panel index,not TX/role/performance',
        response='(R(r+A(delta))-R(r-A(delta)))/2 compared with delta',
        curvature='((R(r+A(delta))+R(r-A(delta)))/2-R(r)) relative squared norm(delta)',
        oracle='Known-A fixed scalar simulator variance;zero-prior softMAP with original SCALE^2;CG64;diagnostic only',
        observable='direction valid;delta^T A^T (nI)^-1 A delta>=9;oracleNMSE<=.1;relativeCGresidual<=.001',
        source_rule=dict(primary_amplitude=.01, primary_view='practical_low_urban',
            physical_directions=list(DIRECTIONS[:-1]), minimum_observable_records=100,
            median_response_NMSE_min=.25, median_gain_max=.8, median_curvature_NMSE_max=.1,
            minimum_supported_directions=2, interpretation_only=True, target_promotion=False),
        limitations=['Fixed-AGC linear counterfactual,not complete regenerated physical receive chain',
            'PA/image/delay are directions at recorded-IQ coordinates,not identified true transmitter derivatives',
            'Counterfactual change has no known TX class;no classification accuracy reported for perturbed IQ',
            'Local response cannot prove real TX fidelity or that future Jacobian training improves generalization'],
        epochs=0, optimizer_updates=0, TX_F_target_training_reads=0,
        counterfactual_F_predictions=0, F_access='Original F50 only in two-Ls unmodified-IQ no-query smoke',
        TX_labels='Only independent CPU grouping after every counterfactual output complete',
        capacity_lock=parent.configuration()['capacity_lock'], max_own_GPU_workers=1,
        stop_rule='Protocol/path/provenance/shape/nonfinite or owned worker failure;preserve outputs;no automatic restart',
        goal_complete=False)
