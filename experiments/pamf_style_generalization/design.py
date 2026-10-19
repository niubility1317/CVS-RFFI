from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES
from .mixing import ARMS,P,ALPHA,EPS,LAYERS,SOURCE_RX

RUN='20261020-phase1-restoration-style-generalization-source-s2026092701-m2-r01'
ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)));MIX_SEED=2026102051
PRECEDING_RUN='20261020-phase1-restoration-semantic-feedback-source-s2026092701-m2-r01'
PSEUDO_RUN='20261020-phase1-restoration-consensus-identity-source-s2026092701-m2-r01'
PSEUDO_COMMIT='623b3f3cb93f56fe77bf680e71570def9170ed2e'
TRAINABLE_PARAMETERS=2140840


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
                F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False)


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered protected style arm required')
    value=base_configuration('posterior03')
    value.update(schema='style_generalization_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        posterior_weight=.3,teacher_aggregation='mean',trainable_parameters=TRAINABLE_PARAMETERS,
        style_layers=list(LAYERS),style_probability=P,style_Beta_alpha=ALPHA,style_epsilon=EPS,mix_seed=MIX_SEED,
        style_pairing='Same unanimous pseudo identity;'+('same source RX,exclude self' if arm==ARMS[0] else 'different source RX'),
        style_matching='Only records with legal partners in both regimes;fixed random draw counts,identical physical selection/coefficient;donor choice differs',
        style_halves='Same physical pairing/coefficient in observed/clean halves,donors remain in corresponding half;detached means/stds',
        style_numeric='Float32 lambda=1 attempts logged but skipped;all original records remain in base losses',
        pseudo_source=PSEUDO_RUN+'/consensus_native/source_identity_bank.pt',pseudo_method_commit=PSEUDO_COMMIT,
        pseudo_provenance='Only pseudo/accepted fields derived from original frozen threeF on same R02 clean L/U;no refined weights or prototype inputs consumed',
        pseudo_confidence=.9,source_RX=list(SOURCE_RX),
        loss='Original posterior03 paired IQNMSE+.1logpower+mean3F(finalcos+.3T2KL);no extra CE/prototype/focus/FiLM',
        inference='Original IQ network only,augmentation off,no RX/pseudo/bank/clean/otherframes/runtime updates',
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean + six practical;three original native heads independently',
        control_run=RUN,control_arm=ARMS[0],control_reuse='New matched same-RX style arm trained once in this fixed matrix',
        target_matrix=target_matrix(),target_selection='NONE;both fixed arms automatically complete all target views',
        matched_comparison='Same originalR03/parameters/E10/data/order/routes/optimizer/threeFloss/activephysicalrecords/coefficients;only donor RX regime differs')
    return value
