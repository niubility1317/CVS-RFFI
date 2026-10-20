from experiments.pamf_unrolled_restore import design as base
from experiments.pamf_unrolled_action_fidelity import design as lineage

RUN='20261020-phase1-restoration-observable-response-selfsup-s2026092701-m2-r32'
ROWS=('pair_aug_control','observable_response')
ALL_ROWS=ROWS
CANDIDATE='observable_response'
KIND=base.KIND
ACTION=base.ACTION
REFERENCE=base.REFERENCE
MODEL_SEED=base.MODEL_SEED
DATA_SEED=base.DATA_SEED
GPUS=base.GPUS
UNROLLED=lineage.PARENT
UNROLLED_COMMIT=lineage.PARENT_COMMIT
REFERENCE_CANDIDATE=lineage.CANDIDATE
RESPONSE_SEED=2026102032
DIRECTIONS=('pa_amplitude','pa_phase','iq_image','random_control')
RESPONSE_VIEWS=('clean','practical_mid_urban','practical_low_urban')
RESPONSE_WEIGHT=.01


def configuration():
    value=base.configuration()
    value.update(schema='observable_response_prior_v1',run=RUN,rows=list(ROWS),
        prior_architecture='Original three unshared Core9 U-Nets;single-IQ inference,CG2/2/2',
        prior_initialization='Scratch;one cloned initial Core9 for all steps and both rows;only frozen R02/R07 ancestors',
        loss='Mean IQNMSE over original/plus/minus pairs;intervention adds .01*fullbatchmean(mask*response_to_oracle_NMSE)',
        response=dict(seed=RESPONSE_SEED,directions=list(DIRECTIONS),amplitudes=[.01,.02],
            sampling='Independent deterministic SeedSequence(seed,epoch,source-LU-index);uniform direction and amplitude',
            random_control_stream='R31 seed2031 with training index+2^32;source validation retains original panel indices;disjoint draws',
            generation='Reuse original actual cached source A/noise/AGC;r+-Adelta,x+-delta',
            oracle='Original SCALE^2,knownA and scalar simulator n,zero prior CG64;target is oracle direction response,not delta',
            eligibility='Valid direction,Fisher>=9,oracleNMSE<=.1,CGrelativeResidual<=.001;no TX/F/target',
            weight=RESPONSE_WEIGHT,average='Full original batch,not accepted-only mean;zero differentiable loss if no eligible items',
            numerics='R31 corrected FP64 two-pass projection,relative residual-energy threshold1e-8,FP32 IQ',
            shared_pair_augmentation=True,additional_constraint_only_row=CANDIDATE,
            source_validation_views=list(RESPONSE_VIEWS),source_validation_amplitude=.01,
            source_primary='Pooled observable mean oracle-response error,three physical directions,lowurban;ratio<=.95 vs matchedcontrol'),
        response_seed=RESPONSE_SEED,microbatch=64,effective_batch=256,
        gradient_accumulation='Four microbatches,one AdamW step/global clip1 per original batch;final batch24 preserved',
        training_forward_count='Three input views per physical visit per row;L378000/U3402000 waveform forwards;unique-physical visits L126000/U1134000',
        matched_control=ROWS[0],source_selection='Only observable_response eligible;fixed matchedcontrol and originalR17 guards plus response ratio<=.95;otherwise0target',
        target='Only source-qualified observable_response;original168000x7,F50/defaulthead,independent truthlast;no target feedback',
        source_thresholds=dict(mean6_NMSE_ratio_max=1.05,clean_NMSE_max=.02,low_and_mid_gain_min_pp=.5,
            clean_drop_vs_original_max_pp=.1,worst_lowurban_TX_delta_min_pp=-1.,worst_other_scene_delta_min_pp=-.5,
            response_error_ratio_max=.95,response_min_observable_per_physical_direction=100),
        question_origin='Completed R31 source response evidence,not formal target ranking',
        scientific_scope='TX-unlabeled simulated-paired self-supervision;synthetic observed-IQ response control;realTX fidelity unproven',
        capacity_lock='receiver_residual_capacity4.lock',max_own_GPU_workers=2,
        stop_rule='Protocol/path/parent/outputcollision/nonfinite/unlaunchable/illegalprediction failure;preserve owned outputs and healthy peers;no automatic restart')
    return value
