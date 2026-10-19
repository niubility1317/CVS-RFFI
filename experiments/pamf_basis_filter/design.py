from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-basis-filter-source-s2026092701-m2-r01'
ARMS=('time_basis','frequency_basis');ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)))
PRECEDING_RUN='20261020-phase1-restoration-temporal-context-source-s2026092701-m2-r01'
FILTER_WIDTHS=(16,32,48,64);FILTER_PARAMETERS=210628;TRAINABLE_PARAMETERS=2351468
MAIN_GAIN=2.;PHASE_LIMIT=1.5707963267948966;IMAGE_LIMIT=.25


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
                F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False)


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered time/frequency residual-filter arm required')
    value=base_configuration('posterior03')
    value.update(schema='basis_filter_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        trainable_parameters=TRAINABLE_PARAMETERS,filter_parameters=FILTER_PARAMETERS,
        trainable='Original R03 coarse/correction plus new residual filter;three original F/defaultheads frozen',
        filter_epochs=EPOCHS,filter_updates=UPDATES,filter_location='After full R03 output,one current IQ frame',
        filter_basis='time' if arm==ARMS[0] else 'fftshifted full256 complex FFT',filter_widths=list(FILTER_WIDTHS),
        filter_main_gain_range=[1/MAIN_GAIN,MAIN_GAIN],filter_phase_limit=PHASE_LIMIT,filter_image_component_limit=IMAGE_LIMIT,
        filter='Core4to4 predicts bounded complex delta-main/image coefficients;output projection zero;delta applied to current R03 signal/conjugate',
        existing_spectral_path='Original coarse already uses bounded spectral correction;new comparison changes only added final-filter basis',
        new_learning_rate=2e-4,new_final_learning_rate=2e-5,parameter_groups='Original3e-5/newfilter2e-4;warm247/cos to0.1,WD1e-4,clip1',
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean + six practical;three original native heads independently',
        control_run=RUN,control_arm=ARMS[0],control_reuse='New matched time-basis filter trained once in fixed matrix',
        inference='One current IQ frame,one coarse/corrector/filter forward;no runtime F inside restorer,no iteration/RX/pseudo/bank/clean/otherframe/update',
        loss='Original posterior03 paired IQNMSE+.1logpower+mean3F(finalcos+.3T2KL);no extra CE/prototype/focus/style/FiLM',
        target_matrix=target_matrix(),target_selection='NONE;both fixed arms automatically complete all target views',
        matched_comparison='Same original R03 plus matching zero-output filter weights/parameters/E10/data/order/routes/optimizer/threeFloss;only time vs frequency basis differs;FFT cost reported')
    return value
