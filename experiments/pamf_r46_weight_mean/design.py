"""One source-only training path; three prespecified final parameter readouts."""
import copy
from experiments.pamf_r46_multipoint_fast import design as baseline

RUN='20261021-phase1-restoration-R46-weight-mean-s2026092701-m3-r66'
SOURCE_ARM='source_path'
ARMS=('final','late_mean','step_ema')
EPOCHS,UPDATES,UPDATES_PER_EPOCH,BATCH,MICRO=20,4940,247,256,256
EMA_START_EPOCH,EMA_DECAY=11,.99
MEAN_EPOCHS=(16,17,18,19,20)
SOURCE_GPU=3
GPUS=dict(final=3,late_mean=2,step_ema=3)
PARAMETERS={a:273765 for a in (*ARMS,SOURCE_ARM)}
CONTEXT_PARAMETERS={a:185615 for a in (*ARMS,SOURCE_ARM)}
WIDTHS={a:(12,16,28,40) for a in (*ARMS,SOURCE_ARM)}
SPECTRAL_WIDTHS={a:(8,12,16,28) for a in (*ARMS,SOURCE_ARM)}
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=baseline.COMPACT_PARENT_RUN,baseline.COMPACT_PARENT_COMMIT,baseline.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=baseline.PARENT_RUN,baseline.PARENT_COMMIT,baseline.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=baseline.F_RUN,baseline.F_SEED,baseline.VIEWS,baseline.KIND
MODEL_SEED,DATA_SEED,SCORING_RUN=baseline.MODEL_SEED,baseline.DATA_SEED,baseline.SCORING_RUN
teacher_weight,learning_rate=baseline.teacher_weight,baseline.learning_rate


def configuration():
    value=copy.deepcopy(baseline.configuration())
    for key in ('max_own_across_R48_R49','healthy_R48_unchanged','throughput_control'):value.pop(key,None)
    value.update(schema='R46_single_source_path_parameter_means_v1',run=RUN,source_arm=SOURCE_ARM,arms=list(ARMS),
        parameters={a:PARAMETERS[a] for a in ARMS},context_parameters={a:CONTEXT_PARAMETERS[a] for a in ARMS},
        widths={a:list(WIDTHS[a]) for a in ARMS},context_widths={a:list(WIDTHS[a]) for a in ARMS},
        spectral_widths={a:list(SPECTRAL_WIDTHS[a]) for a in ARMS},
        architecture='Existing273765 GN4 context/three tied-long-weight priors/sixCG;all readouts same graph,no extra inference tensors/modules',
        hypothesis='Compare terminal weights with fixed late trajectory averages under the legal source objective;parameter smoothing may change restoration stability but no recognition gain is presumed',
        initialization='Only direct humanM35;same original GN4 weight-importance maps as static R53;no trained child weights',
        teacher='The original frozen M35 IQ teacher for the entire shared source path;parameter means are not training teachers',
        objective='Unchanged pairedclean IQNMSE+.05 relative-power multiscale STFT+linear1to.1 three-stage M35 IQteacher',
        source_training_paths=1,shared_source_epochs=20,shared_source_updates=4940,
        source_IQ_examples_total=1260000,teacher_IQ_examples_total=1260000,L_exposures_total=126000,U_exposures_total=1134000,
        cost_scope='One source trajectory counted once across three correlated readouts;averaging time/memory and each readout source/profile/formal costs separate',
        readouts=dict(final=dict(rule='Parameters after global update4940'),
            late_mean=dict(rule='Equal arithmetic mean of parameters after complete epochs16to20',epochs=list(MEAN_EPOCHS),snapshots=5),
            step_ema=dict(rule='First eligible updated student copied;then lerp toward each next updated student with weight1-decay',
                start_epoch=EMA_START_EPOCH,first_global_update=2471,last_global_update=4940,updates=2470,decay=EMA_DECAY)),
        alias_policy='Average each unique parameter once;preserve all tied prior aliases and separate biases/GNaffine;buffers fixed and copied without averaging',
        source_validation_updates=0,extra_neural_forwards=0,new_calibration_IQ=0,new_trainable_parameters=0,
        TX_labels_read=0,F_training_access=False,V_training=False,target_training=False,
        source_GPU=SOURCE_GPU,GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,
        predecessor_run='20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65',
        predecessor_by_GPU={'3':'prune_after5','2':'prune_after10'},
        freeze_rule='All three prespecified readouts frozen only after shared finalE20/4940;all get source1800x7/profile/formal168000x7 without source selection;seven predictions before independent truthlast',
        source_score_selection=False,target_feedback=False,goal_complete=False)
    return value
