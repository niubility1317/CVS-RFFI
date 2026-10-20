"""Fixed source-only channel reconstruction; no new inference components."""
from experiments.pamf_r46_source_prune import design as predecessor
from experiments.pamf_r46_multipoint_fast import design as previous

parent=previous.parent
learning_rate=previous.learning_rate
RUN='20261021-phase1-restoration-R46-fold-compensate-s2026092701-m2-r55'
ARMS=('fold_full','fold_half')
TASK={a:False for a in ARMS}
MODE={a:'R46_weight_GN4_source_ridge_'+a for a in ARMS}
WIDTHS={a:(12,16,28,40) for a in ARMS}
CONTEXT_WIDTHS=WIDTHS.copy()
SPECTRAL_WIDTHS={a:(8,12,16,28) for a in ARMS}
PARAMETERS={a:273765 for a in ARMS}
CONTEXT_PARAMETERS={a:185615 for a in ARMS}
GPUS={'fold_full':3,'fold_half':2}
FOLD_STRENGTH={'fold_full':1.,'fold_half':.5}
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=previous.COMPACT_PARENT_RUN,previous.COMPACT_PARENT_COMMIT,previous.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=previous.F_RUN,previous.F_SEED,previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
SCORING_RUN=previous.SCORING_RUN
CAL_SEED,CAL_PER_ROLE,CAL_BATCH,CAL_POSITIONS=2026102156,64,8,16
CAL_UNIQUE,CAL_INPUTS,RIDGE_FRACTION=128,256,1e-3
PREDECESSOR_RUN=predecessor.RUN
PREDECESSOR_COMMIT='554c52baf75ab8da2618a5480851b74271018103'
PREDECESSOR_MODULE='experiments.pamf_r46_source_prune.runner'
PREDECESSOR_ROW={2:'projection_sensitivity',3:'activation_energy'}
REUSED_CONTROL=dict(run='20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53',row='prune_decay',
    method_commit='df36c3d3da4e026b747776b65b58d4921dc1a71e',parameters=273765,initialization='weight-only GN4 importance',micro=256,epochs=20,updates=4940)

def teacher_weight(epoch,arm):
    if arm not in ARMS:raise ValueError('Registered reconstruction arm required')
    return previous.teacher_weight(epoch)

def configuration():
    value=previous.configuration()
    value.update(schema='R46_source_channel_reconstruction_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},
        architecture='Existing273765 topology;fixed weight-only GN4 maps;fold dropped-channel linear reconstruction into existing weights;3prior6CG/all joint/no extra runtime branch',
        objective='Unchanged pairedclean IQNMSE+.05relativepowerSTFT+linear1to.1mean3stageIQteacher to frozen humanR46parent',
        teacher_weight_schedule={a:[teacher_weight(e,a) for e in range(1,21)] for a in ARMS},fold_strength=FOLD_STRENGTH,
        calibration=dict(unique_source_IQ=128,L_s=64,U_s=64,received_views=['01','20'],frozen_parent_IQ_inputs=256,batch=8,positions_per_conv_input=16,
            sampling_seed=CAL_SEED,ridge_fraction=RIDGE_FRACTION,device='CPU2',TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,
            method='Uncentered input-channel second moments;retained rows exactidentity;dropped rows ridge regressed on retained channels;no intercept so zero padding commutes',
            ridge='1e-3 times mean retained-channel diagonal,clamped1e-12;single fixed value,no source/target grid selection',
            sharing='Aggregate moments across tied prior Conv weights before solving one projection;untied biases and normalization unchanged',
            folding='W_selected_outputs @ P along input channels;alpha1 or .5 blend with ordinary sliced weights;bias and GN slices unchanged',
            limitation='Local frozen-parent channel reconstruction does not guarantee post-pruning GN responses,full-G fidelity or TX accuracy',
            calibration_reused_between_two_rows=True,calibration_states_not_runtime_inputs=True),
        hypothesis='Plain slicing discards the correlated contribution of dropped input channels;source-only reconstruction can compensate existing weights without extra inference parameters',
        reused_weight_only_control=REUSED_CONTROL,freeze_rule='Both fixedfinalE20/4940;all rows auto source/profile/full7formal then independentCPUtruthlast;no sourcegate or earlystop',
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,max_own_across_R54_R55=2,
        predecessor_by_GPU={str(g):dict(run=PREDECESSOR_RUN,row=PREDECESSOR_ROW[g],method_commit=PREDECESSOR_COMMIT) for g in (2,3)},
        authorization='Standing performancefirst compression goal;GPU2/3only;prior R54 owned workers must be complete and actually exited per card;no healthy-task mutation',
        throughput_control='Same global256/micro256/directfloat32buffer;one CPU-only shared calibration;inference topology/parameters/MAC unchanged',
        target_feedback=False,goal_complete=False)
    value.pop('max_own_across_R48_R49',None);value.pop('healthy_R48_unchanged',None)
    return value
