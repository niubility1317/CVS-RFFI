"""Source-calibrated channel selection in the existing R46 compression topology."""
from experiments.pamf_r46_anchor_refine import design as baseline
from experiments.pamf_r46_multipoint_fast import design as previous
parent=previous.parent
learning_rate=previous.learning_rate
RUN='20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54'
ARMS=('activation_energy','projection_sensitivity')
TASK={a:False for a in ARMS}
MODE={a:'R46_source_GN4_'+a for a in ARMS}
WIDTHS={a:(12,16,28,40) for a in ARMS}
CONTEXT_WIDTHS=WIDTHS.copy()
SPECTRAL_WIDTHS={a:(8,12,16,28) for a in ARMS}
PARAMETERS={a:273765 for a in ARMS}
CONTEXT_PARAMETERS={a:185615 for a in ARMS}
GPUS={'activation_energy':3,'projection_sensitivity':2}
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=previous.COMPACT_PARENT_RUN,previous.COMPACT_PARENT_COMMIT,previous.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=previous.F_RUN,previous.F_SEED,previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
SCORING_RUN=previous.SCORING_RUN
CAL_SEED,CAL_PROJECTION_SEED,CAL_PER_ROLE,CAL_BATCH,CAL_DRAWS=2026102154,2026102155,64,8,2
CAL_UNIQUE,CAL_INPUTS=128,256
PREDECESSOR_RUN=baseline.RUN
PREDECESSOR_COMMIT='df36c3d3da4e026b747776b65b58d4921dc1a71e'
PREDECESSOR_MODULE='experiments.pamf_r46_anchor_refine.runner'
PREDECESSOR_ROW={2:'prune_anchor',3:'retain_anchor'}
REUSED_CONTROL=dict(run=baseline.RUN,row='prune_decay',method_commit=PREDECESSOR_COMMIT,parameters=273765,initialization='weight-only GN4 importance',micro=256,epochs=20,updates=4940)

def teacher_weight(epoch,arm):
    if arm not in ARMS:raise ValueError('Registered source-selection arm required')
    return previous.teacher_weight(epoch)

def configuration():
    value=previous.configuration()
    value.update(schema='R46_source_channel_selection_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},
        architecture='Existing273765 topology from fixedR48two-widthmatrix;only initialization channel maps change;all context/prior jointly trained;3prior6CG/no new branch',
        objective='Unchanged pairedclean IQNMSE+.05relativepowerSTFT+linear1to.1mean3stageIQteacher to frozen humanR46parent',
        teacher_weight_schedule={a:[teacher_weight(e,a) for e in range(1,21)] for a in ARMS},
        calibration=dict(unique_source_IQ=128,L_s=64,U_s=64,received_views=['01','20'],frozen_parent_IQ_inputs=256,batch=8,projection_draws=2,
            sampling_seed=CAL_SEED,projection_seed=CAL_PROJECTION_SEED,device='CPU2',TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
            methods=['mean squared encoder activation','mean squared per-channel-gain random projection derivative of normalized final IQ'],
            projection='Independent Rademacher output directions divided by sqrt512 and sqrt pairedclean IQ meanenergy clamp1e-4',
            maps='Stable within-GN4 top channels;common averaged prior0/1/2 scores and identical prior maps preserve shared long-weight semantics',
            calibration_reused_between_two_rows=True,calibration_states_not_runtime_inputs=True),
        hypothesis='Weight energy alone ignores actual source response and downstream influence;compare fixed activation and output-sensitivity channel selections;no TX/F/target tuning',
        reused_weight_only_control=REUSED_CONTROL,freeze_rule='Both fixedfinalE20/4940,all rows auto source/profile/full7formal then independentCPUtruthlast;no sourcegate or earlystop',
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,max_own_across_R53_R54=2,
        predecessor_by_GPU={str(g):dict(run=PREDECESSOR_RUN,row=PREDECESSOR_ROW[g],method_commit=PREDECESSOR_COMMIT) for g in (2,3)},
        authorization='Standing performancefirst compression goal;GPU2/3only;wait fixed R53 row completion AND actual owned handles exit per card,never coexist with own R53 GPUworker',
        throughput_control='Same global256/micro256/directfloat32buffer;calibration CPU-only shared once;no inference parameter or branch added',
        target_feedback=False,goal_complete=False)
    value.pop('max_own_across_R48_R49',None);value.pop('healthy_R48_unchanged',None)
    return value
