"""Fixed source-only GN-affine correction of the existing compact topology."""
from experiments.pamf_r46_fold_compensate import design as base

parent=base.parent
RUN='20261021-phase1-restoration-R46-GN-response-s2026092701-m2-r57'
ARMS=('gn_full','gn_half')
TASK={a:False for a in ARMS}
MODE={a:'R46_weight_fold_full_GN_response_'+a for a in ARMS}
WIDTHS={a:(12,16,28,40) for a in ARMS}
CONTEXT_WIDTHS=WIDTHS.copy()
SPECTRAL_WIDTHS={a:(8,12,16,28) for a in ARMS}
PARAMETERS={a:273765 for a in ARMS}
CONTEXT_PARAMETERS={a:185615 for a in ARMS}
GPUS={'gn_full':3,'gn_half':2}
GN_STRENGTH={'gn_full':1.,'gn_half':.5}
GN_RIDGE_FRACTION=1e-3
GN_SLOPE_BOUNDS=(.25,4.)
GN_MODULES=168
FOLD_STRENGTH={a:1. for a in ARMS}
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=base.F_RUN,base.F_SEED,base.VIEWS,base.KIND
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
SCORING_RUN=base.SCORING_RUN
CAL_SEED,CAL_PER_ROLE,CAL_BATCH,CAL_POSITIONS=base.CAL_SEED,64,8,16
CAL_UNIQUE,CAL_INPUTS,RIDGE_FRACTION=128,512,base.RIDGE_FRACTION
FOLD_CAL_RUN,FOLD_CAL_COMMIT=base.RUN,'6bea50605a1c2cfb49645b5c2f7ba8306b39791d'
PREDECESSOR_RUN,PREDECESSOR_COMMIT=FOLD_CAL_RUN,FOLD_CAL_COMMIT
PREDECESSOR_MODULE='experiments.pamf_r46_fold_compensate.runner'
PREDECESSOR_ROW={2:'fold_half',3:'fold_full'}
REUSED_CONTROL=dict(run=FOLD_CAL_RUN,row='fold_full',method_commit=FOLD_CAL_COMMIT,parameters=273765,
    initialization='same full-fold input weights and GN4 maps;GN affine remains sliced',micro=256,epochs=20,updates=4940)
learning_rate=base.learning_rate

def teacher_weight(epoch,arm):
    if arm not in ARMS:raise ValueError('Registered GN-response arm required')
    return base.teacher_weight(epoch,'fold_full')

def configuration():
    value=base.configuration()
    value.update(schema='R46_GN_response_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},
        architecture='Existing273765/full-fold topology;fit only existing GN affine initialization;3prior6CG/all joint/no new runtime branch or parameters',
        teacher_weight_schedule={a:[teacher_weight(e,a) for e in range(1,21)] for a in ARMS},fold_strength=FOLD_STRENGTH,GN_strength=GN_STRENGTH,
        calibration=dict(unique_source_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=CAL_SEED,batch=8,positions_per_GN_output=16,
            parent_IQ_inputs=256,student_IQ_inputs=256,total_new_G_inputs=512,reused_R55_parent_G_inputs=256,GN_modules=GN_MODULES,
            reused_fold_run=FOLD_CAL_RUN,reused_fold_commit=FOLD_CAL_COMMIT,GN_ridge_fraction=GN_RIDGE_FRACTION,GN_slope_bounds=list(GN_SLOPE_BOUNDS),
            method='Per retained output channel,fit parent GN post-affine output=a*initial full-fold child GN output+b using paired received inputs,identity-centered slope ridge and free intercept;clip slope to[.25,4]',
            application='All fits collected on unchanged initial child,then fold alpha1/.5 correction into existing GN weight/bias;no Conv/Linear changes beyond reused full-fold;no post-fit source selection',
            coordinates='Identical selected output-channel maps and 16 fixed spatial positions;separate stage GN remains untied',
            device='CPU2',TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0,
            sharing='One frozen parent/child source pass shared between two rows;reuse exact R55 source indices and ridge projections;no trained R55 checkpoint',
            limitation='Local pre-SiLU calibration error cannot guarantee simultaneous downstream/full-G fidelity or TX identification gains',
            calibration_states_not_runtime_inputs=True),
        hypothesis='Pruning changes the channels used for GN mean/variance;source-only affine matching may reduce the resulting retained-channel response mismatch without increasing inference cost',
        reused_full_fold_control=REUSED_CONTROL,GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,max_own_across_R55_R57=2,
        predecessor_by_GPU={str(g):dict(run=PREDECESSOR_RUN,row=PREDECESSOR_ROW[g],method_commit=PREDECESSOR_COMMIT) for g in (2,3)},
        authorization='Standing performancefirst compression optimization;matrix and affine rule fixed before reading R55 formal scores;GPU2/3only;wait predecessor actual exit;healthy tasks unchanged',
        throughput_control='Same global256/micro256/directbuffer/E20;existing GN affine only;reused source fold statistics plus one extra frozen parent/child pass;no runtime calibration',
        target_feedback=False,goal_complete=False)
    for key in ('max_own_across_R54_R55','reused_weight_only_control'):value.pop(key,None)
    return value
