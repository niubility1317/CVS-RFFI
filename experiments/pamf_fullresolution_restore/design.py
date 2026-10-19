RUN='20261020-phase1-restoration-fullresolution-selfsup-s2026092701-m2-r15'
ACTION='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
REFERENCE='20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'
ROWS=('unet_control','tcn_prior')
KIND='native_map_prior'
MODEL_SEED=2026092701
DATA_SEED=2026102001
GPUS=(0,1,5,7)


def configuration():
    return dict(schema='fullresolution_prior_v1',run=RUN,rows=list(ROWS),kind=KIND,action_parent=ACTION,source_reference=REFERENCE,
        model_seed=MODEL_SEED,data_seed=DATA_SEED,seed_scope='After frozen parent loading,before scratch prior construction',
        prior_initialization='Scratch zero-output heads;differenthiddenarchitectures,notidenticalweights;no trained R10/R12prior',
        tcn=dict(stem_kernel=5,width=128,original_blocks=6,dilations=[1,2,4,8,16,32],full_length=256,
            convolutional_receptive_field=269,normalization='Original GroupNorm4 percurrentpacket,no persistentstats'),
        training_TX_labels_read=0,training_F_access=False,target_training=False,V_updates=0,
        epochs=20,updates=4940,batch_size=256,L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,
        optimizer='AdamW',peak_lr=2e-4,final_lr=2e-5,warmup_updates=247,weight_decay=1e-4,gradient_clip=1.,precision='FP32',
        loss='IQNMSE+.1FFTlogpowerL1 only',calibration_count=0,training_routes=dict(clean=.15,residual=.85,full=0.),
        observations='Reuse original R07 complete20cachedepochs,no newnoise/physicalrecords',
        trainable='Onlynewprior;coarse/action frozen',cg_steps=6,prior_cumulative_epochs=20,component_cumulative_epochs=60,
        checkpoint_choice='Fixedlast20,no Vselected checkpoint',
        source_thresholds=dict(mean6_NMSE_vs_control_max_ratio=1.05,mean6_NMSE_vs_reference_max_ratio=1.05,clean_NMSE_max=.02,
            lowurban_gain_vs_control_min_pp=.5,lowurban_gain_vs_reference_min_pp=.5,midurban_gain_vs_control_min_pp=.5,
            midurban_gain_vs_reference_min_pp=.5,clean_drop_vs_original_max_pp=.1,
            worst_lowurban_TX_vs_control_min_pp=-1.,worst_lowurban_TX_vs_reference_min_pp=-1.,
            worst_other_scene_vs_control_min_pp=-.5,worst_other_scene_vs_reference_min_pp=-.5),
        target='OnlyTCNprimaryifallsourcepass;thenbothmatchedrows original168000x7 originalF50/defaulthead truthlast',
        user_goal=dict(lowurban_accuracy_min=70.,other_five_max_gap_vs_original_clean_pp=5.),target_feedback=False,
        GPU_pool=list(GPUS),max_own_GPU_workers=2,all_owner_cap=2,min_free_MB=12000,
        capacity_lock='receiver_residual_capacity4.lock',CPU_threads=2,
        stop_rule='Wrongsource/parent,collision,nonfinite,invalidpredictions:stopfailedrow,holdpending,preservehealthyrows,noautomaticretry')
