from experiments.pamf_r17_task import design as previous

RUN='20261020-phase1-restoration-r17-shared-factor-s2026092701-m2-r06'
ARMS=('shared_half','shared_quarter')
FRACTIONS=dict(zip(ARMS,(.5,.25)))
PARAMETERS=dict(zip(ARMS,(2821345,2582905)))
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,10.
GPUS=dict(zip(ARMS,(4,5)))
GPU_PRECEDING_RUN='20261020-phase1-restoration-r17-compact-target-comparison-s2026092701-m2-r05'


def configuration():
    return dict(schema='R17_shared_factored_v1',run=RUN,arms=list(ARMS),views=list(VIEWS),fractions=FRACTIONS,parameters=PARAMETERS,GPU=GPUS,
        parent_run=PARENT_RUN,parent_commit=PARENT_COMMIT,parent_row='unroll_unshared',parent_epoch=20,
        epochs=EPOCHS,updates=UPDATES,batch=BATCH,model_seed=MODEL_SEED,data_seed=DATA_SEED,
        initialization='OriginalR17source-final only;CPUmeanofthree longkernels thenbalancedtruncatedSVD;no testedstudentweights',
        sharing='Shareonlyleftandrightfactorweights across3steps;bias/GroupNorm/1x1andotherstatesremainstep-specific',
        rank_rule='max(1,floor(fraction*min(Cout,Cin*kernel)));factoronlywhenfewerweights,otherlongkernelsremainmeanshared',
        loss='SamefixedR04:10mean3stageIQNMSE_to_originalR17 + KL_T1_Fposterior + halfsquaredunitF160distance;taskcoefficients1',
        budget='E20/4940/source63000x20/L126000/U1134000;originalR07cache/order/model2701/data2026102001;priorcumulative40/serialrestore80',
        optimizer='AdamW2e-4,warm247/cos2e-5,WD1e-4,clip1,FP32/noTF32',
        data='OriginalR07source63000=L6300/U56700;epochs1..20cachedreceived15%clean/85%residual;noTXlabels/V/targetintraining',
        L_exposures=126000,U_exposures=1134000,inherited_prior_epochs=20,cumulative_prior_epochs=40,
        frozen_coarse_epochs=20,frozen_action_epochs=20,cumulative_serial_training_passes=80,
        training_F_access=True,F_run=F_RUN,F_commit=F_COMMIT,F_seed=F_SEED,F_parameter_updates=0,head_fit_updates=0,
        new_TX_labels=0,calibration_IQ_examples=0,target_training=False,
        evaluation='Prospectivefixedtwo-rowtargetcomparison aftersource-final;waveformandVdiagnosticsonly,notidentityperformancegates',
        target_count_per_view=168000,source_V_count_per_view=1800,original_default_head=True,
        freeze_rule='FinalE20/4940;noearlystop/noVselection;eachrowautomaticallysource7predict/truth-lastscore/profile/target7predict/truth-lastscore',
        GPU_queue='physical4/5oneownworker/card;sourcefree>=18GB/allowner<2;allpreviousR05ownedGPU/CPUhandlesexited;preserveotherowners',
        CPU_queue='2threads/max1ownscorer/ownscore.lock',capacity_lock='.pamf-reception-capacity4.lock',
        stop_rule='Contract/nonfinite/OOM/technicalerrorstopsaffectedrow,preservesoutputs;healthyrowcontinues;nosilentfallback/noautomaticretry;bothsystemicfailuresstoprun',
        inference='CurrentIQonly;3stepsx2CG;noFteacher/clean/TX/RX/scene/otherframes/cache/stateupdatesinrestorer',
        profile_scope='SourceV16IQ/batch1;checkpointpayloadandoptimizerexcludedfromGPU;reportresidentandpeakabovebaseline;nootherownercontrol',
        target_acceptance_allowance_pp=1.,target_feedback=False,
        claim='Source-supervisedFknowledgeassistedcompression;exposedbenchmark;notF-freeunsupervised or globalminimumproof')
