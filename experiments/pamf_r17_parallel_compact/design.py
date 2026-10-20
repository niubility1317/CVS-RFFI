from experiments.pamf_r17_shared_factor import design as previous
RUN='20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07'
ARMS=('shared_three_eighths','shared_eighth','shared_quarter_pointwise','shared_eighth_pointwise')
FRACTIONS=dict(zip(ARMS,(.375,.125,.25,.125)))
POINTWISE=dict(zip(ARMS,(False,False,True,True)))
GPUS=dict(zip(ARMS,(2,3,6,7)))
PARAMETERS={'shared_three_eighths': 2702125, 'shared_eighth': 2463685, 'shared_quarter_pointwise': 2471609, 'shared_eighth_pointwise': 2352389}
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,10.
GPU_PRECEDING_RUN=previous.GPU_PRECEDING_RUN
SCORING_RUN=previous.RUN


def configuration():
    value=previous.configuration()
    value.update(schema='R17_parallel_compact_v1',run=RUN,arms=list(ARMS),fractions=FRACTIONS,pointwise_weight_sharing=POINTWISE,
        parameters=PARAMETERS,GPU=GPUS,
        initialization='OriginalR17source-finalonly;CPUmean3longweights+balancedSVD;twoarmsalsoCPUmeanoriginal1x1weights;noR04/R05/R06studentinit',
        sharing='Longfactorweightsacross3steps;optionaloriginal1x1weightsacross3steps;allstepbias/GN/otherstatesindependent',
        rank_rule='max(1,floor(fraction*min(Cout,Cin*kernel)));fractions3/8,1/8,1/4;factoronlywhenfewerweights',
        evaluation='Prospectivefixedfour-rowtargetcomparison;sourceV/waveformdiagnosticonly;noR06scoresusedforrank/loss/budget',
        source_rationale='UserauthorizesbatchparalleliterationwhileR06stilltraining;composeoriginalR17algebraonly;allfourarmsfixedbeforeanyR06newtargetresult',
        GPU_queue='physical2/3/6/7oneown/card;sourcefree>=18GB/allowner<2;R06healthy4/5continues;6ownGPUworkerstotal/preserveothers',
        CPU_queue='2threads/max1ownscoreracrossR06andR07;sharedR06score.lock',
        claim='Source-supervisedFknowledgeassistedcompression;oneexposedbenchmark;allfourfixedrowsreported;notF-freeunsupervised/globalminimumproof')
    return value
