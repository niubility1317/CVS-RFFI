from experiments.pamf_unrolled_restore import design as previous

RUN='20261020-phase1-restoration-iterative-controls-selfsup-s2026092701-m2-r18'
UNROLLED=previous.RUN
UNROLLED_COMMIT='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
REFERENCE_CANDIDATE='unroll_unshared'
SINGLE=previous.SINGLE
SINGLE_COMMIT=previous.SINGLE_COMMIT
FFT=previous.FFT
FFT_COMMIT=previous.FFT_COMMIT
ACTION=previous.ACTION
REFERENCE=previous.REFERENCE
KIND=previous.KIND
MODEL_SEED=previous.MODEL_SEED
DATA_SEED=previous.DATA_SEED
GPUS=previous.GPUS
CANDIDATES={
    'current_anchor':dict(anchor='current',CG_schedule=[2,2,2]),
    'terminal_dc':dict(anchor='current',CG_schedule=[0,0,6]),
}
ROWS=tuple(CANDIDATES)
ALL_ROWS=ROWS


def configuration():
    value=previous.configuration()
    value.update(schema='iterative_control_v1',run=RUN,rows=list(ROWS),
        unrolled_source_run=UNROLLED,unrolled_source_commit=UNROLLED_COMMIT,
        unrolled_source_candidate=REFERENCE_CANDIDATE,CG_per_step='candidate_specific',
        candidate_designs={k:dict(anchor=v['anchor'],CG_schedule=list(v['CG_schedule'])) for k,v in CANDIDATES.items()},
        prior_anchor='candidate_specific',prior_architecture='Three unshared original Core9 U-Nets',
        shared_initialization='All full initial weights match R17 unshared epoch0;different solver paths,not equal full outputs',
        parameter_control='2507526 trainable,three Core calls and six CG iterations;CG restart/preparation costs differ',
        source_selection='All R17/native AND R16/native AND R15/native guards;rank lowmidmean,lowestTX,lowestRX,lexicalname',
        target='Onlysource-selectedcandidate original168000x7 F50/defaulthead truthlast;reuse fixed R17 formal predictions after new scoring,no repeated control prediction')
    return value
