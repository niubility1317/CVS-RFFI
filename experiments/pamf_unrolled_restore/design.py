from experiments.pamf_iqonly_restore import design as single
RUN='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
SINGLE=single.RUN
SINGLE_COMMIT='50587946b6c83b7b32500696af7fa7b9086484b5'
FFT=single.PAIRED
FFT_COMMIT=single.PAIRED_COMMIT
ACTION=single.ACTION
REFERENCE=single.REFERENCE
ROWS=('unroll_shared','unroll_unshared')
KIND=single.KIND
MODEL_SEED=single.MODEL_SEED
DATA_SEED=single.DATA_SEED
GPUS=single.GPUS
CONTROL='single_shot_reference'
ALL_ROWS=(*ROWS,CONTROL)

def configuration():
    value=single.configuration()
    for key in ('paired_source_run','paired_source_commit','primary','secondary_source_only','formal_rows','initial_pairing'):value.pop(key,None)
    value.pop('tcn',None)
    value.update(schema='unrolled_prior_v1',run=RUN,rows=list(ROWS),loss='Final IQNMSE only;FFT auxiliary loss not computed',
        single_source_run=SINGLE,single_source_commit=SINGLE_COMMIT,FFT_source_run=FFT,FFT_source_commit=FFT_COMMIT,
        steps=3,CG_per_step=2,CG_total=6,warm_start=True,prior_anchor='frozen coarse',incoming=9,
        prior_architecture='Original Core9 U-Net',prior_initialization='Scratch Core9;cloned initial states across all steps/rows,no pretrained prior',
        current_inputs='received2,coarse2,current2,unitRMS adjointinnovation2,clampedlognativevariance1',
        stage_gradients='Endtoend,no detach',shared_initialization='Clone one seeded Core9 into all unshared steps;equalinitialfulloutput',
        source_selection='Both eligible iff all R16/native and R15/native guards;rank lowmidmean,lowestTX,lowestRX,lexicalname',
        target='Onlysource-selectedcandidate and fixedR16single-shot IQ original168000x7 F50/defaulthead truthlast')
    return value
