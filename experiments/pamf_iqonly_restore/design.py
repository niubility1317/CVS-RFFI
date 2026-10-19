from experiments.pamf_fullresolution_restore import design as paired
RUN='20261020-phase1-restoration-iqonly-selfsup-s2026092701-m2-r16'
PAIRED=paired.RUN
PAIRED_COMMIT='26a8b7f5fc5471719f600ad880bc465c4b544521'
ACTION=paired.ACTION
REFERENCE=paired.REFERENCE
ROWS=paired.ROWS
KIND=paired.KIND
MODEL_SEED=paired.MODEL_SEED
DATA_SEED=paired.DATA_SEED
GPUS=paired.GPUS
PRIMARY='unet_control'
FFT_CONTROL='unet_fft_reference'
TARGET_ROWS=(PRIMARY,FFT_CONTROL)
ALL_ROWS=(*ROWS,FFT_CONTROL)

def configuration():
    value=paired.configuration()
    value.update(schema='iqonly_prior_v1',run=RUN,loss='IQNMSE only;training FFT not computed',paired_source_run=PAIRED,
        paired_source_commit=PAIRED_COMMIT,primary=PRIMARY,secondary_source_only='tcn_prior',formal_rows=list(TARGET_ROWS),
        initial_pairing='Each architecture exact full epoch0 state equals R15,not inherited trained prior',
        target='OnlyUNetIQprimaryifallsourcepass;thenIQandfixedR15UNetFFT original168000x7 originalF50/defaulthead truthlast')
    return value
