from experiments.pamf_attention_restore import design as previous
from experiments.pamf_attention_restore.design import (KIND,ACTION,REFERENCE,MODEL_SEED,DATA_SEED,GPUS,
    SINGLE,SINGLE_COMMIT,FFT,FFT_COMMIT,UNROLLED,UNROLLED_COMMIT,REFERENCE_CANDIDATE,
    ALTERNATING,ALTERNATING_COMMIT,ALTERNATING_CANDIDATE,CONTROL_RUN,CONTROL_COMMIT,
    CONTROL_CANDIDATE,PRECEDING,PRECEDING_COMMIT)

RUN='20261020-phase1-restoration-wavelet-selfsup-s2026092701-m2-r29'
ROWS=('lowpass_control','haar_allbands')
ALL_ROWS=ROWS
CANDIDATE='haar_allbands'


def configuration():
    value=previous.configuration()
    for key in ('tcn','attention'):value.pop(key)
    value.update(schema='wavelet_prior_v1',run=RUN,rows=list(ROWS),
        prior_architecture='Three unshared Haar analysis/synthesis U-Nets;9x256 in,2x256 out;CG2/2/2 unchanged',
        prior_initialization='Scratch cloned initial steps;only frozen original R02/R07 context inherited',
        shared_initialization='Identical complete states and initial deployed outputs in both rows;identical parameter count',
        wavelet=dict(widths=[32,64,96,128],levels=3,low='(even+odd)/sqrt2',high='(even-odd)/sqrt2',
            bands='low then high on channels;synthesis interleaves original even/odd',
            ablation='Control zeros analysis high coefficients before each encoder block;allband retains them',
            learned_synthesis='Conv1x1 projects current to2skipwidth then IWT;full-resolution skips remain',
            normalization='Original per-packet GroupNorm4;no batch stats',thresholding=False,overall_network_invertible=False),
        source_selection='Only haar_allbands eligible;all six original source guards plus matched lowpass control guards;no target feedback',
        matched_control='lowpass_control;source-only comparator,not an eligible target candidate',
        target='Only source-qualified haar_allbands;original168000x7 F50/defaulthead truthlast;otherwise0target',
        question_origin='R28 source waveform/identity mismatch plus original Core average-pool detail loss;not R27 target ranking',
        scientific_scope='TX-unlabeled simulated-paired self-supervision;feature-band retention ablation,not literal TX-frequency identification',
        literature='MWCNN CVPRW2018 image restoration;1D Haar adaptation is a new IQ hypothesis,not an established RFFI result')
    return value
