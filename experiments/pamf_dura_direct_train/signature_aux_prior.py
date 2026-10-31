"""Matched train-only prior-SIG auxiliary control for the context prior."""
from .signature_context_prior import configure,base,AUX_RUN

if __name__=='__main__':
    configure(decoder_context=False,run=AUX_RUN,module='experiments.pamf_dura_direct_train.signature_aux_prior')
    base.main()
