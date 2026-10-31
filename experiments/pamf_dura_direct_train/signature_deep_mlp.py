"""Companion nonlinear deep-SIG residual predictor with identical E30 losses."""
from .signature_deep_prior import configure,base


if __name__=='__main__':
    configure(mode='mlp');base.main()
