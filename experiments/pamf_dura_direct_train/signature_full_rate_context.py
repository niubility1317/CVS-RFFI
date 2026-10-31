"""Matched full-rate IQ branch conditioned on each prior's predicted SIG."""
from .signature_full_rate import configure, base

if __name__ == '__main__':
    configure(mode='conditioned')
    base.main()
