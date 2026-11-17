# MoPC-HR retained implementation

For future work the retained MoPC-HR registration implementation is the author
repository behavior (`public_code_semantics_matched_IQ_extension`). The author
source used for the ADS-B check is commit `ae65543` in the locally retained
author-source checkout. The Phase2 IQ implementation is an adaptation to the
project's backbone and data interfaces; it is not the untouched repository.

Prototype correction uses the repository's dot-product softmax, hierarchical
regularization uses the repository behavior, and the retained configuration
does not add gradient clipping. The former paper-formula and clipped stable
implementations are retired. Old registered results without verified author
version metadata must not be reported as author-version results.

Retiring the clipping variant does not establish numerical stability. A
nonfinite training state remains a failure and must not be silently accepted.
Old Git history and research records are historical, not active method choices.
This decision does not launch or authorize a new experimental matrix.
