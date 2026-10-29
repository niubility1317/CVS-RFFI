# Native external comparison snapshot

This subtree contains the dependency closure used by `external_native.py`.
It comes from the tracked `comparison_suite` and `paper_reproduction` modules
in this checkout (the same matched WiSig extension used by the immutable
`native_comparisons_20260930_r01` release). Imports are namespaced to this
subtree so the Phase1 activation code cannot silently replace these classes.
The CSIL zero-bias module retains only its exact inference primitive.

The local changes are explicit: generator-compatible Kaiming initialization
for PyTorch 2.1, and an optional MoPC public-code mode (raw-dot softmax prototype
correction, unsquared layer L2 regularization, temperature-2 prototype CE).
The default MoPC paper mode remains cosine correction, squared layer L2, and
temperature-1 prototype CE. Both are matched IQ/task extensions, not claims
of exact original ADS-B dataset reproduction. Source state dictionaries keep
their historical parameter names and shapes.
