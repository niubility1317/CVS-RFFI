# CSIL Class-Incremental IoT Reproduction

This directory is the paper-faithful surface for `Class-Incremental Learning for Wireless Device Identification in IoT`.

Scope:

- ADS-B wireless device identification, not WiSig/ManySig.
- Five class-incremental batches: 20 initial classes and four increments of 20 classes.
- CSIL mechanics: zero-bias cosine similarity classifier, stage-wise channel expansion, old-channel masking, KD, EWC, and DoC tracking.
- Claim boundary: `paper_faithful_adsb_class_incremental_only`.
- Official MATLAB reference: `E:\type10-7\local_artifacts\external_refs\pcwhy_CSIL` at commit `8ce8637daf4dc60eeb1c56bff64c050c5b2353e9`.
- Zero-bias classifier follows the official MATLAB layer convention: normalized fingerprint score scaled and shifted as `5*cosine+5`.

Out of scope:

- CVS Stage2-A/B/C deployment evidence.
- Satellite/LEO success claims.
- Unknown/open-set Phase3 claims.

Server/CVS handoff:

- For ADS-B paper reproduction, use this module only as the paper-faithful layer.
- For CVS Stage2-B/C experiments, build a separate `paper_reproduction/cvs_aligned` adapter with `cvs_extension=true`, explicit `R_s/R_t`, `Y_old/Y_new/Y_unknown`, `k_shot`, support/query splits, and satellite/LEO view.
