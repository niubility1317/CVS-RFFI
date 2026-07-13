# MoPC-HR Non-Exemplar CIL SEI

Paper: `Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification`

Method short name: `MoPC-HR`

Official code: https://github.com/xmuLdz/MoPC-HR.git

Local workspace PDF: `E:\type10-7\paper_reproduction\mopc_hr_non_exemplar_cil_sei\MoPC-HR_Non-Exemplar_Class-Incremental_Learning_for_SEI.pdf`

Scope: paper-faithful, model-agnostic core for non-exemplar class-incremental SEI. The implementation covers class prototypes (eqs. 7-9), momentum prototype correction (eqs. 10-14), Gaussian prototype augmentation (eqs. 15-16), hierarchical regularization (eqs. 19-21), and the integrated loss (eq. 22).

The default correction path uses cosine similarity exactly as stated in the paper. An explicit `official_code_dot_softmax` audit mode captures the public trainer's dot-product/softmax divergence. Knowledge distillation is not part of the optimized loss because equation (22) omits it and the public trainer computes the term without adding it to the backward objective.

Paper defaults are 20 epochs for both base and incremental phases, batch size 16, SGD with learning rate 0.01, prototype noise standard deviation 0.05, and prototype momentum 0.97. The PDF is kept in the local workspace and is not force-added because release `.gitignore` excludes PDFs.

No completed CVS Stage2-C, satellite/LEO deployment, or open-set result is claimed by this module alone.
