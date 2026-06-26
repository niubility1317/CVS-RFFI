# FedCVS-RFFI-VMB Traceability

Source design report: `C:/Users/lh594/Downloads/FedCVS_RFFI_VMB_design_report.md`

Implementation policy:
- Preserve existing centralized, FedAvg, FedProx, StyleBank, ProtoBank, and satellite defaults.
- Add FedCVS-RFFI-VMB as an opt-in federated route.
- Keep all N607-facing changes local first; no remote sync or launch was performed in this implementation turn.
- Because `E:/type10-7` is not a git repository, the touched files were copied into `E:/type10-7/code/snapshots/20260527_185647_fedcvs_vmb/` and listed in `SYNC_MANIFEST.txt`.

Verification summary:
- `conda activate ssr-gpu; python -m py_compile train.py model_dual_cvsincnet.py federated\fed_trainer.py federated\fedcvs_vmb.py federated\fed_aggregate.py evaluation\fedcvs_vmb_probe.py evaluation\fedcvs_vmb_analysis.py`
- `conda activate ssr-gpu; python -m pytest tests/test_fedcvs_vmb.py tests/test_federated_aggregation.py tests/test_federated_train_integration.py tests/test_federated_trainer_smoke.py -q`
- Result: `29 passed, 1 skipped`.
- `conda activate ssr-gpu; python train.py --help | Select-String -Pattern "fedcvs_vmb|fl_vmb_stage1_local_steps|fl_vmb|use_tx_adv"`

Sub-agent review closure:
- Implementation review initially found unresolved traceability, missing snapshot, incomplete Stage 1 semantics, incomplete diagnostics, config ablation mismatch, and a multi-batch gradient scaling issue.
- Follow-up fixes added traceability closure, snapshot/manifest, explicit Stage 1 local-pretrain state averaging, `--fl_vmb_stage1_local_steps`, VMB full defaults for `--train_mode fedcvs_vmb`, original A0-A14 ablation mapping, client drift/domain loss variance/communication diagnostics, and a multi-batch gradient rescale guard.
- Cross-domain rationale review concluded the mechanism is promising and testable, but any improvement claim must wait for ablations and four-probe evidence.

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| VMB-R01 | Sec. 2, 4, 18 | Add an opt-in FedCVS-RFFI-VMB path without changing old behavior. | `train.py`, `federated/fed_trainer.py` | verified | pytest + help output | `train_mode` defaults stay centralized; VMB is selected only by `--train_mode fedcvs_vmb`. |
| VMB-R02 | Sec. 3, 4, 11 | Keep final TX classifier using `z_t`/`z_id` only; `z_r`/`z_dom` stays auxiliary. | `model_dual_cvsincnet.py`, `federated/fed_trainer.py` | verified | smoke tests | The optional TX adversary reads `z_dom`; final `tx_logits` remain from the ID/TX branch. |
| VMB-R03 | Sec. 5.2 | Preserve transmitter CE loss as mandatory base objective. | `federated/fed_trainer.py` | verified | smoke tests | `loss_cls` remains the base term for VMB and legacy routes. |
| VMB-R04 | Sec. 5.3, 6.3, 8.1, 11.6 | Add server TX prototype bank with EMA, counts, and normalized class prototypes. | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py` | verified | `tests/test_fedcvs_vmb.py` | Class counts gate active prototypes. |
| VMB-R05 | Sec. 5.4, 6.3, 8.1, 11.6 | Add server RX/client prototype bank with EMA and normalized receiver prototypes. | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py` | verified | `tests/test_fedcvs_vmb.py` | One receiver/client prototype is maintained per FL client. |
| VMB-R06 | Sec. 5.3, 7.5 | Add TX prototype contrastive CE loss on `z_t`. | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py` | verified | helper + smoke tests | Controlled by `lambda_vmb_tx_proto`; VMB defaults set it to 0.1 unless explicitly overridden. |
| VMB-R07 | Sec. 5.4 | Add RX prototype contrastive CE loss on `z_r`. | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py` | verified | helper + smoke tests | Controlled by `lambda_vmb_rx_proto`; VMB defaults set it to 0.1 unless explicitly overridden. |
| VMB-R08 | Sec. 5.5 | Continue receiver adversarial loss on `z_t`. | `model_dual_cvsincnet.py`, `federated/fed_trainer.py` | verified | smoke tests | Existing GRL receiver adversarial path is reused under `receiver_agnostic_bex02`. |
| VMB-R09 | Sec. 5.6 | Add transmitter adversarial head on `z_r` with GRL. | `model_dual_cvsincnet.py`, `train.py`, `federated/fed_trainer.py` | verified | smoke tests + help output | `use_tx_adv_on_zdom` remains opt-in for old routes and auto-on for VMB unless explicitly disabled. |
| VMB-R10 | Sec. 5.7 | Keep cross-covariance/orthogonal loss between `z_t` and `z_r`. | `federated/fed_trainer.py` | verified | smoke tests | Existing `_covariance_orth_loss` remains available through `lambda_orth`. |
| VMB-R11 | Sec. 5.8 | Keep consistency hooks optional and conservative. | `federated/fed_trainer.py`, `train.py` | verified | existing smoke coverage | Satellite/consistency hooks remain opt-in and are not forced by VMB. |
| VMB-R12 | Sec. 5.9, 7.5, 19 | Expose stage/loss/temperature/warmup configuration. | `train.py`, `configs/fedcvs_rffi_vmb.yaml` | verified | help output + source tests | Includes stage, Stage1 local steps, Stage2 batches, prototype temperatures, loss weights, and adversarial warmup. |
| VMB-R13 | Sec. 6.2, 6.4 | Add transmitter-balanced VMB batch sampling. | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py` | verified | helper + smoke tests | Falls back to normal loader when metadata is unavailable. |
| VMB-R14 | Sec. 7.1, 7.2, 7.6 | Implement synchronized one/few-batch gradient collection from the same global model. | `federated/fed_trainer.py`, `federated/fedcvs_vmb.py` | verified | Stage2 smoke test | Stage2 uses server-side gradient SGD; insufficient batch counts are rescaled to avoid under-weighted gradients. |
| VMB-R15 | Sec. 7.3 | Add domain-balanced gradient aggregation. | `federated/fedcvs_vmb.py`, `federated/fed_aggregate.py` | verified | aggregation tests | Equal receiver/domain contribution is supported for gradients and state averaging. |
| VMB-R16 | Sec. 7.4 | Add domain-balanced client sampling. | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py` | verified | helper tests | Receiver/client IDs are default domains; prototype-cluster sampling is left as a future refinement. |
| VMB-R17 | Sec. 7.5, 9.1, 11.4 | Support Stage 1 local pretraining and Stage 2 freezing or slow update of RX backbone. | `federated/fed_trainer.py`, `train.py` | verified | Stage1 + Stage2 smoke tests | Stage1 uses local optimizer steps and state averaging; Stage2 freezes RX-domain prefixes by default. |
| VMB-R18 | Sec. 8.1, 8.2, 12.5 | Log prototype counts, gradient norm/cosine, drift, domain balance, communication diagnostics. | `federated/fed_trainer.py` | verified | smoke tests + metrics.csv checks | Logs include prototype counts, grad norm/cosine, client drift norm, per-domain loss variance, payload bytes, and domain weights. |
| VMB-R19 | Sec. 10 | Keep FedAvg/FedProx compatibility and avoid renaming existing semantics. | `federated/fed_trainer.py`, tests | verified | legacy smoke tests | Default paths are outside `self.vmb_enabled`; existing FedProx smoke still passes. |
| VMB-R20 | Sec. 11.5 | Record normalization recommendation in config/template without forcing old model defaults. | `configs/fedcvs_rffi_vmb.yaml` | verified | config test | Prototype losses normalize features internally; model defaults are unchanged. |
| VMB-R21 | Sec. 12.3 | Provide probe helpers for `Acc(y|z_t)`, `Acc(d|z_t)`, `Acc(d|z_r)`, `Acc(y|z_r)`. | `evaluation/fedcvs_vmb_probe.py`, tests | verified | `tests/test_fedcvs_vmb.py` | Script expects a saved feature payload; automatic feature export is not forced into training. |
| VMB-R22 | Sec. 12.4, 12.5 | Provide prototype and gradient analysis helpers. | `evaluation/fedcvs_vmb_analysis.py`, tests | verified | `tests/test_fedcvs_vmb.py` | Covers prototype drift/separation, gradient cosine, and VMB log summaries. |
| VMB-R23 | Sec. 13 | Document ablation route A0-A14 in runnable config comments/docs. | `configs/fedcvs_rffi_vmb.yaml` | verified | config test | Config now mirrors the design-report A0-A14 table and adds a shorter pragmatic sequence. |
| VMB-R24 | Sec. 14 | Preserve risk controls: no raw IQ upload, prototype clipping hooks, logging of assumptions. | `federated/fedcvs_vmb.py`, `configs/fedcvs_rffi_vmb.yaml` | verified | helper tests + config review | Clipping and normalization are implemented; secure aggregation/noise are explicitly optional config risks, not claimed implemented. |
| VMB-R25 | Sec. 15, 16 | Support MVP-1 VMB first, then pretrain/full route through configuration. | `train.py`, `federated/fed_trainer.py`, `configs/fedcvs_rffi_vmb.yaml` | verified | Stage1 + Stage2 smoke tests | `stage2` still gives explicit MVP VMB; the A5 two-stage template now uses `auto + pretrain_rounds=20` to run Stage1 then Stage2 inside 200 rounds. |
| VMB-R26 | Sec. 18.2, 18.3 | Add modules using existing package layout rather than intrusive rewrites. | `federated/fedcvs_vmb.py`, `evaluation/*`, `configs/*` | verified | py_compile | New modules are isolated and removable. |
| VMB-R27 | Sec. 19 | Add recommended YAML config template. | `configs/fedcvs_rffi_vmb.yaml` | verified | config test | Includes hard constraints, command skeleton, losses, diagnostics, and ablations. |
| VMB-R28 | Project instructions | Use `receiver`, WiSig ratio 0.1, epochs/fl_rounds 200 defaults for FL launch paths. | `train.py`, `configs/fedcvs_rffi_vmb.yaml` | verified | source tests | Existing parser defaults and VMB config preserve `0.1/200/200/receiver`. |
| VMB-R29 | Version management | Snapshot changed code/config/script files because this tree is not git. | `code/snapshots/20260527_185647_fedcvs_vmb/` | verified | snapshot manifest | `SYNC_MANIFEST.txt` records local-only mappings; no N607 sync was performed. |
| VMB-R30 | User request | Run two sub-agent reviews: implementation correctness and cross-domain rationale. | sub-agent results | verified | Parfit + Noether completed | Review findings were addressed where code/config changes were appropriate; rationale remains evidence-bounded. |
