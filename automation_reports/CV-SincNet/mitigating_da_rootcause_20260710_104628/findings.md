# Findings: Mitigating receiver impact DA

## Confirmed starting point

- Prior best strict final reproduction results were 70.97% for `14-7->3-19`, 83.47% for `1-1->1-19`, and 86.61% for `7-7->8-8`.
- The corresponding paper values previously transcribed were 92.42%, 95.44%, and 99.74%.
- `14-7->3-19` remained dominated by low-precision pseudo-labels for classes `20-15` and `20-19`.
- Target-label-based checkpoint selection is diagnostic/oracle and cannot support a strict UDA reproduction claim.

## Evidence to collect

- Exact paper text, equations, tables, and stated/unstated implementation details.
- Complete local and remote artifact inventory and full learning curves.
- Code paths for data construction, model, MINE, E/C steps, pseudo-labeling, class weighting, normalization, checkpointing, and evaluation.

## Confirmed paper-to-code mismatches

- Paper Section II requires subtracting the signal mean before power normalization. `code/dataset_wisig.py::_rms_normalize_iq` currently divides by RMS without centering.
- Paper Algorithm 1 resets `sigma_0(k)`, `sigma'_0(k)`, and `n^t_0` at the start of each outer iteration. The strict runner default is `pseudo_state_scope=global`; epoch-scoped state is only enabled by the official-compat switch.
- Paper Algorithm 1 pairs source and target for `min(Ns/b,Nt/b)` batches. The strict runner default is `cycle_target`; `zip_min` is only enabled by the official-compat switch.
- Paper Section IV-A starts from an initial model `h0`, for example a model learned from Rx-1. Most prior best-final runs adapted from random initialization; the paper does not state the required source-pretraining duration.
- The public trainer performs one source/target model forward before MINE and reuses those outputs/features. The reproduction E/C step separately calls `_estimate_outputs` and then `model(...)`, so BatchNorm running statistics are updated twice per domain in the E/C step.
- The public trainer uses `torch.nn.CrossEntropyLoss(weight=...)`, whose weighted mean is normalized by the selected sample-weight sum. The reproduction uses `mean(weight * CE)`. The latter matches the literal paper Eq. 10; the former matches the released trainer and may be necessary to reproduce its reported numbers.
- The public trainer resets MINE's moving average to its default inside each of the `m` estimate updates, then passes only the final returned value to the E/C KL call. The reproduction carries the moving average across all `m` updates.

## N607 data audit

- Direct preflight passed at 2026-07-10 10:53 CST; all eight RTX 3090 GPUs were idle and no related training process was active.
- For receivers `14-7`, `3-19`, `1-1`, `1-19`, `7-7`, and `8-8`, equalized ManySig contains exactly 4000 samples per TX and every sample is shaped `(256,2)`.
- Mean subtraction removes only 0.052%-0.061% of average signal power for these receivers. Centering is a real paper mismatch but cannot plausibly explain a 20-50pp gap by itself.

## Public-code boundary

- The authors' public repository exposes only `mine_pseudo_classweight_trainer.py` and points to `YannLeo/Pytorch-Template`.
- It does not publish the experiment TOML, WiSig dataset wrapper/split, model wrapper returning `(output, feature)`, MINE class, or exact ResNet18 constructor arguments. Exact author-code reproduction therefore remains underdetermined even after matching the exposed trainer.

## First repaired matrix

- All eight runs completed without runtime-error markers. The five-row strict-equation mean is 62.37%, versus the paper mean 96.14%, a gap of -33.77pp.
- Correcting `d01->d23` materially improved traceability and produced 85.73%, only -7.61pp from the paper. The receiver-pair tasks remain much worse, so the residual problem is not a generic optimizer failure.
- Source pretraining converged to 98.54%-99.92% source-batch accuracy. Initial target accuracy nevertheless ranged from 22.19% to 84.11%, proving that the source classifier is converged but its representation is receiver dependent.
- On every strict task, epoch-1 CPL selects 97.78%-99.60% of target samples. On the hard receiver pairs pseudo-label precision is only 46.16%-59.14%, so the nominal `tau=0.7` curriculum threshold behaves as an almost-unfiltered self-training loop.
- Per-class results show class-permutation collapse rather than uniform degradation. For example, `14-7->3-19` ends at `[0.28%,12.38%,39.38%,62.88%,60.73%,5.55%]`; `1-1->8-8` nearly loses classes 2 and 4 while retaining four classes near 100%.
- Confusion-matrix argmax mappings make the mechanism explicit: `1-1->8-8` swaps classes 1 and 3 at 93.55%/99.68%; `7-7->8-8` forms a 0->3, 1->0, 3->1 cycle at 98.08%/91.75%/88.03%. Because every target class has 4000 samples, marginal class weighting cannot detect or repair a balanced semantic permutation.
- The exposed released-trainer semantics produce 28.29% and 46.07% on the two hard pairs, so weighted-CE/MINE/public-threshold details do not close the gap.

## Architecture and ablation localization

- An independent model audit confirmed that the public trainer cannot directly instantiate the linked public `ResNet1D`: the trainer expects `(output, feature)`, while the template returns only dense output, and the public `MINE` class/config are absent.
- Added a fail-closed `pytorch_template_resnet18_hypothesis_v1` diagnostic profile: 8 two-convolution residual blocks, SAME-padding stem/blocks, 64/128/256/512 stages, template shortcut semantics, ELU three-layer classifier, and LeakyReLU three-layer estimate network.
- Added exact Table III component switches. Any component ablation or inferred architecture is automatically `diagnostic_only`; neither can be presented as the paper's full Proposed result.
- Table III component scaling follows the exposed trainer: source CE is multiplied by `mu` only when class weighting is enabled; target CE is multiplied by `1-mu` only when CPL is enabled. The all-disabled diagnostic does not update target BatchNorm, but formal Source-only remains the independent `method=source_only` path because paired loaders would otherwise cap its batch count.

## Round-2 completed results

- 完整解析`remote_artifacts_round2`中的8组JSON/`.out`。每组JSON与对应`.out`反序列化后完全一致，均包含20个source-pretrain epoch和20个adaptation epoch，未发现Traceback、RuntimeError、OOM、NaN、Killed或参数错误marker。
- 8个结果的`claim_status`和`result_claim_status`均为`diagnostic_only`，row状态均为`completed_diagnostic_only`。template架构是假设性实现，3个组件消融不是论文完整Proposed，均不得进入正式复现或部署证据。
- template五任务均值为69.9439%，standard五任务均值为62.3742%，论文五任务均值为96.1440%。template相对standard提高7.5697pp，但仍低于论文26.2001pp，不能解除架构与数据划分欠定问题。
- Table III组件定位显示DA-only为68.5667%对论文76.36%（-7.7933pp）；DA+CW为40.9875%对77.02%（-36.0325pp）；CPL+CW为22.2750%对77.11%（-54.8350pp）。在当前实现与数据上，移除CPL后DA-only最接近论文消融值，CPL/CW路径会显著放大错误伪标签，而不是修复receiver-dependent类置换。

## Full multiseed validation design

- `launch_full_multiseed_validation_20260713.sh`定义2个seed（20260711、20260712）、2个profile（`standard_resnet18`、`pytorch_template_resnet18_hypothesis_v1`）、5个Table II任务，共20个Proposed-only运行。
- 三波分配为wave1=8、wave2=8、wave3=4；每波必须独立显式调用，不能跨wave并发。launcher固定并验证20行`expected_matrix.tsv`，拒绝重复组合、重复run ID和8/8/4计数漂移。
- 初版监督审查因缺少防覆盖、跨wave/GPU上限守卫和20项完整性manifest给出NO-GO。修正版增加no-clobber目录/文件检查、`flock`启动锁、其他wave活跃PID/命令检查及GPU计算进程数小于2的门控，独立复审结论为GO。
- 当前共享计划假设每张GPU已有1个无关`phase1_dgleo`训练，再增加本矩阵1个进程后达到每GPU最多2个训练进程的允许上限。共享GPU可能增加时延和运行时方差，因此结果比较必须保留wave、GPU、并发占用和时间信息，不能把运行时差异解释为算法收益。
