# Margin query 终态评分与 BranchLocalRidge 配对方案

现有 CLI 直接支持本任务：先由 Margin 独立 scorer 完成全部新预测的 truth-last 评分，再给 reporter 显式传入 rx3、rx1 各一套旧 `SCORED/results` 与 `method/spec/startup/complete`。Reporter 校验旧方法完整矩阵后，只为当前 2026092701、2026092702 的同物理 parent 生成 C 阶段对照。旧四 seed 整体均值不参与减算，旧方法缺失的 A、B 及遗忘指标保持 N/A。

本文仅读 scorer/reporter 源码、相应合成 API 示例及三个配置的路径/seed 元数据；没有读取或核实任何真实 prediction、truth、评分、cache、registry 或结果报告。下面是待 root 执行的调用方案，不是已执行评分或已确认旧基准完成的记录。

## 1. 配对来源与已知路径

新 spec 为 `configs/d92_margin_joint_repeat_20261001.json`，新 run 为 `20261001-phase2-d92-margin-joint-repeat-m2-r01`。新 remote run root：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/20261001-phase2-d92-margin-joint-repeat-m2-r01
```

旧方法明确为 `D92-BranchLocalRidge-v1`，两套配置及声明的根目录如下。目录是否存在、是否完成均未由本文读取。

| Cohort | 旧配置 | 声明的 remote run root | 声明的 release |
| --- | --- | --- | --- |
| rx3 | `configs/d92_branch_local_ridge_repeat_rx3_20260929.json` | `/home/szu2070436088/2510044040/CV-SincNet/runs/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01` | `/home/szu2070436088/2510044040/CV-SincNet/releases/d92_branch_local_ridge_repeat_rx3_20260929_r01` |
| rx1 | `configs/d92_branch_local_ridge_repeat_rx1_20260929.json` | `/home/szu2070436088/2510044040/CV-SincNet/runs/20260929-phase2-d92-branch-local-ridge-repeat-rx1-m4-r01` | `/home/szu2070436088/2510044040/CV-SincNet/releases/d92_branch_local_ridge_repeat_rx1_20260929_r01` |

两旧配置均声明四个 model seed：2026092701、2026092702、2026092703、2026092704；新配置只声明前两个。新旧对应模型的 checkpoint SHA 为：

| Model seed | SHA256 |
| --- | --- |
| 2026092701 | `f7ea5064d56711c3173b16636a52af27018ba11a459f0205de950c134362e53b` |
| 2026092702 | `72f26413e495dcf82317898f8729caa3d35499bf0b41e9671ffdceab2c90924b` |

同一 cohort 的旧 `rows[].reuse_row_root` 与新 `rows[].row_root` 逐模型相同：rx3 使用 `20260928-phase2-d92-scv-confirmation-manytx-m4-r02` 下的对应模型目录，rx1 使用 `20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01` 下的对应模型目录。跨 cohort 不要求这两个目录相同。

Capsule ID 逐 cohort 相同：rx3 为 `residual-noeq-76121e6f34363fa612ec25fb`，rx1 为 `residual-noeq-d0a99fede324159c5a4750fd`。旧、新矩阵均为 K=1/5/10/20、新增 0/2/5/10/20、三场景和 support seed 2026092711 至 2026092715。rx3 声明三个 receiver，rx1 声明一个 receiver；新配置覆盖两 cohort×两 model 的四行。

## 2. 全部新预测固定后，root 要核实什么

以下是现有完成状态和身份接口所需的信息，不是新增审批、签名、receipt 或数据重验。

### 新 run

独立 scorer 会读取新 run 的 `startup.json`、`complete.json`，要求全部四行 `COMPLETE`、全局状态 `MARGIN_QUERY_BENCHMARK_PREDICTIONS_COMPLETE`、`all_predictions_fixed=true`，并核对 actual runtime commit、完整 resolved spec 和每行 marker。它先校验并再次读取全部 A/B/C 预测及对应来源/状态引用，才第一次连接 spec 中声明的 truth；不拟合、不执行模型、不挑选 parent。API 与顺序见 `score_d92_margin_joint_benchmark.py:328` 至 `:401`。

### 两套旧 baseline

Root 需显式给出以下四个文件和方法名，每个 cohort 一套；不能只给摘要均值或旧配置中的 PLANNED/RUNNING 文字。

| CLI 输入 | 文件与字段要求 |
| --- | --- |
| `--baseline-score` | 实际已完成 `SCORED/results` JSON。顶层 `status='SCORED'`、`selection_feedback_forbidden=true`，且含完整 `results`。两个旧配置没有声明这个 JSON 的实际文件名，因此需 root 明确实际路径，不能猜成 `score.json` 或 `scored.json` |
| `--baseline-spec` | 对应旧 run 的原完整 spec；必须精确等于其 startup 的 `spec`。配置相对路径已知，使用旧 release 中相应配置或 root 核实的完全相同副本 |
| `--baseline-startup` | 对应旧 run 的 actual startup JSON。需含 `spec` 与实际 `commit`；旧 run 根下 `startup.json` 是待 root 核实的候选路径 |
| `--baseline-complete` | 对应旧 run 的终态 JSON。需 `status='SCORED'`、`selection_feedback_forbidden=true`、`commit==startup.commit`、`model_rows==len(spec.rows)`、`records==len(scored.results)`；旧 run 根下 `complete.json` 是待 root 核实的候选路径 |
| `--baseline-method` | 两 cohort 共用 `D92-BranchLocalRidge-v1`，应等于旧 spec 的 `confirmation.candidate_method` |

实际 startup/complete 的 commit 一致用于证明同次完成；旧 `spec.code.commit` 是准备 parent，不要求它等于实际 runtime commit。Reporter 不自动搜索历史路径，也不从目录名称推断完成。Legacy 分支不重新读取旧 prediction streams，因此旧基准的正式完成和对应预测根仍由 root 按既有证据独立核实。旧配置声明 `confirmation.candidate_folder='branch_local_ridge'`，实际每行预测目录须以原完成记录为准，本文不推断其子目录结构。

## 3. 如何只比较当前两 seed

`report_d92_margin_joint_benchmark.py:115` 至 `:180` 先检查每套旧 score 中该方法的完整矩阵：旧 spec 的全部 model row×receiver×scenario×K×new_count×support_seed。不要人为截取两 seed 的旧 `results`，同时继续传四 seed spec/complete；这会与完整性契约不一致。

通过完整性检查后，Reporter 遍历新 score 的 `parents`，按下列键查找旧 parent：

```text
(capsule_id, checkpoint_sha256,
 model_seed, receiver, scenario, k, new_count, support_seed)
```

因此 2026092703/04 留在旧完整 score 中用于证明原运行覆盖，实际比较只取新 score 的 2026092701/02。每个配对还必须满足：

- 旧 record 的 `row_id` 绑定旧 spec 中同 model row；旧 `reuse_row_root` 等于新 row 的 `row_root`。
- `split_id` 相同，注册类集合相同，旧 record 的前六类集合等于当前 old classes，`class_count=6+new_count`。
- `query_count` 相同且为正。旧 record 若含 `query_ids`，其集合必须等于新 parent 的 old/new query ID 并集；若不含，使用同一 `VALIDATED_ONCE` capsule+split_id 的既有身份绑定，不补造 ID、不重验数据。
- 旧 record 若显式含 capsule/checkpoint 字段，其值须与旧 spec 绑定一致；未含时按该旧 spec 绑定，不从其他 run 猜补。

指标值只在这些身份检查全部完成后读取。逐 parent 对照 `old_accuracy→C_old_accuracy`、`new_accuracy→C_new_accuracy`、`harmonic_mean→C_h`，差值为“当前 Margin−旧 BranchLocalRidge”。旧 H 必须与该 parent 的 C旧/C新一致；不把整体旧均值与整体新均值直接相减。完整当前范围声明为 2400 个 parent；实际完成覆盖由 scorer 判定，本文未读取产物确认。

## 4. 具体调用方案

下面只供 root 在终态固定后使用。以新 release 为 CWD，使用 spec 声明的项目解释器；先写独占的新 score，再生成独占报告。示例中的 `ROOT_VERIFIED_*` 是必须由 root 填入的实际旧 scored JSON 路径，不是文件名建议。

```bash
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  tools/score_d92_margin_joint_benchmark.py \
  --spec configs/d92_margin_joint_repeat_20261001.json \
  --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261001-phase2-d92-margin-joint-repeat-m2-r01/score_margin_query.json

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  tools/report_d92_margin_joint_benchmark.py \
  --score /home/szu2070436088/2510044040/CV-SincNet/runs/20261001-phase2-d92-margin-joint-repeat-m2-r01/score_margin_query.json \
  --report /home/szu2070436088/2510044040/CV-SincNet/runs/20261001-phase2-d92-margin-joint-repeat-m2-r01/margin_query_report.md \
  --interpretation /home/szu2070436088/2510044040/CV-SincNet/runs/20261001-phase2-d92-margin-joint-repeat-m2-r01/margin_query_report.json \
  --baseline-method D92-BranchLocalRidge-v1 \
  --baseline-score ROOT_VERIFIED_RX3_SCORED_JSON \
  --baseline-score ROOT_VERIFIED_RX1_SCORED_JSON \
  --baseline-spec /home/szu2070436088/2510044040/CV-SincNet/releases/d92_branch_local_ridge_repeat_rx3_20260929_r01/configs/d92_branch_local_ridge_repeat_rx3_20260929.json \
  --baseline-spec /home/szu2070436088/2510044040/CV-SincNet/releases/d92_branch_local_ridge_repeat_rx1_20260929_r01/configs/d92_branch_local_ridge_repeat_rx1_20260929.json \
  --baseline-startup /home/szu2070436088/2510044040/CV-SincNet/runs/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/startup.json \
  --baseline-startup /home/szu2070436088/2510044040/CV-SincNet/runs/20260929-phase2-d92-branch-local-ridge-repeat-rx1-m4-r01/startup.json \
  --baseline-complete /home/szu2070436088/2510044040/CV-SincNet/runs/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/complete.json \
  --baseline-complete /home/szu2070436088/2510044040/CV-SincNet/runs/20260929-phase2-d92-branch-local-ridge-repeat-rx1-m4-r01/complete.json
```

四类 repeatable baseline 参数分别按出现顺序 zip，因此以上每个列表都固定为 rx3、rx1 顺序，不能交叉。Reporter CLI 已直接支持 `action='append'`，见 `:238` 至 `:260`；不需要修改入口或加入 seed 过滤参数。三个新输出均不得已经存在，调用前由 root 核实独占输出。

`--run-root` 是 scorer 的可选参数，只重映射本次新 run 的预测/控制面文件；它不重映射 spec 中 capsule/truth，也不重写实际 source identity。若在完整原远端路径下评分，省略即可。不能为了本地副本可读而改动已固定 spec 的 source/truth 路径。

对应 Python API 为：

```python
score_benchmark(spec=current_spec, output=new_score_path)

report_benchmark(
    score=current_complete_score,
    baseline_score=[rx3_scored, rx1_scored],
    baseline_metadata=[
        {"method": "D92-BranchLocalRidge-v1", "spec": rx3_spec,
         "startup": rx3_startup, "complete": rx3_complete},
        {"method": "D92-BranchLocalRidge-v1", "spec": rx1_spec,
         "startup": rx1_startup, "complete": rx1_complete},
    ],
)
```

这是数据对象 API 示例，不要求 root 编写新的加载器。CLI 会读取这些显式文件并组装同一结构；合成示例见 `tests/test_report_d92_margin_joint_benchmark.py` 的 `legacy_fixture` 与 `test_explicit_legacy_baseline_pairs_C_only_and_missing_stage_NA`。

## 5. 报告范围与当前缺失信息

当前 Margin 保留完整 A旧/B旧/C旧/C新/H、B−A、B−C旧和绝对新旧差距；旧 BranchLocalRidge baseline 只提供 C旧/C新/H。Reporter 的 legacy `baseline_metrics` 和 `differences` 对其余指标写 null，正文标记 N/A，并给出 `LEGACY_REGISTRATION_ONLY_SCORE_HAS_NO_MATCHED_A_B_OR_FORGETTING`。不借用当前 A 或 B 补造旧方法阶段，不把拟合 R0 当 A，也不从 C旧反推遗忘。新增 0 的旧、新双方 C新/H 均为 N/A，相应差值也为 N/A。

准确率与 H 的正文单位为百分比；C 对照差值为百分点，JSON 内保留 0 至 1 的原值和差值。当前报告保留 K×新增类数及模型、接收机/场景、support seed 分层；legacy C 对照逐 parent 保留，不利用其四 seed 整体统计代替本次匹配集合。重复冻结数据的性质继续明确标注，不称为新的独立确认；结果不回流到参数、checkpoint、seed 或重跑选择。

当前仍缺、需 root 在实际终态后核实的信息为：两套旧 scored JSON 的确切路径；旧 startup/complete 的实际存在与终态/运行 commit；对应旧完整 spec 与真实预测根；本次四行完整固定及独立读回证据；三个新评分/报告输出的独占状态。本文不声称这些已确认，也不额外启动评分或复算旧方法。
