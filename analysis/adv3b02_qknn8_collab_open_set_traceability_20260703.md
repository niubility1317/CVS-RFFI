# ADV3B02 + qknn8 卫星群协同开集识别需求追踪

日期：2026-07-03  
工作区：`E:\type10-7`  
状态：Phase A 离线 evidence evaluator 已本地实现并验证；未启动 N607 实验。

## 当前证据来源

| 来源 | 结论 |
|---|---|
| `项目.md` | CVS 主协议要求 `R_t` 与 `R_s` 不相交；`Y_old`、`Y_new`、`Y_unknown` 互斥；未知类 query 只能评估，不能调阈值。 |
| `code/evaluation/collaborative_inference_eval.py` | 已支持闭集协同推理，`--collab_counts all` 可覆盖 1 到接收机数量；当前只输出闭集 rescue/harm/accuracy。 |
| `code/federated/reliability_fusion.py` | 已有 `soft/adaptive/conservative` 概率融合，但没有 old/new/unknown 三路开集语义和资源遥测。 |
| `automation_reports/CV-SincNet/phase1_adv3b02_open_set_reject_20260702/report.md` | Phase1 低 FAR 门控可做到 `unknown_FAR<=5%`，但旧类覆盖/有效准确率严重下降。 |
| `automation_reports/CV-SincNet/phase1_adv3b02_satonly_open_set_reject_20260702/report.md` | satellite-only 单观测拒识仍存在 FAR 与旧类保留冲突。 |
| `automation_reports/CV-SincNet/phase2_adv3b02_qknn_support_select_k10k15_no_unknown_20260703_1312/report.md` | qknn8/量化 KNN 可提升少样本新类，但该系列明确 `未知类不导出、不评估`。 |
| `code/scripts/phase2_compressed_proto_knn_sweep.py` | 当前 qknn/压缩原型评估是闭集 old/new 头，不处理 unknown query。 |
| `卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md` | 按标题和关键词未在当前工作区找到；资源约束原文缺失，需后续补证。 |

## 需求追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| CIOSR-001 | 用户目标 | 底座模型必须是 `ADV3B02_CORE90_SOFT_E200`。 | 设计报告；后续 launcher/evaluator | pending | 报告引用；后续需 checkpoint identity/hash 校验 | 已有 Phase1 报告记录 checkpoint 路径。 |
| CIOSR-002 | 用户目标 | 在轨少样本域适应和新类学习方法使用 `qknn8`。 | `code/scripts/phase2_compressed_proto_knn_sweep.py`；后续协同 evaluator | pending | qknn8 当前只覆盖 old/new no-unknown | 需要把 qknn8 变成每节点 evidence head。 |
| CIOSR-003 | 用户目标 | 结合卫星群协同推理解决叠加星地信道下未知类拒识。 | `code/evaluation/collaborative_open_set_qknn_eval.py` | implemented | `test_collaborative_open_set_qknn_eval.py` mock evidence 通过；真实 ADV3B02/qknn8 evidence 待接入 | 已实现协同 unknown risk 聚合接口；尚未证明真实指标。 |
| CIOSR-004 | 用户目标 | 旧类总体性能目标 99%，每类不低于 95%。 | 后续实验矩阵/metrics | pending | 当前 qknn8 最大 query 旧类约 85%；单节点未达 | 目标远高于当前证据，需多节点和门控验证。 |
| CIOSR-005 | 用户目标 | 新类总体性能目标 97%，每类不低于 93%。 | 后续实验矩阵/metrics | pending | qknn8 no-unknown 最大 query K15 新类约 86.15% | 需要协同新类原型增强。 |
| CIOSR-006 | 用户目标 | 未知类拒识目标 99%。 | 后续开集 evaluator | pending | Phase1 low-FAR 能到 99% reject，但旧类覆盖差；sat-only 未解决双目标 | 不能单独拿低 FAR 作为完成。 |
| CIOSR-007 | 用户目标 | 报告参与推理数量。 | `code/evaluation/collaborative_open_set_qknn_eval.py` | verified | `pytest code\tests\test_collaborative_open_set_qknn_eval.py`：3 passed | open-set evidence evaluator 支持 `collab_counts=all` 输出 `1..receiver_count`。 |
| CIOSR-008 | 用户目标 | 报告时延。 | `code/evaluation/collaborative_open_set_qknn_eval.py` | verified | `pytest code\tests\test_collaborative_open_set_qknn_eval.py`：3 passed | 当前为 evidence 行输入的 `latency_ms_p50/p95`；真实端到端 latency 待 N607/feature 管线实测。 |
| CIOSR-009 | 用户目标 | 报告资源约束。 | `code/evaluation/collaborative_open_set_qknn_eval.py`；设计报告 | implemented | evaluator 输出 `bytes_per_event/total_bytes`；资源说明原文未找到 | 已实现通信字节遥测；state size/VRAM 仍待真实管线补充。 |
| CIOSR-010 | 项目协议 | `R_t` 与 `R_s` 不相交，old/new/unknown TX 互斥。 | evaluator/launcher | pending | `项目.md` 已定义 | 后续 CLI 必须显式检查。 |
| CIOSR-011 | 项目协议 | unknown query 不参与阈值拟合。 | `code/evaluation/collaborative_open_set_qknn_eval.py` | verified | `test_rejects_threshold_fitting_from_unknown_query_rows` 通过 | evaluator 对 unknown calibration/threshold_fit 行 fail closed。 |
| CIOSR-012 | 现有代码 | 复用闭集协同 `collab_counts all` 能力。 | `code/evaluation/collaborative_open_set_qknn_eval.py`; `code/evaluation/collaborative_inference_eval.py` | verified | 新 open-set tests 3 passed；旧闭集协同 tests 5 passed | 新增 sibling，不破坏旧闭集入口。 |
| CIOSR-013 | 现有报告 | 不得把 `no_unknown` qknn8 结果当开集完成。 | 设计报告；最终报告 | verified | qknn8 报告明确未知类不导出、不评估 | 本追踪表已标注。 |
| CIOSR-014 | 资源约束 | 每事件通信负载需要上限。 | `code/evaluation/collaborative_open_set_qknn_eval.py`; 设计报告 | implemented | evaluator 输出 `bytes_per_event` 和 `total_bytes` | 上限策略仍待资源约束原文核对。 |
| CIOSR-015 | 实验协议 | 必须同时报告 full denominator、coverage、accepted-only、confusion。 | `code/evaluation/collaborative_open_set_qknn_eval.py` | implemented | tests 覆盖 old/new/unknown、coverage、FAR、per-class floor | confusion 明细矩阵尚未输出；当前为核心指标实现。 |

## 关键缺口

1. 当前新增 evaluator 接收 qknn-style evidence，但还没有直接从真实 ADV3B02/qknn8 feature npz 生成 evidence。
2. 当前 qknn8 最强证据是 `no_unknown`，不能证明未知类拒识。
3. 当前低 FAR 拒识证据以牺牲旧类覆盖为代价，不能满足旧类 99%/每类 95%。
4. 未找到用户指定的资源约束设计说明原文；本轮设计只能先给可执行资源预算，后续需与原文核对。
5. 当前目标指标明显高于已有单节点证据，需要多接收机协同、样本对齐和选择性 abstain/defer 才有可能接近。

## 2026-07-03 Phase A 实现记录

| 文件 | SHA256 | 作用 |
|---|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | `CF0008F28460719B6C1AB38350F664710A152BFCE5F59DF3E3091289C4DA09F5` | 离线 evidence 级协同开集 qknn evaluator；支持 `collab_counts=all`、old/new/unknown、coverage、FAR、per-class floor、bytes、latency。 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | `943D8975B77A0EA2500C4D53E873532AC5F7310AB41FECBED40A38DE5351D3D6` | TDD 回归测试，覆盖 `1..N` 输出、资源遥测、unknown query 禁止阈值拟合、真实 TX label 不依赖 `new` 前缀。 |

验证命令：

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:CONDA_REPORT_ERRORS='false'
conda run -n ssr-gpu python -m pytest -q code\tests\test_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m py_compile code\evaluation\collaborative_open_set_qknn_eval.py code\tests\test_collaborative_open_set_qknn_eval.py
conda run -n ssr-gpu python -m pytest -q code\tests\test_collaborative_inference_eval.py
```

结果：

- 新 open-set evaluator tests：`3 passed`。
- `py_compile`：通过。
- 旧闭集 collaborative evaluator tests：`5 passed`。
- 备注：并行 `conda run` 会触发 Windows 临时文件锁，已改为串行执行；pytest cache 目录仍有权限 warning，不影响测试结论。

## 下一步验证入口

1. 新增 `collaborative_open_set_qknn_eval.py`，读取已导出的 per-receiver feature/evidence。
2. 复用 `collab_counts=all`，输出 `K=1..N` 的 old/new/unknown 指标和资源指标。
3. 第一阶段只做离线 feature 级协同，不训练主干，不改 qknn8 记忆格式。
4. 第二阶段再接 N607 全量测试：5 个 target receiver、2 组 unknown TX、K-shot `{5,10,15}`，再扩展协同节点数。
