# Phase3 CARE-PoE G0合成技术闭环报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase3_care_poe_g0_synthetic_20260808_v1`
- 状态：`ARTIFACTS_COMPLETE_NO_PERFORMANCE_RESULT`
- 证据等级：`TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT`
- 时间：2026-08-08
- 操作者：Codex主代理；N607唯一runner已完成
- 实现Git commit：`7c94aface3d9ef7b8f3c9db83da8c186df5774fa`

## 1.目标与假设

目标是验证真值无关本地证据封存、CARE-PoE相关性融合、`N_sat=1..5`、同输入A/B/C/D、独立评分和`anonymous→外部授权→fresh-K`状态机能完整运行。合成fixture不验证任何性能阈值。

假设：同组重复证据不增加独立组数；一个emission event无论有多少reception都只产生一个shot；`N_sat=1`时`C=A,D=B`；历史unknown事件不能转成support。

正式对照目标为未来合法物理事件数据上的A/B/C/D矩阵。现有R8的event构造依赖`role/true_label`，本run不读取R8，也不产生same-emission或真实多星主张。

## 2.本地版本与验证

| 文件 | 目的 | SHA256 |
|---|---|---|
| `code/cvsrffi/phase3_care_poe.py` | seal、CARE-PoE、矩阵、scorer、生命周期 | `5a81e95dc730bb66fa150c2bb7d643068f7be166e62866c57ba7e177a43e73c5` |
| `code/scripts/phase3_care_poe_fixture.py` | 确定性合成fixture | `7f1587f9213336d5270efbf7bd751d0891d1cfa1f2cc8d18881bbc77b53474a8` |
| `code/scripts/phase3_care_poe_predict.py` | 不读真值的A/B/C/D预测入口 | `ff54f02e66d1db3f4996deec24cd3638d58b754f57e44196d698fb18587fcf98` |
| `code/scripts/phase3_care_poe_score.py` | 独立真值评分入口 | `137bc6d60937ff650824b625a2aa4f80c75c52db1cf044022654fc658fc15a21` |
| `code/scripts/phase3_care_poe_lifecycle.py` | anonymous、授权和fresh-K入口 | `333912b935e8c4a0cfbbee4386da8138e096cd5710fab269380d2b5fc9b19149` |
| `code/tests/test_phase3_care_poe.py` | 14项聚焦协议/因果负测 | `0c3acb8439b5d9d3a196ba3144f9c3b5bb56bd8b9bcfbe7bd4599cc32e8d41ab` |

本地环境：`ssr-gpu`。验证结果：`python -m pytest code/tests/test_phase3_care_poe.py -q`为`14 passed`；五个入口均通过`py_compile`；一次确定性本地CLI闭环生成60条prediction、独立metrics和fresh-K receipt。测试曾发现验证端二次概率归一化导致合法seal误拒，已改为先验证收到的canonical payload hash，再进行数值语义检查。独立复审先发现跨bundle物理reception未逐项绑定、相关组混合可受新增相关记录影响和scorer重复行/非法role未拒绝；定点修复后独立复测为`P0=0,P1=0,14 passed,ALLOW_N607_SYNTHETIC_G0=YES`。

## 3.冻结矩阵

| Arm | 本地证据bundle | 部署读取 | node budget |
|---|---|---|---|
| A | base | roster首节点 | 1..5 |
| B | new | roster首节点 | 1..5 |
| C | base | CARE-PoE | 1..5 |
| D | new | CARE-PoE | 1..5 |

冻结roster：`SAT-01..SAT-05`。合成event为2个registered、1个unknown，每个bundle每event有5个reception。相关组为3组，其中两组分别包含相关节点。每个event/arm/budget只输出1条prediction且`shot_count=1`。

## 4.N607预注册

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase3_care_poe_g0_synthetic_20260808_v1_7c94afac`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase3_care_poe_g0_synthetic_20260808_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase3_care_poe_g0_synthetic_20260808_v1`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：CPU技术闭环，不占用GPU；不得影响任意现有GPU任务
- retry：不授权自动重试；入口失败返回主代理
- 同步：从固定commit的Git archive落地新release，不修改远端已有release/run/log

冻结命令顺序仅执行一次：

```text
python scripts/phase3_care_poe_fixture.py --output-dir <run>/fixture
python scripts/phase3_care_poe_predict.py --base-evidence <run>/fixture/base_evidence.jsonl --new-evidence <run>/fixture/new_evidence.jsonl --output-dir <run>/prediction
python scripts/phase3_care_poe_score.py --predictions <run>/prediction/predictions.jsonl --truth-sidecar <run>/fixture/truth_sidecar.jsonl --output <run>/metrics.json
python scripts/phase3_care_poe_lifecycle.py --predictions <run>/prediction/predictions.jsonl --credential-template <run>/fixture/credential_template.json --fresh-support <run>/fixture/fresh_support.jsonl --output <run>/lifecycle.json --k 5
```

预期artifact：fixture下6个小文件、prediction下2个小文件、`metrics.json`、`lifecycle.json`、四入口stdout/exit receipt和manifest。不得下载数据、checkpoint或大型artifact。

## 5.成功、停止与证据读取

技术成功要求：四入口exit=0；prediction=60行；A/B/C/D覆盖`N_sat=1..5`；全部`shot_count=1`；`C=A,D=B`在N=1成立；prediction manifest写`truth_sidecar_opened=false`；lifecycle状态到`FRESH_K_READY_FOR_STAGE2_C`；错误指纹为0。

停止仅限：hash/commit不匹配、run/log目标已存在、协议字段泄漏、prediction不是60行、确定性异常或写入覆盖风险。不得根据合成metrics停止或晋级方法。

## 6.N607技术结果

- 四个冻结入口各执行一次，`fixture/predict/score/lifecycle`均`exit=0`，无retry。
- base/new各15条本地证据，truth sidecar为3条，fresh support为5个独立事件。
- prediction为60行：A/B/C/D各15行，完整覆盖`N_sat=1..5×3 events`；全部`shot_count=1`。
- `N_sat=1`逐event满足`A=C`、`B=D`；prediction manifest确认`truth_sidecar_opened=false`。
- lifecycle到达`FRESH_K_READY_FOR_STAGE2_C`；历史unknown event未进入fresh support。
- 四份stdout无异常指纹；运行后无活跃评估进程，8卡均空闲，SSH/TCP22连接已清理。
- 本地16项小artifact与远端manifest逐项hash匹配；Git archive/worktree的CRLF双hash口径均已记录。

| candidate | category | receiver/TX split | K-shot | seed | old/seen-new/unknown | coverage/rollback/defer | mechanism | verdict |
|---|---|---|---:|---:|---|---|---|---|
| CARE-PoE-G0 A/B/C/D | synthetic technical | synthetic roster/event | 5 fresh events仅状态机 | deterministic | 不作性能解释 | 60/60 prediction；defer/rollback不作性能统计 | seal+correlation-aware fusion+lifecycle | `ARTIFACTS_COMPLETE_NO_PERFORMANCE_RESULT` |

## 7.已知风险与下一步

最大科学风险不是代码，而是缺少标签可见前生成的物理事件/接收ID。G0技术接口已推进到`ARTIFACTS_COMPLETE_NO_PERFORMANCE_RESULT`；正式性能仍等待合法事件绑定。收到合法数据后，直接发布冻结A/B/C/D×`N_sat=1..5`完整矩阵，不再增加额外审查层。
