+# Phase3 CARE-PoE G0合成技术闭环报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase3_care_poe_g0_synthetic_20260808_v1`
- 状态：`LOCAL_VERIFIED / WAITING_INDEPENDENT_P0_P1`
- 证据等级：`TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT`
- 时间：2026-08-08
- 操作者：Codex主代理；N607唯一runner待交接
- Git commit：`334dd23acf2ec7ff7d17ba5e2c54b7cf588836fc`

## 1.目标与假设

目标是验证真值无关本地证据封存、CARE-PoE相关性融合、`N_sat=1..5`、同输入A/B/C/D、独立评分和`anonymous→外部授权→fresh-K`状态机能完整运行。合成fixture不验证任何性能阈值。

假设：同组重复证据不增加独立组数；一个emission event无论有多少reception都只产生一个shot；`N_sat=1`时`C=A,D=B`；历史unknown事件不能转成support。

正式对照目标为未来合法物理事件数据上的A/B/C/D矩阵。现有R8的event构造依赖`role/true_label`，本run不读取R8，也不产生same-emission或真实多星主张。

## 2.本地版本与验证

| 文件 | 目的 | SHA256 |
|---|---|---|
| `code/cvsrffi/phase3_care_poe.py` | seal、CARE-PoE、矩阵、scorer、生命周期 | `a5d3cb5f0a271149605ab60e7c747bdb6b6897980bd54cb4cfe8c60d471446e7` |
| `code/scripts/phase3_care_poe_fixture.py` | 确定性合成fixture | `88a2863442c0d7281f82c28d5d0c39f8cc338d4998a359fe3e3aceb3ea3e1ab3` |
| `code/scripts/phase3_care_poe_predict.py` | 不读真值的A/B/C/D预测入口 | `ff54f02e66d1db3f4996deec24cd3638d58b754f57e44196d698fb18587fcf98` |
| `code/scripts/phase3_care_poe_score.py` | 独立真值评分入口 | `137bc6d60937ff650824b625a2aa4f80c75c52db1cf044022654fc658fc15a21` |
| `code/scripts/phase3_care_poe_lifecycle.py` | anonymous、授权和fresh-K入口 | `333912b935e8c4a0cfbbee4386da8138e096cd5710fab269380d2b5fc9b19149` |
| `code/tests/test_phase3_care_poe.py` | 12项聚焦协议/因果负测 | `81cc66cb9c0b29e9a700279766c3d513e264f6c827703d1fe4e0359086301a78` |

本地环境：`ssr-gpu`。验证结果：`python -m pytest code/tests/test_phase3_care_poe.py -q`为`12 passed`；五个入口均通过`py_compile`；一次确定性本地CLI闭环生成60条prediction、独立metrics和fresh-K receipt。测试曾发现验证端二次概率归一化导致合法seal误拒，已改为先验证收到的canonical payload hash，再进行数值语义检查。

## 3.冻结矩阵

| Arm | 本地证据bundle | 部署读取 | node budget |
|---|---|---|---|
| A | base | roster首节点 | 1..5 |
| B | new | roster首节点 | 1..5 |
| C | base | CARE-PoE | 1..5 |
| D | new | CARE-PoE | 1..5 |

冻结roster：`SAT-01..SAT-05`。合成event为2个registered、1个unknown，每个bundle每event有5个reception。相关组为3组，其中两组分别包含相关节点。每个event/arm/budget只输出1条prediction且`shot_count=1`。

## 4.N607预注册

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase3_care_poe_g0_synthetic_20260808_v1_334dd23a`
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

预期artifact：fixture下7个小文件、prediction下2个小文件、`metrics.json`、`lifecycle.json`、四入口stdout/exit receipt和manifest。不得下载数据、checkpoint或大型artifact。

## 5.成功、停止与证据读取

技术成功要求：四入口exit=0；prediction=60行；A/B/C/D覆盖`N_sat=1..5`；全部`shot_count=1`；`C=A,D=B`在N=1成立；prediction manifest写`truth_sidecar_opened=false`；lifecycle状态到`FRESH_K_READY_FOR_STAGE2_C`；错误指纹为0。

停止仅限：hash/commit不匹配、run/log目标已存在、协议字段泄漏、prediction不是60行、确定性异常或写入覆盖风险。不得根据合成metrics停止或晋级方法。

## 6.结果表（待N607回收后填写）

| candidate | category | receiver/TX split | K-shot | seed | old/seen-new/unknown | coverage/rollback/defer | mechanism | verdict |
|---|---|---|---:|---:|---|---|---|---|
| CARE-PoE-G0 A/B/C/D | synthetic technical | synthetic roster/event | 5 fresh events仅状态机 | deterministic | 不作性能解释 | 待回收 | seal+correlation-aware fusion+lifecycle | `NO_PERFORMANCE_RESULT` |

## 7.已知风险与下一步

最大科学风险不是代码，而是缺少标签可见前生成的物理事件/接收ID。N607 G0完成后只能把接口状态推进到`ARTIFACTS_COMPLETE`；正式性能仍等待合法事件绑定。收到合法数据后，直接发布冻结A/B/C/D×`N_sat=1..5`完整矩阵，不再增加额外审查层。

