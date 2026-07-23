# ADV3B02官方对比方法新类数量实验v7

- 实验ID：`adv3b02_official_newcount_scale_20260724_v7`
- 日期：2026-07-24
- 操作方：主代理；N607唯一运行所有者为`no_leo_n607_release`
- 当前状态：`LOCAL_VERIFIED / INDEPENDENT_REVIEW_APPROVED`
- 目标：减少新类数量并运行多个规模；包括CSIL论文新增20类和MoPC-HR官方代码新增3/5/10/25类。

## 假设与比较边界

CSIL和MoPC-HR均以官方论文和官方GitHub实现为准，ADV3B02不冻结，执行各自完整base与increment流程。
仅做CVS接入适配：特征维度改为ADV3B02的160维、类别容量改为当前注册类数、K-shot输入适配。
旧类使用ManyTx物理接收IQ，不叠加LEO；只有新类support/query叠加LEO弱星地信道。对比方法不受阶段二主方法的其余协议约束。

v6已成功生成25/25套mixed-role缓存。本轮只读复用：
`runs/adv3b02_official_newcount_scale_20260724_v6/target_cache_new25`。
不存在历史new20缓存，因此v7不声称历史parity；只生成25份
`same_cache_new20_integrity`收据，证明当前缓存的前20类覆盖、sample ID和信道后IQ摘要自洽。
CSIL论文new20性能在本轮真实重跑，不复用旧性能。

## 冻结矩阵

| 方法 | 新类数 | 来源 | cells | 场景行 |
|---|---|---|---:|---:|
| CSIL | 1、3、20 | 20为论文increment；1/3为缩减诊断 | 300 | 900 |
| MoPC-HR | 1、3、5、10、25 | 3/5/10/25为官方代码设置；1为缩减诊断 | 500 | 1500 |
| 合计 | — | 5接收机×5seed×4个K | 800 | 2400 |

接收机：`20-1,3-19,7-14,7-7,8-8`；seed：`713101..713105`；
K：`1,5,10,20`；场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

## 官方方法参数锁

| 方法 | base | increment | 核心参数 |
|---|---|---|---|
| CSIL | 20epoch、batch128 | 3epoch、batch20 | SGD；`0.01/(1+0.01*iteration)`；momentum0.9；L2=0.05；KD=0.2；EWC=1.0 |
| MoPC-HR | 20epoch | 20epoch、batch16 | SGD lr0.01、momentum0.9、wd=2e-4；16个伪特征；noise std=0.05；logits/2；α=0.97 |

MoPC-HR按官方代码执行`CE+protoAug+逐参数非平方L2 HR`，KD只计算不反传；
MPC使用raw dot+softmax，校正prototype只供后续protoAug，query使用全部已注册类classifier logits。

## 本地实现与验证

| 文件/制品 | 用途 |
|---|---|
| `verify_adv3b02_official_scale_cache_parity.py` | 新增独立same-cache integrity schema，禁止不同路径冒充同缓存校验 |
| `build_adv3b02_official_scale_cache_reuse_manifest.py` | 冻结25份v6只读缓存及25条完整性命令，不生成cache build命令 |
| `build/run_adv3b02_paper_full_ci_plan.py` | plan与runner显式锁定verification mode/schema |
| `cvs_official_csil_newcount_1_3_20_split_20260724.json` | CSIL将论文new20纳入本轮真实矩阵 |
| `cache_reuse_manifest.json` | 25个receiver/seed缓存与receipt路径 |

本地环境：`ssr-gpu`。验证：相关脚本`py_compile`通过；
验证命令为
`python -m pytest tests/test_adv3b02_paper_full_ci_plan.py tests/test_adv3b02_official_repo_ci.py tests/test_paper_reproduction_csil_class_incremental.py tests/test_mopc_hr_non_exemplar_cil_sei.py -q`。
首轮审查后新增2个fail-closed负向测试，重新执行为`59 passed`；`git diff --check`通过。
首轮独立审查为`P0=0,P1=2,P2=1 / REJECT`；两项P1均已修复并重新验证。
最终复审为`P0=0,P1=0,P2=0 / APPROVE`。
release实现提交：`96a32a97f794cc1cc146c673a21852b3aeceeb94`。

## N607预注册

- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_official_newcount_scale_20260724_v7`
- log root：`logs/adv3b02_official_newcount_scale_20260724_v7`
- integrity root：`runs/adv3b02_official_newcount_scale_20260724_v7/cache_integrity`
- base26：`runs/adv3b02_official_newcount_scale_20260724_v7/base26/official_repo_base_state.pt`
- base31：`runs/adv3b02_official_newcount_scale_20260724_v7/base31/official_repo_base_state.pt`
- CSIL输出：`runs/adv3b02_official_newcount_scale_20260724_v7/csil_newcount_1_3_20`
- MoPC-HR输出：`runs/adv3b02_official_newcount_scale_20260724_v7/mopc_newcount_1_3_5_10_25`

执行顺序：preflight/哈希→v6缓存25/25只读复核→25份integrity→base26/base31→
两方法真实checkpoint no-query smoke→CSIL 300 cells与MoPC-HR 500 cells完整矩阵→制品回收与同row分析。
GPU最多每卡两个训练进程；实际PID、GPU、命令、日志和输出在落地后补录。

出现P0协议/安全错误，或至少两个不同row在prediction前产生同一确定性异常指纹时，
停止该精确run-owned进程树并保留产物；绝不因准确率、H、BA或其他性能低而停止。

## 运行结果

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | forgetting | loss/adapter摘要 | coverage/rollback/defer | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| v7-complete | CSIL+MoPC-HR | 5个 | 多个 | 1/5/10/20 | 5个 | 3个 | 见下表 | 见下表 | 见下表 | 见下表 | 官方完整方法 | 800/800 | `ANALYZED` |

## N607执行结果（LEO参考组）

- 状态：`ARTIFACTS_COMPLETE / ANALYZED`。
- 缓存完整性：25/25个receiver/seed收据通过；旧类原始接收IQ保持不变，新类使用LEO弱星地信道。
- 官方base：CSIL base26与MoPC-HR base31均完成并通过严格校验；SHA256分别为`635becd7db2d8041a669cb0ef922429c42ba389846f2252c7cfe4e0f3510a07e`和`306c8dfc767bad93f78f24b675b1c20058629eda8f3153bdf98d03ab6ae26202`。
- 烟测：CSIL 2/2、MoPC-HR 2/2通过，失败0，异常指纹0。
- 完整矩阵：CSIL 300/300 cells、900/900场景行；MoPC-HR 500/500 cells、1500/1500场景行。16个shard均`PASS`退出，失败0，异常指纹0。
- 逐cell closure：800份cell receipt、800份prediction、800份formal score完成；预测SHA、评分SHA、三场景覆盖和receipt唯一性全部通过。
- 同row分析制品：`retrieved/formal_rows.csv`、`retrieved/cell_summary_same_row.csv`、`retrieved/summary_by_method_new_k.csv`和`retrieved/artifact_audit.json`。

跨K、receiver、seed和场景的同row均值如下：

| 方法 | 新类数 | old_acc_after | seen_new_acc | H_old_new | forgetting |
|---|---:|---:|---:|---:|---:|
| CSIL | 1 | 0.83700 | 0.00000 | 0.00000 | 0.00000 |
| CSIL | 3 | 0.62794 | 0.08511 | 0.00034 | 0.20906 |
| CSIL | 20 | 0.38164 | 0.04075 | 0.03417 | 0.45536 |
| MoPC-HR | 1 | 0.80789 | 0.24133 | 0.18173 | 0.06678 |
| MoPC-HR | 3 | 0.72667 | 0.30478 | 0.28525 | 0.14800 |
| MoPC-HR | 5 | 0.61392 | 0.34373 | 0.31784 | 0.26075 |
| MoPC-HR | 10 | 0.57506 | 0.24148 | 0.26343 | 0.29961 |
| MoPC-HR | 25 | 0.56386 | 0.16420 | 0.21083 | 0.31081 |

这些结果只描述LEO参考组。特别是CSIL在多个新类数/K组合中出现`seen_new_acc=0`，不能用旧类准确率单项替代注册后联合性能。

## 新类不叠加LEO诊断

- 目标语义：复用相同physical IDs和split，仅把`target_new`恢复为LEO overlay前的ManyTx原始IQ；`target_old`原始接收IQ保持不变。
- 声明边界：`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`，不属于正式CVS/阶段二结果。
- 首次烟测状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 触发条件：CSIL与MoPC-HR两个不同烟测cell在prediction前产生相同确定性异常`ValueError: '1-16' is not in list`；完成cell 0，prediction 0。
- 根因：启动参数错误地传入基座训练文件`Dataset_WigSig/ManySig.pkl`；专用no-LEO脚本及冻结cache spec要求`Dataset_WigSig/ManyTx.pkl`。
- 处理：两进程均已退出，未启动800-cell完整诊断矩阵；日志与空输出保留，未覆盖、未原路径重试。修正后的独立v2 run见`adv3b02_official_newcount_scale_no_leo_20260724_v2`。
