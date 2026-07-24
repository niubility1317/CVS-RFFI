# ADV3B02-CSIL/MoPC-HR CVS接口优化实验v3

- 实验ID：`adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3`
- 日期：2026-07-24
- 状态：`ANALYZED / ARTIFACTS_COMPLETE`
- 操作者：Codex主代理；N607唯一运行代理：`no_leo_n607_release`
- 目标：保持已冻结CSIL/MoPC-HR方法、LEO数据、query边界与400-cell矩阵不变，修复v2 smoke发现的容量契约和machine status链后完成正式实验。

## 历史技术停止

- v1：仓库外CWD导入失败，0 cell/0 prediction/0 score，`NO_PERFORMANCE_RESULT`。
- v2：CSIL base capacity drift；MoPC partial prediction后status drift；smoke receipt=0、authority plan=0、full launch=0，`NO_PERFORMANCE_RESULT`。
- v1/v2均只读保留，不覆盖、不续跑，所有run-owned进程和SSH连接已清理。

## 本轮最小修复

1. CSIL v2 adapter在builder和runner强制`required_total_capacity=26`；MoPC v2 adapter强制31。
2. runner按method统一校验并写入三类status：strict baseline、interface adapter、ordered-arrival diagnostic。
3. status规则覆盖predictor返回、existing/new cell receipt和smoke artifact authority验证。
4. 未修改`adv3b02_official_repo_ci.py`、predictor训练/判决、LEO IQ入口、query打开顺序、loss或优化器。
5. 本地定向+计划+相邻集成：`54 passed`；`git diff --check`通过。
6. 独立复审：`P0=0,P1=0 / APPROVE`。
7. 修复提交：`f120d4931febc2fffabe193233a0294bca15fd90`。
8. builder SHA256：`d012dfa4b9efb503167cef756519e61a9e51a790d6d5f9cb5eab5ac213f0698b`；runner SHA256：`827aa78bdabba7c648b57d5a3075a6402885aa2ff99f2b2818bb599c14fed836`。

## 冻结矩阵与容量

| 方法 | 新类数 | K | receiver×seed | capacity | cells | 场景行 |
|---|---:|---|---:|---:|---:|---:|
| `csil_official_repo_corefix_cvs_adapter` | 1、3 | 1、5、10、20 | 5×5 | 26 | 200 | 600 |
| `mopc_hr_official_repo_cvs_adapter` | 25 | 1、5、10、20 | 5×5 | 31 | 100 | 300 |
| `mopc_hr_official_repo_sequential5_cvs_adapter` | 25 | 1、5、10、20 | 5×5 | 31 | 100 | 300 |
| 合计 | — | — | — | — | 400 | 1200 |

## N607预注册

- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3`
- log root：`logs/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3`
- base26：只读复用v7，SHA256=`635becd7db2d8041a669cb0ef922429c42ba389846f2252c7cfe4e0f3510a07e`
- base31：只读复用v7，SHA256=`306c8dfc767bad93f78f24b675b1c20058629eda8f3153bdf98d03ab6ae26202`
- 必须显式以26/31构建两份pre-smoke plan；smoke PASS后才生成authority plan和完整矩阵。
- 预期每cell闭环prediction、score、predictor/enrollment/cell receipt、loss trace和3条formal rows。

## 健康、停止和结论边界

- P0或两个不同row在prediction前同一确定性异常指纹时，仅停止v3精确process tree并保留制品；不得因性能低停止。
- 新类support/query只使用固定LEO弱信道IQ；query在模型状态锁定后才打开，训练行数0。
- CSIL差异同时包含官方old-old fingerprint mask修复和small-K接口适配，禁止单因归因。
- MoPC single-stage new25是instrumentation parity；sequential5仅为`ORDERED_ARRIVAL_DIAGNOSTIC`，不与同时注册等价。
- 若v3技术失败，不自动创建v4。

## 结果

### 运行与制品闭环

- 直接N607 preflight通过；同步后builder/runner远端SHA256分别与预注册值一致；仓库外CWD的`--help`与`py_compile`通过。
- smoke：CSIL 2/2 cell、MoPC 4/4 cell，均为`PASS`；随后生成`launch_authority=true`的两份正式plan。
- 正式矩阵使用GPU0—7，每卡同时一个CSIL shard和一个MoPC shard，共16个runner；终态400/400 cell成功、0失败、0异常指纹。
- 正式runner命令结构：`CUDA_VISIBLE_DEVICES=<gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python paper_reproduction/scripts/run_adv3b02_paper_full_ci_plan.py --plan <csil_authorized.json|mopc_authorized.json> --project-root /home/szu2070436088/2510044040/CV-SincNet --stage matrix_shard --shard-index <0..7> --shard-count 8 --device cuda:0`；CWD为`/home/szu2070436088/2510044040/CV-SincNet`。
- CSIL shard0—7 PID依次为1785308、1785310、1785312、1785314、1785317、1785319、1785323、1785326；MoPC shard0—7 PID依次为1785309、1785311、1785313、1785316、1785318、1785322、1785324、1785327。launch receipt、PID文件、GPU映射与启动后进程核验一致。
- 制品审计：400份prediction、400份predictor receipt、400份scoring receipt、400份cell receipt和1200条formal row全部存在且receipt内SHA256一致；16个full shard与2个smoke日志终态均为`PASS`。
- query审计：全部400个predictor receipt均为`query_rows_used_for_training=0`、`query_labels_available_to_predictor=false`、`query_members_opened_before_model_lock=false`。
- 冻结场景为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。首次临时审计命令误把第三场景断言为`leo_urban_weak`，在读取cell payload前即退出；修正后依据两份authorized plan全量审计通过。该事件是audit-tool false positive，不是实验异常，未修改任何实验artifact。
- 远端分析manifest SHA256：`feb30fdb9c63e93eb0500436f50255d25fd11f4362203e09cb224646de8db2b9`；取回后8个受manifest约束文件的本地SHA256复核通过。
- 收尾核验：v3 run-owned进程为0；N607上另有8个约542—544MiB的非本轮GPU进程，未干预；本地`ssh.exe=0`，到N607/bridge的TCP22连接为0。

### 核心机制receipt

| 方法 | 审计范围 | 结果 |
|---|---:|---|
| CSIL corefix adapter | 200 cell×3场景=600资源 | `official_fingerprint_mask_corefix=true`；old-old/new-new mask为1、cross mask为0；每资源optimizer step>0；训练类集合覆盖本cell全部新类 |
| MoPC-HR single adapter | 100 cell×3场景=300资源 | 每资源optimizer step>0；小于官方batch size时`proto_aug_count_per_step=16` |
| MoPC-HR sequential5 | 100 cell×3场景=300资源、1500阶段 | 类顺序在query前密封；每场景5阶段×5类；相邻阶段prototype SHA链一致；每阶段`proto_aug_count_per_step=16` |

因此，本轮不是空跑、零步训练或CVS包装替换核心方法。CSIL使用官方仓库核心并显式修复官方fingerprint mask的类维度问题，同时保留small-K接口适配；MoPC-HR single保留官方训练与原型增强核心；sequential5是顺序到达诊断扩展，不能冒充官方同时注册设定。

### 同row总体结果

下表每行在5个receiver×5个seed×4个K的100个cell内，先对每cell的3个LEO场景求均值，再对cell求均值；所有指标来自同一批cell。

| 方法 | 新类数 | old-before | old-after | seen-new | H | forgetting | min-old |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSIL corefix adapter | 1 | 0.83700 | 0.00103 | 1.00000 | 0.00182 | 0.83597 | 0.00000 |
| CSIL corefix adapter | 3 | 0.83700 | 0.00689 | 0.33883 | 0.01006 | 0.83011 | 0.00000 |
| MoPC-HR single adapter | 25 | 0.87467 | 0.56325 | 0.16459 | 0.21110 | 0.31142 | 0.14500 |
| MoPC-HR sequential5诊断 | 25 | 0.87467 | 0.19425 | 0.10378 | 0.06965 | 0.68042 | 0.00100 |

### K-shot同row结果

| 方法 | 新类数 | K | old-after | seen-new | H | forgetting | min-old |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSIL | 1 | 1 | 0.00411 | 1.00000 | 0.00728 | 0.83289 | 0.00000 |
| CSIL | 1 | 5 | 0.00000 | 1.00000 | 0.00000 | 0.83700 | 0.00000 |
| CSIL | 1 | 10 | 0.00000 | 1.00000 | 0.00000 | 0.83700 | 0.00000 |
| CSIL | 1 | 20 | 0.00000 | 1.00000 | 0.00000 | 0.83700 | 0.00000 |
| CSIL | 3 | 1 | 0.02011 | 0.33911 | 0.02886 | 0.81689 | 0.00000 |
| CSIL | 3 | 5 | 0.00478 | 0.34244 | 0.00687 | 0.83222 | 0.00000 |
| CSIL | 3 | 10 | 0.00100 | 0.33756 | 0.00175 | 0.83600 | 0.00000 |
| CSIL | 3 | 20 | 0.00167 | 0.33622 | 0.00278 | 0.83533 | 0.00000 |
| MoPC-HR single | 25 | 1 | 0.85433 | 0.00712 | 0.01404 | 0.02033 | 0.49133 |
| MoPC-HR single | 25 | 5 | 0.58122 | 0.14501 | 0.22983 | 0.29344 | 0.07133 |
| MoPC-HR single | 25 | 10 | 0.44722 | 0.22285 | 0.29255 | 0.42744 | 0.01733 |
| MoPC-HR single | 25 | 20 | 0.37022 | 0.28336 | 0.30797 | 0.50444 | 0.00000 |
| MoPC-HR sequential5诊断 | 25 | 1 | 0.37022 | 0.06064 | 0.09975 | 0.50444 | 0.00067 |
| MoPC-HR sequential5诊断 | 25 | 5 | 0.34400 | 0.05885 | 0.09658 | 0.53067 | 0.00333 |
| MoPC-HR sequential5诊断 | 25 | 10 | 0.04344 | 0.14125 | 0.05399 | 0.83122 | 0.00000 |
| MoPC-HR sequential5诊断 | 25 | 20 | 0.01933 | 0.15437 | 0.02828 | 0.85533 | 0.00000 |

### 与v7严格基线的同row比较

- MoPC-HR single adapter与v7 strict在同receiver、seed、K、场景上的差值很小：K=1/5/10/20的H差值分别为+0.00052、-0.00445、-0.00068、+0.00567。这表明CVS接口适配没有造成主要性能漂移，当前低新类性能主要属于原方法在“25个新类+LEO弱信道+极少样本”条件下的能力边界。
- CSIL corefix相对v7 strict发生结构性变化：新类从v7多数为0提升到n1的1.0和n3约0.34，但旧类准确率几乎归零。结果说明mask修复确实让新类进入训练与判决，却暴露了官方CSIL目标在该大域偏移增量设定下对旧类的灾难性覆盖；不能把性能问题归因于“代码未运行”。
- sequential5相对v7 single-stage只作为参考，不是等价对照。其K越大旧类遗忘反而越重，说明连续5次全模型增量更新累积覆盖旧决策边界；该诊断路线不应作为正式优胜方法。

### 逐类与最终判断

- 逐旧类96行汇总已生成；CSIL所有method/new/K组的`min-old=0`，不是单一旧类偶发下降。MoPC-HR single在K=1仍保留较完整旧类下界，随后随K增加持续下降到K=20的0；这是“新类学习增强—旧类遗忘加剧”的稳定权衡。
- 本轮没有“best epoch/checkpoint”可选择：正式结论按冻结epoch和完整矩阵终态预测给出，不依据query性能挑epoch或cell。
- 最终状态：`ANALYZED / ARTIFACTS_COMPLETE`。CSIL不具备可推广性能；MoPC-HR single是方法忠实且接口适配稳定的完整对比结果，但整体H仍低；sequential5保留为非等价顺序到达诊断。

### 取回文件

- `retrieved/analysis_v1/artifact_audit.json`
- `retrieved/analysis_v1/formal_rows.csv`
- `retrieved/analysis_v1/summary_by_method_new.csv`
- `retrieved/analysis_v1/summary_by_method_new_k.csv`
- `retrieved/analysis_v1/summary_by_method_new_k_receiver_seed.csv`
- `retrieved/analysis_v1/same_row_delta_vs_v7.csv`
- `retrieved/analysis_v1/summary_delta_vs_v7_by_method_new_k.csv`
- `retrieved/analysis_v1/per_old_class_summary_by_method_new_k.csv`
- `retrieved/logs/`内18份完整日志及两份launch receipt
