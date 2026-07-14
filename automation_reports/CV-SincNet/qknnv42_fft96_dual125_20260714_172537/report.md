# qKNNV42双125-run实验报告

## 1.实验信息

| 字段 | 内容 |
|---|---|
| experiment ID | `qknnv42_fft96_dual125_20260714_172537` |
| 时间 | 2026-07-14 17:25:37+08:00 |
| 操作方 | Codex |
| 当前状态 | 两个分支均125/125完成，artifact完整，结果已回收并完成配对分析 |
| 总目标 | 在完全相同的5个target receiver×5个seed×5档K-shot上，并行评估轻量单视图FFT版与完整legacy-style oracle版，各125次运行 |
| 基线 | 2026-07-13正式单视图、无FFT的125次`cvs_qknnv42`结果 |

## 2.控制文件与协议边界

已按顺序读取：

1. `E:\type10-7\AGENTS.md`
2. `E:\type10-7\项目.md`
3. `E:\type10-7\tools\optimizer_control_manifest.md`
4. `E:\type10-7\tools\optimizer_workflow_contract.md`

本实验保持`项目.md`的Stage2-C协议不变：

- target receiver与source receiver不相交；
- 6个target-old TX为`14-10,14-7,20-15,20-19,6-15,8-20`；
- 2个seen-new TX为`1-16,1-18`；
- support与query均来自同一target receiver的简化LEO视图；
- K∈{1,2,5,10,20}；
- seed∈{713101,713102,713103,713104,713105}；
- receiver∈{20-1,3-19,7-14,7-7,8-8}；
- unknown不参与Phase2主指标、阈值拟合或模型选择。

完整体的old/new角色Oracle和等类别配额Hungarian分配违反普通逐样本部署可用性，因此该分支固定标为`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`，不得进入正式部署排名或论文主结论。

## 3.实验分支

| 分支 | raw IQ视图 | ADV3B02 | FFT | qKNN决策 | 运行数 | 声明边界 |
|---|---|---|---|---|---:|---|
| A：轻量单视图FFT | 每个场景1个LEO视图 | 冻结`ADV3B02_CORE90_SOFT_E200`，输出160维`z_id` | 同一post-channel视图生成96维`fft_logmag`，权重0.34 | 逐样本统一argmax；无角色Oracle、无quota、无场景类别硬筛 | 125 | 正式Stage2-C候选 |
| B：完整legacy-style oracle | 每个物理样本5-view TTA并聚合 | 同一基础checkpoint+60 epoch`id_norm_late_feature`适配器 | 5-view同源96维`fft_logmag`聚合，权重0.34 | label propagation+old/new角色限制+每类等额Hungarian quota；场景候选硬筛在正式数据中因每类覆盖全部场景而无实际排除作用 | 125 | 仅oracle上界诊断 |

两个分支都使用相同receiver、seed、K、query_per_tx=20、support_pool_max_k=20和逐场景split规则。B分支额外导出的target-unknown仅用于兼容历史adapter导出脚本，qKNN125-run不读取、不评估且不参与任何拟合。

## 4.方法与输入输出

### A分支

`single LEO raw IQ→同一视图96维FFT+冻结ADV3B02→160维z_id→qKNNV42分数融合→逐样本预测`

输入：

- 同一target receiver中的target-old和target-new LEO IQ；
- 每类K条带标签support；
- 与support互斥的每类20条query；
- 160维`z_id`和同一post-channel IQ生成的96维FFT描述符。

输出：

- 每个query的预测标签；
- old_acc、seen_new_acc、H_old_new；
- old→new、new→old错误率；
- 按receiver、TX、receiver×TX、receiver×TX×day的详细结果；
- score table、split manifest、loss trace和延迟/存储元数据。

### B分支

在A分支输出契约基础上，使用60 epoch适配特征、5-view聚合和角色/配额Oracle。输入额外包含离线已知的query old/new角色与每类query配额。该信息在普通在线卫星部署中不可获得，因此结果仅用于量化oracle上界。

## 5.本地改动

| 文件 | 目的 |
|---|---|
| `code/export_spaceborne_features.py` | 从与ADV3B02完全相同的post-channel IQ视图导出96维FFT log-magnitude特征 |
| `code/scripts/train_apply_phase1_iq_preadapter_20260703.py` | 完整体adapter导出支持同源FFT、5-view聚合和每个物理样本覆盖全部3种LEO场景 |
| `paper_reproduction/cvs_aligned/cvs_method_runner.py` | qKNN主辅分数融合、receiver×scenario缓存解析、正式argmax与legacy角色配额Oracle双决策模式 |
| `tests/test_cvs_proposed_stage2_runner.py` | 覆盖FFT融合、同视图manifest约束和oracle diagnostic标记 |
| `paper_reproduction/configs/cvs_qknnv42_fft96_singleview_stage2c_20260714_n607.json` | A分支配置 |
| `paper_reproduction/configs/cvs_qknnv42_full_legacy_oracle_stage2c_20260714_n607.json` | B分支配置 |
| `paper_reproduction/scripts/export_cvs_publication_adv3b02_fft96_singleview_20260714.sh` | A分支3场景特征导出 |
| `paper_reproduction/scripts/run_cvs_qknnv42_fft96_singleview_125_20260714.sh` | A分支125-run执行器 |
| `paper_reproduction/scripts/run_cvs_qknnv42_full_legacy_oracle_125_20260714.sh` | B分支60 epoch适配、5-view导出与125-run执行器 |

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`。编辑前分支相对远端ahead1035，已有2个不相关Markdown修改和多组未跟踪`local_artifacts`；本实验未覆盖或清理这些既有内容。

## 6.本地验证

| 命令 | 结果 |
|---|---|
| `conda activate ssr-gpu; python -m py_compile code/export_spaceborne_features.py code/scripts/train_apply_phase1_iq_preadapter_20260703.py paper_reproduction/cvs_aligned/cvs_method_runner.py paper_reproduction/scripts/run_cvs_publication_matrix.py` | PASS |
| `bash -n paper_reproduction/scripts/export_cvs_publication_adv3b02_fft96_singleview_20260714.sh paper_reproduction/scripts/run_cvs_qknnv42_fft96_singleview_125_20260714.sh paper_reproduction/scripts/run_cvs_qknnv42_full_legacy_oracle_125_20260714.sh` | PASS |
| `conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q tests/test_cvs_proposed_stage2_runner.py code/tests/test_phase2_raw_iq_sketch_export.py` | PASS，5 tests |
| 两个JSON配置解析 | PASS |
| A分支导出launcher`--dry-run` | PASS，展开3个场景、3个GPU和`--aux_fft_logmag_dim 96 --satellite_tta_policy none` |

第一次使用`conda activate ssr-gpu; python -m pytest`时命中了基础Python且缺少pytest；按本机Conda包装规则串行改用`conda run -n ssr-gpu`后5项测试全部通过。该问题属于本地环境入口噪声，不是项目测试失败。

## 7.N607执行记录

### 本地到远端映射

| 本地 | 远端 |
|---|---|
| Git承载面上述9个代码/配置/脚本文件 | `/home/szu2070436088/2510044040/CV-SincNet/`下相同相对路径 |
| 本报告Git镜像 | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/qknnv42_fft96_dual125_20260714_172537/report.md` |

### A分支预期路径

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_fft96_singleview_125_20260714`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_qknnv42_fft96_singleview_125_20260714`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714`
- 启动命令：`nohup env GPUS=0,1,2 bash paper_reproduction/scripts/run_cvs_qknnv42_fft96_singleview_125_20260714.sh > paper_reproduction/logs/cvs_qknnv42_fft96_singleview_125_20260714/launcher.out 2>&1 &`
- launcher PID：`462988`；矩阵worker PID：`463333`；状态：125/125完成、0失败。

### B分支预期路径

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_full_legacy_oracle_125_20260714`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_qknnv42_full_legacy_oracle_125_20260714`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_full_adapter5_fft96_20260714`
- 启动命令：`nohup env GPU=3 bash paper_reproduction/scripts/run_cvs_qknnv42_full_legacy_oracle_125_20260714.sh > paper_reproduction/logs/cvs_qknnv42_full_legacy_oracle_125_20260714/launcher.out 2>&1 &`
- launcher PID：`462989`；adapter训练/导出PID：`462993`；矩阵worker PID：`472511`；状态：60 epoch完成、5个receiver缓存完成、125/125完成、0失败。

Conda/Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。启动前N607直连预检PASS，8张GPU均无训练进程；A分支使用GPU0–2做短时特征导出，B分支使用GPU3训练和导出，随后两个矩阵均为CPU侧qKNN计算。结束后检查本地`ssh.exe`和到N607/bridge的TCP22连接，均为0。

## 8.成功标准与风险

| 项目 | 标准 |
|---|---|
| artifact完整性 | 每个分支125/125运行完成；每行均有metrics、split manifest、resolved config、score table、detailed metrics和loss trace |
| 协议完整性 | support/query交集为0；每个target receiver同时覆盖6 old+2 new；3种LEO场景齐全 |
| A分支主要比较 | 与原单视图无FFT基线做同receiver、seed、K、scenario配对比较，报告Δold_acc、Δseen_new_acc、ΔH |
| B分支主要比较 | 只报告相对A分支的oracle上界及其来源，不作部署成功声明 |
| Phase2路线门槛 | 先检查old_acc≥80%；再检查seen_new_acc与H；125行必须按同一运行完整上下文解释 |
| 停止条件 | OOM、NaN、Traceback、协议字段不一致、缓存同视图验证失败、support/query重叠或远端GPU容量不满足 |

已知风险：

- A分支FFT权重0.34来自历史诊断，尚未在正式125-run上预注册验证；结果可能改善，也可能因receiver/场景变化而退化。
- B分支60 epoch适配器耗时显著高于A分支，且角色/quota使用未来批次先验；即使准确率很高也不是可部署证据。
- 完整体场景硬筛在当前正式矩阵中理论上无效，因为每个注册类均覆盖clear、low-elevation和rain；这与历史类别-场景混杂数据不同。
- N607如已有训练任务，只能按AGENTS容量规则选择空闲槽位；不得超过每GPU两个训练进程。
- 完整体checkpoint载入报告存在预期的missing/unexpected/skipped键，来源是ADV3B02训练checkpoint与当前adapter包装头之间的结构差异；训练、导出和125-run均无Traceback、NaN或OOM。该事实不影响artifact完成状态，但必须保留为可复现性风险，不能把完整体解释成“严格逐参数同构加载”。

## 9.核心结果

所有数值均为125个相同receiver×seed×K任务的运行级宏平均；百分比差值为百分点。完整体只作为oracle诊断，不与正式候选混排。

| 分支 | old_acc | new_acc | H | old_acc标准差 | new_acc标准差 | H标准差 | old/new均≥80% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-13原qKNNV42单视图无FFT | 65.59% | 47.94% | 53.26% | — | — | — | 0/125 |
| A：单视图+FFT96正式候选 | 75.12% | 64.64% | 68.56% | 13.46% | 16.79% | 15.25% | 26/125 |
| B：完整legacy oracle诊断 | 82.93% | 93.37% | 87.65% | 9.02% | 8.33% | 8.53% | 85/125 |

配对变化：

| 对比 | Δold_acc | Δnew_acc | ΔH |
|---|---:|---:|---:|
| A−原qKNNV42 | +9.53 | +16.70 | +15.31 |
| B−原qKNNV42 | +17.34 | +45.43 | +34.39 |
| B−A | +7.80 | +28.73 | +19.08 |

## 10.按K-shot结果

| 分支 | K | old_acc | new_acc | H |
|---|---:|---:|---:|---:|
| A | 1 | 62.80% | 48.13% | 52.70% |
| A | 2 | 70.00% | 57.20% | 62.01% |
| A | 5 | 78.03% | 66.80% | 71.33% |
| A | 10 | 81.36% | 73.67% | 76.89% |
| A | 20 | 83.43% | 77.40% | 79.89% |
| B | 1 | 75.02% | 87.87% | 80.44% |
| B | 2 | 80.54% | 91.87% | 85.69% |
| B | 5 | 84.27% | 94.40% | 88.92% |
| B | 10 | 86.27% | 96.20% | 90.88% |
| B | 20 | 88.54% | 96.53% | 92.32% |

结论：A分支随K单调改善，但即使K=20，平均H仍为79.89%；B分支在K=1就达到80.44%H，主要增益来自角色Oracle和等配额批分配对new类错误的强约束，而不是单纯增加support。

## 11.按receiver结果

| 分支 | receiver | old_acc | new_acc | H |
|---|---|---:|---:|---:|
| A | 20-1 | 74.58% | 70.33% | 72.02% |
| A | 3-19 | 57.66% | 42.80% | 47.97% |
| A | 7-14 | 85.96% | 73.50% | 78.58% |
| A | 7-7 | 81.41% | 67.57% | 73.17% |
| A | 8-8 | 76.02% | 69.00% | 71.08% |
| B | 20-1 | 79.32% | 94.07% | 85.84% |
| B | 3-19 | 71.44% | 80.00% | 75.20% |
| B | 7-14 | 91.02% | 97.53% | 94.09% |
| B | 7-7 | 86.16% | 98.20% | 91.60% |
| B | 8-8 | 86.70% | 97.07% | 91.51% |

`3-19`是两个分支共同的主要失败域：A的H仅47.97%，B也只有75.20%。这说明125-run与历史单切分诊断的差距首先来自receiver/domain难度，而不是随机噪声；5-view、adapter和oracle可以缓解，但不能消除该域的old类表示退化。

## 12.按LEO场景结果

| 分支 | 场景 | old_acc | new_acc | H |
|---|---|---:|---:|---:|
| A | clear | 78.57% | 68.86% | 72.51% |
| A | low-elevation | 72.33% | 66.36% | 68.67% |
| A | rain | 74.48% | 58.70% | 64.52% |
| B | clear | 84.89% | 95.72% | 89.80% |
| B | low-elevation | 81.04% | 92.16% | 86.02% |
| B | rain | 82.85% | 92.24% | 87.13% |

A分支对rain最敏感，主要损失落在new_acc；B分支通过5-view聚合和oracle显著压低该退化。375个场景行中，场景候选硬筛实际生效次数为0，因为每个注册类在三个正式场景中都有support；因此完整体增益不能归功于场景筛选。

## 13.完整体训练与artifact审计

| 项目 | 结果 |
|---|---|
| adapter训练 | 60 epoch完成；日志每5 epoch记录一次，共12个观测点 |
| total loss | epoch5的3.5965降至epoch60的2.7627，最低点为epoch60 |
| proxy unknown SupCon | 8.2338降至6.7673 |
| proxy unknown prototype CE | 3.3768降至2.5326 |
| 异常扫描 | Traceback=0、NaN=0、OOM=0 |
| A分支artifact | metrics/split/score/detailed/loss/resolved config均125/125 |
| B分支artifact | metrics/split/score/detailed/loss/resolved config均125/125 |
| support/query重叠 | 两分支均0/125违规 |
| query标签用于训练 | 两分支均0/125违规 |
| TTA与Oracle标记 | A全为1-view且oracle=false；B全为5-view且375/375场景行role/quota oracle=true |

注意：训练日志只按`log_every=5`记录12个点，不是逐epoch结构化曲线；本文结论基于完整launcher日志中的全部可用训练观测，不能声称已经审计60个逐epoch点。

## 14.对历史92.28%H诊断的解释

历史`old_acc=94.52%、new_acc=90.14%、H=92.28%`属于不同切分、20个新类、单seed的legacy diagnostic。当前完整体125-run均值为`82.93%/93.37%/87.65%`，H低4.63个百分点；但当前K=20子集的H为92.32%，与历史92.28%几乎相同。由此可见：

1.历史值不是“同一方法在任意K、receiver、seed上的稳定水平”，而接近高support条件的单切分结果；
2.125-run加入K=1/2/5和困难receiver`3-19`后，old_acc被显著拉低；
3.完整体把new_acc推到93.37%，但old_acc只有82.93%，说明主要短板已从new类识别转为跨receiver的old类保持；
4.角色Oracle与等配额Hungarian直接使用query批次的old/new角色和类别数量先验，能制造很高的new_acc与H，因此历史92.28%只能作为oracle上界诊断，不能视作单星逐样本部署结果。

## 15.最终判定

| 分支 | 判定 |
|---|---|
| A：单视图+FFT96 | 相对原qKNNV42显著改善，125-run的H提升15.31个百分点；但整体old_acc=75.12%未达到80%主门槛，且只有26/125同时满足old/new≥80%，不能宣称已达成正式部署成功。K≥10和receiver`7-14`方向值得继续。 |
| B：完整legacy oracle | 性能高，K=20平均H=92.32%，但角色/quota先验不可在线获得；必须保持`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`，只用于估计上界和定位缺失能力。 |

机器资源方面，A是轻量路线：GPU只用于一次性冻结backbone特征导出，qKNN阶段主要是小规模CPU计算；B不是卫星端轻型路线，60 epoch训练应在地面完成，5-view把在线前向开销约放大5倍，Hungarian还要求批次级角色/配额先验。即使卫星算力足够，B的核心障碍也不是算力，而是部署协议中拿不到Oracle信息。

完整逐运行同排指标见本报告后续250行表；机器可读结果见`analysis/per_run_results.csv`、`analysis/per_scenario_results.csv`、`analysis/paired_full_minus_light.csv`和`analysis/paired_vs_original_qknnv42.csv`。

## 16.250行逐运行结果

| candidate ID | 分支/机制 | receiver | TX split | K | seed | old_acc | new_acc | H | unknown | coverage | rollback/defer | verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713101` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 1 | 713101 | 64.44% | 48.33% | 54.32% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713101` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 10 | 713101 | 85.56% | 80.00% | 82.62% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713101` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 2 | 713101 | 74.17% | 67.50% | 70.47% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713101` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 20 | 713101 | 88.89% | 82.50% | 85.46% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713101` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 5 | 713101 | 79.44% | 75.83% | 77.40% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713102` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 1 | 713102 | 63.33% | 57.50% | 59.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713102` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 10 | 713102 | 81.94% | 85.83% | 83.84% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713102` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 2 | 713102 | 64.44% | 60.00% | 62.10% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713102` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 20 | 713102 | 86.39% | 85.83% | 86.04% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713102` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 5 | 713102 | 75.83% | 75.83% | 75.74% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713103` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 1 | 713103 | 43.33% | 62.50% | 51.10% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713103` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 10 | 713103 | 83.06% | 76.67% | 79.66% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713103` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 2 | 713103 | 60.00% | 60.83% | 60.14% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713103` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 20 | 713103 | 88.06% | 83.33% | 85.61% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713103` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 5 | 713103 | 77.50% | 70.00% | 73.29% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713104` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 1 | 713104 | 53.61% | 60.83% | 56.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713104` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 10 | 713104 | 80.28% | 75.83% | 77.98% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713104` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 2 | 713104 | 68.33% | 67.50% | 67.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713104` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 20 | 713104 | 81.94% | 77.50% | 79.49% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713104` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 5 | 713104 | 73.89% | 70.83% | 72.31% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713105` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 1 | 713105 | 56.39% | 52.50% | 54.16% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713105` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 10 | 713105 | 87.22% | 75.83% | 81.12% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713105` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 2 | 713105 | 74.17% | 54.17% | 62.56% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713105` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 20 | 713105 | 89.72% | 79.17% | 83.77% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713105` | A:1-view+FFT96+argmax | `20-1` | 6old/2new | 5 | 713105 | 82.50% | 71.67% | 76.52% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713101` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 1 | 713101 | 42.22% | 33.33% | 35.97% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713101` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 10 | 713101 | 62.22% | 47.50% | 53.54% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713101` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 2 | 713101 | 53.33% | 32.50% | 38.40% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713101` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 20 | 713101 | 63.33% | 45.00% | 51.61% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713101` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 5 | 713101 | 63.89% | 33.33% | 43.71% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713102` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 1 | 713102 | 46.67% | 40.00% | 42.73% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713102` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 10 | 713102 | 68.61% | 48.33% | 56.60% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713102` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 2 | 713102 | 55.56% | 34.17% | 41.86% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713102` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 20 | 713102 | 69.44% | 57.50% | 62.89% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713102` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 5 | 713102 | 64.17% | 42.50% | 50.84% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713103` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 1 | 713103 | 43.06% | 35.83% | 38.97% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713103` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 10 | 713103 | 59.44% | 48.33% | 52.78% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713103` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 2 | 713103 | 50.83% | 48.33% | 49.47% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713103` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 20 | 713103 | 67.50% | 51.67% | 57.95% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713103` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 5 | 713103 | 56.67% | 50.83% | 53.28% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713104` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 1 | 713104 | 35.28% | 39.17% | 35.57% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713104` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 10 | 713104 | 59.72% | 59.17% | 59.02% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713104` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 2 | 713104 | 55.83% | 35.83% | 43.46% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713104` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 20 | 713104 | 64.44% | 55.83% | 59.53% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713104` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 5 | 713104 | 52.78% | 49.17% | 49.79% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713105` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 1 | 713105 | 55.00% | 18.33% | 26.31% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713105` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 10 | 713105 | 64.44% | 38.33% | 47.37% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713105` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 2 | 713105 | 54.17% | 38.33% | 44.22% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713105` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 20 | 713105 | 70.00% | 48.33% | 56.42% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713105` | A:1-view+FFT96+argmax | `3-19` | 6old/2new | 5 | 713105 | 62.78% | 38.33% | 47.07% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713101` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 1 | 713101 | 80.56% | 53.33% | 63.74% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713101` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 10 | 713101 | 89.72% | 82.50% | 85.72% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713101` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 2 | 713101 | 83.89% | 65.83% | 73.42% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713101` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 20 | 713101 | 91.39% | 87.50% | 89.40% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713101` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 5 | 713101 | 88.33% | 70.00% | 78.06% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713102` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 1 | 713102 | 78.33% | 50.83% | 61.64% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713102` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 10 | 713102 | 86.94% | 83.33% | 85.06% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713102` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 2 | 713102 | 83.61% | 55.00% | 66.35% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713102` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 20 | 713102 | 87.50% | 87.50% | 87.32% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713102` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 5 | 713102 | 87.50% | 67.50% | 76.20% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713103` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 1 | 713103 | 81.11% | 46.67% | 58.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713103` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 10 | 713103 | 91.67% | 85.00% | 87.99% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713103` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 2 | 713103 | 87.22% | 68.33% | 76.19% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713103` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 20 | 713103 | 92.78% | 85.83% | 89.14% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713103` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 5 | 713103 | 88.89% | 67.50% | 76.42% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713104` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 1 | 713104 | 68.33% | 75.00% | 71.42% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713104` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 10 | 713104 | 85.28% | 82.50% | 83.84% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713104` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 2 | 713104 | 78.89% | 79.17% | 78.98% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713104` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 20 | 713104 | 86.39% | 83.33% | 84.77% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713104` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 5 | 713104 | 85.56% | 81.67% | 83.52% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713105` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 1 | 713105 | 86.67% | 59.17% | 69.37% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713105` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 10 | 713105 | 90.28% | 85.00% | 87.54% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713105` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 2 | 713105 | 88.06% | 67.50% | 76.03% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713105` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 20 | 713105 | 90.83% | 85.83% | 88.26% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713105` | A:1-view+FFT96+argmax | `7-14` | 6old/2new | 5 | 713105 | 89.17% | 81.67% | 85.21% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713101` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 1 | 713101 | 71.11% | 50.83% | 59.27% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713101` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 10 | 713101 | 90.00% | 74.17% | 80.86% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713101` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 2 | 713101 | 76.67% | 55.83% | 64.43% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713101` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 20 | 713101 | 89.17% | 82.50% | 85.49% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713101` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 5 | 713101 | 82.50% | 77.50% | 79.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713102` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 1 | 713102 | 70.00% | 46.67% | 55.68% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713102` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 10 | 713102 | 85.28% | 75.00% | 79.57% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713102` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 2 | 713102 | 76.39% | 48.33% | 58.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713102` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 20 | 713102 | 88.89% | 76.67% | 82.15% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713102` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 5 | 713102 | 81.94% | 70.83% | 75.51% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713103` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 1 | 713103 | 41.94% | 35.00% | 37.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713103` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 10 | 713103 | 90.28% | 79.17% | 84.22% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713103` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 2 | 713103 | 61.94% | 69.17% | 65.35% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713103` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 20 | 713103 | 90.28% | 81.67% | 85.70% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713103` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 5 | 713103 | 84.17% | 79.17% | 81.34% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713104` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 1 | 713104 | 68.06% | 60.83% | 63.90% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713104` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 10 | 713104 | 88.89% | 84.17% | 86.41% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713104` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 2 | 713104 | 83.89% | 65.83% | 73.64% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713104` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 20 | 713104 | 85.56% | 89.17% | 87.31% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713104` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 5 | 713104 | 90.00% | 77.50% | 83.12% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713105` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 1 | 713105 | 80.00% | 40.83% | 54.00% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713105` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 10 | 713105 | 92.22% | 72.50% | 81.15% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713105` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 2 | 713105 | 82.50% | 55.83% | 65.45% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713105` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 20 | 713105 | 94.17% | 76.67% | 84.34% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713105` | A:1-view+FFT96+argmax | `7-7` | 6old/2new | 5 | 713105 | 89.44% | 63.33% | 73.75% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713101` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 1 | 713101 | 61.94% | 61.67% | 61.78% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713101` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 10 | 713101 | 85.83% | 75.00% | 79.96% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713101` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 2 | 713101 | 65.00% | 63.33% | 64.09% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713101` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 20 | 713101 | 86.94% | 85.83% | 86.34% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713101` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 5 | 713101 | 78.06% | 71.67% | 74.56% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713102` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 1 | 713102 | 71.11% | 40.00% | 50.29% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713102` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 10 | 713102 | 79.72% | 84.17% | 81.81% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713102` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 2 | 713102 | 70.83% | 58.33% | 63.06% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713102` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 20 | 713102 | 81.39% | 87.50% | 84.23% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713102` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 5 | 713102 | 77.50% | 65.83% | 69.99% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713103` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 1 | 713103 | 70.83% | 49.17% | 55.56% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713103` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 10 | 713103 | 80.00% | 82.50% | 81.06% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713103` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 2 | 713103 | 69.72% | 55.83% | 58.79% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713103` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 20 | 713103 | 82.22% | 90.00% | 85.91% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713103` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 5 | 713103 | 77.78% | 69.17% | 72.41% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713104` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 1 | 713104 | 69.17% | 55.83% | 60.68% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713104` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 10 | 713104 | 85.00% | 86.67% | 85.81% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713104` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 2 | 713104 | 59.44% | 65.00% | 61.92% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713104` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 20 | 713104 | 87.22% | 85.83% | 86.51% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713104` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 5 | 713104 | 80.00% | 79.17% | 79.24% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713105` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 1 | 713105 | 67.50% | 29.17% | 37.34% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713105` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 10 | 713105 | 80.28% | 74.17% | 76.62% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713105` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 2 | 713105 | 71.11% | 57.50% | 63.18% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛FAIL |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713105` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 20 | 713105 | 81.39% | 82.50% | 81.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713105` | A:1-view+FFT96+argmax | `8-8` | 6old/2new | 5 | 713105 | 80.56% | 69.17% | 74.22% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=按门槛 | 正式候选:old门槛PASS |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713101` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 1 | 713101 | 58.89% | 90.00% | 70.92% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713101` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 10 | 713101 | 82.50% | 96.67% | 88.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713101` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 2 | 713101 | 71.39% | 95.00% | 81.47% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713101` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 20 | 713101 | 84.17% | 98.33% | 90.61% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713101` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 5 | 713101 | 80.28% | 96.67% | 87.64% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713102` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 1 | 713102 | 75.00% | 86.67% | 80.37% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713102` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 10 | 713102 | 89.72% | 96.67% | 93.01% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713102` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 2 | 713102 | 75.56% | 95.00% | 84.12% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713102` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 20 | 713102 | 91.11% | 98.33% | 94.58% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713102` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 5 | 713102 | 84.72% | 96.67% | 90.17% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713103` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 1 | 713103 | 68.06% | 90.00% | 77.35% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713103` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 10 | 713103 | 84.17% | 93.33% | 88.48% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713103` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 2 | 713103 | 73.61% | 93.33% | 82.30% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713103` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 20 | 713103 | 91.11% | 98.33% | 94.56% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713103` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 5 | 713103 | 76.67% | 95.00% | 84.81% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713104` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 1 | 713104 | 68.33% | 88.33% | 76.70% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713104` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 10 | 713104 | 80.83% | 95.00% | 87.33% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713104` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 2 | 713104 | 73.06% | 88.33% | 79.93% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713104` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 20 | 713104 | 88.89% | 95.00% | 91.84% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713104` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 5 | 713104 | 80.56% | 93.33% | 86.45% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k1_seed713105` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 1 | 713105 | 69.44% | 85.00% | 75.72% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k10_seed713105` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 10 | 713105 | 86.67% | 96.67% | 91.32% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k2_seed713105` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 2 | 713105 | 76.39% | 93.33% | 83.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k20_seed713105` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 20 | 713105 | 88.89% | 98.33% | 93.37% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx20-1_k5_seed713105` | B:adapter60+5-view+FFT96+role/quota | `20-1` | 6old/2new | 5 | 713105 | 83.06% | 98.33% | 90.03% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713101` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 1 | 713101 | 60.83% | 73.33% | 66.08% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713101` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 10 | 713101 | 75.56% | 76.67% | 76.00% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713101` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 2 | 713101 | 70.00% | 73.33% | 71.62% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713101` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 20 | 713101 | 79.44% | 78.33% | 78.87% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713101` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 5 | 713101 | 74.44% | 71.67% | 72.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713102` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 1 | 713102 | 53.61% | 68.33% | 59.97% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713102` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 10 | 713102 | 76.39% | 86.67% | 80.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713102` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 2 | 713102 | 67.22% | 76.67% | 71.58% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713102` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 20 | 713102 | 77.78% | 88.33% | 82.58% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713102` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 5 | 713102 | 71.39% | 81.67% | 76.05% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713103` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 1 | 713103 | 57.50% | 68.33% | 62.38% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713103` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 10 | 713103 | 75.00% | 90.00% | 81.49% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713103` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 2 | 713103 | 65.28% | 76.67% | 70.43% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713103` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 20 | 713103 | 81.11% | 90.00% | 85.18% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713103` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 5 | 713103 | 72.50% | 85.00% | 78.02% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713104` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 1 | 713104 | 64.44% | 71.67% | 67.29% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713104` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 10 | 713104 | 73.33% | 88.33% | 80.13% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713104` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 2 | 713104 | 64.17% | 73.33% | 67.99% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713104` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 20 | 713104 | 78.89% | 83.33% | 80.98% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713104` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 5 | 713104 | 73.06% | 80.00% | 75.95% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k1_seed713105` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 1 | 713105 | 65.56% | 70.00% | 66.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k10_seed713105` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 10 | 713105 | 78.61% | 93.33% | 85.24% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k2_seed713105` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 2 | 713105 | 70.83% | 76.67% | 73.59% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k20_seed713105` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 20 | 713105 | 83.06% | 91.67% | 87.12% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx3-19_k5_seed713105` | B:adapter60+5-view+FFT96+role/quota | `3-19` | 6old/2new | 5 | 713105 | 76.11% | 86.67% | 80.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 1 | 713101 | 85.83% | 96.67% | 90.81% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 10 | 713101 | 92.22% | 98.33% | 95.18% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 2 | 713101 | 89.17% | 95.00% | 91.93% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 20 | 713101 | 93.33% | 98.33% | 95.77% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 5 | 713101 | 90.83% | 96.67% | 93.66% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 1 | 713102 | 91.67% | 96.67% | 94.08% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 10 | 713102 | 91.67% | 100.00% | 95.64% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 2 | 713102 | 91.94% | 96.67% | 94.23% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 20 | 713102 | 92.78% | 100.00% | 96.25% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 5 | 713102 | 91.39% | 96.67% | 93.89% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 1 | 713103 | 88.06% | 98.33% | 92.86% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 10 | 713103 | 90.28% | 100.00% | 94.88% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 2 | 713103 | 90.28% | 98.33% | 94.09% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 20 | 713103 | 91.67% | 100.00% | 95.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 5 | 713103 | 89.17% | 98.33% | 93.49% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 1 | 713104 | 84.72% | 76.67% | 80.17% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 10 | 713104 | 92.78% | 100.00% | 96.23% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 2 | 713104 | 89.17% | 98.33% | 93.45% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 20 | 713104 | 94.44% | 100.00% | 97.12% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 5 | 713104 | 90.56% | 100.00% | 94.97% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k1_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 1 | 713105 | 92.50% | 96.67% | 94.54% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k10_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 10 | 713105 | 92.50% | 100.00% | 96.09% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k2_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 2 | 713105 | 92.78% | 98.33% | 95.46% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k20_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 20 | 713105 | 93.06% | 100.00% | 96.39% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-14_k5_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-14` | 6old/2new | 5 | 713105 | 92.78% | 98.33% | 95.47% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 1 | 713101 | 77.22% | 85.00% | 79.88% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 10 | 713101 | 87.78% | 100.00% | 93.49% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 2 | 713101 | 85.56% | 95.00% | 89.93% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 20 | 713101 | 91.67% | 100.00% | 95.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713101` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 5 | 713101 | 86.39% | 100.00% | 92.68% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 1 | 713102 | 72.78% | 98.33% | 82.64% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 10 | 713102 | 89.72% | 100.00% | 94.55% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 2 | 713102 | 84.72% | 96.67% | 90.16% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 20 | 713102 | 91.94% | 100.00% | 95.77% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713102` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 5 | 713102 | 86.39% | 100.00% | 92.61% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 1 | 713103 | 71.67% | 95.00% | 81.58% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 10 | 713103 | 88.33% | 100.00% | 93.80% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 2 | 713103 | 83.06% | 98.33% | 90.00% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 20 | 713103 | 89.44% | 100.00% | 94.42% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713103` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 5 | 713103 | 87.78% | 98.33% | 92.71% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 1 | 713104 | 85.56% | 100.00% | 92.17% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 10 | 713104 | 91.67% | 100.00% | 95.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 2 | 713104 | 88.61% | 100.00% | 93.95% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 20 | 713104 | 92.50% | 100.00% | 96.10% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713104` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 5 | 713104 | 89.72% | 100.00% | 94.57% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k1_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 1 | 713105 | 82.78% | 93.33% | 87.56% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k10_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 10 | 713105 | 89.17% | 100.00% | 94.24% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k2_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 2 | 713105 | 81.39% | 96.67% | 88.25% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k20_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 20 | 713105 | 91.11% | 100.00% | 95.34% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx7-7_k5_seed713105` | B:adapter60+5-view+FFT96+role/quota | `7-7` | 6old/2new | 5 | 713105 | 86.94% | 98.33% | 92.25% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713101` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 1 | 713101 | 84.17% | 98.33% | 90.69% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713101` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 10 | 713101 | 91.67% | 100.00% | 95.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713101` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 2 | 713101 | 86.39% | 98.33% | 91.94% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713101` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 20 | 713101 | 90.28% | 100.00% | 94.88% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713101` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 5 | 713101 | 90.56% | 100.00% | 95.04% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713102` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 1 | 713102 | 81.11% | 95.00% | 87.13% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713102` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 10 | 713102 | 86.67% | 100.00% | 92.85% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713102` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 2 | 713102 | 86.11% | 100.00% | 92.49% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713102` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 20 | 713102 | 87.78% | 100.00% | 93.45% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713102` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 5 | 713102 | 86.39% | 100.00% | 92.65% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713103` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 1 | 713103 | 82.78% | 95.00% | 88.14% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713103` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 10 | 713103 | 90.28% | 96.67% | 93.34% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713103` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 2 | 713103 | 86.39% | 95.00% | 90.44% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713103` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 20 | 713103 | 90.56% | 98.33% | 94.27% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713103` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 5 | 713103 | 88.61% | 96.67% | 92.42% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713104` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 1 | 713104 | 75.56% | 91.67% | 82.75% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713104` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 10 | 713104 | 91.67% | 98.33% | 94.87% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713104` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 2 | 713104 | 86.67% | 96.67% | 91.39% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713104` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 20 | 713104 | 90.83% | 98.33% | 94.41% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713104` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 5 | 713104 | 89.17% | 95.00% | 91.97% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k1_seed713105` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 1 | 713105 | 77.50% | 88.33% | 82.47% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k10_seed713105` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 10 | 713105 | 87.50% | 98.33% | 92.59% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k2_seed713105` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 2 | 713105 | 83.89% | 91.67% | 87.60% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k20_seed713105` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 20 | 713105 | 87.78% | 98.33% | 92.75% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
| `cvs_qknnv42_stage2c_rx8-8_k5_seed713105` | B:adapter60+5-view+FFT96+role/quota | `8-8` | 6old/2new | 5 | 713105 | 87.22% | 96.67% | 91.70% | N/A(协议不评估) | artifact完整;3场景 | rollback=none;defer=永久禁止部署声明 | NON_DEPLOYMENT_ORACLE_DIAGNOSTIC |
