# qKNNV42双125-run实验报告

## 1.实验信息

| 字段 | 内容 |
|---|---|
| experiment ID | `qknnv42_fft96_dual125_20260714_172537` |
| 时间 | 2026-07-14 17:25:37+08:00 |
| 操作方 | Codex |
| 当前状态 | 本地实现与验证完成，N607预检和启动待执行 |
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

## 7.N607计划

### 本地到远端映射

| 本地 | 远端 |
|---|---|
| Git承载面上述9个代码/配置/脚本文件 | `/home/szu2070436088/2510044040/CV-SincNet/`下相同相对路径 |
| 本报告Git镜像 | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/qknnv42_fft96_dual125_20260714_172537/report.md` |

### A分支预期路径

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_fft96_singleview_125_20260714`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_qknnv42_fft96_singleview_125_20260714`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714`
- 启动命令：`nohup env GPUS=<preflight-selected-three-gpus> bash paper_reproduction/scripts/run_cvs_qknnv42_fft96_singleview_125_20260714.sh > paper_reproduction/logs/cvs_qknnv42_fft96_singleview_125_20260714/launcher.out 2>&1 &`

### B分支预期路径

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_full_legacy_oracle_125_20260714`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_qknnv42_full_legacy_oracle_125_20260714`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_qknnv42_full_adapter5_fft96_20260714`
- 启动命令：`nohup env GPU=<preflight-selected-gpu> bash paper_reproduction/scripts/run_cvs_qknnv42_full_legacy_oracle_125_20260714.sh > paper_reproduction/logs/cvs_qknnv42_full_legacy_oracle_125_20260714/launcher.out 2>&1 &`

Conda/Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。GPU、PID和实际启动命令将在N607只读预检与占用核对后回填。

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
- 当前阶段尚未启动远端任务，不能报告运行完成、artifact完成或性能提升。

