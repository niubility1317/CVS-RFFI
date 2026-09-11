# CORE90锚定角度几何：实现与确认验收

> 后续复查更新：已发现2项P1、2项P2，下面保留的全量通过结论不再代表当前状态。此前测试通过事实保留，当前缺口及机制开启证据以[再次审查报告](core90_anchored_geometry_recheck_20260911.md)和修订追踪表为准。生产代码尚未修复这些新增问题。

日期：2026-09-11。结论：**IMPLEMENTATION_VERIFIED**。本次完成设计实现计划T1–T9适用代码及技术验收；R01–R42为实现验证通过，R43–R48仍按计划延期。**没有完成正式source矩阵，也没有独立确认结果。**

依据：[设计实现计划](core90_anchored_geometry_implementation_plan_20260911.md)、[用户设计报告全文](core90_anchored_geometry_design_source_20260911.md)、[逐项追踪表](core90_anchored_geometry_traceability.md)。起始提交`d72a5bc083f4ef320fa2ce7d8f3ab0cb9b5ddc57`；隔离Git承载面为`E:/type10-7/code/snapshots/core90_evidence_20260911_wt`，分支`codex/core90-evidence-head-20260911`。既有H0/H1–H4生产入口与既有结果未改写。

## 1.验收摘要

|验收项目|实测结果|证据|
|---|---|---|
|新增行为及受影响旧接口|57项通过，0失败、0错误、0跳过；35.152秒|[JUnit完整记录](core90_anchored_implementation_evidence_20260911/acceptance_tests.xml)|
|真实H0三方前向|六类各3条，clean及三种弱LEO共72行；identity与完整/fast前向最大误差均为0|[真实H0检查](core90_anchored_implementation_evidence_v2_20260911/real_h0_technical.json)|
|G初始锚点|最大绝对误差3.8147×10⁻⁶，小于`atol=rtol=1e-5`；非并列argmax全部一致|同上|
|身份提取开销路径|identity提取对domain分支调用数为0；前后骨干状态逐tensor相等|同上|
|真实源数据上的可学习性|180个合法L_s physical、360个clean/LEO视图；4RX训练、1RX只作诊断；两个优化步后a非零，骨干不变|[拟合history](core90_anchored_implementation_evidence_v2_20260911/technical_head_fit/head_fit_history.csv)、[几何诊断](core90_anchored_implementation_evidence_v2_20260911/technical_head_fit/geometry_diagnostics.csv)|
|真实证据块|joint=160；t/f/pa分别160，总计480维|[实际block schema](core90_anchored_implementation_evidence_v2_20260911/real_block_schema.json)|
|端到端接线|真实架构合成IQ→角色分离缓存→实际两轮拟合→合成V校准→导出/重载；A0/P1完整阶段与A5融合链通过|JUnit中的pipeline/source_stages测试|
|A6无收益分支|固定动作无source净收益时写出`NOT_ACTIVATED_NO_SOURCE_UTILITY`，不生成A6部署系统|source_stages测试|
|一次独立P0/P1审查|发现并修复OOF错接、P1跨坐标统计接入、阶段候选/seed冒标；定点复核通过，无剩余P0/P1|第4节|

这里的两个真实头部优化步是有界实现测试。`budget_exhausted`如实保留，不解释为已收敛，也不根据这些样本的准确率选择配置。真实V与target IQ均未用于该技术检查；合成V拥有独立physical ID和明确的合成契约，不冒充实际V。

最初的前向证据目录[版本1](core90_anchored_implementation_evidence_20260911/real_h0_technical.json)保留。版本2补齐六类均衡前向样本，以及分离的成本测量；本报告采用版本2。其head计时预先加载冻结头，未混入每次反复加载权重的成本。

## 2.按任务交付

|任务|已实现行为|主要验证|
|---|---|---|
|T1|IdentityAnchor只提取原身份aux；标签margin不进入s0；L_s/V缓存绑定checkpoint、契约、预处理及view recipe；一次批量提取普通CPU tensor|真实H072行、cache角色/唯一键/重载反传、6300样本scene分配|
|T2|QR/tanh有界低秩G，a=0锚点，冻结W/tau；部署保存规范Q/a并预计算类别投影/分母；普通角度对照单独实现|稠密值/梯度参考、正缩放、cond≤2、b梯度、数值失败、类置换和重载|
|T3|physical配对权重、完整分层batch、CE/keep/metric分项与加权梯度、固定预算及稳定窗口；A2/A3/A4/C_angle/C_angle_keep配置分离|6300 physical每轮一次，5/4/3RX均35批且分别360/288/216 feature；真实H0两步训练|
|T4|5折头部OOF、最终全source拟合；嵌套门控按训练RX集合复用10个3RX模型；行键/配置/seed/锚点和fold精确绑定|16个唯一训练集合计数；physical/RX隔离；重排、候选、seed错配拒绝|
|T5|五种概率动作、实际动作utility、类别对称线性门控、H0高间隔保护与逐行回退；固定A5报告的保护阈值也只用其他RX的H0间隔|第三类获胜、端点、保护后utility、零收益、并列、NaN单行回退、类置换|
|T6|冻结系统输出之后才拟合V的单一正温度与阈值；内部尺度不变；物理权重及置信度ties完整保留|argmax不变、禁止重复fit/版本错配、A0/G/A5实际接线、P1单独模式校准|
|T7|七个显式source阶段；A0/A1/G独立读出与A5/A6融合模式分开；白名单部署、实际类映射、参数变更检测；无target串联|CLI配置/参数、A0/A5/P1导出重载、批次拆分、只读状态、候选/seed错标拒绝|
|T8|480维命名证据块的固定尺度；E分阶段拟合共享协方差和source配对观测误差；低秩求解复用；版本2pattern/quality校准|NaN/极值缺失值无影响、稠密边缘化对照、一次solver、同维不同pattern不合并、全缺失defer；旧Gaussian/Schur回归|
|T9|分组fusion/class flow、嵌套source评价、三seed判据、policy与confidence曲线、pattern校准/风险、Schur全类对及分阶段profile；中文阶段报告|已知计数、ties和晋级边界测试；真实CPU profile；独立审查与Git交付|

G首版固定r=4、ρ=log(2)/2，实际H0为160维/6类，故B/b合计644个可训练参数。普通角度对照只训练960个W参数。首轮默认80轮、head seeds=392005/392006/392007、view/split seed=392005。参数起点仍是预登记开发起点，未获得最优性证据。

所有源数据样本级特征、labels、RX/day及physical/view键仅保留在训练侧。部署包仅有身份骨干、专家、类别映射、融合/校准和架构状态，声明`stage=Phase1`，不自动取得Phase2权限。模型参数及融合设置在冻结后发生变化会被拒绝；alpha=0直接采用真实s0的决定。

## 3.实现入口和阶段使用

主入口：[run_core90_anchored_geometry.py](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/scripts/run_core90_anchored_geometry.py)。配置：[core90_anchored_geometry_v1.json](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/configs/core90_anchored_geometry_v1.json)。技术复核脚本：[verify_core90_anchored_implementation.py](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/scripts/verify_core90_anchored_implementation.py)。

已验证解释器：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，PyTorch2.10.0+cu128。以下为参数说明，不表示已经启动正式矩阵：

|阶段|输入|输出/约束|
|---|---|---|
|cache|`--ground`合法ground checkpoint、`--contract`、`--source`角色化IQ tensor、`--role L_s/V`|`--output`新目录；joint供G，`--candidate P1`提取t/f/pa固定尺度块；source tensor仅x/y/physical_ids|
|fit|`--cache` L_s；`--candidate A0/A1/A2/A3/A4/C_angle/C_angle_keep/P1`|独立专家或E；A0/A1不进行梯度训练，其他专家按配置固定80轮|
|oof|L_s缓存；五种训练专家之一及单个head seed|五个4RX专家、一个全source专家、带行键/配置/seed的OOF产物|
|fuse|`--input` A4 OOF目录；`--candidate A5-0/25/50/75/100/A6`|A5不重复拟合专家；A6满足source触发条件才增加嵌套3RX拟合；生成实际动作和分组审计|
|calibrate|`--cache` V；`--input`已冻结的system_state.pt，P1为partial_evidence.pt|匹配最终温度/阈值；检查上游候选、seed、配置、checkpoint/契约/坐标；P1另查fit_manifest和block schema|
|export|`--input`校准阶段system_state.pt|frozen_system.pt，白名单重载、H0身份与当前合法ground一致|
|profile|已校准/导出的系统和L_s IQ tensor|分阶段计时、warmup/repeats、设备、CUDA同步/峰值和共享负载限制|

每次显式执行一个阶段。已完成的上游目录可作为下一未完成阶段的输入；相邻已有manifest用于校验身份。已存在的输出目录一律拒绝，不覆盖历史结果、不静默恢复optimizer或重跑已完成阶段。不引入额外审批、数据复验或receipt链。

CLI的`--help`及`--validate-config`已验证。配置拒绝未知字段、target阶段、未登记候选/seed与首轮deferred开关。正式矩阵的90次基础拟合及条件性30次嵌套拟合仍待实验授权与实际执行；P1和H0已有训练成本分开记录。

完整测试执行方式：

```text
python -X utf8 -m pytest code/tests/test_anchored_cache.py code/tests/test_anchored_geometry.py code/tests/test_anchored_fit.py code/tests/test_anchored_crossfit.py code/tests/test_anchored_fusion.py code/tests/test_anchored_calibration.py code/tests/test_anchored_pipeline.py code/tests/test_anchored_source_stages.py code/tests/test_anchored_reporting.py code/tests/test_partial_evidence_patterns.py code/tests/test_partial_gaussian_head.py code/tests/test_pairwise_evidence.py code/tests/test_evidence_integration.py code/tests/test_evidence_pipeline.py -q
```

测试包括真实架构与受影响旧H1–H5数值/部署路径，含已有CUDA autocast回归。剩余警告仅为旧model.py的autocast弃用提示，本次没有为消除提示而改变历史模型默认行为。

## 4.独立审查及修复证据

执行了计划T9的一次独立审查，并在新pipeline补齐后接续覆盖剩余文件；只对发现的问题作定点复核。

|问题|修复|独立复核结果|
|---|---|---|
|P1：同一cache identity下重排行、换候选/seed仍可能复用旧OOF|OOF携带有序physical/view键、完整FitConfig、head seed、H0方向/尺度绑定；复核fold和expert.train_rx|VERIFIED：重排、候选、seed错配分别拒绝|
|P1：裸E统计可能与另一H0坐标拼接后进入V校准|读取已有E fit_manifest，与V对应的L_s checkpoint/contract/feature身份及实际block schema比较|VERIFIED：跨checkpoint E拒绝校准|
|P1：CLI参数可把A0输入冒标A6或另一seed|使用上游protocol_manifest核对阶段、候选、head seed与行为配置，禁止重标|VERIFIED：冒标A6及换seed拒绝，匹配输入允许|
|P2：G单行非有限数导致整批异常|有效H0行的异常G输出归为alpha0；记录原因，其他正常行继续|VERIFIED：NaN行全部动作回H0，正常行仍执行动作|

独立审查结论为无剩余P0/P1或明确source阶段阻断。审查者未修改文件、未启动正式实验；其复现写出均mock。对应回归已进入本次57项测试。

## 5.实测成本及限制

版本2profile：本地CPU、4线程、每次18条IQ，warmup=1、repeats=3；数值为每批平均值。

|阶段|平均耗时|
|---|---:|
|IQ弱LEO增强|31.905ms|
|身份骨干|19.294ms|
|冻结G头|0.385ms|
|融合与保护逻辑|18.217ms|
|最终温度算子|0.034ms|
|冻结状态序列化到内存|11.830ms|

完整样本、范围和环境见[runtime_profile.json](core90_anchored_implementation_evidence_v2_20260911/runtime_profile.json)。这是受控本地调用，未宣称独占GPU或生产吞吐。温度只测T=1算子，没有实际V拟合；保存项使用内存缓冲，不包含磁盘/网络I/O。融合Python控制逻辑的成本不可忽略，不能据此宣称部署“几乎免费”。正式N607/GPU总墙钟、真实V校准、完整source三seed矩阵尚未测量。

## 6.科学状态与未执行项

- **实现正确性：通过。**设计对应代码、角色/身份防错、数值测试、真实H0前向、可学习性、部署和报告链路已有证据。
- **source开发收益：未判定。**没有执行完整80轮、三seed和全部OOF矩阵，也未用两个技术优化步宣称G优于H0。合成OOF计数训练器只验证16个训练集合及接线，不能作为效果证据。
- **独立确认：未执行。**没有对真实target生成新预测或开展truth连接、评分；旧接口回归使用合成capsule。既有target已参与设计反馈，不能因更换head seed变成独立确认集。
- **P1语义：受限。**命名块mask是特征观测诊断，不能宣称恢复了缺失IQ。all_missing/未覆盖模式defer，不自动等于unknown；source V的pattern分组是校准描述，不是跨域风险认证。
- **R43–R48：仍延期。**有限类别方向修正、状态响应及相关误差、support后验/斜率、联合骨干/H5和独立确认条件均未冒充已实现。

N607仅作普通用户只读核验和下载已有H0/source文件；未启动新训练、未恢复自动监控、未干预健康进程。技术checkpoint保留在本地证据目录并由目录内.gitignore排除；Git交付代码、配置、报告与JSON/CSV/XML证据，不上传训练权重或样本cache。

本报告的确认验收限于IMPLEMENTATION_VERIFIED。正式运行是否形成SOURCE_ANALYZED与最终CONFIRMATION_COMPLETE，仍须各自实际产物和科学权限成立。
