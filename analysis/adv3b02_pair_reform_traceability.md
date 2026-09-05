# ADV3B02两份设计报告落地追踪

目标：保持双骨干、身份/域主线和clean+LEO监督，修正约束语义并提供互斥的点对齐/身份安全区域候选。依据用户2026-09-05提供的两份报告；第二份对候选优先级的调整优先。代码基线2a56f851。

本轮交付代码、可执行配置及性质/集成测试。真实数据性能、隔离GPU效率、多seed晋级和P5首异常复现需要运行证据，不由短测试代替。报告明确列为后续探索的机制不叠入默认候选。

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|R01|一2.1、二2|四个同起点差分、冲突方向排除|selective_tangent.py、train_ssdg.py|verified|恒等干预及冲突测试|Route独立对照|
|R02|一2.2–2.3|可靠度接纳率、训练ID稠密索引、重复合并、恢复|orbit_teacher.py|verified|可靠度单调、缺失、恢复测试|缓存可选|
|R03|一2.4、二4.2|缓存只用于历史特征，不伪造分类票或提高物理质量|pair_reform_runtime.py、train_ssdg.py|verified|缓存命中/未命中测试|真实教师分类输出独立|
|R04|一2.5、7|关闭模块不构图、不前向、不更新状态|pair_reform_runtime.py、train_ssdg.py|verified|调用计数与状态测试|按有效阶段和权重判断|
|R05|一2.6、3|确定性成对差分、先采样、稳健量纲、Route-only|train_ssdg.py、selective_tangent.py|verified|模式恢复、子集与公式测试|历史公式显式区分|
|R06|一4、二9.1|共享clean/LEO输出、统一U分类目标、identity-only教师|train_ssdg.py|verified|实际训练helper集成测试|共享视图是方法改变|
|R07|一4.3、二9.2|点对齐/不对称目标与安全区域互斥|pair_reform.py、train_ssdg.py|verified|梯度与路径测试|默认优先B，A为对照|
|R08|二1、3|干预名称/方程/单位/位置审计、时间连续、独立噪声方向|deployment_orbit.py|verified|算子等价与步长收敛测试|探针不宣称真实TX生成|
|R09|二4|r_phys/q_cls/q_cache分责、未知元数据、固定batch分母|pair_reform.py、train_ssdg.py|verified|均匀分类、质量和权重缩放测试|U真值不可见|
|R10|二5|固定无偏置余弦头安全半径、错误锚点关闭、U固定容差|pair_reform.py|verified|几何充分条件与反向测试|不宣称泛化保证|
|R11|一8|AMP尺度、状态提交时序、随机流/恢复、异常信息|train_ssdg.py|verified|FP32参考与跳步测试|不调整保护阈值|
|R12|一9、二9|少量可归因配置、source-only、同预算、配置及入口|configs、analysis|verified|配置与运行入口检查|不恢复A0、不自动晋级|
|R13|一8.1、二9.3|P5首个非有限算子定位|既有失败artifact|deferred|需要原失败batch/模型/RNG|不能由其他缺陷倒推根因|
|R14|一9.4、二9.4|真实效率/性能/多seed确认|后续实验报告|deferred|正式训练及隔离GPU测量|本轮实现不代表收益|
|R15|二6–8|同RX差异保持、条件域头、稀疏U|后续独立候选|deferred|报告明确后续探索|不与安全区域堆叠|
|R16|一6–7|RX/Style/tail/subspace默认关闭；编译/早停后置|候选配置|verified|默认配置检查|保留历史代码与原型导出|

## 执行计划

- [x] 1.物理算子及方向语义：先写连续性/注册表测试，再修正独立算子与显式版本。
- [x] 2.缓存与目标：先写可靠度、无效视图、恢复测试，再实现稠密批量读写与有效目标。
- [x] 3.成对主链：实现固定头安全半径、三类权重、复用前向、互斥候选和状态事务，运行训练helper测试。
- [x] 4.配置与交付：检查逐项覆盖，针对变更测试、独立审查、显式stage并提交push，远端OID读回。

采用设计报告追踪、writing-plans、using-git-worktrees、测试先行及subagent-driven-development流程；复用已核实干净的隔离工作树。任务内分配不同文件，主Agent负责训练入口整合。不创建额外实验许可门槛。

## 实际落地范围

原文已保存为[第一份报告](design_inputs/adv3b02_reform_report_1.md)和[第二份报告](design_inputs/adv3b02_reform_report_2.md)。按第二份报告的优先级实现A点对齐、B安全区域和独立不对称候选；C同RX差异保持、条件域头和稀疏U继续后置。没有把全部建议堆叠为一个新默认。

入口为`--pair_reform point|safe|asymmetric`，默认仍为`off`，旧实验路径保留。所有新候选维持双骨干、MUSE有标签local监督与分类原型、原有域主线和真实LEO监督。新U路径只使用剥离TX字段后的物理元数据；统一软分类目标不调用旧三态路由。预热期仅做clean学生域训练，不生成额外U视图或教师前向。缓存、原型、先验及梯度控制历史仅在优化步成功后提交。

B使用实际EMA模型的无偏置CosFace头及同一`feat_joint`空间；有标签错误锚点关闭辅助约束，真实CE保留。U使用固定容差，不宣称类别安全保证。安全区域与Tangent、Route、缓存互斥。共享的是实际监督输出：`concat_masked`复用已有clean和LEO两次输出；`concat_full`复用拼接输出。共享视图和统一U目标属于方法改变，不宣称与历史多视图目标等价。

修正算子独立提供连续零延拓STO/采样时钟、固定独立噪声、滤波/多径/AGC及去重探针API。当前方向候选刻意只使用STO nuisance和PA/IQ gain接收波形探针；其他API通过性质测试但未在候选中启用。公共相位、净频偏和采样时钟不接受相反身份路由要求。探针不是新TX身份或已验证的发射端生成机制。

`r_phys`只来自现有物理元数据，缺失时显式标未知；默认0.5是工程权重假设，不是测量或恢复概率。`q_cls`是新鲜教师概率的置信度/间隔/JS启发式，不经过V校准。`q_cache`只控制历史特征目标，不能提高当前波形质量或伪造教师分类票。

## 待运行矩阵与使用

配置：[phase1_adv3b02_pair_reform_pending.json](../configs/phase1_adv3b02_pair_reform_pending.json)。生成工具：[pair_reform_dry_run.py](../tools/pair_reform_dry_run.py)，只输出JSON命令，不启动进程。

|候选|相对M0的改变|状态|
|---|---|---|
|M0/A_POINT|修正点对齐底座|PENDING_NOT_RUN|
|M1|Tangent开启|PENDING_NOT_RUN|
|M2|Route开启，Tangent关闭|PENDING_NOT_RUN|
|M3|Tangent与Route开启|PENDING_NOT_RUN|
|B_SAFE|点对齐替换为安全区域|PENDING_NOT_RUN|
|POINT_MEMORY|只开启历史特征缓存|PENDING_NOT_RUN|
|ASYMMETRIC|只约束LEO学生到clean主导教师目标|PENDING_NOT_RUN|

继承ManySig源/目标轴、seed392005、E200、130+70及CORE90初始化。共同将`sat_cons_start_epoch`从原脚本E80改为E1，保证E11成对目标能够复用LEO监督；这是所有新候选共同的显式改变，不能与历史分数作无混杂比较。新增系数为待验证候选，不冒称历史梯度等价权重。CORE90已见过源RX，当前设计不声称未见源RX验证。

示例（替换实际数据与checkpoint路径）：

```text
python tools/pair_reform_dry_run.py --root E:/path/to/repo --python E:/path/to/python.exe --dataset E:/data/ManySig.pkl --checkpoint E:/checkpoints/CORE90.pth --output-root E:/runs/pair --rows A_POINT,B_SAFE
```

## 验证与边界

性质测试覆盖算子连续性、重复/冲突方向、固定头几何、固定batch分母、未知质量、缓存隔离、恢复、确定性模式恢复及关闭分支调用计数。真实双骨干前向/反向与合成source-only主循环测试覆盖masked监督、U目标、实际优化步、checkpoint及日志读取。合成主循环只做一个batch，不进入晋级；heldout样本数为0，不能解读为评测完成或新准确率。

新日志提供L/U分开的质量与权重、额外前向样本数、RX×物理权重档统计；B额外记录固定教师头margin、有效锚点、安全半径内比例和翻转率。档位是`r_phys`权重档，不冒称实测信道强度。训练既有跳步与梯度投影日志继续保留。

发现并修复AMP放大梯度归约误判、预热多余前向、pair路径旧日志未初始化变量和checkpoint中frozenset导致受限读取失败。新run首异常保存`pair_first_anomaly.pth`，记录损失分组、首非有限梯度参数、源物理ID、当前IQ与状态；它是失败后观测快照，不是完整前向RNG重放，也不定位首个失败算子。历史P5根因仍需原始失败现场。

未运行N607正式实验、隔离GPU计时、source配对多seed、条件RX探针或新数据评分。最高风险仍是真实训练稳定性与实际收益；本次修复不能解释历史P5，也不能据此晋级B。交付是按报告核心公式和优先级实现的可运行候选，包含明确的工程选择与后置项，非所有探索建议的无差别复刻。

## 最终本地核验

2026-09-05：184项测试通过，4条既有AMP弃用警告。追踪项共16项：verified=13、deferred=3、rejected=0、blocked=0。7行真实训练入口dry-run、真实双骨干前反向及1batch主循环包含在测试中。

执行命令：

```text
conda run -n ssr-gpu python -X utf8 -m pytest -o addopts= tests/test_pair_reform_operators.py tests/test_pair_reform_directions.py tests/test_pair_reform_memory.py tests/test_pair_reform_objectives.py tests/test_pair_reform_runtime.py tests/test_pair_reform_training.py tests/test_pair_reform_execution.py tests/test_pair_reform_telemetry.py tests/test_pair_reform_configs.py tests/test_pair_reform_main.py tests/test_pair_failure.py tests/test_daot_rx_v2_control.py tests/test_daot_rx_v2_core.py tests/test_daot_unlabeled_trust.py tests/test_adv3b02_daot_stn.py -q
git diff --check
```

Git交付使用当前独立分支，不合并默认分支；最终提交OID与远端读回结果见本次任务回复。
# E11技术恢复补充（2026-09-05）

后续r2的B_SAFE检查暴露EMA通过.data更新导致分类头缓存未失效。r3修复Parameter版本更新，新增先失败后通过的缓存生命周期回归及每步EMA smoke，仅恢复B_SAFE3行。旧21行继续；新旧版本比较标记NO_PROMOTION，不能把缓存修复混同为机制增益。详见`automation_reports/CV-SincNet/phase1_adv3b02_safe3_manysig_e200_20260905_r3/report.md`。

原24行中12行在机制启用时暴露AMP头校验和缺失采样率字段错误；修复前5个回归测试复现，修复后通过。真实parser＋CUDA AMP＋L/U checkpoint smoke替代原人工namespace／纯CPU覆盖。新run仅恢复失败12行、其余健康12行继续；完整映射与证据见`automation_reports/CV-SincNet/phase1_adv3b02_pair12_manysig_e200_20260905_r2/report.md`。实际性能、多seed结论仍待E200完成。
