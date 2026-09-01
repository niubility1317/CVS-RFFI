# Phase1 PairBiCAD-CV2设计—实现追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|CV01|设计2|复用ADV3B02双骨干、160维`z_id/z_dom`|`code/SSDG/train_ssdg.py`、`code/model_dual_cvsincnet.py`|verified|真实checkpoint smoke4/4|未改变Phase2导出维度|
|CV02|设计2|严格Clean/LEO同物理样本单前向|`phase1_bicad_xr/pair.py`、trainer|verified|CV2 trainer19/19及全套回归|`backbone_forward_count=1`|
|CV03|设计2|16L+32U普通batch与每4步结构化batch|入口、sampler|verified|batch count RED/GREEN及入口回归|4RX=24L+24U；5RX=30L+18U|
|CV04|设计3.1|覆盖周期ledger|`convergence.py`、入口|verified|coverage计划、实际计数和convergence测试|4RX每4步实际120U；按实际batch计数|
|CV05|设计3.1/4|候选特定科学停止与安全停止状态|`convergence.py`、入口|verified|状态机和入口终态测试|安全上限不标科学完成|
|CV06|设计4|`S_DG`与平台检测|`convergence.py`、`metrics.py`|verified|数值和边界测试|只用source V_cal/LORO|
|CV07|设计4|ReduceLROnPlateau覆盖周期调度|`convergence.py`、入口|verified|scheduler测试|factor0.3/patience3/min_lr1e-6|
|CV08|设计4.4|SWAD窗口和权重平均|`swad.py`、入口|verified|窗口/平均/拒绝测试|0.50pp准入与floor保护|
|CV09|设计3.2|detached判别器与编码器分步更新计划|`adversarial_game.py`、trainer、入口|verified|反向计划和入口可达测试|一次backbone前向|
|CV10|设计3.2|双时间尺度参数组|`adversarial_game.py`、入口|verified|optimizer组测试|判别器LR为1.5倍|
|CV11|设计3.2|动态GRL双比率控制|`gradients.py`、入口|verified|双控制器及入口测试|conditional和TXadv分开|
|CV12|设计3.2|局部冲突梯度投影|`gradients.py`、trainer|verified|局部allowlist和历史D6回归|排除domain/adversarial heads|
|CV13|设计3.3|低权重pair identity hinge候选|`config.py`、trainer|verified|candidate diff和trainer测试|epsilon0.05、权重0.02，VICReg/delta关闭|
|CV14|设计3.4|Margin-REx/CVaR|`tailguard.py`、trainer|verified|损失和有限性测试|仅CV2-T2/T3|
|CV15|设计3.4|困难组采样上限30%|`tailguard.py`、trainer|verified|cap测试与审计|不取代均衡样本|
|CV16|设计6|冻结`CV2-B0…CV2-T3`共12个唯一配置|`config.py`|verified|registry和diff测试|D0/T0静态继承B3，历史D0不变|
|CV17|设计6|24行fold1/fold8/seed392002矩阵|新launcher|verified|dry-run24行核对|每GPU最多两个槽位|
|CV18|设计6|不可覆盖row root和队列调度|新launcher|verified|collision/GPU调度测试|16并发槽位、8行排队|
|CV19|设计7|主线与TailGuard晋级分析|新analyzer|verified|合成artifact测试|同row比较|
|CV20|设计9|四场景严格artifact闭合|`metrics.py`、launcher、analyzer|verified|closure与worker测试|真实评估后才ARTIFACTS_COMPLETE|
|CV21|协议|Phase1 source-only fail closed|入口、launcher|verified|聚焦负测|Phase2/target/query/truth禁止|
|CV22|验收|真实checkpoint无query smoke|smoke脚本|verified|N607真实P1-F1-S392002 checkpoint PASS|新鲜optimizer step、严格恢复、四场景、无query|
|CV23|发布|最小预登记与N607每GPU两个实验|报告、launcher|verified|release SHA、远端编译和启动回读|GPU0因无关任务冻结1槽，其他卡2槽|

## 当前计数

- verified：23
- deferred：0（设计级延期项不进入本轮实现清单）
- rejected：0
- blocked：0
- pending：0

实现与发布追踪项已闭合；正式24行当前为`RUNNING`，科学结果和artifact闭合需等待训练完成后另行分析。

Ruling：新候选统一使用`CV2-`前缀，因为现有注册表已经冻结历史`D0-D3`；若复用短ID会静默改变既有实验语义。

Ruling：为允许24行并发且不读取运行中冠军，`CV2-D0`和`CV2-T0`均在发布前静态复制`CV2-B3`配置；D/T增量只修改各自声明机制。

## 独立P0/P1审查闭合

- 配置、静态矩阵和launcher审查：`NO_BLOCKING_FINDINGS`。
- 覆盖率审查发现结构化步导致U覆盖高估；已改为累计实际`x_u.size(0)`，定点复审`RESOLVED`。
- 结构化loader审查发现旧32U硬编码会拒绝24U/18U；已允许冻结的18/24/32U入口，定点复审`RESOLVED`。
- 动态终态审查发现真实stop update会被静态6500验收拒绝，且非整点安全上限缺少终评；已绑定selection/curve的真实停止步并强制计划末步评估，定点复审`RESOLVED`。
- 最终本地验证：`code/tests/phase1_bicad_xr`共454项通过；仅3条既存PyTorch autocast弃用警告。

## 2026-09-01 E200修复版重新开放

旧追踪表中的`verified=23/pending=0`只表示旧冻结实现具备可运行入口，不能继续解释为设计报告的严格完整实现。针对实际训练路径反向检查后，旧run已按用户要求停止并保留artifact；修复版以[PairBiCAD-CV2-E200修复版冻结说明](../docs/superpowers/specs/2026-09-01-pairbicad-cv2-e200-repair.md)为当前验收口径。

|ID|修复要求|当前状态|严格验收|
|---|---|---|---|
|E200-01|全部候选完整200epochs，无update/coverage/24小时正常提前终止|verified|固定E200循环、忽略科学停止请求、epoch200断言及launcher命令测试|
|E200-02|strict Pair每个物理样本都是真实LEO，三种LEO从早期可达|verified|真实LEO fail-closed测试及epoch1三场景可达测试|
|E200-03|CoverageLedger接入真实sample ID和L分组|verified|标签可见批次使用五元物理ID；MUSE U批次使用与冻结subset `selected`同源的不可变`base_index`，不恢复TX真值；U唯一覆盖和L组暴露行为测试|
|E200-04|coverage warmup后再Plateau|verified|warmup前不step、warmup后Plateau参数和状态测试|
|E200-05|`no_early_freeze`成为运行约束|verified|逐训练步`requires_grad`审计和冻结参数fail-closed测试|
|E200-06|`adversarial_two_time_scale`成为显式运行分支|verified|入口直接消费该开关及契约，独立optimizer与1.5倍LR测试|
|E200-07|pair梯度比例不超过5%|verified|raw/effective ratio、scale和实际effective weight行为测试|
|E200-08|困难组权重质量不超过30%|verified|T2/T3真实Margin-REx/CVaR使用bounded权重的行为测试|
|E200-09|动态GRL消费四类反馈|verified|判别器准确率、TX margin、对抗梯度比、冲突信号及独立有界剂量测试|
|E200-10|`V_cal/V_select`物理隔离及职责分离|verified|确定性物理ID分割、重叠fail-closed和角色审计测试|
|E200-11|final/EMA/SWAD一次`V_select`选择|verified|EMA更新、候选集合和单次selection不可反馈测试|

当前计数：verified=11，deferred=0，rejected=0，blocked=0，in_progress=0。这里的`verified`表示本地实现与行为测试已闭合；N607真实200epoch运行、每行机制遥测和四场景artifact仍须由新run给出运行时证据。

独立P0/P1审查发现并修复一项P1：分离优化器入口原先仍由`detached_adversarial`间接触发，现已改为直接消费`adversarial_two_time_scale`及其运行契约；同时把新launcher默认run ID与本报告冻结ID对齐。定点RED测试先失败，修复后通过。最终本地回归为470项全部通过，仅3条既存PyTorch autocast弃用警告；所有改动模块通过`py_compile`和`git diff --check`。

## 2026-09-01 r3最终选模包装器故障与r4修复

- r3在两条`CV2-B0`行完成200epoch后进入final/EMA/SWAD的`V_select`评估时，重复触发`_forward_unimplemented() got an unexpected keyword argument 'y_tx'`；0/24行形成完整artifact，按系统技术失败规则精确停止并保留partial artifact。
- 根因是训练期评估传入可调用的底层模型，而最终候选评估传入已加载候选状态的`BiCADXRTrainer`；该类继承`nn.Module`但缺少`forward`，导致统一source-LORO入口无法执行。
- 修复在`BiCADXRTrainer.forward`中只把推理参数转发给其当前`self.model`，使final/EMA/SWAD候选继续使用已严格加载的trainer状态，同时不改变训练损失、优化器、数据划分、候选配置或source-only边界。
- 新增行为级回归测试直接用`CV2-B0` trainer调用真实`_evaluate_bicad_xr_source_loro`。修复前精确复现远端异常，修复后完成推理且底层模型仅调用一次。
- r4继续冻结同一24行、seed392002、fold1/8、完整200epoch矩阵；仅修复运行时评估接口并使用新不可覆盖run ID。
