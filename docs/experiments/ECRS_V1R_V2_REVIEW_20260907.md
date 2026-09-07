# ECRS-V1R/V2第二轮实现与训练审查

审查基线：`93545228ff7da7f1b6b808d8bd02c9bd4c58bf63`。依据原始《ECRS_V2_optimization_spec.md》、2026-09-06实施计划和本机`项目.md`。只处理本地实现与验证，不启动或干预N607任务。

|ID|来源|要求/审查点|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|R01|设计§2/5/6/9|空有效集合不产生伪optimizer/EMA/LR更新|ecrs_training.py,ecrs_runtime.py|verified|带动量和weight decay的AdamW反例及修复验证|保留有效零hinge正常更新，与空集合区分|
|R02|设计§6/9|恢复必须同训练配置|ecrs_config.py,train.py|verified|6类参数改变拒绝、输出路径变化允许、真实恢复检查|加入训练语义版本；旧语义只能新run加载模型权重|
|R03|设计§5/6/8|U未启用时不取样；真实LEO才执行U目标|train.py,ecrs_config.py|verified|关闭U、p=0重复clean两条真实入口；缺增强配置检查|区分取样覆盖和成功LEO更新；不改变既有p日程|
|R04|计划§11/设计§9|source融合净收益完整分层|ecrs_evaluation.py|verified|同RX救回/破坏相抵反例、全局和分层守恒|原计划门槛明确为RX，不额外增加day门槛|
|R05|设计§2/5/8|V1R响应CE复用真实clean配对输出|train.py|verified|检查实际输出来源，B2/B2-V1入口回归|原入口用增强main；现在复用已计算clean，不增加前向|
|R06|设计§3/4|真实参考对齐、TX/RX/view probe与区间覆盖|实施报告|deferred|缺真实IQ与合法probe结果|不能由合成正确性推导科学收益|
|R07|设计§2/8|动态gate/compile条件性后置|ecrs_config.py|deferred|维持原有明确拒绝|不因代码审查扩大实验范围|
|R08|设计§9|非默认辅助记忆的精确恢复声明|ecrs_config.py|rejected|未持久化辅助状态时明确拒绝exact resume|默认15行不启用这些可选状态；不禁止新run训练|

共8项：verified=5，deferred=2，rejected=1，blocked=0。这里统计本轮审查项，不替代前一报告的32项设计追溯。

## 结论与具体问题

本轮确认并修复了训练执行语义问题；没有证据支持立即改变0.15/0.05/0.03/1.0这组候选损失系数。此前127项检查通过只证明当时覆盖的行为，并不能证明未覆盖的边界没有错误。

**P1：空集合损失仍导致参数更新。**原先总损失初始化和融合零项依赖`z_resp`的空切片，产生连接encoder的零梯度。有效配对为0时，AdamW仍会施加weight decay或已有动量，随后有效step时钟和EMA也可能推进。复现中`valid_count=0`，encoder最大参数变化约0.00055349。修复后空集合/关闭/零权重项不接入总损失图；有合法triplet但hinge恰为0的情况仍保留原来的正常优化语义。辅助单项函数可以保持可微零返回，但实际总损失只接入执行过的目标。

**P1：恢复检查不完整，且旧训练语义不能冒充原run恢复。**旧检查只枚举部分前缀，允许改变普通增强、label smoothing、基线loss和其他日程。现在比较所有双方共有的训练参数，仅放行明确的运行环境及输出路径；runtime增加训练语义版本。此次修复前的runtime包会明确拒绝exact resume，可在新run通过model-only初始化继续研究，不能保留原run名称宣称严格续训。

**P2：U数据使用与日志存在歧义。**V2即使未启用U目标也取U批次；日程跳过增强时，clean复制仍进入单向EMA损失，被记成U配对执行。现在没有U或其他实际无标签目标时不取U；仅`applied=true`才执行LEO一致性，p=0时有效U数为0。队列覆盖仍表示取到的物理样本，不表示成功LEO更新，日志写明这一区别。缺少同步增强而启用正权重U目标时提前报错。

**P2：分层准确率不足以还原rescue/harm。**同一个RX可能救回1个、破坏1个，净收益为0；仅准确率差看不到这种抵消。新增`gain_by_group.per_rx/per_day/per_rx_day`，分别保存救回数、破坏数、样本数及净收益。source truth-blind输出不生成这些依赖真值的统计。原计划B6门槛是RX保留，不因本次审查擅自新增day数值门槛。

**P2：V1R响应CE用错配对输入来源。**使用普通增强时`out_main`对应增强输入，而旧物理损失已经另算了真正clean的`out_ecrs_clean`。现在响应CE直接复用后者，保持clean/LEO来源一致。入口回归覆盖E1可运行；此处E91响应CE的来源同时通过源码分支检查，不把E1训练说成E91收敛测试。

## 训练策略与损失设置核对

|内容|当前实际设置|审查判断|
|---|---|---|
|V1R|保留旧课程与物理目标；独立响应头；raw额外CE默认0|属于修复路线，不能当作完整SSDG-Core90复现|
|V2基础目标|响应CE=0.15；跨RX=0.05；U=0.03|跨RX/U独立开关；系数是候选值，没有真实数据最优性证据|
|响应CE归一化|有标签clean/LEO拼接后按有效样本均值|不是两个CE分别乘0.15再相加；掩码排除无标签样本|
|跨RX|margin=0.2；同TX不同RX正例；同RX/day/view异TX负例；先每anchor平均再跨anchor平均|没有合法正负集合时不训练该项；稀疏batch可能降低有效anchor数，应看日志|
|U|LEO学生→clean EMA，decay=0.99，teacher无梯度|现在只计实际施加的LEO；CE开启是设计中的防坍塌支撑，不能把U-only消融称为具备同等保障|
|固定融合|rho=0.05；零初始化切向投影；独立融合头；CE=1.0；默认E91启动|默认融合梯度不回传raw/encoder；raw基线继续训练，并未机械冻结全主干|
|S-LR/S-LEO/S-BATCH|分别改变有效step时钟、提前CE、结构化采样|保持独立消融；不组合后再把收益归因于单个模块|
|默认LEO日程|E1–40:p=.30；E41–90:p=.60；E91+:p=.80；默认辅助CE从E80开始|维持既定协议；本轮只修正“跳过增强”对应的U损失执行|
|source选模|固定raw或fused路径，clean+三LEO，200轮48个检查点|没有改选头、调阈值、访问target或增加额外门槛|

以下限制仍需明确：V1R保留历史全模型U/clean额外前向，这可能影响BN和耗时，因此B2不是相对B0的纯分类头消融；V2非融合行默认按raw选模，不能由raw平台证明响应encoder收敛；同row融合门槛不能替代独立匹配B0的退化检查。完整TX/RX/view probe、真实参考对齐、区间覆盖和真实资源结果仍缺失，当前最高科学风险是“响应是否学到跨RX的TX信息并真正补充raw”，不是代码是否能运行。

非默认的全模型EMA/SWA/SWAD、prototype、Meta-SSL教师/原型及平滑GroupDRO辅助记忆目前没有完整恢复持久化，新增exact-resume检查会明确拒绝这些配置，不再静默重建。它们仍可在新run启用，但不能声称已具备本轮验证的严格恢复能力。默认展开矩阵没有启用这些状态。

## 验证与交付

本轮57项针对性检查通过：51项损失/runtime/配置/评估检查，1项新增缺增强配置检查，以及5条真实训练入口检查。入口包括关闭U、clean重复视图、B2/B2-V1回归和非融合保存/恢复；合成数据不包含target观测，不产生真实识别收益结论。

实际执行入口（使用已验证`ssr-gpu`解释器）：

```text
python -X utf8 -m pytest code/tests/test_ecrs_second_review.py code/tests/test_ecrs_training_helpers.py code/tests/test_ecrs_runtime.py code/tests/test_ecrs_revision_evaluation.py code/tests/test_ecrs_v1r_v2_config.py -q
python -X utf8 -m pytest code/tests/test_ecrs_second_review.py -q -k missing_leo
python -X utf8 -m pytest code/tests/test_ecrs_revision_train_entrypoint.py -q -k "sampling_and_applied or nonfusion or B2"
```

本次是既有ECRS路线的设计语义修复，不是严格全量设计性能复现；未启动N607，未重新解释或覆盖历史实验。
