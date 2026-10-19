# R16：IQ 重建目标与 FFT 辅助目标的配对消融

持续目标不变：单一恢复 IQ 供原 PAMF 全部分支；训练完整利用合法 L/U，不读取 TX 标签或 F 权重/输出。冻结原 F50/defaulthead 验证输入对应 TX。正式 low urban≥70%，其余五个受损场景各≥同 F/seed 原始 raw clean−5pp；clean 损伤另报。N607 默认、多 GPU、其他 owner 健康任务保留。

R15 的源证据表明 TCN 的训练总损失更低、FFT 项更低，却有更差的 IQ NMSE 与 TX 保真。这只是相关证据，不能归因于 FFT；R16 直接检验移除 FFT 辅助目标的影响，不改结构、随机初始状态、数据顺序或预算。复用 R15 两组已完成的 FFT 对照及其 source-only 评分；新增两组 scratch prior，仅 IQ NMSE。每种结构内与 R15 比较才是严格配对，不能把不同结构称为相同隐藏权重。

两组新 prior 为原 U-Net 和原 TCN，冻结 R07 coarse/action、噪声估计、原 SCALE² 与 6CG。只加载 R07 完整合法来源，没有 R10/R15 训练后 prior 继承。加载父后重置 model seed2026092701；实际入口比较每种结构完整初始化状态与 R15 epoch0 smoke checkpoint 逐值一致。比较旧 epoch0 只用于验证配对，不作为初始化来源。相同 data seed2026102001、原 R07 20 轮缓存、6300L/56700U、15%clean/85%residual、20轮/4940、batch256、AdamW2e-4→2e-5、warm247、WD1e-4、clip1、FP32。每行 L126000/U1134000 曝光，组件 E60，F50 历史 E50 另列。IQ-only 训练不计算 FFT 项；冻结评估仍报告 FFT 误差和原 PAMF 全分支。

U-Net IQ-only 为唯一主候选，TCN IQ-only 为源机制对照，不能因结果较好改为晋级。固定最后20轮，每行七种 source V 预测全固定后独立CPU评分。主候选需同时相对 R15 U-Net FFT 对照及原 R10 native：六场景平均 NMSE≤1.05倍、cleanNMSE≤.02；low/mid urban各≥+.5pp；clean相对原raw下降≤.1pp；最差低urban TX变化≥−1pp；其他五场景最差变化≥−.5pp。门槛在新训练前固定。源 V 标签只用于单独披露的选择，整个选择过程不称完全无标签。

全部通过才自动正式确认两行：新 U-Net IQ-only 与固定 R15 U-Net FFT 训练后权重；后者仅在独立 target-only 行读取，不继续训练。两行原168000物理记录×七视图，各行全部预测固定后独立truth-last评分。否则0target，不救跑TCN或U-Net FFT。正式基准此前暴露，不称新的盲测，不依据目标反馈设计。固定目标clean基准84.6761904762%，另外五场景下限79.6761904762%。

资源沿用池0/1/5/7择空卡、最多两条本任务worker，每卡一条；全owner cap2/free≥12GB、capacity4共享锁到CUDA PID可见。一个launch owner、不可覆盖run/release、普通N607账户；技术错误停失败行并保留输出，健康任务继续，pending暂停，无自动重试。入口真实2Ls/原父+新checkpoint+原F smoke通过立即训练。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A16-01|R15源损失/身份矛盾|IQ-only与FFT配对；不计算训练FFT|training.py/design.py|verified|实际IQ损失梯度/禁FFT及训练注入接口检查通过|不提前归因|
|A16-02|无监督/来源|全L/U、0TX/F/target训练、只继承冻结R07|runtime.py/shared training|pending|来源负测、实际曝光日志|不是无clean配对学习|
|A16-03|配对公平|同结构初始化全状态/顺序/预算与R15一致|controller.py|pending|实际epoch0逐值比较、20/4940日志|复用对照不新增预算|
|A16-04|选择权限|固定U-Net主候选及双护栏，TCN不可救跑|evaluation.py|verified|source角色/最差TX/非法target及旧完整护栏负测通过|V选择标签单独披露|
|A16-05|冻结F与正式对照|新IQ和旧FFT不同schema正确加载；七预测后truth|evaluation.py/runner.py|implemented|两loader全七视图/原F版本计数接口通过，真实父+F smoke为启动第一步|默认头不拟合|
|A16-06|执行保护|单owner、共享容量、不可覆盖、N607发布|controller.py/runner.py|implemented|20项聚焦检查、compile/help、一次P0P1无阻断、N607 preflight通过，待poststate|无额外gate|
|A16-07|完整证据|三郊区三urban/clean、TX/RX/day/损坏/成本|run报告/汇总|pending|完整40epoch和条件正式结果|不以source替代正式70/5|
|A16-08|外部有效性|多seed/独立物理族/逐细节真实|后续确认|deferred|单seed首轮证伪，候选通过再扩展|完整持续目标不缩减|
