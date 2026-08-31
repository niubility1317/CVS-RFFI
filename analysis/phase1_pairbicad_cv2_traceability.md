# Phase1 PairBiCAD-CV2设计—实现追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|CV01|设计2|复用ADV3B02双骨干、160维`z_id/z_dom`|`code/SSDG/train_ssdg.py`、`code/model_dual_cvsincnet.py`|pending|真实checkpoint smoke|不得改变Phase2导出维度|
|CV02|设计2|严格Clean/LEO同物理样本单前向|`phase1_bicad_xr/pair.py`、trainer|pending|pair和trainer测试|已有基础，需核对同尺度语义|
|CV03|设计2|16L+32U普通batch与每4步结构化batch|`sampler.py`|pending|sampler测试|fold自适应24L/30L|
|CV04|设计3.1|覆盖周期ledger|`convergence.py`|pending|RED/GREEN测试|新模块|
|CV05|设计3.1/4|候选特定科学停止与安全停止状态|`convergence.py`、入口|pending|状态机测试|不得把安全上限写成完成|
|CV06|设计4|`S_DG`与平台检测|`convergence.py`|pending|数值和边界测试|只用`V_cal`|
|CV07|设计4|ReduceLROnPlateau覆盖周期调度|`convergence.py`、入口|pending|scheduler测试|至少两次LR下降|
|CV08|设计4.4|SWAD窗口和权重平均|`swad.py`|pending|窗口/平均/拒绝测试|V_select只在候选形成后使用|
|CV09|设计3.2|detached判别器与编码器分步更新计划|`adversarial_game.py`、trainer|pending|反向计划测试|一次backbone前向|
|CV10|设计3.2|双时间尺度参数组|`adversarial_game.py`、入口|pending|optimizer组测试|默认1.5倍|
|CV11|设计3.2|动态GRL双比率控制|`gradients.py`|pending|控制器测试|labeled和TXadv分开|
|CV12|设计3.2|局部冲突梯度投影|`gradients.py`、trainer|pending|冲突/非冲突测试|只允许identity尾部|
|CV13|设计3.3|低权重pair identity hinge候选|`config.py`、trainer|pending|candidate diff测试|VICReg/delta保持关闭|
|CV14|设计3.4|Margin-REx/CVaR|`tailguard.py`|pending|损失和有限性测试|只在结构化batch|
|CV15|设计3.4|困难组采样上限30%|`tailguard.py`、sampler|pending|cap测试|不取代均衡样本|
|CV16|设计6|冻结B0-B3/D0-D3/T0-T3配置|`config.py`|pending|registry和diff测试|不得动态读取冠军|
|CV17|设计6|24行fold1/fold8/seed392002矩阵|新launcher|pending|dry-run测试|每GPU两个槽位|
|CV18|设计6|不可覆盖row root和队列调度|新launcher|pending|collision/GPU测试|16并发槽位|
|CV19|设计7|主线与TailGuard晋级分析|新analyzer|pending|合成artifact测试|同row比较|
|CV20|设计9|四场景严格artifact闭合|`metrics.py`、analyzer|pending|closure测试|不得只用LEO均值|
|CV21|协议|Phase1 source-only fail closed|入口、launcher|pending|聚焦负测|Phase2/target/query/truth禁止|
|CV22|验收|真实checkpoint无query smoke|smoke脚本|pending|N607前本地/远端smoke|严格恢复|
|CV23|发布|最小预登记与N607每GPU两个实验|报告、launcher|pending|preflight和启动回读|普通账户|

## 当前计数

- verified：0
- deferred：0（设计级延期项不进入本轮实现清单）
- rejected：0
- blocked：0
- pending：23

最高风险项为CV09—CV12：双向对抗的分步反向和局部梯度保护必须保持一次backbone前向，并且不能污染TX主梯度或shared stem。
