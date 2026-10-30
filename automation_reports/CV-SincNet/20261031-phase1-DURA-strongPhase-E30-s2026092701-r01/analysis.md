# 强频点相位保真：两景改善，但联合验收未通过

相对同预算E30 physical读出，本版mid urban提高0.347619个百分点（+584条正确），low urban提高0.604167个百分点（+1015条）；clean及其余四景下降，六星地仍全部低于默认DURA。六景3pp与clean保持均未达到，不更换默认。

## 正式结果与完整分组

同一原F50/defaulthead、同168000物理样本×7景，raw clean84.6761904762%。全部预测固定后独立truth-last；再次独立重算126个总体/TX/RX/day、MacroF1、纠错/损伤计数一致。完整分组及相对E30控制的变化见results.json。单seed、既有暴露工程基准，不称独立盲测或多seed稳定改善。

|场景|本版准确率%|相对E30 physical（百分点）|相对默认（百分点）|距raw clean（百分点）|
|---|---:|---:|---:|---:|
|clean|84.436905|-0.405357|-0.241071|0.239286|
|practical_high|79.075595|-0.942262|-3.749405|5.600595|
|practical_mid|78.014286|-0.726190|-3.546429|6.661905|
|practical_low_suburban|76.792262|-0.608929|-3.266071|7.883929|
|practical_high_urban|78.345238|-0.616071|-3.400595|6.330952|
|practical_mid_urban|73.198810|+0.347619|-2.287500|11.477381|
|practical_low_urban|67.311310|+0.604167|-1.578571|17.364881|

clean相对raw纠回290条、损伤692条，净−402，下降0.239286个百分点；较E30 physical少681条。旧clean最多1pp容忍通过，但不等于本目标clean不降低。low urban相对默认净少2652条，距rawclean仍17.364881个百分点，不能因比另一E30提高就晋级。六景3pp=False，clean不低于raw=False，R17逐景最多下降1pp=False，三个条件分别判断。

## 已执行的训练变化

仍noVar的IQ1/prior0.1/STFT0.05/action0.05、variance辅助0，从零单阶段30轮/7410步。唯一新增损失为每条sourceclean最强64 FFT点上的精确IQ误差相位分量，再加一次使权重1→2；不作全局相位商，不忽略幅度约束。sourceclean排名mask仅训练使用，部署不读clean或mask。

L189000/U1701000曝光，20份固定源腐蚀缓存后复用01至10，非30份独立腐蚀；训练F/TX/学习教师/V/target访问与G祖先均0。方法选择参考过源F机制诊断，不能把整个研发过程称为完全未见F。30条加权loss逐项重算一致，phase实际启用；最终IQNMSE0.02279791、额外phase分量0.00590673，强64点clean能量占比约88.4719%。

![完整30轮与七景表现](E:/type10-7/automation_reports/CV-SincNet/20261031-phase1-DURA-strongPhase-E30-s2026092701-r01/training_and_formal.png)

源oracle曾在512训练池样本两缓存上提高10/6条F自身clean选择、一景仍−1；它不是TX准确率或目标保证。本版正式结果只支持“该固定配方在本seed两种严重城区场景观察到改善”，不能扩成六景有效，也不能据测试改相位权重、mask或场景路由。

## 实际成本

训练循环与保存1002.686482秒（16.711分钟），控制器至完整分析1187.861899秒（19.798分钟），目标七景预测160.036987秒。source/model准备未单独计时，不伪填0，包含在全流程；原模拟缓存和F既有训练为共同历史成本。

新增2个FFT只属于训练loss，autograd也有实际开销；部署仍266951参数/2prior4CG27FFT，ConvLinearMAC25068072、FP32参数1067804字节、checkpoint1305508字节。训练峰值allocated1084666880字节。GPU4同卡交替G+F B1中位默认5.238818ms、本版5.575699ms，完整P95/吞吐/B128见JSON。共享负载下不作跨run因果加速结论，profile显存含共同驻留模型和图。

E30 physical是同预算优化参照：物理参数无F梯度、独立裁剪且G/probe RNG已匹配，但原整个进程含F指导增强分支与成本，不能称作另一场独立F-free训练。预算、知识与实际成本分别保留。

## 验证与后续边界

执行84c5f08b3f334781cad4cb96f9e22bd33d9000eb，8项聚焦检查、唯一P0P1、真实2Ls source梯度/strict reload/原F noquery通过。全日志无Traceback/OOM/非有限数，controller1134232/worker1134322自然退出；其他owner不变。全部负面与正面结果保留，不选择性重跑。新宽coarse对照已在本次正式分数读取前依据旧source拟合差距与容量盘点固定，独立比较旧coarsehint、复用原冻结IQ缓存，不混本相位损失。默认与逐景3pp/clean/效率目标继续ACTIVE。
