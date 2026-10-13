# 合法源证据驱动的浅中层 RX 对抗

目标仍为普通cos的clean三primary均值84%，未达成。本候选由冻结源健康诊断驱动：合法L的probe heldfold2，TX条件time_down RX识别96.3%至96.7%、t2.pool97.5%至97.9%（5RX随机基线20%）；证明可观测RX预测信息，不证明纯硬件或目标因果。V只健康，U真值隐藏。方法/参数在新target访问前固定，不从已知target错例或分数选择强度。

与中心对齐/样本拉近/有限均值位移不同，本候选由训练期RX判别器约束浅中层分布，不做硬投影或重建身份分类方向。相关原理可参考[CDAN](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html)，本实现是L真TX分组选源RX的source-DG变体，不冒充论文原目标域适应配方。

9模型三组×三primary，各从零30/6660身份更新，普通cosW6×160scale30保持。辅助两个MLP D96/D192→64 ReLU→30 reshape B6×5，22460参数，全部方案一致。DETACHED_AUX训练辅助头但identity不受梯度；MARGINAL_GRL平均六个head的RX CE；TX_COND_GRL选合法L真实TX对应head。RX1/3/4/6/8显式映射0..4，辅助不读取U TX或V。捕获一次原L身份forward，不增加身份视图。域CE系数1、GRL只在反向前5轮0，第6至15轮ramp至.1；aux持续训练以跟随当前特征坐标。

主/辅optimizers及clip严格分离：主原AdamW/WD.0001/cosLR.0002→.000001/clip10，辅AdamW/WD.0001/cosLR.001→.00001/clip5。辅助init用primary+271828隔离RNG，无Dropout；原L/U顺序、弱目标、强扰动、两pass95门控、KL.25和各随机性保持。零GRL/DETACHED要求实际primary损失梯度、clip后梯度、单步更新、RNG等价；aux头实际更新，域loss对普通cosW直接梯度None/0。

源训练199485参数，推理身份177025参数，辅助全弃掉；总59940main和59940aux更新分别报告，不能把辅助训练成本藏掉。日志每步域CE/准确率/层位/系数、main CE/KL、aux/main gradnorm；每轮首末步记录净domain主梯度与base的范数比/夹角/直接cosW路径检查。只影响时间浅/中及其祖先，不声称直接消除频率/PA/参考分支RX信息或完整未来TX身份保留。

L6300/U56700隐藏/V27000原契约，全3primary，noVforward/gate、无EMA/外部checkpoint/卫星或物理分支。每row固定FINAL30后自动clean168000预测，全部prediction固定后exec独立truth-last；总1512000预测/162same-row指标，负结果完整保留。无性能停止或target反馈。

一owner/不可覆盖root/普通N607/每GPU总≤3/free≥12000MB、健康任务保持。单代码config提交、聚焦协议负例/真实checkpointnoquery首步、一次P0/P1、单archiveSHA/compile/postlaunchPID-CWD-argv-GPU-log。只有系统技术失败停止。命令、路径、环境、GPU与输出见experiment.json。

|ID|验收|目标|状态|边界|
|---|---|---|---|---|
|A1|两层真实capture/descriptor/class-RX6x5|auxiliary/source|verified（组件与完整实际运行证据）|L-only|
|A2|仅GRL反向系数、条件/边缘CE正确|auxiliary|verified（组件与完整实际运行证据）|非硬投影|
|A3|分离optimizer/clip及zeroGRL完整等价|checks/source|verified（组件与完整实际运行证据）|训练额外cost披露|
|A4|普通cos查询无aux、从零30/U隐藏/无Vgate|design/predict|verified（组件与完整实际运行证据）|query无RX路由|
|A5|每row自动clean/独立truthlast/完整三seed|dispatch/score|verified（组件与完整实际运行证据）|不按target调整|

本地CPU/CUDA actualsource.objective检查全部PASS：177025身份/22460辅、实际time maps48×128/96×32、GRL一阶符号及一次系数、条件/边缘CE、auxCPU独立RNG/noDropout、普通cos160/9类refit。四个零强度/DETACHED（含非零KL合成U）核对base loss/主梯度/触发clip10后的梯度/AdamW完整主状态与RNG精确torch.equal，辅同时真实更新；directcosW均None/0/True。9row完整synthetic聚合162指标/3汇总/3配对及不完整truth-last拒绝PASS。唯一独立P0/P1 PASS，无发现；remote compile/真实 checkpoint no-query/9 个真实训练预算与完整 clean 独立评分均已核实。实现有效执行不等于性能提升。

发布已独立核实：实际执行提交 `4f6258bb230389198f7036e1c57a7f740d4dcb50`，release archive 本地/远端单次 SHA 一致、remote compile、真实 checkpoint no-query 首步 PASS。最后保存远端证据时间 2026-10-13T04:16:35.115Z：9 个健康 worker，完成轮数 25/24/25/24/24/23/23/22/22，无完成或失败标记。此后远端查询被执行工具的 pwsh 启动器限制阻止，当前状态 UNKNOWN；保留原任务与产物，无重启或补跑。完整性能证据待取回后更新。

完成取回与全量独立核验 VERIFIED：9×30 轮、59940 主更新与 59940 辅更新、119880 层记录、540 净梯度记录、9 主＋9 辅权重及 1512000 原始 clean 预测、162 项 same-row 指标和三种子配对全部核对一致。DETACHED_AUX 80.0458%，MARGINAL_GRL 79.6978%，TX_COND_GRL 80.0605%；条件对匹配对照 +0.0147 个百分点，仅 2/3 种子正向，84% 仍未达成。A1 至 A5 均 verified。结果见 [完整分析](final_analysis.md)。此前客户端启动限制通过项目 Python 直接调用已验证 SSH 客户端解决，全程未启动 pwsh、未重启实验或重做预测/评分。
