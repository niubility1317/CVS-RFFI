# 逐视图二阶矩 × 原 practical：E30 四行预登记

状态：PLANNED／LOCAL_VERIFIED，未发布、未完成真实训练或正式评分。唯一假设：在不改变原 ordinary/As 监督目标的情况下，二阶矩使用两种视图的离散分歧项，能否保护 clean 并增加 mid/low urban 实际正确率。本项是训练配方，不是 IQ A_g；本四行只报告“优化配方 × A_s”交互，不能冒称 Ag×As 四格完成。整体联合目标 ACTIVE。

## 依据与固定算法

已独立读取此前 MEASURE 行的全部 2220 步合法源梯度标量：加权视图分歧平方和／两视图和梯度平方和=0.7307394981；逐步比值 p10/p50/p90=.4568307581/.7239100125/1.0407558418。Sinc 使用物理 Hz 梯度，其余原坐标；这里不是逐坐标 Adam 状态、不是含 SupCon/U 的总梯度方差，也不是模型收益。0 新 IQ／CP／NN／更新／target。来源和十轮全量结果见 source_gradient_basis.json。

原 L 目标 .5 RD(ordinary)+.5 RD(secondary)+原 SupCon 不变，secondary 在 NoAs 为同一份 ordinary，在 As 为完整 practical(raw)。两 RD 对各自两次前向共享同一已生成 IQ；不重抽 ordinary/As。不改变任何 CE/KL/对比项系数，不新增门控、头或部署分支；所有 128 合法卫星 L 样本参与监督。

设 gO=grad(.5 RDordinary)，gS=grad(.5 RDsecondary)，原 native 总梯度 g=gO+gS+gSupCon+gU，c 为 native 物理 clip5 的实际统一缩放。VIEW_SECOND 使用

    m <- beta1*m + (1-beta1)*(c*g)
    v <- beta2*v + (1-beta2)*((c*g)^2 + c^2*(gO-gS)^2)

MEASURE 保留 native v 的原公式；两组均额外求 gO/gS，各 2 VJP，均无额外 NN。候选分歧系数固定为 1，来自两种等权完整视图梯度的离散方差恒等式，不拟合、扫描或随质量门控。实际 AdamW 仍在原存储参数坐标工作，Sinc 先按原物理坐标全局裁剪，再回存储梯度；候选分歧也乘同一个 c²。原 Sinc 分组 lr/eps/物理 weight decay 不改。候选通过原 step pre-hook 加入 v，在 AdamW 的 beta2 衰减前补偿该衰减；不改原一阶矩或总梯度，也不投影梯度。原 U 损失、teacher/门控/前向规则不改；优化器变化会改变后续参数及 U 预测，不能声称整个 U 更新轨迹不变。

[Adam 原论文](https://arxiv.org/abs/1412.6980)提供自适应矩估计的基本背景；上述逐视图项是本项目待检验扩展，不是论文已证明的 CVS 方法。原 AdamW 对实际 batch 梯度平方的估计没有被判定为 bug；候选不是全梯度方差的无偏估计或有限步下降保证。

## 固定最小矩阵与输入边界

|行|二阶矩|As|初始化|
|---|---|---|---|
|NOAS_MEASURE|原 native；只测量视图分歧|无|原全源 REPEAT20＋自身 EMA|
|AS_MEASURE|原 native；只测量视图分歧|完整原 practical|原全源 CONCAT20＋自身 EMA|
|NOAS_VIEW|逐视图二阶矩|无|同 NOAS_MEASURE 父|
|AS_VIEW|逐视图二阶矩|完整原 practical|同 AS_MEASURE 父|

每行自身 E20＋10 至 E30／6660 累计更新，新增 10 轮／2220 更新，重置 AdamW 与 cos10，不声称原优化轨迹连续恢复。源选定原固定 final20，不加载 SAM/PCGrad/其他按目标选出的候选。跨 As 处理的父权重不同，但初始化规则相同；同 As 两行严格同父。全部新行 modelseed2026092701、split392005、全部 RX[1,3,4,6,8]／day1/2/3、L6300／U56700 TX 隐藏／V27000、256 点 IQ。不留出训练 RX、不使用 L5580/dev720、不重新验证 VALIDATED_ONCE 数据。

F 三 SIG_TIME／统一 Sinc CENTER25K+UNITS／joint160／默认 cosine6×160 scale30、NoExplicitCFO 和必要内部补偿保持，骨干与原头联合更新。父原阶段13L+7U，新增10轮继续原U，总13L+17U；ordinary 固定原实际 E13 强度，全部四行共用原物理顺序、隔离 L/U loader 流、原 SupCon 日历、原 own EMA、原U门控和原教师费用。A_s 完整几何/遮挡/传播/噪声/接收质量/补偿依赖、residual/post_sync/NoEQ即G1/ref30/reflect 保持，训练/源资格/正式随机流分离。不存在 ordinary 后再叠完整 As。正式 received IQ、信道、SNR、困难样本均不改。

固定源资格先于 Agent 读取目标指标：原 V1800、90 格各20包、全部5RX、2 个预先固定 practical draw；AS_VIEW 相对 AS_MEASURE clean 不降、每个 midurban draw 不降、每个 lowurban draw 至少+.5pp、每个 suburban/draw 不低于−.25pp；NOAS_VIEW clean 相对 NOAS_MEASURE 不低于−.25pp。只加1e−10pp浮点容差。任一失败关闭该固定系数/坐标/父/seed/配方/预算，不调参救跑或自动扩种子。通过也只说明单 seed 源资格，仍不证明跨 RX Ag 或联合目标。

全部四行每行训练/冻结后自动原 clean＋6 practical 各168000 默认头预测，七视图 prediction 固定后独立 CPU truth-last。源科学失败也不跳过预登记正式评价。报告 accuracy/macroF1/worstRX、每个 suburban 绝对值与自身 clean 差、修复/新错/clean与场景联合正确、二阶矩配方独立差、As 独立差及优化器×As交互。不得把本交互标为 Ag×As。clean 净正／low>60／suburban 各差≤2pp（探索目标）及跨 seed 仍是未完成的全目标。

## 成本、验证和执行

四行均每物理 L 4 次 NN 前向和每步2次额外 VJP，无 replay／新 IQ／新 teacher／新参数。新增 4×10=40 modelE、8880 更新、17760 额外 VJP；两个唯一原 E20 父计40 modelE/8880更新一次，唯一累计80 modelE/17760更新，各模型累计30。不得把4个读出各自30相加后又重复计父；原教师、V评价、源资格（每行23400NN/21600物理处理）、真实CP smoke、正式4×7×168000=4704000NN、优化器额外算术与墙时分别记录。VJP并非免费。

本地19个聚焦负例／独立16步多步AdamW递推／同视图3步与native参数及状态精确相同／4个真实F合成集成通过：2048F NN、4次F合成更新、8额外F VJP，900722参数、原默认头和物理Sinc clip/step核对通过。纯张量递推19次合成更新/38VJP另列；开发中closure钩子边界失败在首update前终止（2VJP），随后fused CPU负例构造失败发生在19次张量更新/38VJP之后，修正测试构造再完成本次23更新/46VJP全套。开发累计42合成更新/86VJP/2048F NN，0真实IQ/CP/更新；现有Torch/NumPy桥接警告保留，无安装/环境修改。

执行仅既有最小流程：一份实际代码/config Git提交、一份最小预登记、一次限定直接P0/P1独立审查、launcher首步两个匹配原CP/4合成NN无query smoke，PASS即继续；普通N607一次preflight／一份archive本远单次SHA比较／一次remotecompile、独立PID/CWD/argv/GPU/loggrowth和技术停止规则。通用单独设计审批等为 REJECTED_EXTRA_GATE，不增加审批、签名、成员hash或重复审查。无健康旧任务修改/终止/重启。

唯一 owner codex/root/F-viewMoment-E30；CWD PROJECT/releases/cvs_F_viewMoment_E30_20261018_r01，命令 CVS-RFFI/bin/python -u experiments/cvs_view_moment30/runner.py；本地发布入口 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_view_moment30.publish，CWD当前工作树。不可覆盖root PROJECT/runs/20261018-phase1-F-crossRX-viewMoment-e20plus10-s2026092701-m4-r01；按既有capacity chooser选择GPU，每GPU所有owner总进程≤3，CPU/OMP/MKL/OPENBLAS各2。精确逐行配置、路径与command见 experiment.json。只因权限/来源/非有限/错误预算/输出冲突/不可合法闭合预测/系统技术故障停止所属行并保留输出，不因低分停任务，不自动重试。

预期保存真实 source/result/finalcheckpoint/ownEMA、全部原生epoch与step/sequence/sampler记录、view_moment_steps的每步三坐标一阶/二阶矩原值/总梯度/分歧/新值见证、每行源资格、源矩阵判定、完整七视图预测/独立评分、同row报告/登记/Git。原IQ、逐包ID/预测/完整log/权重/私有runtime不提交；只提交代码/config/小型聚合。

唯一独立 P0/P1 审查 PASS，未发现直接影响下一真实运行的缺陷；核对了二阶矩/裁剪/Sinc单位/原step顺序、EMA、四父映射、发布依赖和七视图truth-last。未重复完整测试，未访问真实IQ/CP/Utruth/target或远程。当前仍未发布真实训练。
