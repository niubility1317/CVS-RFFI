# 下一步：固定原始 IQ 监督，检验普通增强的训练组织

状态：DESIGN_NOT_RUN。NoPA RX1 的一个新 scratch E20 已完整收尾，N0/N1/N2 的 heldV 准确率为 76.2222/82.6296/79.2778%，N2 比 N1 低 3.3519pp、macroF1低3.5120pp，未保留原有收益，按原规则结束，不扩展该候选。不能把冻结头的 PA 输入损伤当作增强训练无效或删除 PA 必然有效的证据。

当前有两个互补事实：原 ordinary 替换 L 视图的五fold中，3折改善、2折下降，整体稳定性FAIL；删除 PA 又损失RX1的训练收益。下一问题改为**保留稳定原始 IQ 监督，是否能保留普通增强的正则作用，同时减轻其跨 RX 的错误转移**。这是独立的训练组织检验，不调 PA 强度、不开新硬件算子、不重排RX或seed，不将已失败的替换式配方直接作为 Ag 晋级。

## 匹配的三行最小实验

固定顺序首折 heldRX1、modelseed2026092701，三行都从零总 E20=13L+7U、4440 optimizer updates，L5040/U45360/seenV21600，heldRX1原V5400仅FINAL；heldL1260/U11340不训练。CVS/SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos、LS.01、原RDrop、AdamW/cos、oldgate/ownEMA/U规则不变。

定义 RD(v)=原有同一份IQ两次随机子网络前向的 mean CE＋对称KL，仍使用同一真实TX标签。每个物理L的两对损失各归一化为.5；总监督损失尺度和RDrop系数保持1，不新增跨视图一致性或特征对齐。

|行|第一对输入|第二对输入|有标签目标|
|---|---|---|---|
|A0|原始x|同一原始x|.5RD(x)+.5RD(x)|
|A1|单次生成的原ordinary G(x)|同一G(x)|.5RD(G(x))+.5RD(G(x))|
|A2|原始x|同一物理样本的G(x)|.5RD(x)+.5RD(G(x))|

三行均为每个L样本4次完整原网络/默认头前向，不用历史2前向N0/N1/N2充当匹配控制。A1识别额外随机子网络前向的影响；A2与A1直接检验原始监督的作用。A0关闭无用增强生成，A1/A2每L批次只生成一次G，保留20261016051私有增强流及原loader流，核对三行实际完整L/U物理序列。U学生/教师算法及权重保持；其伪标签和接受量可以随模型改变，不能用U真值评价。

A2第一对原始IQ的.5监督份额固定；第二对视图明确披露为原始或增强，不通过拼接batch隐式改权重。原ordinary仅作为已授权旧配方的固定研究视图，保留已审计真实开关/顺序/范围/schedule，不把其中PA等逐项称为标签保持，也不因本次保留PA就声称硬件识别成立。源域训练终点仍必须验证实际身份收益；冻结输入分数只作解释。

## 固定条件与失败处理

首轮A2相对新匹配A0：held accuracy至少+1pp、macroF1不下降、seenV均值下降不超过.5pp、全部有限更新及6TX×4RX×3day覆盖；相对新A1，accuracy及macroF1各不下降超过.5pp。主候选只有A2，不按结果切换主行或换强度/RX/seed救援；A2失败则关闭该原始监督组织假设。A1即使得分更高也不自动变成发布候选。

首轮PASS只允许另登记固定剩余RX3/4/6/8的同三行，每个仍scratch E20。五fold对A2−A0沿用均值>=+1pp、>=4折正差、最差>=−1pp、mean macroF1>=0、每fold seenV均值损失<=.5pp。完整报告A2−A1的逐fold/mean/worst和TX/day正负差；没有正收益时不宣称原始监督改善了替换式训练。

第一轮新增3×20=60model-epoch/13320updates，每行2273280有标签网络前向，是历史2前向行的2倍；U/EMA/源评价/heldV/CPsmoke另外实测。不能把epoch相同写成前向成本相同。旧源研究/诊断成本另列，不继承旧权重。

本阶段明确source-only，无As或target，不改正式received IQ/信道。若源身份收益建立，才独立固定full-source Ag/As因子矩阵：从原IQ分别生成raw、G和原practical，避免ordinary AWGN/多径先叠在As前；As(G(x))仍是单独后续问题。任何正式训练row继续自动冻结、原clean+6practical预测和truth-last。当前没有该组织实验模型或成绩，clean/urban双提升及跨seed目标保持未完成。
