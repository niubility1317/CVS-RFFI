# 冻结 E20 实际表示的身份可读出性

**VERIFIED：三个固定 E20 源域诊断完成。** 身份／辅助头更新为零，不读取 U、V 前向或目标。每模型两种输入策略各6300条L／50前向，共300模型前向／37800样本模型读出；发布预检另24L／3前向。

historical_explicit_CFO 为历史原模型函数；no_explicit_CFO 屏蔽参考向量第27坐标，其他28个坐标、参数、缓冲、RNG保持不变，内部频偏补偿仍执行。后者是原权重上的功能诊断，不是重新训练的无CFO基线，也不代表原始IQ不含频偏。

每个读出使用L内3折×5RX；查询fold/RX不参加方向、中心和原型拟合，每PID每方法恰好读出一次。冻结骨干已经见过全部L和五源RX，本结果仅是统计读出迁移，不能称未知RX验证。

|输入策略|层／读出|2701|2702|2703|三种子均值|
|---|---|---:|---:|---:|---:|
|historical_explicit_CFO|time_down/raw_merged_prototype|44.1746%|48.0476%|48.3651%|46.8624%|
|historical_explicit_CFO|time_down/raw_RX_separate|50.6032%|53.7302%|51.6190%|51.9841%|
|historical_explicit_CFO|time_down/stable5_merged_prototype|60.9524%|67.9841%|58.6667%|62.5344%|
|historical_explicit_CFO|time_down/stable5_RX_separate|60.6508%|67.9524%|58.7778%|62.4603%|
|historical_explicit_CFO|t2.pool/raw_merged_prototype|70.2540%|63.5397%|58.2381%|64.0106%|
|historical_explicit_CFO|t2.pool/raw_RX_separate|67.4444%|58.0317%|57.4921%|60.9894%|
|historical_explicit_CFO|t2.pool/stable5_merged_prototype|78.0952%|68.6825%|72.9841%|73.2540%|
|historical_explicit_CFO|t2.pool/stable5_RX_separate|77.9524%|68.6190%|72.9524%|73.1746%|
|historical_explicit_CFO|id_feat_joint/raw_merged_prototype|93.9524%|93.6190%|95.2698%|94.2804%|
|historical_explicit_CFO|id_feat_joint/raw_RX_separate|94.1587%|93.8254%|95.0635%|94.3492%|
|historical_explicit_CFO|id_feat_joint/stable5_merged_prototype|95.9841%|94.4286%|95.9365%|95.4497%|
|historical_explicit_CFO|id_feat_joint/stable5_RX_separate|96.0000%|94.4603%|95.9206%|95.4603%|
|historical_explicit_CFO|native_identity_head/native_frozen_head|98.2063%|97.8889%|98.2063%|98.1005%|
|no_explicit_CFO|time_down/raw_merged_prototype|44.1746%|48.0476%|48.3651%|46.8624%|
|no_explicit_CFO|time_down/raw_RX_separate|50.6032%|53.7302%|51.6190%|51.9841%|
|no_explicit_CFO|time_down/stable5_merged_prototype|60.9524%|67.9841%|58.6667%|62.5344%|
|no_explicit_CFO|time_down/stable5_RX_separate|60.6508%|67.9524%|58.7778%|62.4603%|
|no_explicit_CFO|t2.pool/raw_merged_prototype|70.2540%|63.5397%|58.2381%|64.0106%|
|no_explicit_CFO|t2.pool/raw_RX_separate|67.4444%|58.0317%|57.4921%|60.9894%|
|no_explicit_CFO|t2.pool/stable5_merged_prototype|78.0952%|68.6825%|72.9841%|73.2540%|
|no_explicit_CFO|t2.pool/stable5_RX_separate|77.9524%|68.6190%|72.9524%|73.1746%|
|no_explicit_CFO|id_feat_joint/raw_merged_prototype|93.9524%|93.5556%|95.2381%|94.2487%|
|no_explicit_CFO|id_feat_joint/raw_RX_separate|94.1270%|93.8095%|95.0159%|94.3175%|
|no_explicit_CFO|id_feat_joint/stable5_merged_prototype|95.9841%|94.3810%|95.9365%|95.4339%|
|no_explicit_CFO|id_feat_joint/stable5_RX_separate|95.9841%|94.4286%|95.9365%|95.4497%|
|no_explicit_CFO|native_identity_head/native_frozen_head|98.3016%|97.8889%|98.2222%|98.1376%|

native_frozen_head 是已见L上的原身份头读出，不能与crossfit读出当作相同训练路径对照。raw/stable原型均只由fit L建立，所有TX统一竞争。RX_separate对四个RX各自的原型余弦分数作logmeanexp；merged平均RX/day单元后建立共同原型。固定tau=.1，不由成绩选择。

|输入策略／seed|层|TX占比|RX占比|交互占比|within占比|全L TX秩|闭合相对误差|
|---|---|---:|---:|---:|---:|---:|---:|
|historical_explicit_CFO/2026092701|time_down|32.4095%|5.8319%|23.6268%|38.1317%|5|4.13e-16|
|historical_explicit_CFO/2026092701|t2.pool|45.3591%|5.6180%|16.6515%|32.3714%|5|3.72e-16|
|historical_explicit_CFO/2026092701|id_feat_joint|80.2675%|1.6954%|5.2608%|12.7763%|5|3.06e-16|
|no_explicit_CFO/2026092701|time_down|32.4095%|5.8319%|23.6268%|38.1317%|5|4.13e-16|
|no_explicit_CFO/2026092701|t2.pool|45.3591%|5.6180%|16.6515%|32.3714%|5|3.72e-16|
|no_explicit_CFO/2026092701|id_feat_joint|80.2170%|1.7389%|5.2574%|12.7867%|5|4.08e-16|
|historical_explicit_CFO/2026092702|time_down|27.8315%|5.7130%|25.9725%|40.4830%|5|4.1e-16|
|historical_explicit_CFO/2026092702|t2.pool|41.7397%|6.2771%|18.9382%|33.0450%|5|2.99e-16|
|historical_explicit_CFO/2026092702|id_feat_joint|78.7591%|1.8753%|6.4349%|12.9307%|5|3.24e-16|
|no_explicit_CFO/2026092702|time_down|27.8315%|5.7130%|25.9725%|40.4830%|5|4.1e-16|
|no_explicit_CFO/2026092702|t2.pool|41.7397%|6.2771%|18.9382%|33.0450%|5|2.99e-16|
|no_explicit_CFO/2026092702|id_feat_joint|78.6911%|1.9301%|6.4316%|12.9471%|5|2.43e-16|
|historical_explicit_CFO/2026092703|time_down|34.3323%|5.5053%|24.2737%|35.8887%|5|4.72e-16|
|historical_explicit_CFO/2026092703|t2.pool|41.5994%|5.7217%|18.6585%|34.0204%|5|3.73e-16|
|historical_explicit_CFO/2026092703|id_feat_joint|76.6641%|2.3779%|6.3980%|14.5600%|5|2.83e-16|
|no_explicit_CFO/2026092703|time_down|34.3323%|5.5053%|24.2737%|35.8887%|5|4.72e-16|
|no_explicit_CFO/2026092703|t2.pool|41.5994%|5.7217%|18.6585%|34.0204%|5|3.73e-16|
|no_explicit_CFO/2026092703|id_feat_joint|76.6336%|2.3878%|6.4096%|14.5690%|5|2.83e-16|

这里统计实际表示，不是虚拟作用敏感度。TX/RX均值按RX/day单元等权，within包含同TX/RX的day变化；全局6TX均值的判别秩≤5。分母使用有限总体PSD外积、固定alpha=beta=1、shrinkage=.1和相对eps1e-6。全L方向只作几何描述，绝不回用于crossfit查询评分。所有45个拟合中的特征值、分母最小特征值、方向分量能量、fit/query数量和残差保留原JSON。交互可能承载身份响应，不把存在交互等同污染，不据该比值删掉全部交互。

所有三种子、两输入、三层、四读出方法与全部fold/RX混淆矩阵、NLL、margin保留在complete_source_evidence.json。源诊断不会证明84%目标达成，也不据目标特定TX加权。后续训练需要同代码路径CE/SSL控制、原身份头与关系任务分别报告，并由固定后独立clean测试验证。
