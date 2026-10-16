# F＋跨 RX：周期差异残差相关形状源检验

状态：PLANNED，已登记，尚未启动。基线为 F＋跨 RX 同 TX 对比学习，后续模型预算 E20；当前不构造模型、不修改 IQ、不训练。

唯一矩阵与固定配置见 [experiment.json](experiment.json)。Run 为 20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01，owner 为 codex/root/F-cycle-covariance-source-20261017。

固定源统计问题：IQ[20:180] 的八个20点周期，经分析专用 lag20 公共相位抵消及周期均值移除，其周期内部 lag1至4 归一化复相关形状是否有跨 TX/day 的 RX 关联。残差不预设为纯接收机热噪声，也可能含 TX 随机误差、传播、均衡和重采样影响。已核查的原文与推导见 [研究说明](../../../analysis/cvs_cycle_residual_covariance_basis_20261017.md)。区别于另一对话 practical 正负噪声采样方向。

输入仅复用原 VALIDATED_ONCE 源载体的 fit L5040，RX3/4/6/8、day1/2/3、6TX、每格70包，split392005。原缓存反序列化 L6300、未使用 V 和历史特征；RX1原L1260在转换统计数组前排除，U仅角色ID元数据，没有U IQ/TX标签。U/V/heldRX1/target均不参与统计，不消费父权重或缓存特征；不是完整L6300或内部L_train5580训练结果。

18折同时留出一个TX和一个day，其余5TX×2day给每RX估计均值，训练和留出分别去除四RX公共均值；以零RX对比为MSE参照，另设三个固定错误RX循环指派。已知源RX条件统计，不拟合身份或接收机分类器。全部5040包、72格、18折参与，没有置信筛选或删包。

原固定条件必须全部通过：每格lag20 coherence中位数≥.90、q10≥.80；RX形状对比基准RMS≥.02；合并风险改善≥.20；至少15/18折改善；每个RX风险改善≥0；正确RX风险改善超过最佳错误指派≥.02。风险改善不是身份准确率。任一失败即关闭固定窗口、周期、lag、阈值、RX、seed依据问题，不改条件救跑；PASS也只支持源统计依据，不自动训练或证明Ag。

本地22个针对性非法配置/输入及源条件反例检查PASS，包含TX-only、day反转、低周期一致性、弱RX差异失败。两次本地检查累计296个合成IQ观察、0真实IQ、0NN/更新。一次独立P0/P1审查PASS，见review.json。无模型/CP，真实CP无查询smoke不适用；无重复数据验证、额外审批或签名。

发布入口：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_cycle_covariance17.publish，CWD为当前受管codex/cvs-iq-timing-e20-20261016工作树。远端普通N607既有CVS-RFFI Python，命令 CVS-RFFI/bin/python -u -m experiments.cvs_cycle_covariance17.run；CWD PROJECT/releases/cvs_F_cycle_covariance_fitL_20261017_r01。实际Python绝对路径、机器私有信息仅在私有runtime与submit中保存。代码标识是本次实际Git commit，由release_commit.txt和resolved_config记录，不新增报告/配置哈希。

CUDA_VISIBLE_DEVICES为空，CPU/OMP/MKL/OpenBLAS各2线程，GPU none。不可变输出 PROJECT/runs/20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01；独立发布根 PROJECT/releases/cvs_F_cycle_covariance_fitL_20261017_r01；日志source_cycle_covariance.stdout.log。一次普通账户/存在性/空间预检，一份release archive单次传输比对，一次远端编译与启动后PID/CWD/argv/CUDA/日志或终态读回。0模型、0CP、0NN、0更新、0教师、0另训头。

角色不匹配、缺缓存、非有限值、错矩阵或异常时保存failure和原产物停止，无自动重试，不干预健康旧任务。科学FAIL不等于技术失败。

预期产物：active/resolved_config/completion、全部72格/18折与wrongRX的cycle_covariance.json、私有5040逐包shape/coherence/residualratio/metadata、fit物理ID、stdout、独立全量复算、同row报告和登记。只取回统计与日志，不取回IQ或权重。生成输出及私有机器状态不进入Git。

尚缺有效新Ag、独立As稳定收益、同契约B00/B10/B01/B11与交互项、跨seed和正式clean＋六practical。low urban>60%与suburban各场景距自身clean≤2pp探索目标均未完成。本检验不能替代真实身份识别收益。
