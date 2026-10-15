# HQ 教学：三个固定模型的源域冻结归因预登记

本分析为已授权 HQ E50 小试的冻结机制交付，不增加训练。三个角色在训练结果产生前固定：普通拼接 NOMINAL、HQ 标签监督 BRIDGE_CE、额外局部教学 BRIDGE_SIG。当前两条新模型仍在训练，尚未发布本源分析、加载其未完成权重或产生结果。既有 HQ 64.86%→70.97% 仍仅是旧冻结模型的源域受控诊断。

唯一矩阵、数据、路径/命令、环境/CWD、checkpoint来源、seed、stop及预期产物维护于 experiment.json 和 experiments/cvs_urban_bridge_mechanism/registration.json。代码身份是实际发布 Git HEAD，发布时写入 preflight/resolved；不增加报告或成员 SHA。

源角色复用 L_s6300 内部 L_train5580/L_dev720，fold2026101503，source RX1/3/4/6/8、day1/2/3、全部6TX，split/data392005。只前向物理互斥的 L_dev720；不打开 U，V 容器可能由既有源 cache 物化但前向为0，不调用任何目标数组或 truth。原冻结 final50/11100 默认头及 own EMA 严格加载，不拟合头、不更新 optimizer/EMA/normalization。新行还必须具备真实 VERIFIED 五视图预算及自己的 epoch5/5760 前向源校准；不匹配折、来源不明或目标接触的权重拒绝。

固定 single SIG_TIME、内部门控/参考响应、默认 cosine6×160 scale30。采样率25MHz、IQ256中心裁剪、原native RMS后模拟器unitRMS；真实 adv3b02_xuc/code/leo_practical residual/post_sync、EQfalse G1、reference_snr_db30及原质量适配/相位/IQ补偿/reflect/FIR24不改。实际 __file__ 写入resolved。正式七环境IQ从不读取或重新生成。

六种源 practical 场景全部保留，每个物理样本构造full和独立噪声HQ。两者共享传播/RX session/接收质量/lock/补偿/残余核/AGC/FIR/边界，只将模拟新增噪声幅度乘10^(-6/20)。原录音噪声与未知TX硬件不反演；新增噪声SNR不冒充总SNR。coupled实际 full逐值一致、状态/算子冻结证明须通过。源流2026101509独立于训练1401、验证1402、旧HQ1504、own-cal1508及正式固定流，RX/day/eq session不由TX定义。

比较固定 NOMINAL student clean 正确集合上的 full/HQ 默认头保持率、对 NOMINAL full 的错误修复/新增，以及 SIG 对真实 BRIDGE_CE full 的修复/新增；间隔为true logit−最强wrong logit后除30。NOMINAL先处理时没有CE预测，显式null/count0，不冒充已观察对照。各RX/LOS/PARTIAL/BLOCKED/lock/full新增噪声SNR区间分别汇总，三模型使用完全相同的720条源视图与分层。

实际 raw112 只取4窗的increment6、feature-area15、time-area6，排除dt。报告full→同H HQ ownEMA的有界组误差，同时报告同packet−最近wrongTX组余弦、既有各分支的身份关系/方差、实际gain贡献。跨模型共同误差使用BRIDGE_CE自己epoch5冻结的HQ RMS；各自训练尺度误差单列，不重新校准。几何相似增强与默认头间隔改善分别解释，不能互相替代。NOMINAL有不同训练CE/前向成本，局部增量主要看SIG−CE；随机优化仍限制唯一因果解释。

实际预算56882前向：真实NOMINAL checkpoint烟测2＋固定clean集合720＋3×（clean student/EMA1440＋6场景×full/HQ student/EMA2880）。新增epoch、optimizer/EMA/head-fit更新均0；原三个E50/11100及实际训练秒数保留，源分析另记墙钟。不执行或声称新训练收敛、跨seed重现或正式测试提升。训练梯度/门控/曲线读取原主实验完整日志，源分析本身没有回放optimizer更新。

执行在N607普通账号，CPU2、CUDA_VISIBLE_DEVICES为空。精确命令为既有解释器 `-u -m experiments.cvs_urban_bridge_mechanism.probe`，CWD为本run的独占release，路径见experiment.json。首步真实checkpoint默认头无query烟测PASS后继续；全套实际final50就绪才发布。一次archive比较/远端compile/preflight，独立PID/CWD/argv/log和最终results/complete读回，不以退出码单独证明完成。源/CP/物理算子/非有限/状态/预算/输出碰撞/技术错误只退出本owned源行，保留所有产物、不自动重试、不触碰健康训练或其他owner。

本地5个唯一检查分批执行：首4项中实际组/统计/真实冻结模型3项PASS，源资格fixture因Windows路径字符串比较出现StopIteration；改为Path比较后定向PASS.010s，并补入真实calibration status检查。发布接线定向PASS.006s，证明无需本地读取远端checkpoint且临时运输模块恢复。已知torch/NumPy ABI警告保持原环境，检查exit0。compileall通过。未用合成模型结果冒充源或目标准确率。

独立源候选一次有界P0/P1审查PASS：已核对三模型严格加载、同H/full状态、L_dev边界、默认头/ownEMA、统计/56882预算及发布/输出保护；图工具已实现但没有运行或图路径声明。本分析没有进行远端mutation。真实56882预算/状态保护、分层结果、图和完成读回均待三个final50实际冻结。追踪表为analysis/cvs_bridge_teaching_mechanism_traceability_20261015.md，当前verified4/pending2/deferred0/rejected0/blocked0。最高风险：局部误差下降但默认头间隔不增、正确证据没有恢复；HQ仍可能错，单seed源代理不足以证明正式跨RX收益。
