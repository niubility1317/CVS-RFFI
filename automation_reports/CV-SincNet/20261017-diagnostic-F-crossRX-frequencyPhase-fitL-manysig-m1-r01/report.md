# 固定频域相位测度：全包源依据预登记

状态：IMPLEMENTED／PREREGISTERED_NOT_LAUNCHED。唯一run为20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01，唯一owner为codex/root/F-frequency-phase-source-basis-20261017。

基线F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine头正常联合训练／E20／原U不变；本阶段无模型、训练或增强。另一对话研究As时序，本阶段仅频率相位观测。

上一固定合成诊断252/252格、63/63组通过，仅支持测量基础。当前唯一假设：同一固定测度在实际合法fitL5040中有跨TX/day保持的RX特异性。数字观察不等于真实RX替换，也不将RX关联直接作为身份保持或DG证据。

实际数据为已有源carrier的EQ1／RMS／256点／25Msps。原source契约L6300、U56700、V27000，源RX1/3/4/6/8，day1/2/3、六TX；本观测只用fitL5040、RX3/4/6/8，72格各70包。原source_cache.pt反序列化包含未用V和历史特征；只保留labeled x/y/rx/day/ids，RX1在任何观测前剔除；U仅opaque角色ID，不读U IQ/TX真值。不是L_train5580/L_dev720，不改变训练契约。逐项比较已有source_contract和preparation句柄及原L物理ID，不额外验证数据字节；不加载旧权重或把历史features输入测度。U、V、RX1、target均不作统计／校准／选测度。

固定依赖experiments/cvs_frequency_phase17/estimator.py的已验证observe/summarize，不改其窗口80:160、lag20自身CFO干扰补偿、周期20的12谐波、8相邻谐波闭合量、载波噪声比25、单元与RX参考圆合向量模长0.8、相对数值下限1e-12。原float IQ仅转换Complex128，不新增归一化。估计CFO单独作为私有质量审计量，不进入相位身份坐标；未删除或修改原模型内部补偿。

全部5040包和72格必须保留。任一包必要频点不满足条件，则整个共享RX估计不可辨识并判FAIL；仍保存全部包闭合量／质量和全部格均值，不剔除难包、不选择可用子群。全部可辨识时，70包等权均值，先对每个TX/day的4RX作圆参考对齐，再作RX均值中心化。18折同时留出TX和day，仅其余5TX×2day估计RX轮廓；零RX为风险基线，3种非平凡循环错误RX映射为特异性控制。没有分类器、投影、拟合scale或数据挑选。

固定判据与合成检测一致：全包／全格可辨识，轮廓RMS≥0.02rad，交叉留出风险改善≥0.5，至少15/18折严格改善，每RX不劣于零基线，正确映射相对最佳错误映射改善余量≥0.05。全部条件须同时通过。任一科学FAIL关闭此固定源依据；不改窗口、周期、频点、CFO方法、质量条件、幅度、阈值、TX/day/RX、包或seed救跑，不进入IQ增强或E20。技术输入／角色／非有限／碰撞／执行异常保留失败产物，不自动重启，不干预健康任务。

PASS仅为源RX相关观察证据；仍须另外固定物理合理、有界IQ算子与原默认头身份保护，然后才能固定匹配E20。不得从观察轮廓直接推断增强幅度，不自动启动增强或训练。

本地验证21个协议负例／4个科学用例PASS，生成5040合成包／1290240复点、1次测度调用与3次直接汇总，无真实IQ／CP／NN／更新。首次负例夹具使用整体标签循环移位，实际保持合法平衡，已改为单条标签破坏；失败原产物保留，未修改科学测度或阈值。正式源数据尚未读取。

一次限定P0/P1审查后、代码和配置提交，使用普通N607账户发布。一个归档／一次传输SHA比较／一次CPU句柄空间预检／remote compile；零模型阶段不执行不存在的checkpoint smoke。命令CVS-RFFI/bin/python -u -m experiments.cvs_frequency_phase_source17.run；CWD为PROJECT/releases/cvs_F_frequency_phase_fitL_20261017_r01。独占输出PROJECT/runs/20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01，CPU2／CUDA隐藏，无GPU，所有旧任务保持。

正式成本预算为1次sourcecache加载、5040原IQ包／1290240复点、60480谐波投影／40320闭合量、最多18折；模型／checkpoint／NN前向／更新／教师／另训头／新IQ视图／target均0。实际数、主体墙钟分别记录。产物包括active／resolved／completion或failure、全部逐包观察和质量、fitL物理ID、72格均值／方差／可用折风险、完整stdout。原始IQ／ID／逐包量／日志／私有runtime不入Git。

独立stdlib reader从全部保存观察完整重算质量判定、72格均值／方差、全部可用RX轮廓与18折风险及原判据；不导入候选估计器／Torch／模型，不冒称再次读取原IQ。随后同一row报告和登记，保留全部负结果。REJECTED_EXTRA_GATE：当前独占最小流程之外的设计审批、多轮审查／hash或seal链不实施。联合目标ACTIVE，尚无有效Ag、稳定As、四因子交互、跨seed或正式七视图成功。
