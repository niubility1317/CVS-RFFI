# 接收条件不变机制报告追踪

来源：用户附件 `69f454a7-232c-4be5-bca1-f195927716e8/已粘贴的文本.txt`。最新用户覆盖：只用固定后的clean测试验证性能，不设sourceV晋级门槛；三种子，每新模型从头30；单常规cos供Phase2 refit。目标clean三seed均值≥84%，尚未达成。

|ID|报告位置|要求|目标文件|状态|验证|边界/备注|
|---|---|---|---|---|---|---|
|P1|2.1|TX/RX等权条件均值、S_ID/S_N/S_E与广义特征值投影|experiments/cvs_report_projection/statistics.py|pending|待数学/实际源fit/clean测试|保留160维全宽，已知6TX的均值秩≤5，不当未来完整身份空间|
|P2|2.1|条件干扰正交投影、15TX均值差保留率|experiments/cvs_report_projection/statistics.py|pending|待P²/PD/15对ρ重算|训练L统计代理，不证明纯RX或全部身份信息|
|P3|2.2|最小改动斜投影、病态/放大率报告|experiments/cvs_report_projection/statistics.py|pending|待PD/P²、二维例子、实际fit|Σ正定正则；不声称统计独立|
|P4|2.1/6|可逆软谱抑制＋明确分类方向正则，对照原坐标正则|experiments/cvs_report_projection/statistics.py|pending|待ridge目标/soft几何与clean|闭式带正则LS估计普通cos方向，不冒充CE最优解；不额外训练专用分类器|
|T1|2.3|可信数字动作的归一化身份方向导数约束|experiments/cvs_report_mechanisms/objective.py|pending|待真实梯度/尺度/直接clean|有限差分近似Tangent Prop；数字滤波/插值不当真实换RX|
|T2|2.3|有界切空间匹配边界|experiments/cvs_report_mechanisms/diagnostics.py|pending|待有界求解/对称性|只作诊断；推理不替换用户指定的普通cos，也不声称严格度量|
|S1|3.1|子带包络二阶调制/S1归一化分支|experiments/cvs_report_mechanisms/frontends.py|pending|待平坦子带gain条件与低能量测试|离散滤波/有限窗近似散射，不宣称任意RX非线性不变|
|S2|3.2|多个lag局部四阶幅相多项式序列|experiments/cvs_report_mechanisms/frontends.py|pending|待增益/相位/线性频偏代数检查|是四阶乘积，不是累积量；有限字典不当完整不变量|
|S3|3.3|中心化IQ轨迹Gram逆、杠杆及指定lag项|experiments/cvs_report_mechanisms/frontends.py|pending|待GL2+DC、满秩/近共线保护|固定无记忆仿射作用，不能消一般CFO/卷积；压缩不等同完整行空间|
|F1|4.1|明确低维对数频响基投影|experiments/cvs_report_mechanisms/frontends.py|pending|待PB/同阶相乘/零频谱测试|有限窗/噪声/谱底近似；也可能删TX平滑响应|
|F2|4.2|标量跨频相干三节点闭环|experiments/cvs_report_mechanisms/frontends.py|pending|待节点gain/退化/实际源普查|同包8片段仅作为可比较统计假设，不保证真实RX稳定或非平凡|
|F3|4.2|[X(f),X*(-f)]增广矩阵闭环的trace/det|experiments/cvs_report_mechanisms/frontends.py|pending|待非交换GL2相似不变量/逆病态|先实报相关图/秩一/单位环退化，数值保护可能破坏严格不变|
|G1|5.1|共享非线性编码器完整C4相位平均|experiments/cvs_report_mechanisms/model.py|pending|待离散群、非零特征、逐包性|只保证有限C4群；不平均rawIQ、不称连续全相位/任意非紧群不变|
|I1|5.2|有效Fisher信息的干扰Schur补|experiments/cvs_report_projection/statistics.py|pending|待明确Gaussian局部位置模型与PSD|诊断连续均值路径；TX离散，不伪造现成神经分类损失|
|B1|全篇/6|数学边界、非交换作用、信息冗余、内容控制|analysis/phase1_receiver_invariance_report_20261013_traceability.md|pending|待模型/变换反例与实际诊断|分别不变不当联合不变；不信任eq0/eq1同sig_i配对，不虚构前导或新增长度|
|C1|用户覆盖|固定common cos、160身份接口、9类refit smoke|experiments/cvs_report_mechanisms/checks.py|pending|待真实模型接口检查|不创建RX专家或类别特定margin；projection原始源协方差/样本缓存不供Phase2 runtime|
|E1|用户覆盖|全源三seed、30轮、无sourceV性能门槛、clean-only自动truth-last|experiments/cvs_report_mechanisms/design.py/dispatch.py/predict.py/score.py|pending|待完整实际矩阵与测试重算|包括元反馈9模型与SWA3轨迹，保留旧源诊断；不追加practical|
|E2|报告6|冻结原CVS的投影zero-gradient对照|experiments/cvs_report_projection/design.py/dispatch.py|pending|待parent真实角色/祖先与18clean rows|三CE30父模型固定复用，不借oracle80权重或sourceU标签|

方法总览：9类探索空间全部映射到具体实现；有效Fisher另有明确定义的Gaussian工作模型诊断。新报告不要求替换commoncos为切空间距离，故有界距离只诊断，不作为query决策。教师目标调整核心先保留未注册备选，不发布新的源留出矩阵；优先完成当前报告和已经承诺的元反馈/SWA正式clean验证。

本记录用于完整性追踪，不是额外发布审批、source性能筛选或hash gate。状态随实际代码/测试/远端证据更新；目前不能宣称已完整实现或已达到84%。
