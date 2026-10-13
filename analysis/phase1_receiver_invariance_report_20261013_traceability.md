# 接收条件不变机制报告追踪

来源：用户附件 `69f454a7-232c-4be5-bca1-f195927716e8/已粘贴的文本.txt`。最新用户覆盖：只用固定后的clean测试验证性能，不设sourceV晋级门槛；三种子，每新模型从头30；单常规cos供Phase2 refit。目标clean三seed均值≥84%，尚未达成。

|ID|报告位置|要求|目标文件|状态|验证|边界/备注|
|---|---|---|---|---|---|---|
|P1|2.1|TX/RX等权条件均值、S_ID/S_N/S_E与广义特征值投影|experiments/cvs_report_projection/statistics.py|verified|数学PASS；源6300独立重拟合；24clean完成|保留160维全宽，已知6TX的均值秩≤5，不当未来完整身份空间|
|P2|2.1|条件干扰正交投影、15TX均值差保留率|experiments/cvs_report_projection/statistics.py|verified|P²/PD数学PASS；三实际源fit/15对ρ完成|训练L统计代理，不证明纯RX或全部身份信息|
|P3|2.2|最小改动斜投影、病态/放大率报告|experiments/cvs_report_projection/statistics.py|verified|PD/P²/二维例子PASS；实际fit完成|Σ正定正则；不声称统计独立|
|P4|2.1/6|可逆软谱抑制＋明确分类方向正则，对照原坐标正则|experiments/cvs_report_projection/statistics.py|verified_approximation|source LS普通cos方向；三seedclean完成|闭式带正则LS估计普通cos方向，不冒充CE最优解；不额外训练专用分类器|
|T1|2.3|可信数字动作的归一化身份方向导数约束|experiments/cvs_report_mechanisms/source.py|verified_approximation|实际梯度/代理尺度/3seedclean完整重算|有限差分近似Tangent Prop；数字滤波/插值不当真实换RX|
|T2|2.3|有界切空间匹配边界|experiments/cvs_report_mechanisms/source.py/frontends.py|verified_diagnostic|二维box LS单测PASS；3seed各6300源L、135组有界残差核对|只作诊断；推理不替换用户指定的普通cos，也不声称严格度量|
|S1|3.1|子带包络二阶调制/S1归一化分支|experiments/cvs_report_mechanisms/frontends.py|verified_approximation|平坦子带gain/低能量数学PASS；3seedclean完整|离散滤波/有限窗近似散射，不宣称任意RX非线性不变|
|S2|3.2|多个lag局部四阶幅相多项式序列|experiments/cvs_report_mechanisms/frontends.py|verified|增益/相位/线性频偏PASS；3seedclean完整|是四阶乘积，不是累积量；有限字典不当完整不变量|
|S3|3.3|中心化IQ轨迹Gram逆、杠杆及指定lag项|experiments/cvs_report_mechanisms/frontends.py|verified|GL2/DC与退化保护PASS；3seedclean完整|固定无记忆仿射作用，不能消一般CFO/卷积；压缩不等同完整行空间|
|F1|4.1|明确低维对数频响基投影|experiments/cvs_report_mechanisms/frontends.py|verified_approximation|固定B消除/零谱/逐包PASS；3seedclean完整|有限窗/噪声/谱底近似；也可能删TX平滑响应|
|F2|4.2|标量跨频相干三节点闭环|experiments/cvs_report_mechanisms/frontends.py|verified_approximation|节点gain/常数loop/逐包PASS；6300源普查/3seedclean|同包8片段仅作为可比较统计假设，不保证真实RX稳定或非平凡|
|F3|4.2|[X(f),X*(-f)]增广矩阵闭环的trace/det|experiments/cvs_report_mechanisms/frontends.py|verified_approximation|非交换GL2 trace/det及秩1保护PASS；源全秩有效率100%/3seedclean|先实报相关图/秩一/单位环退化，数值保护可能破坏严格不变|
|G1|5.1|共享非线性编码器完整C4相位平均|experiments/cvs_report_mechanisms/model.py|verified|实际C4特征平均/普通cos/逐包PASS；3seedclean完整|只保证有限C4群；不平均rawIQ、不称连续全相位/任意非紧群不变|
|I1|5.2|有效Fisher信息的干扰Schur补|experiments/cvs_report_projection/statistics.py|verified|Gaussian连续均值路径明示；PSD数学PASS/实际15pair完成|诊断连续均值路径；TX离散，不伪造现成神经分类损失|
|B1|全篇/6|数学边界、非交换作用、信息冗余、内容控制|analysis/phase1_receiver_invariance_report_20261013_traceability.md|verified_diagnostic|一般filter反例/退化/全部6300源内容及15组CKA完成|分别不变不当联合不变；不信任eq0/eq1同sig_i配对，不虚构前导或新增长度|
|C1|用户覆盖|固定common cos、160身份接口、9类refit smoke|experiments/cvs_report_mechanisms/checks.py|verified|12arm真实模型普通cos及9类refit smokePASS|不创建RX专家或类别特定margin；projection原始源协方差/样本缓存不供Phase2 runtime|
|E1|用户覆盖|全源三seed、30轮、无sourceV性能门槛、clean-only自动truth-last|experiments/cvs_report_mechanisms/design.py/dispatch.py/predict.py/score.py|verified|36scratch30/39cleanreadouts/6552000预测与702指标、239760步及2010SWA张量完整重算|包括元反馈9模型与SWA3轨迹，保留旧源诊断；不追加practical|
|E2|报告6|冻结原CVS的投影zero-gradient对照|experiments/cvs_report_projection/design.py/dispatch.py|verified|三真实CE30角色/祖先/noquerysmoke和24clean rows核实|三CE30父模型固定复用，不借oracle80权重或sourceU标签|

方法总览：9类探索空间全部映射到具体实现；有效Fisher另有明确定义的Gaussian工作模型诊断。新报告不要求替换commoncos为切空间距离，故有界距离只诊断，不作为query决策。教师目标反馈已完成正式 clean 验证，学生主种子均值仍未达到 84%；负结果已保留。

本记录用于完整性追踪，不是额外发布审批、source性能筛选或hash gate。状态随实际代码/测试/远端证据更新；投影24行已完整独立重算，NATIVE79.4562%，其他73.0067%至76.0157%；84%未达成。其余36条新30轮轨迹/39clean读出预登记，数学、真实梯度、protocol负测与唯一独立P0/P1已PASS，正式39组clean已完成及独立重算。最高AFFINE_IQ80.3286%，META_SSL79.9605%，SWA79.8272%，84%仍未达成。实现状态不等于性能成功。

当前18项均已实现并完成组件/实际证据核验；其中6项为明示近似、2项仅诊断，其余按所列理想条件核验。性能目标尚未达成。第一报告的教师目标调整备选已形成独立6联合row/12模型/30轮/clean方案，参数沿原未发布核心固定，不使用本轮target反馈调整；独立P0/P1与实际梯度PASS，已完成并独立重算，学生 FEEDBACK_TEACHER 为 79.9466%，同成本 TEACHER_SSL 为 79.9837%；教师actualseed为primary+10000，仅诊断，不能冒充学生primary三seed目标。

2026-10-13 恢复：再次提供的附件与当前九类机制报告一致，复用上述已验证实现及实际结果。浅中层 TX 条件矩对齐与无条件矩对齐另作 6 模型固定实验，登记 `20261013-phase1-conditional-alignment-clean-manysig-m6-r01`。保持普通 cos、三主种子、从零 30 轮、仅 clean、无源 V 性能门槛；当前本地检查 PASS、上一聊天唯一独立 P0/P1 PASS，正式性能待完成。此为条件均值代理约束，不宣称严格消除真实 RX 作用。

条件矩扩展正式完成 VERIFIED：6 模型从零30轮，共39960更新、1008000 clean预测和108条same-row指标完整重算；TX_CONDITIONAL79.9931%、UNCONDITIONAL79.8954%、匹配UNIFORM_SSL80.0313%，目标未达成。完整报告 `automation_reports/CV-SincNet/20261013-phase1-conditional-alignment-clean-manysig-m6-r01/final_analysis.md`。

下一固定扩展为 `20261013-phase1-crossrx-contrastive-clean-manysig-m6-r01`，两种合法 L 样本级监督对比（ALL_TX/CROSS_RX_TX），原九类实现保持。配方在当前矩对齐 target 汇总之前固定，属于报告思想的训练扩展，非严格原投影公式；CPU/CUDA与普通 cos接口检查PASS。三项追踪见其report.md，完整性能尚待实际训练/clean独立评分。
