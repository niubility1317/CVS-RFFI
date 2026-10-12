# 完整波形＋关系互补分支：源端30轮预登记

继续84% clean三seed目标，从头训练30轮。上一质量FiLM15模型完整评分：CE61.5222%、FREE61.3370%、QUALITY60.4481%，未过源条件，无目标访问，不放宽阈值。此前独立几何表示较原IQ差，但不是完整主干的互补分支试验。源端现有固定prediction显示COMPLEX关系能纠正7.1704%的native错误，RAW关系7.3111%，普通IQ11.0259%；这是oracle上界，不能声称可部署或可学习融合优势，不用它选目标参数。

保留原native完整波形/时频/PA/参考/内部门控与cos头，仅新增同包关系CNN，通过joint160+.2*tanh(gain160)*branch160接入原cos分类头。gain0初始化，初始完整模型等价。RAW_REL和COMPLEX_REL使用完全同结构197177参数，相对原177025增加20152（11.38%），不是无限堆容量。

两个分支使用同一固定数值预处理：包RMS归一化，在80:160区间以lag20相位斜率数值补偿全256点，再固定公共32维投影，归一化投影能量。无CFO数值送分类器；不把窗口/lag称为已核实前导、硬件参数或真实RX作用。RAW为II/QQ/IQ/QI外积，COMPLEX为复Gram实部/虚部重复，4x32x32经相同Conv2d/GroupNorm/SiLU和固定池化映射160维。弱幅相不变性只属于新增通道，完整网络仍保留原波形，不宣称全模型或任意RX滤波不变。

五个源RX整机留出，所有身份/分支从零，只用其余四RX fitL5040；heldRX完全不见。原heldV5400只在final30冻结后predict/score。两个分支x五RX=10个新模型，各30x222=6660，共66600更新，保留全部15TX类对/day/混淆。上一相同seed、fitL、heldV、native、优化器与LR的五CE30结果只复用比较，共历史33300更新，不加载其权重。没有U/teacher/EMA、星地/其他训练增强、额外CE或域损失。

每个分支独立晋级：源均值相对CE至少+1个百分点，至少4/5RX非负，最差RX下降不超过1个百分点。合格者按源均值最高选择，完全并列偏RAW，不接触目标反馈。弱不变机制归因还需相对RAW均值+0.5个百分点、4/5RX非负、最差RX不劣1个百分点；普通容量收益不得当作不变机制收益。合格方案再进入全源从头30轮三seed及自动clean+六practical，逐row全部预测固定后独立truth-last。

run/output根不可覆盖，单owner，每GPU最多3个总进程，实际代码/config一次Git提交、一次P0/P1审查、聚焦负测、真实checkpoint无querysmoke先行PASS即续、一次N607preflight/归档传输SHA/远端compile/启动进程log核对。仅预登记系统技术故障停止所属run，不因低性能停止、重启或影响其他owner。精确矩阵、环境/CWD/命令/路径/stop rule见experiment.json；实际commit见release_commit.txt及启动后execution_commit。

REJECTED_EXTRA_GATE：brainstorming单独设计/spec/plan多轮审批不在项目最小流程白名单中；依据用户已有继续研究授权直接执行，未新增门槛。现有目标基准已暴露，本研究不是新的独立盲测；84%尚未达到。

## 技术失败：训练前预登记路径

FAILED：独立远端核对 dispatcher/worker 均已退出，run仅logs目录，训练/源预测/目标预测均未启动。唯一差异是 baseline_ref 用str(Path)在Windows登记成反斜杠，Linux运行生成正斜杠。本地同一断言修复前失败，改为as_posix后10行全部PASS。旧run/release/归档保留，不覆盖或重启；相同模型、seed、数据、source-only权限、30轮与晋级阈值在r02发布。该失败没有性能结果。
