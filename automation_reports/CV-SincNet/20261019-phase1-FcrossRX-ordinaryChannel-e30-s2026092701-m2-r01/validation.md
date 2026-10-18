# P78 聚焦验证与真实执行证据

已执行本地 local_checks.py：27 个负例、实际 native OFF 输出/RNG 精确、强制 MP＋AWGN 旁路对独立重放精确、原 U strong 的256/末124包透传输出/RNG精确、finally恢复及原P43 own20/E21/E30边界均PASS。命令：E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 experiments/cvs_ordinary_channel78/local_checks.py --dependency-root E:/type10-7/code/worktrees/cvs_iq_timing_20261016；CWD为当前worktree。exit0，wall2.700472秒，原tool chunk4355d0。19py及2段发布代码AST PASS，0.2726702秒，chunk2bc1a0。未为证据重跑；原输出合并流如实记录在local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/local_checks_actual.json。

这些是原语与真实函数定义的合成检查，不是实际CVS模型/CP或真实IQ测试；本地NN/CP/真实IQ/更新均0。发布launcher首步真实CP＋默认头＋自身EMA与两图1540NN smoke已PASS并直接继续训练。合成augmentor临时强制两原语p=1，finally恢复，不改变训练概率。

一次既有官方N607 ordinary只读preflight已完成，项目存在，当时8GPU空闲；实际worker启动仍遵守每GPU最多2个全owner进程共享容量检查。原始本地证据仅保留local_artifacts，不提交私有运行细节。

本节以下记录最终分析检查、唯一独立P0/P1及实际发布结果。

分析实际14项CPU fixtures PASS（suite5.714秒/process6.1426361秒，chunk3e23ac），另新增最终报告链1项PASS（suite.312秒/process.7288701秒，chunkb6068d），均exit0且NN/IQ/CP/updates0，原合并输出保存在analysis_checks_actual.json，未为证据重跑。

本候选唯一持续独立P0/P1审查PASS，未解决直接问题0。两处直接缺陷（最终报告interaction键、旧runner进程匹配）在同次审查内修复并定点确认；分析改显式source/final，源决定先保存、owner观察后才final连接truth。审查及证据在local_artifacts/.../p0p1_review.md；不要求提交后重复审查。源panel频率未经本轮独立核实，不给出猜测值；训练2.462GHz、固定正式2.45GHz分别记录，原manifest六景fs25MHz/ref30/residual/EQfalse已只读核实，不改信道。

实际launcher首步真实CP1540NN smoke PASS，两原语真实执行后旁路且其余ordinary仍保留，头/Sinc/骨干梯度有限非零；无realIQ/query/update。单归档比较、25py remotecompile及两worker所属PID/CWD/argv/CUDA/实际GPU已独立核实；随后两row首epoch各222更新、日志6714→11879/11882字节，确认实际训练日志增长，未见Traceback。原始私有证据留local_artifacts，不提交。

两条新增row均完成E10/2220更新/累计E30；各七视图prediction完成后独立truth-last评分，原参照0重预测/0重评分。source决定先保存并观察，FAIL不扩E50/种子。最终所有所属句柄已消失，技术评价缺项0。

结果聚合独立算术复核PASS、issues=[]：504正式CM、84正式分区、28最差RX，600源基础CM、2904状态CM与708源分区，源20+3630配对、正式35+595配对、726+7交互、4成本行和650固定NoAs保持率口径检查一致。该复核只检查现有聚合的一致性，不冒充重新truth-last评分；NN/IQ/CP/updates/remoteaccess均0。四张PNG已实际查看，四PDF存在且%PDF-1.4。详细原证据留local_artifacts，不提交原始日志/私有路径。
