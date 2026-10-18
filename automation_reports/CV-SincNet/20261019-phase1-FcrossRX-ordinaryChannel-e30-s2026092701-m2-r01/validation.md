# P78 聚焦验证与真实执行证据

已执行本地 local_checks.py：27 个负例、实际 native OFF 输出/RNG 精确、强制 MP＋AWGN 旁路对独立重放精确、原 U strong 的256/末124包透传输出/RNG精确、finally恢复及原P43 own20/E21/E30边界均PASS。命令：E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 experiments/cvs_ordinary_channel78/local_checks.py --dependency-root E:/type10-7/code/worktrees/cvs_iq_timing_20261016；CWD为当前worktree。exit0，wall2.700472秒，原tool chunk4355d0。19py及2段发布代码AST PASS，0.2726702秒，chunk2bc1a0。未为证据重跑；原输出合并流如实记录在local_artifacts/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01/local_checks_actual.json。

这些是原语与真实函数定义的合成检查，不是实际CVS模型/CP或真实IQ测试；本地NN/CP/真实IQ/更新均0。真实CP＋默认头＋自身EMA与两图1540NN smoke将在发布launcher首步运行，PASS直接继续。合成augmentor临时强制两原语p=1，finally恢复，不改变训练概率。

一次既有官方N607 ordinary只读preflight已完成，项目存在，当时8GPU空闲；实际worker启动仍遵守每GPU最多2个全owner进程共享容量检查。原始本地证据仅保留local_artifacts，不提交私有运行细节。

分析聚焦fixtures与同一次独立P0/P1最终结论尚待收口；无实际启动或性能结论。

分析实际14项CPU fixtures PASS（suite5.714秒/process6.1426361秒，chunk3e23ac），另新增最终报告链1项PASS（suite.312秒/process.7288701秒，chunkb6068d），均exit0且NN/IQ/CP/updates0，原合并输出保存在analysis_checks_actual.json，未为证据重跑。

本候选唯一持续独立P0/P1审查PASS，未解决直接问题0。两处直接缺陷（最终报告interaction键、旧runner进程匹配）在同次审查内修复并定点确认；分析改显式source/final，源决定先保存、owner观察后才final连接truth。审查及证据在local_artifacts/.../p0p1_review.md；不要求提交后重复审查。源panel频率未经本轮独立核实，不给出猜测值；训练2.462GHz、固定正式2.45GHz分别记录，原manifest六景fs25MHz/ref30/residual/EQfalse已只读核实，不改信道。
