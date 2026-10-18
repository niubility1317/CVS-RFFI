# As 首批 GPU 梯度与裁剪零更新技术诊断

状态 PLANNED；不是性能实验。原 E30 run 20261019-phase1-FcrossRX-splitAdam-e30-s2026092701-m3-r01 保留不重启：完整逐步日志核实 N01 成功16次更新，C01/S01在首批更新前失败。无完成epoch／prediction／score。

唯一目的：区分重复 total-loss VJP、分量VJP和实际backward的数值差异，以及原 physical clip 的单位／缩放差异。复用已发布代码37cce29bb9bf65e372b681242705a1837728b3f2及原As20 sourceFINAL20 parent；取原 C01 的第一个合法 L128/U256 source batch，原6L/As64、original U/RDrop/ORD128、FP32 policy不改。额外两个 totalVJP，总共4VJP；读取clip前后梯度，输出每参数的范数、误差、RMS、cosine及现判据失败坐标数，不输出梯度/IQ/原始ID。

诊断必须在第一批clip之后、首次optimizer.step之前主动抛专用完成异常，核对参数完全未更新、optimizer/EMA成功更新0，保存实际NN和耗时。不会更改原生产容差或实际数值路径，不连接target、truth或正式评估。CPU2；普通N607账户；每GPU所有owner计算进程至多3／free>=12000MB，资源不足等候，不动其他job。

独立不可变release PROJECT/releases/cvs_FcrossRX_splitAdam_clipdiag_20261019_r01，run PROJECT/runs/20261019-diagnostic-FcrossRX-splitAdam-firstclip-s2026092701-m1-r01，唯一owner codex/root/splitAdam-clipdiag。命令 CVS-RFFI/python -u experiments/cvs_split_adam30/diagnose_first.py。实际辅助代码commit随发布保存；原训练依赖仍固定37cce29。输出diagnostic.json、原source初始化/角色metadata和stdout；无final model或性能指标。

意外异常保留原产物、停止，不盲目重试。根因确认后只修该问题，针对性回归及原P0/P1问题定点复审；真实E30需另建r02，不复用失败输出。
