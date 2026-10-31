# MILKR K/N 改进验证

状态：预登记，未启动。基座固定完整 Tri-SIG-TIME SSL（原 REPL 全机制与固定重排），不重新训练 F/G。

## 矩阵与判定

唯一矩阵与命令见 experiment.json。单模型种子 2026092701，P01/MIX01，全部七个既有目标 RX；K=1/5/10/20 × N=2/5/10/20，全补集 query。原版从 r08 权威 jobs 复用对应 r04/r05 冻结结果。没有按历史成绩挑 RX。

R1 为新几何直接拟合；R2 为默认头残差直接 AC；R3 在 R2 上继承 B 函数。S0/S1 分别严格只拟合 raw/restored 支持。R4/R5 是同物理双视图，单列 SD，不能改标 S1。query 两路径分别报告。

主目标为 R3 S0Q0 配对宏 H 至少提高 5 个百分点，另给全部 16 个 K/N 格的差值。仅全部格都达标才称各 K/N 均提高 5pp。S1Q0、SDQ0、S1Q1 作为独立结论，不能用失败接入基线替代主对照。先在每单元计算旧新宏准确率的调和均值，再等权平均。报告旧/新准确率、Macro-F1、floor、逐 RX 负差与资源。

## 源端固定与边界

先运行一次真实 source-checkpoint 无 query smoke，再用合法 L_s 建立五源 RX × 两 episode 代理。每类5 support、32互斥held，仅复用源端 epoch091/092 的 p80 已冻结增强。四 lambda={0.001,0.01,0.1,1}，按全部源路线的 B6/B3/C6-H 等权均值选一个；精确平局取大值。所有目标路线/K/N 共享。代理六类曾被 F 见过，不代表真实新类确认。

C 采用完整 FP64 分解（本轮未实现 Schur 加速），但真实继承 B 函数和 tau。没有 query 拟合或 query 参数选择。全部七 RX 全预测固定后，独立进程才连接 score_only。该 r10 开发基准已曝光，不声称独立盲测。

## 执行约束

N607 GPU0/2/4/6，每卡至多两个新增 worker，逐启动检查至少 4096 MiB 剩余显存。其他 owner 进程不停止、不修改。唯一 owner/run/output root；失败产物保留，技术错误不原地重启。技术停止与预期 artifact 见 experiment.json。低分不触发停止。额外设计审批为 REJECTED_EXTRA_GATE，用户已授权执行。

运行状态、源码 commit、发布 SHA 比较、进程与日志读回、结果在下方追加。

发布 preflight：首选041/042不存在，且release/run root均未创建。改用已核实存在的091/092 p80源缓存；首归档保留，未启动、未访问目标。
独立 P0/P1 审查 PASS；三项聚焦测试 PASS。
