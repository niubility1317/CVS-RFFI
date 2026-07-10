# Phase1 DGLEO P1 Verify128 Experiment Design

正式运行报告位于：
`E:\type10-7\automation_reports\CV-SincNet\phase1_dgleo_p1verify128_20260710\report.md`

## 设计摘要

- 128个Phase1 source-only实验。
- 32个机制单元×4个配对seed。
- 8张GPU每卡总计16个实验。
- 每卡最多2个并发；任一实验terminal后同卡队列自动补位。
- 机制组覆盖联合强度、satellite、L_s解耦、U_s解耦、局部组件、U_s三态、直接open-set指标和闭集/open-set梯度冲突。
- 所有正式行使用`ManySig.pkl`、`rho_label=0.10`、final-only checkpoint和独立full-physics satellite冻结评估。
- local/global、L_s/U_s等关闭行是诊断性消融，预期可能被P1 fail-closed门控标记为non-promotable。
- Phase1指标只说明DG、known几何和proxy risk，不声明真实unknown或Stage2成功。

## 机器可读证据

- 矩阵生成器：`code/scripts/launch_phase1_dgleo_p1verify128_20260710.py`
- 完整矩阵：根目录正式报告的`artifacts/candidate_matrix.json`
- 调度事件：远端`logs/phase1_dgleo_p1verify128_20260710/scheduler_events.tsv`
- 每候选日志：远端`logs/phase1_dgleo_p1verify128_20260710/<candidate>.out`
- 每候选结果：远端`runs/phase1_dgleo_p1verify128_20260710/<candidate>/`

## 验证

- 36项focused测试通过。
- 128条命令全部通过当前trainer参数解析。
- 128个candidate ID和命令均唯一。
- 每GPU恰好16行，调度器CLI只接受`--max-active-per-gpu 2`。
- satellite train/eval family保持独立，checkpoint只允许`final_ssdg.pth`。

## Server launch status

- First attempt `phase1_dgleo_p1verify128_20260710` was stopped after a pre-training import failure caused by stale remote `phase2_prototypes.py`; 21 attempted rows are retained as failed startup evidence and excluded from the formal matrix.
- Full P0 runtime dependencies were synchronized and hash-verified before retry.
- Formal run: `phase1_dgleo_p1verify128r1_20260710`.
- Scheduler PID: `1945261`.
- Startup health: 16 active compute processes, exactly two per GPU, 16 non-empty metric files, no fatal markers.
- The scheduler backfills the same GPU after any candidate writes a terminal exit until all 128 formal rows have run.
- Current state is startup PASS only; no performance or promotion claim is made.
