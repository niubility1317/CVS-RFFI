# Phase1 P1后续112实验r2队列交接

## 决策

旧scheduler`1945261`已单独停止，r1当前16个训练进程保留并自然完成。后续候选使用新run`phase1_dgleo_p1verify112r2_20260710`，精确排除当前16个candidate，仅运行剩余112行。

## 修复

1. 每轮clean source-val保留为轻量健康检查。
2. heavy source-val tail geometry和六场景satellite评估改为E10-E180每10轮、E182-E200每2轮，E200强制。
3. tail状态机只消费本轮新heavy评估，缓存值不能重复推进窗口或触发决策。
4. `source_val_sat_hmean`best遥测只在本轮heavy评估完成时更新。
5. 新矩阵写入`eval_schedule_cohort=source_val_heavy_E10_180_every10_E182_200_every2_v1`。
6. 新scheduler把r1训练视为外部占槽；即使尚无自己启动的进程，只要队列仍有候选也会按poll间隔休眠，不再空转。

## 验证

- `py_compile`通过。
- focused pytest：70 passed。
- 新队列dry-run：112候选、排除16、剩余30个机制单元、每GPU14、112条唯一命令、并发硬限制2/GPU。

## 证据边界

r1的16行和r2的112行训练机制一致但评估频率不同。最终分析必须保留cohort字段；tail状态机触发次数、逐epoch曲线和运行时不可直接合并。Phase1结果仍只支持DG、satellite stress、known几何和proxy风险分析，不支持真实unknown或Stage2成功声明。

## 发布状态

- Git实现提交：`d4a270c`。
- 新scheduler PID：`2130580`。
- 新run：`phase1_dgleo_p1verify112r2_20260710`。
- 启动53秒后：scheduler存活且CPU为0.0%；矩阵112行、每GPU14行；`LAUNCHED=0`、r2 candidate目录为0；r1的16个训练进程仍全部存活，GPU compute总数16。
- 判定：新scheduler处于正确等待态，不会重复启动当前16行；后续槽位释放时按同GPU上限2自动补位。
