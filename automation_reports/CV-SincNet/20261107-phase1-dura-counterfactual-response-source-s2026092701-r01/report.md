# DURA 合成响应约束 GPU 训练

- run_id：`20261107-phase1-dura-counterfactual-response-source-s2026092701-r01`
- group_id：`phase1-dura-counterfactual-response`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED。独立核实的 N607 队列 PID 为 3907664，等待本任务 GPU 3 的训练进程 PID 3823069 结束；新版本尚未训练。

## 目的与对照

冻结 DURA，对每条合法源 clean IQ 构造相对均方根 1% 的合成扰动，通过该记录的 DURA 估计物理作用注入接收 IQ，并对扰动后信号重新估计 DURA 上下文。四种扰动按固定源缓存顺序循环分配，每个源视图使用一种。多尺度修正网络以原源 IQ 保护损失加 0.01 倍归一化响应误差训练；扰动上下文训练前只计算一次。部署只使用当前接收 IQ 与 DURA，不依赖源 clean、扰动或其他视图。

合成方向是源域代理，不等同于真实 TX 的身份变化；该损失不保证两个目标同时下降，也不证明目标识别收益。固定 32768 个源物理 ID × 3 视图、5 轮、batch 32 和最终轮 checkpoint；不使用 F、TX 标签、V 或目标来训练或选模。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本地 RTX 5070 Ti 真实 DURA 权重及合法源缓存小规模 GPU smoke 通过：96 条源视图、6 次更新，损失和梯度有限、DURA 保持冻结；不生成正式训练权重。12 项针对性测试通过，包括扰动配对、响应梯度和冻结 checkpoint 元数据拒错。一次独立 P0/P1 审查未发现阻塞问题。N607 发布包校验一致且远端编译通过；正式 N607 源 smoke 将由训练控制器在容量允许后自动执行。

此处暂无七景识别结果。冻结后由控制器自动完成 [Phase1 Main split 七景测试](../20261107-phase1-dura-counterfactual-response-ground-test-s2026092701-r01/report.md)，全部预测固定后再独立 truth-last 评分。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
