---
name: cvs-experiment-workflow
description: Execute or monitor CVS-RFFI/CV-SincNet experiments in this repository, including local verification, N607 release, technical recovery, and truth-last scoring. Use for experiment work, not general Windows commands or ordinary document edits.
---

# CVS实验执行入口

根据用户请求选择一个入口，沿用已有授权；不要创建额外审批或固定数量的Agent。

- **设计/启动**：读取仓库根[AGENTS.md](../../../AGENTS.md)、当前科学协议及[最小流程](../../../tools/optimizer_workflow_contract.md)，以当前目标和候选报告确定实际矩阵。完成本地工作后按[N607操作](../../../docs/workflows/n607.md)发布。
- **状态/监控**：读取当前run报告和N607操作，仅核实所属进程、进度和产物；健康任务继续监控。SSH超时先只读核实原run，不重复启动。
- **技术失败**：仅既有周期实验监控或当前明确授权修复的任务，按N607操作保留失败产物、本地复现、修复并使用新run；一次性状态检查只报告。不能复现或相同指纹经一次修复仍重现时停止盲目重启。
- **评分/报告**：读取最小流程的完成与指标段；prediction完整后独立评分，给出同row结果和证据边界，完成Git交付。

只读相关引用；已经读过且未变化的规则无需重读。全局旧N607技能的Bash强制路由和额外gate由当前仓库规则取代。外部对比方法权限按当前科学协议处理。
输出应能区分当前状态、已证明的产物、缺项和下一步，不能把本地验证或训练结束写成完整实验结果。
