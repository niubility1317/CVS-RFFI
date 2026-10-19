# 接收作用与 prior 的源身份瓶颈诊断：R13

状态：**FAILED_TECHNICAL / 未评分**。实际方法 `855d4e12e813810015fa5a50740e99a0854d1341` 已提交推送且远端 OID 一致；9 项聚焦检查、compile/help 和一次独立 P0/P1 通过；普通账户 N607 的原 source/parent/action 句柄与新路径预检已核实。

固定原 R10native 完整末权重与原 F50/defaulthead，0训练/0sourceV持久更新。原 1800 条 source V 七视图及匹配作用缓存复用；5 个新增读出和原 raw/coarse/R07/native 参考回答：学习 prior 自身是否已补偿作用偏差，单独替换 action、单独替换 variance proxy、同时替换以及 coarse oracle 各有何效果。每个恢复仅处理同一received，不读取clean作为预测输入。known action包含实际AGC/lock，只是oracle条件诊断；原生白输入实方差代理不等于完整真实协方差，冻结prior替换作用也不是理论上界。

5×7 组新预测全部完成且有限、IDs匹配后，独立CPU才连接V标签和clean评分；三郊区、三城区、clean、TX/RX/day、repair/damage、NMSE/频谱/配对特征逐项保留。新run不读取目标数据或目标分数，也不自动晋级任何oracle候选。0新增L/U曝光，原父训练完整使用6300L/56700U、组件E60，F50历史E50/11100另列。具体五读出、固定解释门槛、命令、环境、路径、GPU和停止规则见[登记](experiment.json)，设计追踪见[R13](../../../docs/PAMF_ACTION_FIDELITY_R13_20261019.md)。

源解释规则为known_action_simvar相对native的六场景平均NMSE至少降低5%、低城区≥+1pp、中城区≥+0.5pp、clean相对原输入下降≤0.1pp、最差低城区TX≥−1pp。通过只支持后续改善作用估计，不宣称达到正式 lowurban70%/其他五场景与原始clean差≤5pp；失败也保留全部结果。GPU池0/1/5/7择空卡，最多一条本run worker，全owner cap2/free≥12GB，共享capacity4锁至CUDA PID可见；故障保留产物、不自动重试、不动其他健康任务。


实际发布层：单次归档传输SHA比对、远端compile/help、真实源2Ls检查均通过；随后的首视图预测失败，不能宣称完整发布验证或科学效果。根因是新增predict外层inference_mode在构造原F Sinc参数时移除版本计数器，F缓存读取_version报错。控制器和GPU5worker均已退出；0完整视图、prediction/complete与source summary均未产生，未连接源TX评分标签或目标。部分输出及日志保留，不热改或重启R13。

真实本地F50/defaulthead和2条Ls已重现同一错误，no_grad构造/推理正常；新增七视图五读出回归先RED后GREEN，全10检查通过。唯一计算修复为使用既有成功预测器的no_grad；数据、父权重、矩阵、CG和解释门槛均不变。一次定向P0/P1复核无残留问题。技术替代使用新不可覆盖run 20261019-diagnostic-restoration-action-fidelity-F50-s2026092701-m5-r14，原方法提交855d4e12e813810015fa5a50740e99a0854d1341和失败产物保持。用户70%/5pp目标未完成。
