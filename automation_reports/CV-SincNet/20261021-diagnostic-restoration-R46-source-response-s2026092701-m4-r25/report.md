# R25：冻结恢复器的源 IQ 有限扰动响应诊断

状态：LOCAL_VERIFIED，29 项聚焦检查及唯一一次独立 P0/P1 审查 PASS；尚未发布，无新训练或正式识别分数。

本诊断固定原 R46 350521、R18 factor625_stage、R22 complex625 和 complex625_identity 四份源域冻结权重。后面三版都是 278180 参数，用于检查输出拟合与局部响应是否存在不同取舍；不按目标分数重新挑选权重。原父 source-final E20/4940；三学生均 E20/19700。历史正式结果已经暴露，属于工程机制诊断，不声明独立盲测。后续训练唯一初始化及 IQ 教师仍是用户指定的原 350521 父版。

输入只用原 7/63/30 的训练角色：128 条 L_s 和 128 条 U_s，按不透明角色位置均匀取样，复用 R21 面板及 R07 的 clean、received epoch01、received epoch20。源 RX 索引 1/3/4/6/8、day1/2/3。既有 READY/EXACT_MATCH 直接复用，不重划分、不重验数据。TX 标签、V、target、F 权重及输出读取均为 0；无新网络、优化器更新或分类头拟合。

每个 batch 的四个模型共享同一随机方向和 i*x 相位切向方向，扰动与输入的复 RMS 比固定为 0.005、0.010，均执行正负成对前向。odd=(G(x+delta)-G(x-delta))/2；even=(G(x+delta)+G(x-delta))/2-G(x)。相对响应误差为 power(student_odd-parent_odd)/max(power(parent_odd),0.01*power(delta),1e-8)。另记 clean 归一化响应误差、学生/父响应增益、even/odd 比、父响应分母 floor 命中率。两个步长的 odd/epsilon 差异以 max(low_step_scaled_power,0.01*input_power,1e-8) 归一化，检查有限步长一致性。有限扰动响应不是精确 Jacobian，也不证明 TX 真实性。

每个模型有 768 个中心输入、6144 个扰动输入，诊断共 6912 条；另有首步真实 2L_s 无 query smoke，PASS 后立即继续。四模型合计 27648 个诊断输入和 8 个 smoke 输入，唯一物理源位置仍为 256。每版 L_s/U_s 诊断曝光各 3456；不新增模拟信道。全量矩阵 4 模型×3 视图×all/L_s/U_s，响应再分 2 方向×2 步长，输出 1224 个指标汇总行，含 mean/median/p90/p95/max。只保存汇总和私有不透明面板位置，不发布 IQ。

全程 N607 普通账户、CPU2、一条诊断进程、GPU0 条 worker，不修改健康任务。完整 coarse/action/prior/三阶段及六 CG 均实际执行。逐位核对每份模型全部状态前后不变、无梯度、eval；核对实际 2/768/6144 输入计数。CPU 总壁钟包含加载/检查/面板/诊断，不能作为部署延迟。R23 保持原训练和逐版完整七景测试，R24 等待所有既定正式结果；本诊断不产生新权重。

使用一个实际代码/配置 Git 方法 commit、新不可覆盖 run/release/output，一次 N607 预检、一次发布归档两端 SHA 比较、一次远端 compile。发布后以实际 PID/CWD/完整命令、CPU/GPU 状态、smoke 与日志增长核实。系统性技术失败保留产物，不自动重启、回退或修改其他 owner。环境、路径、启动命令、检查点来源、固定矩阵和停止规则见 experiment.json。

唯一独立审查由 /root/review_r14_prior_compact 完成，独立运行 29 项测试，未发现 P0/P1；未读真实 IQ/远端、未编辑文件。通用技能的独立设计审批属于 REJECTED_EXTRA_GATE/NONBLOCKING，不另加发布门槛。低 urban≥70%、其他五景距 raw clean≤5pp、clean 保持及真实 TX 细节目标仍未完成；源指标不替代正式测试。
