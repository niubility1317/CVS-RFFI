# 技术失败：教师构建重入

状态 FAILED，独立后态 VERIFIED。已启动的 NOAS_FT、NOAS_ADAPTER、S0 三行及 dispatcher 均退出；S1/S2/S3 未启动。失败发生在第一条有标签 hook 内、第一次 backward/optimizer update 之前，成功更新为 0。未产生 final checkpoint、source completion、正式 prediction 或 truth-last score；没有科学性能结果。

根因：lazy Tc 加载调用旧 own_model→feature_sequence50.build→nested runtime.installed。内层捕获当前学生 checked_model，CPU 教师被误送入学生训练包装器，最终 get_device_name(cpu) 抛错，并可能覆盖当次初始元数据。因此 r01 的 initial_smoke/初始化元数据不用于新的训练证据或恢复。所有 r01 产物保留，不覆盖。

技术恢复 r02：双教师在任何学生 installed/build/optimizer/deepcopy 范围前预载，checked_model 只挂接既有冻结 holder，hook 缺少 holder 则直接拒绝。数据、物理 ID、seed、损失、矩阵和 E20＋10 不变。无需也不允许启用额外机制修补旧门槛；不按目标表现选择性重跑。

证据：local_artifacts/F_dualTeacher_adapter_E30_20261019/failure_details.json 保存全部三行错误栈、后态、文件清单及 0 prediction/score。技术失败的初始化和首 L 计算开销单列，不计作完成训练或独立模型 seed。
