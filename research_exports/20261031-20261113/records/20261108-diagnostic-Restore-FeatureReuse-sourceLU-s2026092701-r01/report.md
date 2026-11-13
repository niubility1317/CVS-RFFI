# 冻结特征复用：源样本计算图一致性检查

- run_id：`20261108-diagnostic-Restore-FeatureReuse-sourceLU-s2026092701-r01`
- group_id：`source-F0-feature-reuse-exactness`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

合法源L_s/U_s各一条、四种固定扰动，比较在线与缓存参考特征、重复与共享F0图的数值、输入和代理参数梯度及AdamW单步；不修改在训模型。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定的检查和停止条件

唯一候选：在 CPU 上固定原三视图 Tri-SIG-TIME SSL F0，取原源训练缓存中 L_s 位置 0 与 U_s 位置 6300，按原确定规则覆盖四种 1% 扰动。对每个方向比较在线参考与四方向缓存特征，以及重复和共享 F0 计算图的损失、两个 IQ 输入梯度、简化 Conv 参数梯度和一次 AdamW 更新；代理 Conv 固定 seed193，不是生产恢复器。只读取原 L_s/U_s IQ 和原源特征尺度，不访问 TX 标签、源 V/OE 或目标，不更新 F0 或实际 G。

代码 commit ddb59c1c285059698edd64964b287f72c93462ad；命令、CPU2 nice10/CUDA empty、不可覆盖运行与日志目录见 experiment.json。原 F0 和角色边界不符、结果非有限或 rtol 1e-5 / atol 1e-6 一致性断言失败即保留日志停止。两项本地针对性检查通过，一次独立 P0/P1 审查 PASS；实际源 F0 是远端检查首步。完成仅说明两条样本、CPU 和代理 G 的检查结果，不证明生产 G/GPU 轨迹等价、训练提速或正式识别质量。

## 检查完成

VERIFIED / ANALYZED：真实 F0 与源 L_s/U_s 各一条的 CPU 检查四方向通过。特征和损失最大绝对差为 0，IQ 输入梯度最大绝对差 1.303852e-8，代理参数梯度最大绝对差 1.192093e-7，一次 AdamW 更新后的代理参数最大绝对差为 0。GPU、生产 G 与实际训练加速尚未验证；详情见 analysis.md。
