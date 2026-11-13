# 生产恢复器与冻结识别器特征复用：源样本梯度检查

- run_id：`20261108-diagnostic-Restore-ProductionFeatureReuse-sourceLU-s2026092701-r01`
- group_id：`source-production-G-F0-feature-reuse-exactness`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原JointPrior固定E30权重载入相同FeatureResponse G结构；合法源L/U各一条、四方向，CPU比较原训练图和共享F0特征图的损失、真实G梯度和一次局部AdamW。

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

## 固定检查与停止条件

固定原 JointPrior 从零源训练 final E30/7410，科学 commit 4bba9df3；该结构与 FeatureResponse 相同，参数名、形状和 dtype 已核对，原固定 E30 先于目标访问选定，仅作源端数值诊断。加载状态到两个独立本地真实 G 副本，不改保存文件或在训模型。源 R07 固定 L_s 第 0 条和 U_s 第 6301 条，按原信道记录和奇数轮 1/3 覆盖四方向；同一原冻结 F0、源尺度、全源先验、完整物理与响应损失。

逐项比较原始路径与缓存源扰动特征、共享恢复特征计算图的前向损失、两路 IQ 输入梯度、所有真实 G 参数梯度及裁剪后新建 AdamW 一步；任何非有限值、后验残差大于 1e-8、来源不符或超过 rtol 1e-5 / atol 1e-6 均保留并停止。[计算环境] CPU2 nice10/CUDA empty，命令与输出见 experiment.json；三项本地核心检查及一次独立 P0/P1 审查 PASS。检查首步加载真实源 F0/固定 G。

只测试源两样本、CPU、全源先验和新建优化器，不能证明 GPU、其他 RX 先验、完整训练轨迹等价或实际提速，也不能代替正式识别测试。

## 原训练图与共享图的一步更新未通过预设容差

FAILED / PRESERVED。真实固定源 G、原 F0 和源样本检查执行到局部 AdamW 更新比较；预先设定的 rtol 1e-5 / atol 1e-6 在一个 3200 元素的参数张量上有 1 个元素不满足，最大绝对差 4.3734908e-6、最大相对差 4.0315412e-5。进程已退出、原始日志保留；没有结果完成标记。原容差不能放宽后宣布等价。该失败发生在 CPU 单步数值检查，不是任何已运行训练或正式成绩失败；差异来源与全部参数统计尚待定位。
