# 生成流源训练样本积分路径诊断

- run_id：`20261101-diagnostic-identity-flow-source-paths-s2026092701-r01`
- group_id：`phase3-identity-flow-source-paths`；类别：`diagnostic`；阶段：`Phase3-source-trajectory-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 E20、F0，比较原源 L_s/U_s 与 G10 L_s 的 1/2/4/8/16 步反向噪声轨迹，不读取类别标签、V/OE 或目标 IQ。

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

## 固定源训练样本诊断：VERIFIED

每组固定 128 条源训练记录，冻结 E20 恢复器和原 F0，未读取 TX、V/OE 或目标 IQ。G10 L_s 在 1、2、4、8、16 步的平均 IQ NMSE 依次为 0.8920、0.9001、0.9289、0.9726、1.0145；R07 L_s 分别为 0.6494、0.6572、0.6826、0.7223、0.7615。增加步数未解决源训练样本的恢复误差。

G10 的 4 步两条反向噪声轨迹输出差异 MSE 为 0.08836；固定起点的反向噪声差异 MSE 约为 0.001625。两者量纲相同，提示对初始扰动敏感，但不能单凭比值断言因果或后验不确定度。clean 源输入的 4 步平均 NMSE 为 0.00853。

这些源训练样本仅作固定诊断；原 E20 和独立 E30 均不因该结果选择步数或重训。下一步可在合法源训练输入上区分轨迹起点敏感性与训练插值路径上的误差。
