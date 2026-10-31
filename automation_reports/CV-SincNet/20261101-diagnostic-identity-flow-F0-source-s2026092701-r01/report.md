# 生成式恢复网络接入真实冻结 F0 的源域梯度诊断

- run_id：`20261101-diagnostic-identity-flow-F0-source-s2026092701-r01`
- group_id：`phase3-identity-flow-source-gradient`；类别：`diagnostic`；阶段：`Phase3-source-F0-gradient-feasibility`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

两条源 L_s 记录检查独立 F0 身份梯度与流匹配梯度；无优化、无目标数据。

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

## 实际源域梯度检查：VERIFIED

真实 Phase3 F0 E50/11100 与源角色契约匹配；两个源 L_s 样本、两条噪声轨迹。F0 身份损失到恢复场的梯度范数为 0.515009522，流匹配损失的独立梯度范数为 8.169138908；两端点梯度范数分别为 0.035222556、0.034696039。F0 参数和缓冲区保持不变，无优化更新、TX 标签、V/OE 或目标 IQ 读取。

这是数值可行性检查，不代表准确率改善、损失权重已确定或达到 3%% 目标。R07 源视图为 2.462 GHz，不是 G10 2.45 GHz；正式训练需要单独固定合法源数据和评测方案。模型为条件生成流，区别于其他对话的确定性时频残差方案；历史 R20 也曾研究不同结构的条件流且结果不佳，不宣称条件流本身新颖。

证据：`E:/type10-7/local_artifacts/pamf_identity_flow/20261101-diagnostic-identity-flow-F0-source-s2026092701-r01/complete.json`；远端完成文件独立读取，训练代码 `881e6f6db099fa6eb57de58c4ebb736f367c5e24`；独立 P0/P1 复核通过。
