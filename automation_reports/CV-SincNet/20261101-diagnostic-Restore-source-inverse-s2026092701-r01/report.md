# 恢复性能诊断：原L_s/U_s二十份信道的盲恢复与已知信道反演

- run_id：`20261101-diagnostic-Restore-source-inverse-s2026092701-r01`
- group_id：`restoration-source-blind-vs-known-action`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源端诊断已知信道时的可恢复程度，识别信道估计与细节去除问题；不训练G、不访问目标、TX或验证集，不称理论上限。

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

## 源端性能诊断

只用原 L_s/U_s 的历史 R07 二十份模拟训练观测。最小输入核对确认 63000 个物理 ID 集合相同、角色/RX/day 按 ID 一致；顺序不同必须对齐，clean 最大差 4.77e-7。R07 的 2.462 GHz 与六景均衡采样保持其真实身份，不改标为 G10。固定 128 L_s、384 U_s，二十份共 10240 个源端样本实例，保留全部 clean/六景。

冻结实际原 Tri-SIG-TIME SSL 与默认 DURA。源均值只由原 63000 条 clean 无标签 IQ 计算；oracle 读取源模拟器的真实 FIR、主/镜像调制和噪声，采用固定白先验或全局源均值先验、64/128 次 CG。它只回答已知信道时的有限数值对照，不是可部署恢复器或理论上限。识别器对 clean 的预测一致率不是 TX 准确率。无 G 优化、V/OE/目标/标签读取；性能目标、默认恢复器保持。

## 已完成

VERIFIED：20/20 draw、10240/10240 source cases，0更新/标签/V/OE/目标；实际F0与DURA smoke PASS，独立全部聚合复算 PASS。见 analysis.md、results.json、independent_check.json。真实信道高斯MAP弱于默认DURA，64→128CG逐实例变化为0，不称理论上限。源训练内诊断，不替代Main split性能；默认及目标ACTIVE保持。
