# 参考响应与时序签名互补性：原头源域联合替换诊断

- run_id：`20261031-diagnostic-DURA-source-response-SIG-s2026092701-r01`
- group_id：`phase1-dura-source-response-SIG-complementarity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定默认DURA/source512×01/20，复用实际加权四分量，联合替换response+SIG的clean源分量，接同一冻结原头；无新G/模型训练/TX-V-target。

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


## 固定源域 response＋SIG 互补假设

固定默认 DURA/source512（128 L/384 U）×缓存 01/20，复用原 F50 的实际四个加权分量。native、response、SIG、response＋SIG、allclean 五案例均接同一原头；联合案例保留 restored identity/PA，按原顺序相加。原 PA、PA＋SIG、源 IQ 资格 FAIL 均保留。本次不读 G 权重，不训练模型，不读 TX/V/target。

两视图联合案例均须较 native 净增至少 6/512、较 SIG 净增至少 2；五个 RX 较 native 不负，每 RX 较 SIG 最多损失 1 且至少 4/5 不负；各非 clean 场景较 native 最多损失 1、clean 不损失。判定不改主比较、不搜索分支或阈值。PASS 仅支持后续另行固定学习假设，不自动启动 G。潜在混合特征未必存在对应 IQ，源 clean F 选择不是真实 TX 准确率，allclean 不是物理恢复上限。

3 项聚焦检查、实际入口和唯一 P0/P1 均 PASS。首步真实 2 Ls 完整 F 四加权分量及原头回放，PASS 立即继续。主完整 F0/head5120，smoke F2/head2，CPU2/空 CUDA。一次 preflight、单归档 SHA 比较、远端编译，实际 PID/CWD/argv/空 CUDA/日志增长核实；独立 NumPy 全1024观测、5120头输入及52分组复算。默认DURA与六景3pp/clean/效率目标保持。


## 完整response＋SIG源诊断

VERIFIED / ANALYZED，SOURCE_ONLY。默认DURA/source512（128L/384U）×缓存01/20，CPU2主循环0.067106秒自然结束。response＋SIG联合替换的clean F选择一致数509/509，单独SIG507/509，额外净增+2/0。第二视图纠2损2、RX1−1/RX3+1，低城区+2与中城区−2抵消。未达到两视图均较SIG至少+2的固定要求，主判定FAIL；其他事前保护条件通过。不以更低特征误差或更高margin覆盖判定。主完整F0/head5120，smoke真实2Ls/F2/head2，0G/优化/TX-V-target；F不变，1024观测/5120头输入/52组独立复算。不新增response训练约束；潜在分量oracle不是可实现IQ、真实TX或恢复上限，默认DURA与完整目标保持。

详见[完整分析](analysis.md)。
