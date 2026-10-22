# 最终T1-E50＋DURA：17契约三种子完整对比

- run_id：`20261021-phase1-T1-DURA-comparison-m153-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Final T1 correction: 20(13L+7LU)+own10LU+own20LU, AsCE/KL0, sixL, nativeEMA/U, defaulthead. Existing F0 final T1 three seeds reused. Old wrong33L17U unstarted179 withdrawn; all healthy old workers/results preserved as historical. 150mainF+63mechanismF;35same-contract fullG+21Gstudents;249prediction/scoring rows;2exclusive timing panels.

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

Final version correction and exact matrix: see docs/goals/CRIC_DURA_COMPARISON_AND_LATENCY_20261021.md. Final T1 is13L37U, not old33L17U. Current state PLANNED until independent PID/log/CUDA evidence.

时延兼容故障：统一面板因原生F的baselines命名空间遮挡外部RIEI模块失败，尚无计时样本。已本地复现，独立latency-r02恢复这一面板；旧失败行保留并明确替代，不重启主队列或健康worker。机制附面板无外部模型，不受同一导入错误影响。

**VERIFIED / RUNNING**：8张卡均有本矩阵真实CUDA worker，F源训练/完整G谱系/G消融实际步数增长；不接管旧错误配方健康worker。当前首段F监督约85至90秒/轮，U和G后续阶段仍待校准；整批暂估3至7天。统一时延r01兼容失败已独立r02恢复等待独占卡，主队列不变。

已闭合最终T1＋DURA的F0三Fseed/一个固定G、七景42读出格。详见[子矩阵结果](final_T1_F0_results.md)。clean84.4839±0.9660%，六景77.6468±0.8461%，lowurban67.6562±0.7124%；原T1 raw六景70.6013±1.0624%，配对增益7.0455±0.5949个百分点。只表示F0完成，不把17契约整批标ANALYZED，不按结果调整其他实验。

## 2026-10-22 源输入绑定技术缺陷与替代

已只用源侧证据确认F0 IQ实际存储顺序与新metadata的物理ID顺序不一致，L_s6300条中5291条标签错配；按既有ID索引对齐后为0。旧F01至F21全部三seed的产物保留并标记INVALID_INPUT_BINDING，不能用于组件贡献；旧低分不是移除组件的有效负结果。6条F0匹配监督/普通PL控制有同一缺陷。

技术替代为20261022-phase1-T1-DURA-F-ablation-binding-m63-r01及20261022-phase1-T1-DURA-F0-controls-binding-m6-r01；原63/6科学设置和三seed预算保持，从零训练，不继承错误checkpoint。原最终T1三seed及全部合法G/其他16源契约不重训。原运行健康PID保持，未启动错误下游撤下；新owner和分析replacement映射见20261022-phase1-T1-DURA-F-binding-control-r01。最终汇总必须按替代映射使用新预测/评分，旧结果单列历史，不混入正式表格。
