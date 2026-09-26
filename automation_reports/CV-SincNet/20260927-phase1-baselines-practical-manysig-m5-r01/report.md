# 外部对比方法：统一残差信道、五模型种子源域训练

- run_id：`20260927-phase1-baselines-practical-manysig-m5-r01`
- group_id：`phase1-external-practical-residual-matched`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

8条方法路线，固定L/U/V与200轮预算；5个模型种子。392005是历史优化参考，其余4个在本次目标评分前固定。

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

## 发布前验证

用户已明确确认Phase1增强以及Phase2 support/query统一使用residual_noeq（post_sync残差、无额外均衡）。该授权覆盖本次运行对旧LEO_WEAK默认的替换；不修改历史数据或结果。

源域角色依据N607已有source_contract逐物理ID匹配，L/U/V为6300/56700/27000。仅构建源域loader；U输出标签为-1，元数据不携带TX、隐藏真值或capture_group。训练和选模不读取目标。所有行从零训练、固定E200末轮权重，无继承checkpoint。数据划分不随model seed改变。

首轮包括392005历史优化参考种子；2026092701至2026092704为预先固定的新模型种子。所有方法全量报告同组种子。不得以这些新种子中CVS表现好坏删行；392005单列历史探索，不将其与新种子混合冒称完全独立确认。当前尚无CVS新版本对齐结果，不能据此承诺性能优势。

28项聚焦测试通过，包括5个骨干真实反向更新、实用残差信道确定性与批次顺序不变性、物理角色互斥、U真值隔离、source_only目标拒绝与最终权重保存。独立P0/P1审查发现RIEI PL仅更新head；已改为两个互不重叠的优化器同时更新，并通过骨干参数变化测试和唯一定点复审。其余最终配置/防覆盖/GPU队列检查通过。

POSTER和RadioNet使用作者结构的PyTorch移植，并与本地Keras作者模型进行相同CPU float32权重映射比较：最大概率绝对误差均为1.4901161193847656e-08。冻结前缀在真实微调更新后完全不变，证据见../../../docs/evidence/li_torch_port_parity_20260927.json。POSTER保留作者代码3个Dense层重新初始化/微调的行为（与论文“最后层”措辞有差异）；RadioNet DF仅分类头可训练。源训练200轮和星地增强是明确的统一条件扩展，不冒称逐项原论文设置复现。

作者来源：[POSTER](https://github.com/SmartHomePrivacyProject/RadioFingerprinting)、[RadioNet](https://github.com/UCdasec/RadioNet)。固定作者版本在baselines/common/li_backbones.py标注。RadioNet本run仅DF有标签微调所需源骨干，不等于已运行其ADA/triplet所有变体。

Phase1训练完成后还需单独进行冻结目标预测和评分。Phase2数据builder须为新residual_noeq received IQ建立单次观测的物理ID一致capsule；原LEO_WEAK的received IQ不能静默改名复用。Phase2及CSIL/MoPC尚未启动。

## 启动读回

VERIFIED：发布提交d8d0f333d968a7ec4df3b85636c474c634c6c8cb，本地与远端归档SHA256一致，远端编译通过。dispatcher PID889485；8个首seed训练进程的PID/CWD/argv/GPU及日志已独立核实，32行排队。证据：evidence/launch_readback.json。所有当前日志tested=0；仅源域训练，尚无目标成绩。
