# CVS全主干有界复FIR前置：静态与逐包动态×四seed

- run_id：`20261003-phase1-cvs-frontfilter-identity-manysig-m8-r01`
- group_id：`cvs-clean-frontfilter-identity-ce`；类别：`cvs`；阶段：`Phase1-CVS-source-research`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

所有身份通路仅消费前置有界滤波后的IQ；比较全局静态与逐包神经预测的复系数。无原IQ身份旁路。

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

## 冻结实施说明

[设计与限制](../../../docs/CVS_FULL_BACKBONE_FRONTFILTER_20261003.md)。本轮为两个前置滤波候选各4seed，16源记录比较后只测试被选中的新候选；旧控制胜出则核验复用已有clean。原始IQ不直接进入身份主干，static新增48参数、dynamic新增320参数，均保持原单一CE和训练策略。源代码与实际远端commit在发布后补实测值。

既有source全量1440单元分析显示，RX3是全部16模型的最差RX；crosspath两候选对Anchor在RX3分别新增76和64次四模型累计错误，RX6分别减少20和23次。该分布说明退化有接收条件差异，但不证明信道或RX因果；统计口径与全量结果见[源单元核对](evidence/prior_source_cell_audit.json)。本轮架构依据仍是此前辅助G可被原IQ主干绕过的源归因事实，不按这些单元重新采样或加权。

本地公开纯DC输入的FP32相位检查首次失败已定位到原频域log-ratio对极弱边缘谱的数值放大。设计记录原误差及同state FP64确认；不修改训练精度或原主干，也不宣称整网对全部FP32输入都有统一相位误差界。

## 本地验证

状态LOCAL_VERIFIED，尚未启动。模型20项、源协议58项、全V诊断5项、源分析87项，共170项独立测试通过；另完成8行公开输入三步CE冒烟与一次P0/P1发布审查。检查不代表性能提升。纯DC数值边界保留原始失败和分层核验。

[本地验证](evidence/local_validation.json) · [8行冒烟](evidence/local_model_smoke.json) · [数值根因](evidence/constant_phase_roundoff.json) · [独立审查](evidence/source_p0_p1_review.json)。
