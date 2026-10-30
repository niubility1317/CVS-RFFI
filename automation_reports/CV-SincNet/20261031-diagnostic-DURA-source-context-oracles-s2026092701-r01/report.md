# 恢复器完整两阶段源域参考替换与信道平滑诊断

- run_id：`20261031-diagnostic-DURA-source-context-oracles-s2026092701-r01`
- group_id：`phase1-dura-source-full-context-oracles`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

纯repeat E30主对照和默认DURA描述对照；512源样本×clean/缓存01/20，完整两阶段固定五种context，源F一致性与IQ误差；0新训练/TX/V/target。

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


## 固定诊断与晋级规则

主对照为从零单阶段 repeat_E30（8cf8e22，30/7410），默认 DURA 仅作描述。固定 512 条源训练池记录（128L/384U）× clean/原缓存01/20，替换完整两阶段 context：原生、实际模拟算子、源 clean 粗恢复、二者，以及固定 DCT32 平滑预测系数。原生方差保持，两个 prior、四步 CG 保持；不更新模型，不读取 TX/V/target。固定原 F50/defaulthead，源 clean 特征为引用，主对照 native 新计算。旧诊断仅改末级 CG16 且无 F，本次定位完整阶段的身份保真。

主对照 known_action 或 rank32 必须在缓存01/20各自使 IQNMSE 下降至少10%、源clean F选择多恢复至少6/512条，每个非clean场景最多少1条、clean场景不少，全部512clean输入选择不下降。rank32另要求实际源算子的两白噪声探针相对误差在两view总体及逐七景均≤1%。优先合格rank32，否则合格knownaction仅支持另行登记源算子课程研究；均失败则不晋级，不放宽。默认和knowncoarse/both仅描述。这是源一致性代理，不是TX准确率、目标收益或恢复上限。无自动训练；未来任何新G仍须独立从零联合30epoch和七景正式预测后truth-last。

CPU2/CUDA空，3072 context、15360完整stage样本、13824新增F特征、1536复用defaultnativeF，首步每模型2条Ls真实G/F smoke。六份源IQ/特征NPZ私有保存，不提交。4聚焦测试已通过，唯一P0/P1审查后发布；首步smoke通过即继续。默认DURA及目标保持。
