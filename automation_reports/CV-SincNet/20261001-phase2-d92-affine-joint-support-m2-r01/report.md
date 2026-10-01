# 解析自由截距残差LocalRidge与support监督adapter联合适应注册

- run_id：`20261001-phase2-d92-affine-joint-support-m2-r01`
- group_id：`d92-affine-joint-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ARTIFACTS_COMPLETE（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留旧类核尺度、实际B函数先验和函数坐标adapter，以带解析自由类截距的残差LocalRidge经完整隐式梯度联合训练；保留完整interaction，不做参数网格。

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

LOCAL_VERIFIED/VERIFIED：单一解析自由截距结构已实现，79个不同相关检查通过（core/entry/ops58、summary10、reporter11）；唯一独立P0/P1审查无未解决问题。完整Schur和非零g_b伴随、当次B→C状态、全部RHS与状态字节、真实事件顺序经合成检查。实际四缓存/160parent输入绑定及新输出无冲突已核实。方案在旧AJLR完整评分前按数学推导确定，无部分结果选模/参数扫描。未发布或启动，真实性能尚未知；A及B−A=N/A，query/source样本不读，目标ACTIVE。

RUNNING/VERIFIED：AffineJoint已单次发布启动，实际runtime81a226a8d1cef34ea817ada87070bd89912d027d；supervisor451253及workers451265/451266的PID/argv/cwd独立读回一致。实际生效算法、缓存身份、run/row和CPU/BLAS参数匹配。首次读回16/160parents，两个rx3运行、两个rx1待排队，详细训练文本已增长；尚无完整性能结论。query/source样本0，encoder/checkpoint不加载，新增地面摘要0B。禁止重复publish/停止/重启/热修改；旧AJLR分析r02保持原PID394399/handle99201，目标ACTIVE。

COLLECTOR_VERIFIED/VERIFIED：独立只读AffineJoint训练诊断工具已完成，12个不同合成检查通过（9.02s），累计91个不同相关检查。两阶段目录/训练引用快照与完整训练档案提取、13项RHS/intercept费用、gCE=gZ−Z及实际B→C状态诊断已覆盖；不读取外层评分、不做新拟合/求解/SVD/前向。真实采集尚未执行，当前训练release/runtime不变，run继续RUNNING；不能据此宣称性能改善。

ANALYZER_READY/VERIFIED：未来Affine独立分析器的完整中心化VJP以代数等价的求和/外积展开，O(n³+hn²)改为O(n²+hn)，保留移动reference项、非零g_b及原有核验与容差；64MiB数组预算/64entry缓存按实际nbytes管理，每lane释放，缓存命中仍核查文件状态且不跳过数学核验。17个新增不同检查及10项既有summary回归通过，累计108个不同检查。首轮1项fixture混淆非均匀q的伴随零方向，原独立矩阵oracle已一致；仅修正fixture后受影响8项通过。健康训练runtime81a226a8d1cef34ea817ada87070bd89912d027d及AJLR分析r02不变；Affine分析尚未启动，无真实加速或性能结论，目标ACTIVE。

ARTIFACTS_COMPLETE/VERIFIED：AffineJoint四row/160parent训练已结束，supervisor及全部workers退出；完整marker/state/manifest元数据和实际计数独立读回一致。实际更新2592次、头拟合35988次、latent SVD3240次。完整数学档案仍由独立分析核验；完成不代表性能改善，query/源样本0。下一步唯一analysis release d92_affine_joint_analysis_20261001_r01，root sole owner，目标ACTIVE。

ANALYSIS_R01_FAILED/VERIFIED：Affine首次独立分析PID489447已退出，无summary；入口阶段记录含schema/method，汇总重建漏字段导致严格比较失败，B→C顺序正确。原160训练及失败档案保留；只修复分析器和真实入口回归，不放宽核验、不改变训练。新analysis release d92_affine_joint_analysis_20261001_r02尚未启动；等待相关测试、Git交付后单次分析，目标ACTIVE。

ANALYSIS_REPAIR_READY/VERIFIED：Affine分析器严格阶段比较已补齐真实入口schema/method；B→C顺序、全部字段、数学核验和容差保持原样。真实evaluate→core写盘及篡改回归在内的11项相关测试通过（106.31s），其中1项新增、10项既有回归，累计109项不同检查。r01失败和完整训练产物保留，新release d92_affine_joint_analysis_20261001_r02尚未启动；提交推送核对版本后单次独立分析，不重跑训练，目标ACTIVE。

ANALYSIS_R02_RUNNING/VERIFIED：Affine完整独立分析修复版本已单次启动，PID501716/argv/cwd与预登记命令独立读回一致，实际analysis commit 1acc83a584ace40f294a433ace80629472ee2525，本地handle2931。原训练runtime81a226a8d1cef34ea817ada87070bd89912d027d，四row/160已结束；未来分析器采用已验证完整等价VJP和64MiB数组缓存，不跳过数学核验。完整summary尚未生成，无性能结论；A/B−A=N/A，query/源样本0，目标ACTIVE。

MATH_SCOPE_AUDIT/VERIFIED：独立query-blind复核已补充联合目标的可证明范围；未读取真实评分或snapshot，不改方法、参数和健康进程。Affine r02独立读回仍PID501716/start5870553/handle2931，完整summary未产生；训练160已完成，保持ARTIFACTS_COMPLETE，目标ACTIVE。

RELATED_DEVELOPMENT/SYNTHETIC_VERIFIED：正核条件Affine解析头及完整低秩伴随通过20项独立KKT/差分/反例测试；联合SFT上层仍在实现，尚未启动该方法实验、无性能结论。Ground-A原分类头通过43项合成回归，但真实head行绑定、factory/s/weight包未接入，A与B−A仍N/A。实际source_contract classes及dual/no-sat-adapter元数据已只读核实。Affine诊断传输16项检查通过，真实snapshot/extract未启动；当前健康analysis保持PID501716/start5870553/handle2931/源1acc83a，无summary，不修改release或重启。目标ACTIVE。
