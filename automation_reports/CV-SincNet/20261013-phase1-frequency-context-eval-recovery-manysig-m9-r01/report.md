# 频域上下文：终轮权重评估恢复

- run_id：`20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01`
- group_id：`phase1-frequency-context-eval-recovery-20261013`；类别：`cvs`；阶段：`phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原9模型均scratch50已完成；仅修复原生checkpoint参数类型加载顺序，追加独立clean预测/评分，不重训、不改权重/候选/预算

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

恢复最小预登记

代码提交：本报告随实际加载修复和恢复入口同次提交，发布archive内release_commit.txt记录实际提交。原执行提交43bd20494aa470568a68553ce664dc76811b9707。

候选/矩阵：原 LEGACY50 / LOCAL_FREQ / GLOBAL_FREQ 各3种子，固定同9个终轮checkpoint、clean168000/row。不新增训练，不改变源划分、旧伪标签门控/EMA/cos、权重、终轮选择或测试场景。source_proof继续对照原run config。

失败原因：所有9行训练50轮/11100更新后，torch.load缺少原生baseline_origin_sat_view模块。真实终轮CPU复现失败；预先调用native_modules后strict load与零输入烟雾PASS，无query访问。

环境/CWD：N607普通账号的CVS-RFFI Python；/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_frequency_context_eval_recovery_20261013_r01。命令：python -u -m experiments.cvs_frequency_context_recovery.dispatch；共享容量锁，每GPU总进程最多3，空闲显存至少12000MB，owner codex/root/frequency-context-eval-recovery-20261013。当前AGENTS三进程覆盖旧n607文档的二进程数字。

输入：/home/szu2070436088/2510044040/CV-SincNet/runs/20261013-phase1-frequency-context-legacycos-clean-manysig-m9-r01/<row>/source/final_ssdg.pth；scratch来源无祖先，原SOURCE_FROZEN E50，与本契约实际源角色一致。预检逐行strict加载真实终轮，无query；全部旧worker已退出且原prediction/scores不存在才放行。数据仍复用VALIDATED_ONCE clean capsule，物理ID不变。

输出：/home/szu2070436088/2510044040/CV-SincNet/runs/20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01；逐行prediction全部固定后独立scorer读truth。原训练目录和失败记录保留不覆盖，恢复run单独登记。

停止规则：任何来源/预算/物理角色检查失败、原worker存活/已有预测、非有限输出、缺prediction或评分失败停止该新run并保留产物；不影响其他owner、不重训、不得按目标成绩重排/重跑。

预期：9x168000固定预测、每row18个clean指标、同seed配对差、完整原训练日志核验。仅说明Phase1原头结果；丢头迁移与Stage2-C旧新统一识别尚待独立固定验证。

聚焦验证：本地配置/源checkpoint身份不变、新输出隔离及9个错误预算拒绝 PASS；真实原终轮权重先复现依赖错误，再预导入native_modules后CPU严格加载与零输入PASS。一次针对性P0/P1复审PASS，无重复全面审查。
