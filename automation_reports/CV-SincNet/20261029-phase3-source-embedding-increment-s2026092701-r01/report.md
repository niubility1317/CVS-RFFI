# PAMF第三阶段：原生监督与均匀OE特征增量配对

- run_id：`20261029-phase3-source-embedding-increment-s2026092701-r01`
- group_id：`pamf-phase3-source-embedding-increment`；类别：`cvs`；阶段：`Phase3-source-native-open-embedding-increment`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：FAILED；优化前父兼容检查技术失败，无新模型性能

## 目的与对照

完整F6 E50各独立新增222步：原生L监督对照vs监督+uniformOE；原head/骨干联合CENTER_UNITS训练，U关闭，MSP预指定拒识，原F6同输入闭集保护。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

无科学性能结果。两行在新增父兼容检查失败，原native祖先校验已过；梯度smoke、优化和评估均未发生。原始失败日志和post-state保留，状态FAILED不变。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 版本与本地验证

实际 code/config commit：`22cdc714eab234dceadee917887cc0e9d6debdf5`。完整新候选77项测试和52个subtests通过；一次独立P0/P1 PASS并独立同77项通过，含原生full6合成CPU梯度和CENTER_UNITS实际单步。原库autocast旧API warning保留，不修改历史代码。两行各从原E50独立初始化，原生L监督对照不是纯CE；新增222/累计11322，原head/骨干joint、冷CENTER_UNITS恒1e-4，U/V/novel/target不训练。首实际L2梯度smoke，OE行加OE2，state/RNG/buffer全还原无step，然后立即正式222。固定MSP唯一候选，原F同输入baseline保护，阈值先于novel读取，完整矩阵后判推进。N607一次preflightPASS，reuse已在远端的6134public data，不再传数据或重验；2row并行优先不同空闲容量卡，每卡≤2新增，不停foreign，单行失败不终止健康行。

## 技术失败事实

新增检查误要求parent完成config含split_seed，并要求旧cache路径与已验证复用路径相同。实际seed392005在source contract，cache物理角色由原contract/存储ID绑定；原native E50来源检查已通过。两行0新增优化／0评估、无新checkpoint性能，foreignstop/target0，不作为科学负结果。r02仅定向修正兼容假设，原权重、种子、222步、数据角色和门槛不变；r01产物不覆盖。见[后续r02实际结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_RESULTS_20261029.md)。
