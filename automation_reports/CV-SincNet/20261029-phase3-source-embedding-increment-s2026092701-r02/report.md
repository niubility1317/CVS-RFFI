# PAMF第三阶段：原生监督与均匀OE特征增量配对

- run_id：`20261029-phase3-source-embedding-increment-s2026092701-r02`
- group_id：`pamf-phase3-source-embedding-increment`；类别：`cvs`；阶段：`Phase3-source-native-open-embedding-increment`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；VERIFIED / ANALYZED_SOURCE_ONLY，NO_PROMOTION

## 目的与对照

完整F6 E50各独立新增222步：原生L监督对照vs监督+uniformOE；原head/骨干联合CENTER_UNITS训练，U关闭，MSP预指定拒识，原F6同输入闭集保护。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

两行完整222步、预测和独立评分。固定MSP监督未知47.8095%、OE59.9683%（配对+12.1587pp）；OE closed98.4963%、open97.9593%、FRR1.5185%、原F总损伤0.5074pp。已知保护总体和每RX均过，未知总体和各RX均未95，NO_PROMOTION。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 版本与本地验证

实际 code/config commit：`28d53ea9b38c19f642c59e31f178a3f65f40894d`。原完整候选77项/52subtests与独立P0/P1已通过；本技术修复仅针对父兼容字段，2项/3subtests及一次本P1定向复核PASS，12项未变训练测试未重复，不声称修复后重跑整套77。原库autocast旧API warning保留，不修改历史代码。两行各从原E50独立初始化，原生L监督对照不是纯CE；新增222/累计11322，原head/骨干joint、冷CENTER_UNITS恒1e-4，U/V/novel/target不训练。首实际L2梯度smoke，OE行加OE2，state/RNG/buffer全还原无step，然后立即正式222。固定MSP唯一候选，原F同输入baseline保护，阈值先于novel读取，完整矩阵后判推进。N607一次本release preflightPASS，reuse已在远端的6134public data，不再传数据或重验；2row并行优先不同空闲容量卡，每卡≤2新增，不停foreign，单行失败不终止健康行。

## 实际完成与证据

两个原F6 E50独立增量各222/累计11322步，原头和编码器联合；监督为原native KL/CRIC/As骨架，非纯CE。真实原L2／OE2梯度smoke与最终checkpoint smoke均PASS，state/RNG恢复无额外优化；每行28416 L采样/170496包视图/14208 As，OE另14208。恒lr.0001、epoch51、CRIC.1，原U_s/V/novel/target不训练；source knownV只冻结风险阈值。全部5分数总体/每RX的encoder≤.5pp、totalopen≤2pp、FRR≤5%保护通过；未知均未95，仅uniform_oe::msp可晋级，eligible空。GPU2监督/GPU1OE，在跑owned各1496MB的工具观察转录标明来源；最终post-state已退出，foreignstop0。两完整trace有随机波动，不宣称收敛。全流水线85.065秒，无新技术故障。

[完整结果报告](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_RESULTS_20261029.md)；[444步、全部分组、smoke和10份日志入口](E:/type10-7/local_artifacts/pamf_phase3_source_embedding_increment_20261029_r02/analysis_summary.json)。同knownV经验风险/同novel重复source开发非新盲测，也非跨RX/星地/适应/注册成功。不改阈值或MSP规则；下一固定编码器＋原4OE/consensus读出组合尚未实施。
