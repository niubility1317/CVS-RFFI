# WISER-RF历史D92 E0因果诊断实验报告

## 当前结论

截至2026-08-31 00:26 CST，8条run均已按预登记规则进入`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。失败原因不是低性能，而是共享manifest把query指向不存在的`before/enrollment_only/query_leo_*_weak.npz`；实际文件位于同row的`before/apply_only_staging/`。所有部分artifact已保留，未评分、无性能结论、未重启。

## 冻结版本与协议

- 诊断代码提交：`563bbb30041fe8c673fa13ac80def0225b05dad5`，已push且远端OID一致。
- 唯一release：`wiser_rf_cause_suite_20260831_v1_563bbb30.tar.gz`；本地/远端SHA256均为`b05a5ea2d7759552315eef8f403f50cc711cbeb898d0ece58ee0439e6897fbaf`；远端编译回读通过。
- 真实ADV3B02 checkpoint无query smoke：`PASS`，`query_opened=false`。
- 数据保持历史pilot`rx_3_19__seed_713102__k_10__new_5`、seed713102、3个LEO场景，以及同一`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`。所有run均先冻结全部support状态，再读取query；prediction完成后才由独立scorer连接truth。
- `--arms`变更经过TDD：CLI、support/query边界、pilot/scorer arm注册表和不可覆盖根失败路径共28项聚焦测试通过；唯一P0/P1审查的2个发现经定点修复后结论`READY`。

## 8卡矩阵

|物理GPU|run ID|同run矩阵|PID|状态|
|---:|---|---|---:|---|
|0|`wiser_rf_abc_hist_e0_pilot_20260830_v1`|`B0+A+B+C+ABC`|2401124|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|1|`wiser_rf_cause_nol2sp_20260831_v1`|`B0+A(lambda_sp=0)`|2423555|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|2|`wiser_rf_cause_noproto_20260831_v1`|`B0+A(lambda_proto=0)`|2423578|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|3|`wiser_rf_cause_l2sp01_20260831_v1`|`B0+A(lambda_sp=0.1)`|2423554|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|4|`wiser_rf_cause_l2sp20_20260831_v1`|`B0+A(lambda_sp=2.0)`|2423583|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|5|`wiser_rf_cause_vsw01_20260831_v1`|`B0+B(lambda_vsw=0.1)`|2423564|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|6|`wiser_rf_cause_vsw10_20260831_v1`|`B0+B(lambda_vsw=1.0)`|2423579|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|7|`wiser_rf_cause_short_20260831_v1`|`B0+A+B(stage_steps=500/1000/1500)`|2423635|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|

启动后独立回读曾确认7个新增PID的CWD、cmdline、run root和物理GPU映射正确。短训练run最先到达首次query读取并确定性报错；随后5条run自然复现同一缺失路径，剩余GPU0、GPU5、GPU6的run在确认共享必现故障后按无prediction闭合规则精确`TERM`。最终回读为8张GPU均空闲，无残留WISER进程。

## 后续闭环

本批次没有`pilot_result.json`和合法prediction，因此不得启动scorer。若继续，必须先在本地修复matrix manifest的query包选择，使用新Git提交、新release、新run ID和新输出根发布；现有run不得覆盖或重启。
