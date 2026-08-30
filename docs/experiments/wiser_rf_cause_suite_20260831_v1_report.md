# WISER-RF历史D92 E0因果诊断实验报告

## 当前结论

截至2026-08-31 00:11 CST，7条单因素因果诊断均已发布并进入`RUNNING`。它们与GPU0已有的完整ABC pilot构成8卡矩阵，每张物理GPU严格1个正式实验。当前只有运行与绑定证据，尚无性能结论；任何低性能都不会触发停止。

## 冻结版本与协议

- 诊断代码提交：`563bbb30041fe8c673fa13ac80def0225b05dad5`，已push且远端OID一致。
- 唯一release：`wiser_rf_cause_suite_20260831_v1_563bbb30.tar.gz`；本地/远端SHA256均为`b05a5ea2d7759552315eef8f403f50cc711cbeb898d0ece58ee0439e6897fbaf`；远端编译回读通过。
- 真实ADV3B02 checkpoint无query smoke：`PASS`，`query_opened=false`。
- 数据保持历史pilot`rx_3_19__seed_713102__k_10__new_5`、seed713102、3个LEO场景，以及同一`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`。所有run均先冻结全部support状态，再读取query；prediction完成后才由独立scorer连接truth。
- `--arms`变更经过TDD：CLI、support/query边界、pilot/scorer arm注册表和不可覆盖根失败路径共28项聚焦测试通过；唯一P0/P1审查的2个发现经定点修复后结论`READY`。

## 8卡矩阵

|物理GPU|run ID|同run矩阵|PID|状态|
|---:|---|---|---:|---|
|0|`wiser_rf_abc_hist_e0_pilot_20260830_v1`|`B0+A+B+C+ABC`|2401124|`RUNNING`|
|1|`wiser_rf_cause_nol2sp_20260831_v1`|`B0+A(lambda_sp=0)`|2423555|`RUNNING`|
|2|`wiser_rf_cause_noproto_20260831_v1`|`B0+A(lambda_proto=0)`|2423578|`RUNNING`|
|3|`wiser_rf_cause_l2sp01_20260831_v1`|`B0+A(lambda_sp=0.1)`|2423554|`RUNNING`|
|4|`wiser_rf_cause_l2sp20_20260831_v1`|`B0+A(lambda_sp=2.0)`|2423583|`RUNNING`|
|5|`wiser_rf_cause_vsw01_20260831_v1`|`B0+B(lambda_vsw=0.1)`|2423564|`RUNNING`|
|6|`wiser_rf_cause_vsw10_20260831_v1`|`B0+B(lambda_vsw=1.0)`|2423579|`RUNNING`|
|7|`wiser_rf_cause_short_20260831_v1`|`B0+A+B(stage_steps=500/1000/1500)`|2423635|`RUNNING`|

启动后独立回读确认：7个新增PID的CWD均为提交`563bbb30`的release目录；cmdline分别绑定唯一run root；`nvidia-smi pmon`确认PID与物理GPU1–7逐一对应；目标目录开始增长且日志未出现`Traceback`、`RuntimeError`、`ValueError`或OOM指纹。GPU0原pilot保持健康且未被修改。

## 后续闭环

每条pilot出现`pilot_result.json`后先核对冻结arm注册表及3个LEO场景prediction完整性，再写入不可覆盖`pilot_score`并独立truth-last评分。结果按同row的B0/候选和场景报告；A/B为正式候选，C/ABC仅为非正式模型反演诊断。全部分析完成后在本报告追加因果结论，不跨run按truth调参或重跑。
