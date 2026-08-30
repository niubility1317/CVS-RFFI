# WISER-RF历史D92 E0因果诊断实验报告

## 当前结论

v1的8条run均已按预登记规则进入`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。失败原因不是低性能，也不是manifest缺项：manifest同时提供`before_enrollment`和`before_apply`，但旧WISER解析器错误地让support与query都使用`before_enrollment`，因此无法找到只存在于`before_apply`的query。v1部分artifact已保留，未评分、无性能结论、未覆盖或重启。

v2修复了query根解析后，已有2条因果run完成全部prediction和独立truth-last评分，但均未通过预注册门槛；另5条run在P3旧D92特征中出现同一`feature row is degenerate`确定性技术失败，主ABC pilot仍在GPU0运行。当前不授权扩展到完整Target125，也不为占满每GPU3条的容量上限复制实验。

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

v1没有`pilot_result.json`和合法prediction，因此不得启动scorer。

## v2定点修复重发

- 修复提交：`b5fb479032c371d0df016c19e67b25aa3c94d600`；support固定解析`before_enrollment`，query固定解析`before_apply`。新增测试先在旧代码上准确失败，修复后29项聚焦测试通过；独立P0/P1审查结论`READY`。
- 新release：`wiser_rf_queryfix_suite_20260831_v2_b5fb4790.tar.gz`；本地/远端SHA256均为`cd1e5778abab1cb2909d9667b957c3f71ff4a98c03d8ac541d99cd174b357807`；远端编译与真实ADV3B02无query smoke均通过，smoke记录`query_opened=false`。
- 2026-08-31 00:43 CST已用8个全新`*_20260831_v2_queryfix1`run ID重发，GPU0–7各1条。用户授权的并发上限为每GPU最多3个训练实验，本批次每GPU只占1个，不为占满容量复制科学row。
- v2 PID/GPU映射：GPU0=`2439930`完整ABC；GPU1=`2439927`无L2-SP；GPU2=`2439998`无LOO原型；GPU3=`2439999`弱L2-SP；GPU4=`2439997`强L2-SP；GPU5=`2440025`弱VSW；GPU6=`2440029`强VSW；GPU7=`2440008`短训练。独立回读确认CWD、cmdline、run root和物理GPU正确，目录增长且无异常指纹。
- 当前状态：主ABC pilot仍为`RUNNING`；2条因果run为`ANALYZED`；5条为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。prediction完整后每个run分别由独立truth-last scorer绑定其`pilot_result.json`冻结arm集合，不跨run用truth调参或重跑。

### v2运行状态

|物理GPU|run ID|同run矩阵|PID|状态|
|---:|---|---|---:|---|
|0|`wiser_rf_abc_hist_e0_pilot_20260831_v2_queryfix1`|`B0+A+B+C+ABC`|2439930|`RUNNING`|
|1|`wiser_rf_cause_nol2sp_20260831_v2_queryfix1`|`B0+A(lambda_sp=0)`|2439927|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|2|`wiser_rf_cause_noproto_20260831_v2_queryfix1`|`B0+A(lambda_proto=0)`|2439998|`ANALYZED`|
|3|`wiser_rf_cause_l2sp01_20260831_v2_queryfix1`|`B0+A(lambda_sp=0.1)`|2439999|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|4|`wiser_rf_cause_l2sp20_20260831_v2_queryfix1`|`B0+A(lambda_sp=2.0)`|2439997|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|5|`wiser_rf_cause_vsw01_20260831_v2_queryfix1`|`B0+B(lambda_vsw=0.1)`|2440025|`ANALYZED`|
|6|`wiser_rf_cause_vsw10_20260831_v2_queryfix1`|`B0+B(lambda_vsw=1.0)`|2440029|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|
|7|`wiser_rf_cause_short_20260831_v2_queryfix1`|`B0+A+B(stage_steps=500/1000/1500)`|2440008|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|

5条技术失败均没有`pilot_result.json`，没有连接truth或评分。它们在适配臂P3只读query推理中复现`OldOnlyERBTError: feature row is degenerate`；这说明query路径修复有效，但部分适配状态把D92消费的identity/FFT特征推入退化行。partial artifact原位保留，未因低性能停止或重启。

### 已分析同row结果

|候选|P1变化（clear/low-elev/rain）|P2变化（clear/low-elev/rain）|P3变化（clear/low-elev/rain）|P3 floor中位变化|几何比中位变化|结论|
|---|---|---|---|---:|---:|---|
|A，`lambda_proto=0`|`+5.00/-0.83/+6.67pp`|`+6.67/+0.83/+5.00pp`|`+0.83/-3.33/-0.83pp`|`-5.00pp`|`+0.3423`|未通过，不晋级|
|B，`lambda_vsw=0.1`|`+7.50/-0.83/+6.67pp`|`+8.33/+0.83/+6.67pp`|`+0.83/-3.33/+0.83pp`|`-5.00pp`|`+0.7676`|未通过，不晋级|

两条run均完成3场景×2臂共6组prediction后，才由独立scorer连接truth；`score_collection.json`均为`ANALYZED`且`truth_join_after_prediction_only=true`。两种设置都能改善P1/P2与表示几何，但P3旧类收益不足且类别下限下降，因此`passed=false`、`next_experiment_authorized=false`。其中弱VSW比去原型约束更接近P3正向一致性，但仍不能作为性能晋级结果。

## v3零模态修复预登记

- 根因：可微D92的`F.normalize`允许单个模态为零并将其保持为零，但独立精确D92此前会直接拒绝任一零范数模态，导致适配后零identity行无法利用仍有效的FFT块完成P3。
- 修复提交：`4e51e29b393cba723c2e79ed0d3314ed64d6369f`，已push且远端OID一致。单模态零范数现在安全置零；identity与FFT同时退化时，联合归一化仍确定性拒绝，避免无信息prediction。
- TDD与验证：新增零identity/有效FFT回归测试先准确失败；修复后相关37项测试通过。唯一独立P0/P1审查结论`READY`。
- release：`wiser_rf_zeromodal_suite_20260831_v3_4e51e29b.tar.gz`；本地/远端SHA256均为`9ac739632a48d91600b41ca1eb005c7e16b8a12a6f327695001dcefffb267521`，远端编译和真实ADV3B02无query smoke均通过，smoke记录`query_opened=false`。
- 新run只重跑没有合法结果的6条row：GPU2主`B0+A+B+C+ABC`、GPU1无L2-SP、GPU3弱L2-SP、GPU4强L2-SP、GPU6强VSW、GPU7短训练。已`ANALYZED`的去原型和弱VSW不重复跑；GPU0旧v2主pilot保持只读。每GPU最多3个训练实验，本批次每张目标卡只新增1条。
- 2026-08-31 01:32 CST启动并完成首次绑定回读：GPU2主pilot PID=`2463277`，GPU1无L2-SP PID=`2463288`，GPU3弱L2-SP PID=`2463290`，GPU4强L2-SP PID=`2463276`，GPU6强VSW PID=`2463289`，GPU7短训练PID=`2463291`。6条均为`RUNNING`，CWD、cmdline、run root和物理GPU映射正确，每条初始artifact文件数为2；GPU0旧v2 PID=`2439930`继续只读运行。
