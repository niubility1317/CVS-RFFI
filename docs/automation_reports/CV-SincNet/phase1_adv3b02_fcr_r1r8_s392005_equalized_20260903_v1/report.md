# ADV3B02→FCR R1-R8重新发布预登记

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v1`
- 顺序：先训练`ADV3B02_CORE90_SOFT_E200_S392005`，再由其`final_ssdg.pth`初始化R1-R8。
- 数据：`ManySig.pkl`，equalized=`true`，split_mode=`tx_rx_day_1_7_2`，seed=`392005`。
- source：receiver=`[1,3,4,6,8]`，day=`[1,2,3]`，pool=`90000`；`L_s=6300`、`U_s=56700`、`V=27000`。
- target test：receiver=`[0,2,5,7,9,10,11]`，day=`[0,1,2,3]`，TX=`[0,1,2,3,4,5]`，每个scenario=`168000`。
- scenario：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- checkpoint：全部固定使用E200最后一个epoch；禁止按source V或target test筛选checkpoint。
- target边界：训练过程不评估target test、target指标不进入checkpoint选择；E200冻结后由标签屏蔽的独立prediction进程生成四scenario预测，随后独立scorer连接truth sidecar。
- GPU：每张卡同时至多运行本批次一个训练任务；先单卡ADV3B02，完成后R1-R8按可用GPU排队。
- 技术停止：仅错误数据角色/seed/receiver/day/scenario、错误checkpoint、输出覆盖、确定性执行异常、无prediction闭合或scorer连接失败；低性能不停止。
- 预期artifact：每个任务`final*.pth`、`train.log`、四scenario prediction、独立truth sidecar、独立score JSON。

## 设计追踪

|要求|实现入口|验证证据|
|---|---|---|
|单一V，不使用V_cal/V_select说法|source-only grouped split|精确计数测试与启动日志|
|ADV3B02先行|批次调度器第一阶段|ADV3B02 E200 final存在后才解锁R1-R8|
|R1-R8同一初始化|R1-R8启动器checkpoint锁|checkpoint seed/candidate/epoch检查|
|target不参与筛选|训练policy=`never`且final-only|测试与训练日志中target eval count=0|
|跨进程truth连接|稳定opaque sample_id+独立truth sidecar+scorer|缺失/重复/错配负测与完整覆盖测试|

## 当前状态

`LOCAL_VERIFIED`

## 本地变更与验证

- `dataset_wisig.py`：允许在source/target receiver严格不重叠时保留重叠日期。
- `training_test_eval.py`、`train.py`、`SSDG/train_ssdg.py`：新增严格`never`策略、final-only checkpoint和外部target evaluation路径。
- `cvsrffi/truth_last.py`：稳定opaque ID、独立truth sidecar、完整覆盖scorer。
- `predict_phase1_truth_last.py`：prepare、prediction、scorer职责分离；正式prediction进程不接收ManySig路径，只读取不含标签的独立IQ包和opaque sample_id manifest。
- `launch_phase1_adv3b02_fcr_r1r8_s392005_20260903.sh`：ADV3B02串行先行，随后R1-R8映射GPU0-7，并在训练结束后执行独立prediction/scorer。
- 聚焦回归：新协议测试5项通过；本次最终组合回归17项通过；此前FCR checkpoint/review测试17项、SSDG/post-stage测试25项通过；Python编译通过。
- 独立P0/P1审查：发现prediction进程仍可触达带标签ManySig的P1；修复后定点复审结论为`FIXED`。truth仅由prepare生成，prediction只读标签自由包，scorer最后独立连接truth。
- `REJECTED_EXTRA_GATE`：审查提出的“每个物理样本在全部LEO scenario中只能出现一次”属于`p2_min_v1` Phase2规则；本批为Phase1且用户明确要求每个scenario 168000样本，因此未擅自改变冻结矩阵。
- 本地Git Bash：`FAILED`，桌面shell适配器错误路由至损坏WSL，fail-closed且未执行payload；将在N607目标原生Bash完成`bash -n`与干跑。
- N607首次干跑发现历史队列器仍会按现有GPU进程数等待；按用户明确授权将本批ADV3B02的容量检查设为`999`以旁路等待，不终止或修改任何既有进程，并新增回归断言。
- N607真实ADV3B02 checkpoint无query烟测首次发现旧checkpoint缺省物理特征源的兼容错误；加载默认已修正为模型原生`raw_fft/raw_iq`并增加定点回归，修复后必须重新通过真实checkpoint严格加载和dummy forward。
- 同一真实checkpoint烟测继续发现历史`sample_rate_hz=0`自动推断哨兵未被加载器归一化；现按WiSig协议解析为25MHz并纳入同一定点回归。
- prediction最终改为复用已验证的精确SSDG重构器，从checkpoint权重推断域头宽度并严格加载，移除`num_domains=15`硬编码；该重构器已在N607真实ADV3B02 checkpoint上完成无query dummy forward。

## 最终状态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

- ADV3B02在进入训练前因历史`best_metric=joint_safe`被source-only保护拒绝，exit=1；未生成checkpoint、prediction或性能结果。
- run树已自行退出，专属失败日志与空输出目录保留；未终止或修改任何无关进程。
- 修复仅把final-only模式下最后生效的兼容best metric固定为`clean_val_tx`，实验矩阵、seed、数据配置、训练预算和选择规则不变；替代run为`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v2`。
