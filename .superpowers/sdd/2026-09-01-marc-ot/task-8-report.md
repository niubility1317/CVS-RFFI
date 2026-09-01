# Task8：Phase1多episode/K软件闭合与R2 SupCon

## 设计追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|T8-01|brief Phase1|Canonical K严格为`(1,2,5,10,20)`，确定性覆盖receiver holdout、day/capture holdout、clean→三种LEO及全部有序LEO cross-scene|`meta_episodes.py`及聚焦测试|PASS|完整schedule为55个cell；移除任一必需cell后audit拒绝|这是软件覆盖，不是实际训练或性能证据|
|T8-02|brief Phase1|严格校验kind、partition物理ID互斥、每类K/query数、class role、source role/receiver allowlist|`validate_episode_semantics()`|PASS|伪造kind/K、partition overlap及coverage缺口负测|不从opaque Phase2 token推断语义|
|T8-03|brief Phase1|真实可调用schedule→validate→`run_meta_bank_step()`→strict bundle save/load路径|`marc_ot_phase1.py`|PASS|完整synthetic integration执行一个K2 episode并回读bundle|独立入口未调用旧`run_meta_train_step()`；输入来源标为`CALLER_SUPPLIED_UNCLAIMED`|
|T8-04|brief SupCon|normalized support-only SupCon；K1可微0；温度有限正数；无query/queue/sample embedding持久化|`stage2_marc_ot.py`|PASS|K1 mode/0梯度及K2正loss/非零梯度测试|diagnostics保留tensor loss及detached scalar/mode/valid anchors|
|T8-05|brief R-arm|R0/R1关闭SupCon；R2/R4/R6/R8使用同一冻结正权重|runner、pilot、config及测试|PASS|R1梯度为0、R2梯度真实非零且不同；CLI config消费weight/temperature|R8 primary projection保留SupCon，只把OT/statistics作为auxiliary|
|T8-06|brief formal K|K1保守且不伪造cross-fit证据；K2/5/10/20确定性合法fold|`stage2_marc_ot_runner.py`及测试|PASS|K1返回0 step、`held_out_support_evidence=false`；其余K重复构建fold完全相同|K1不崩溃、不选择full-support adapter|
|T8-07|brief pilot|软件K、实际训练覆盖、正式pilot K及是否执行分别记录|pilot config、Phase1 closure及测试|PASS|`software_supported_k=[1,2,5,10,20]`、`training_coverage_k=[]`、`pilot_k=10`、`pilot_executed=false`|未运行正式pilot，未生成性能结论|
|T8-08|brief delivery|聚焦及Task1–7回归、help、compile、精确stage、指定commit、push/OID回读|本Task实际文件|READY|356项通过；help/compile退出0|Git提交与远端OID在提交后独立回读|

## Phase1编排与覆盖语义

- 新入口`run_marc_ot_phase1_bank_training()`先生成冻结55-cell schedule，再逐episode执行严格语义校验与完整coverage audit；只有显式selector返回且属于该冻结schedule的episode才进入训练。
- 每个实际选择episode由调用方构建`MetaEpisodeBatch`，入口要求batch保留完全相同的episode，再调用Task3/Task7真实`run_meta_bank_step()`。该trainer继续通过共同685D support feature ABI构建source support特征并执行bank inner/outer step。
- 输出bundle路径必须不存在且父目录已存在；训练完成后调用严格`save_meta_weight_bundle()`并立即以base checkpoint/block spec绑定执行`load_meta_weight_bundle()`回读。
- closure分别记录`software_coverage`与`training_coverage`。合成测试实际执行一个K2 bank step，但仅证明调用链；`input_provenance=CALLER_SUPPLIED_UNCLAIMED`禁止把合成输入升级为真实source训练声明。
- 当前正式pilot配置仍为K10，`training_coverage_k=[]`且`pilot_executed=false`。因此本Task只交付软件能力；没有真实source训练、N607或性能证据。

## SupCon与formal K边界

- SupCon输入仅为当前support feature tensor与old-class label；内部先L2归一化。每个anchor的positive是其它同类support row，分母是所有其它support row。
- K1或其它无positive集合返回`support_features.sum()*0`，mode为`K1_NO_POSITIVE_PAIRS`且`valid_anchor_count=0`，因此保留可微图但梯度严格为0。
- K≥2返回`SUPPORT_ONLY_SUPCON`；测试以真实autograd证明loss大于0且梯度非零。R-arm映射固定为R0/R1权重0、R2/R4/R6/R8共享配置正权重。
- formal K1 runner不进入任何adapter更新/选择，恢复初始model state，记录`optimizer_steps=0`、`crossfit_fold_count=0`、`held_out_support_evidence=false`和`query_rows_used=0`。K2/5/10/20使用opaque support token参与确定性fold排列，但不解析token语义。
- SupCon无memory queue、query/source输入、持久prototype或sample-level bundle state；Phase2仍只读合法support并保持query隔离。

## TDD记录

- 初始RED覆盖K20旧缺口、完整schedule/audit缺失、伪造kind/K/overlap未拒绝、真实MARC-OT bank入口缺失、SupCon/R2 effect缺失、K1 formal fallback及K2/5/10/20 fold缺失。
- Phase1 GREEN先完成55-cell确定性schedule与严格semantic validator，再以真实`run_meta_bank_step()`闭合synthetic schedule→step→bundle round-trip。
- SupCon GREEN完成K1可微0、K2非零梯度、R-arm权重映射和runner total接入；相邻旧progressive测试由不再合法的K1 callback fixture改为K2，生产旧meta-adapter路径未改。
- pilot三态分离先新增`pilot_executed`断言得到2个预期失败，再加入exact config字段与`false`校验后定点转绿。
- 声明边界定点RED证明synthetic closure仍暴露`source_training_executed`假声明；改为`training_step_executed=true`与`input_provenance=CALLER_SUPPLIED_UNCLAIMED`后转绿。

## 最终验证与交付

- Task1–8完整新鲜回归：`356 passed in 151.09s`，退出码0。
- `python code/scripts/run_stage2_marc_ot_pilot.py --help`：退出码0，显示`smoke/pilot/score`。
- 对Task8相关7个生产入口执行`python -m compileall -q`：退出码0。
- `git diff --check`：退出码0，仅有当前Git换行策略提示，无whitespace error。
- 未访问N607、未修改`VALIDATED_ONCE`数据、未运行正式K10 pilot、未读取Phase2 query用于训练/选择、未生成性能声明。
- 精确stage仅纳入Task8代码/config/测试与本报告；既有Task6`analysis/marc_ot_traceability_20260901.md`、`docs/experiments/marc_ot_k10_pilot_20260901_report.md`、`conversation_index/`及`local_artifacts/`继续保留且不stage。
- 提交主题固定为`feat: close MARC-OT meta episodes and SupCon`；本报告所在提交无法自引用最终OID，提交后的local/remote OID一致性由交付状态记录。
