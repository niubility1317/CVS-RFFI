# MARC-OT设计落地追踪

设计来源：`docs/superpowers/specs/2026-09-01-marc-ot-design.md`、`docs/superpowers/plans/2026-09-01-marc-ot.md`、`项目.md`及Task1～9实现报告。冻结代码提交为`12a6379823776b8ce5e8d3c6d3313f618f878fc4`。

状态口径：`verified`表示生产文件与本轮测试共同证明相应软件要求；`pending`、`blocked`和`deferred`分别表示证据尚待取得、受前置科学结果约束和明确不在首轮范围。`software supported`只表示接口、调度和失败语义已由测试覆盖；`training coverage capability`只表示生产训练入口能消费合法调度、执行真实优化步并如实记录所训练子集；两者都不等于真实Phase1训练，更不等于`pilot_executed=true`或存在性能结果。

|ID|来源章节|要求|生产文件|状态|本轮验证证据|边界|
|---|---|---|---|---|---|---|
|PROTO-01|设计2、10|Phase2只读合法support；训练与选择接口不可达query/truth/role/quota；全部support状态先冻结，prediction闭合后才能独立连接truth|`code/cvsrffi/stage2_marc_ot.py`、`stage2_marc_ot_runner.py`、`stage2_marc_ot_pilot.py`、`stage2_marc_ot_scoring.py`|verified|公开训练面负测、先冻结后prediction及scorer拒绝路径测试；最终聚焦回归332项通过|未运行真实checkpoint或N607生命周期|
|BANK-01|设计4|稳定block schema、delta提取、确定性低秩重构、base checkpoint和bundle geometry绑定、禁止成员检查|`code/cvsrffi/meta_weight_bank.py`、`meta_weight_bank_checkpoint.py`、`marc_ot_phase1.py`|verified|bank/bundle测试覆盖canonical block、bitwise冻结、SVD、round-trip、geometry、禁止成员、训练后严格bundle回读，以及与`bank.task_keys`逐行绑定的int8多物理样本聚合685D task-domain descriptor|测试bundle来自合成fixture；未生成真实Phase1训练bundle，不声明bank数值质量|
|ENC-01|设计5|permutation-invariant SupportSetEncoder输出`q/u/gamma/eta`，并使用多层、分支、多view、CFO/SFO proxy、PSD、K和mask统计|`code/cvsrffi/marc_ot_support_features.py`、`meta_support_set_encoder.py`、`meta_bank_trainer.py`、`stage2_marc_ot_runner.py`、`meta_weight_bank_checkpoint.py`|verified|固定ABI=`marc_ot.support.row.v1`、维度`685=640+3+2+6+4+16+10+1+3`；`test_marc_ot_support_features.py`覆盖布局、五K、确定性view、proxy、PSD/RF-lite、K/mask、梯度、拒绝漂移及module mode恢复；encoder测试覆盖masked逐类raw mean/diag-var/norm/availability和DeepSets置换不变性；Phase1/Phase2均走同一builder，bundle拒绝旧160D几何|CFO/SFO明确为`PROXY_ONLY`；确定性view不增加物理row或K；未从真实IQ训练或评估|
|CAL-01|设计5、6.1|按`(1-u)gamma_bB_bq`有界组合bank初始化；绑定、geometry或有限性失败时原子回退`theta_0`|`code/cvsrffi/meta_weight_calibrator.py`|verified|calibrator测试覆盖异构rank、LR边界、允许block、非有限状态和完整原子回退；最终聚焦回归通过|真实Phase1 bundle尚未生成|
|META-01|设计7|显式bank fast state的一阶inner loop；outer梯度到达encoder、gate/LR与允许bank basis；receiver/day/scene holdout和K=`1/2/5/10/20`episode覆盖；生产入口执行bank step并保存bundle|`code/cvsrffi/meta_bank_inner_loop.py`、`meta_bank_trainer.py`、`meta_episodes.py`、`marc_ot_phase1.py`|verified|canonical调度对每个K覆盖1个receiver holdout、1个day/capture holdout、3个clean→LEO和6个有向LEO跨场景，共`5×11=55`个精确语义单元；coverage审计拒绝缺失、重复或额外单元；`run_marc_ot_phase1_bank_training()`测试实际执行`run_meta_bank_step()`、证明必需bank/encoder tensor改变并严格保存/回读bundle|55-cell是`software supported`调度覆盖；训练入口测试仅训练1个K2合成episode，provenance=`CALLER_SUPPLIED_UNCLAIMED`。正式配置`training_coverage_k=[]`，未执行真实Phase1全覆盖训练|
|OT-01|设计2、6.3|OT只发生在support与冻结bank task features之间；FP32 log-Sinkhorn、bank detach、边际和有限性检查|`code/cvsrffi/stage2_marc_ot.py`、`meta_weight_bank_checkpoint.py`、`code/scripts/run_stage2_marc_ot_pilot.py`|verified|OT测试覆盖均匀边际、确定性、bank detach、非法输入和未收敛失败；Phase1 descriptor量化、strict load、解量化与R6/R8默认685D stage链测试通过|task-domain descriptor为int8多样本聚合知识；bundle不保存source IQ或sample-level source embedding|
|LOSS-01|设计6.2、6.3|冻结head、cross-fit、LOO、SupCon、类风险、K条件统计、OT与信任域组成合法support-only目标|`code/cvsrffi/stage2_marc_ot.py`、`stage2_marc_ot_runner.py`|verified|`marc_ot_losses()`包含support-only SupCon及标量/mode/anchor/temperature/weight诊断；R0/R1关闭，R2/R4/R6/R8启用；K≥2测试证明非零SupCon梯度，production单步测试证明权重引入预期梯度增量；其余测试覆盖冻结head、prototype/D92 cross-fit、LOO、类风险、OT、K条件统计和trust项|K1无正样本对时保守返回可微零损失，mode=`K1_NO_POSITIVE_PAIRS`、anchors=`0`；不伪造正对|
|GRAD-01|设计6.4|按canonical block独立执行主任务优先投影和ratio cap；非有限梯度立即失败|`code/cvsrffi/stage2_marc_ot.py`、`stage2_marc_ot_runner.py`|verified|blockwise投影测试覆盖冲突/同向block、同block多参数、None、零primary、极端尺度和非有限值；R8接线测试通过|未改动WISER历史全局投影|
|SAFE-01|设计6.5|固定四阶段渐进开放；support-only cross-fit选择；`alpha=0`精确恢复原model、dual和非浮点buffer；退出后refreeze|`code/cvsrffi/stage2_marc_ot_runner.py`、`code/scripts/run_stage2_marc_ot_pilot.py`|verified|runner与真实`_adapt_unit()`测试覆盖fold隔离、早期接受/后期拒绝、全拒绝回退、mode恢复、异常refreeze；K=`2/5/10/20`生成确定性fold|K1保守路径不做adapter update或选择：`optimizer_steps=0`、`folds=0`、`held_out_support_evidence=false`、`query_rows_used=0`；Sinc未开放|
|MATRIX-01|设计8|正式首轮矩阵精确为`R0/R1/R2/R4/R6/R8`、K10、Target5 pilot outer和三个LEO弱场景|`code/cvsrffi/stage2_marc_ot_pilot.py`、`configs/marc_ot_k10_pilot_20260901.json`|verified|arm registry、config完整性、`VALIDATED_ONCE`绑定和软件/训练/pilot三层状态分离测试；JSON解析校验通过|`software_supported_k=[1,2,5,10,20]`、`training_coverage_k=[]`、`pilot_k=10`、`pilot_executed=false`；未执行真实三场景pilot|
|MRIOR-01|设计3、8|MRIOR-H/B/HB只作为显式permission/claim scope控制；拒绝历史MRIOR-SDA数值倒填|`code/cvsrffi/stage2_marc_ot_pilot.py`、pilot config|verified|MRIOR scope、历史字段、`MRIOR-SDA/history/backfill`字符串注入负测；最终聚焦回归通过|三项控制不进入首轮正式R矩阵，未运行控制实验|
|SCORE-01|设计2、8、10|prediction纯预检完成后只打开truth一次；输出REG0绝对指标、同row增量、per-class、help/harm和资源状态；晋级要求zero-id为0且support-CV/query方向一致|`code/cvsrffi/stage2_marc_ot_scoring.py`、`code/scripts/run_stage2_marc_ot_pilot.py`|verified|scoring与CLI测试覆盖prediction闭合、truth-last、同row重算、资源schema、old-only REG0、truth前zero-id重算与跨receipt绑定、真实held-out support-CV证据和两项promotion负门|科学门失败只产生`ANALYZED/NO_PROMOTION_TO_TARGET25`；未连接真实truth sidecar，无性能结果|
|CLI-01|设计9、10|CLI固定`smoke/pilot/score`；smoke无query路径；pilot/score output root不可覆盖；checkpoint identity与bundle绑定失败即关闭|`code/scripts/run_stage2_marc_ot_pilot.py`、pilot config|verified|CLI`--help`显示三个固定入口并exit0；14个新增/相关模块`py_compile` exit0；最终聚焦回归通过|未执行真实checkpoint无query smoke|

## 生命周期状态

|ID|要求|状态|证据或原因|
|---|---|---|---|
|VERIFY-LOCAL-01|Task1～9完整聚焦pytest、CLI help、相关模块compile、JSON和最终diff检查|verified|最终回归`403 passed`、0 failures、exit0；CLI help、13个相关生产入口`py_compile`、JSON和`git diff --check`均exit0|
|REVIEW-01|一次最终独立P0/P1审查及唯一一次定点复审|verified|最终审查发现2个P1；唯一修复轮关闭R6/R8共同685D OT空间及zero-id/support-CV方向晋级门，定点复审结论`APPROVE`、无新P0/P1|
|RELEASE-SMOKE-01|release归档一次本地/远端SHA比较、远端compile和真实checkpoint无query smoke|pending|本Task不访问N607；真实Phase1 MARC-OT bundle也尚未生成|
|PILOT-01|K10/Target5三场景六arm prediction闭合和独立truth-last评分|pending|正式配置仍为`pilot_executed=false`；未启动N607、未产生prediction或score|
|T25-01|pilot达到预登记科学门槛后运行Target25|blocked|等待PILOT-01；低性能只产生`NO_PROMOTION_TO_TARGET25`，不构成技术失败|
|STAGEC-01|Stage2-C冻结`phi_D`并另训`phi_R`|deferred|设计明确排除在首轮交付之外|

## NONBLOCKING P2

- 极端但有限的大幅IQ输入在RMS中间平方运算上仍存在潜在浮点溢出边界。生产builder会拒绝最终非有限row，本地常规与压力测试均通过；该项不改变合法数据协议、启动能力或prediction闭合，记为`NONBLOCKING P2`，不得提升为额外实验gate。
- R2 production单步测试已证明SupCon权重对真实更新路径贡献非零且梯度差满足独立autograd预期；测试未严格断言两个不同非零权重最终保存state彼此不同。该剩余断言强度属于`NONBLOCKING P2`，不否定生产梯度接线，也不得阻断最终P0/P1审查、release或pilot。

当前反向审计统计为`verified=15`、`implemented=0`、`pending=2`、`blocked=1`、`deferred=1`、`rejected=0`。软件能力层已闭合首轮设计范围；最高本地状态为`LOCAL_VERIFIED / FINAL_P0P1_REVIEW_APPROVED / N607_NOT_RUN`。这不代表真实Phase1训练覆盖、真实bundle、checkpoint smoke、K10 pilot、truth-last评分或任何性能结论；`pilot_executed=false`保持不变。
