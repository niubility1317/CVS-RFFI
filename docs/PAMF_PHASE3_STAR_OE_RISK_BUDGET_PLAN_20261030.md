# G10 源端 OE 风险预算阈值比较

Run `20261030-phase3-star-oe-risk-budget-s2026092701-r01`。原联合 OVA r02 已完整结束且 NO_PROMOTION；本实验是新的源端方法比较，不追改旧判定，不新增目标访问。完整 Phase3 仍 ACTIVE，采用用户 v3 指标。

## 问题与选择

原 `joint::ova_max` 的总体 C_K98.46%、R_U24.92%。固定 common-tail 校准的下一候选被 RX3/mid 的 2pp 身份保护绑定，虽然该组 C_K仍94.60%。这只证明当前阈值族的运行点受限，不能证明六个独立阈值均无可行改善。

可选路线是继续改变神经表征，或先在完全相同模型／原始分数下检验阈值族。这里由Agent选择后一条以隔离校准影响；不放宽任何42组保护，也不利用novel选择阈值。两套冻结模型均来自合法原F0独立训练的 `20261029-phase3-star-joint-ova-s2026092701-r02`，实际code `5aa041122396d48c49b8cb3d5912649bdc3217e1`；不是以较高novel分数改换主模型。

## 固定矩阵与数据

两个arm仍为detach_control、joint；每arm比较 `common_tail` 与 `oe_risk_budget`，唯一可晋级主为 `joint::oe_risk_budget`。模型、默认身份头、966参数aux和raw max(aux)均冻结，身份由原默认头argmax给出。common_tail精确复用r02的ova_max六阈值，不重校准。

新拟合仅source V27000的合法旧六标签与originalF0同P1闭集参照，以及OE6134的outside-six成员关系／15个RX/day域。OE真实TX、真实类别数不提供。V/novel的原rawlogits、身份、unitfeatures和IDs复用r02公共prediction；新F只对同一缓存OE P1前向，每arm6134实物＋2synthetic G10 P0 smoke，合计12272 F，G新前向0，神经/optimizer更新0。L、原Us、target U/Q、formalRX24-18读取0。

全部42584原记录仍为同一G10实现、同五个已见source RX、同三日，三个novel已有历史开发曝光。novel3150在新阈值全部冻结后才打开；它只用于冻结预测和独立truth-last评分，不参与任何风险分配。新方法不宣称未见目标RX或未知设备盲测。

## OE风险预算分配（预先固定）

对六个模型预测类别分别设置阈值，接收条件raw max(aux)≥τ[predicted class]。所有相同分数完整处理。每个OE记录的收益权重为1/(15×该域OE数)，目标是source OE的域宏拒识代理；不声称它等于未知真实TX宏拒识。

初态六阈值有限且接受全部V/OE。某预测类别的下一候选为下一个尚未拒绝的OE分数层v；τ取float64 nextafter(v,+inf)，同时计入该层以前及同层全部V的增量拒绝。没有OE增量收益的阈值移动不单独采取。对全部42组按真实旧TX等权，累计拒绝率≤0.10，原F0闭集−新正确接收≤0.02；预算以原F0而非候选闭集定义。若初态本身不合格，保留有限接受全部回退、constraints_passed=false，仍完成冻结novel预测，不隐藏失败。

每轮在六个下一候选中，只考虑不违反任一约束者。代价是新增两种风险分别除以其当前剩余预算后的最大值；零增量风险贡献为0。选择OE收益/(代价+1e-12)最高者，同分按预测类index从小到大。更新后重新计算剩余预算；某类下一候选已不可行则其更高单调阈值均不可行。直到无可行OE收益步骤。使用float64累计风险和1e-12数值容差，最后复用原exact-Fraction风险核对全部42组并如实报告约束结果。该算法是确定性的贪心可行分配，不宣称全局最优或分布外覆盖保证。

输出仅六个float64阈值；训练期队列、OE行bank、域标签和风险曲线不部署。保存选择轨迹、每类prefix、域宏收益、实际V风险、停止时下一候选失败组，供解释，不据novel追加搜索或换规则。

## 实现接口与分工

1. `star_oe_risk_budget.py`及对应测试由算法worker负责；纯NumPy函数 `calibrate(ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain)->(state,report)`。state仅含`conditional_thresholds` float64[6]；report含`constraints_passed`、`groups`（42组原risk_group结构）、`conditional_thresholds`、`objective_before/after`、`trace`、`coverage`、`source_only`、`target_reads=0`、`novel_inputs_used=false`。真TX不路由阈值。函数 `apply(score,closed,state)->float64[N]`。
2. `star_oe_risk_run.py`及对应测试由运行worker负责；`validate_config(config)`、`run(config,device='cuda:0')`。精确8配置字段为run_id/code_commit/arm/parent_eval_config/parent_prediction_root/OE_root/output_root/batch_size128。parent_eval_config保持原r02完整身份，可直接复用原eval._backend/_smoke/_predict、原source OE loader及已验证角色句柄；不改变父模块。真实checkpoint smoke在任何新OE/V消费之前，完成后立即继续。
3. 主Agent负责`star_oe_risk_score.py`与`star_oe_risk_launch.py`及聚焦测试／登记／发布。source进程全部预测完成才独立启动CPU scorer；每arm即时闭合，健康另一arm不受已返回科学结果影响。

新prediction schema=`pamf_phase3_star_oe_risk_prediction_v1`、status=`STAR_OE_RISK_PREDICTIONS_COMPLETE`，cal status=`STAR_OE_RISK_THRESHOLDS_FROZEN`。两方法顺序固定common_tail/oe_risk_budget；thresholds.npz字段`conditional` float64[2,6]。known/novel_predictions.npz保留原r02相同字段，但scores为[N,2]，由同rawmax减各自条件τ；novel包含原candidate和baseline单位特征供原结构guard复核，无新F0 novel前向。先写完整校准和known文件，再打开novel，最后complete。

校准complete包含run/code/arm、method_names、conditional_thresholds、methods[两法各report，含constraints_passed/groups]、group_count42、source_V_labels_used=true、source_OE_membership_used=true、source_OE_TX_or_K_used=false、novel_inputs_used=false、target_reads0、source_onlytrue。原common_tail report来自父r02 calibration.methods.ova_max，独立预算report来自上述函数。OE_scores.npz仅ids/rx_i/day_i/domain/closed/score/default_logits/aux_logits，不写TX。

prediction complete须包含source_only/target_reads0、parent_run_id/parent_code_commit/parent_model_identity、parent_prediction_root、known_count27000/novel_count3150/OE_count6134、all_thresholds_frozen_before_novel=true、model_parameters_buffers_unchanged=true、aux_parameters_buffers_unchanged=true、neural_updates0、F_forward_rows6136/G_forward_rows0/source_L_reads0/source_U_reads0、method_names及classification_path=P1。独立scorer核对实际rawmax/默认头身份/条件margin、42组宏风险后连接private truth；复用原跨sourceRX1NN结构guard，目标A_N/H_OW仍N/A。

## 验证、发布与判定

聚焦测试覆盖完整ties、零风险收益、局部预算绑定、原F0参照、OE域权重、独立类预算可改善common-tail的可证伪夹具、无novel/target进入fit、有限回退、实际checkpoint first-step smoke与不可覆盖输出。只做本候选一次独立P0/P1审查，不重复父候选测试或添加报告审批。

N607每arm用一张真实free≥4096MB卡，至多2arm同时、每卡≤2新增，foreign任务不变。一个新Git code/config commit、最小登记、一个新run/release、一次preflight/archiveSHA比较/compile，启动后PID/CWD/cmd/GPU/log增长读回，预测固定后独立评分。任何技术失败保留产物，科学低分不停止或重启。

只有主joint::oe_risk_budget的known42保护、unknown总体＋5RX＋6scene R_U≥90%与原结构guard均通过，才具备另行预登记小目标实验的资格。否则NO_PROMOTION；不换主、不按novel选阈值、不扩seed。source晋级永远不等于完整跨RX星地目标、适应或未来匿名注册完成。
