# ERBT-IDR M2.8局部共形翻转风险screen正式实验报告

## 当前状态

`ANALYZED / SCREEN_NEGATIVE_NO_FULL125`

本轮实验已完成本地验证、N607不可覆盖发布、真实checkpoint无query smoke、16行truth-blind prediction、独立truth-last评分、完整汇总和证据回收。实验没有技术失败，但两个候选均未通过预登记科学门槛，因此按计划不启动full125。本报告不覆盖M2.7原始证据，也不把4个paired identity的screen结果外推为完整125结论。

## 候选与因果问题

主基线固定为去RF32 D92 E0（B0），性能分支固定为M2.5 B3。M2.8仅判断B3相对B0的单条query翻转是否可由target support的局部证据支持。

- C1：`M28-C1-B3-MGD-PAIR-POSTERIOR`
- C2：`M28-C2-B3-MGD-LOCAL-CONFORMAL-RECALL`

候选只读取当前`p2_min_v1`已接收IQ导出的IF256/FFT96与已注册support标签；query truth、query角色、批量类别数和全局重分配均不可用。query不更新任何状态。

## 冻结矩阵

- matrix kind：`screen`
- receiver：`3-19`、`8-8`
- method seed：`7282101`
- 条件：`K5/new20`、`K10/new5`
- arm：B0、B3、C1、C2
- 配对identity：4
- 方法row：16
- 场景单元：48

## 冻结方法

1. B0/B3完全复用既有实现，RF32保持移除。
2. 从FFT96构造MGD96；以旧类support逐类中位中心的类平衡均值估计目标域中心。
3. 通过严格support leave-one-out生成类别条件非一致度、目标类稳定度和top1/top2类别对事件。
4. 采用global→destination→pair的Beta-Binomial收缩，避免稀疏pair直接过拟合。
5. C1仅接受高置信top1；C2允许满足更严格后验与共形条件的top2。
6. 每条query输出必须精确等于完整B0或完整B3分数行。
7. `K<5`、全局事件不足或任何风险条件失败时回退B0。

## 技术停止规则

仅在协议/query泄漏、错误matrix/checkout、输出碰撞、运行命令不能执行、进程归属不清、prediction不闭合、scorer连接错误或至少两行出现相同确定性prediction前异常时停止。不得因低性能停止。

## 晋级门槛

候选必须同时达到`ΔH vs B0≥0.002`、`ΔH vs B3≥0.0002`、`N_help>N_harm`、`Δmin_old≥-0.005`、`Δmin_new≥-0.005`，才运行完整125。未达门槛则发布`SCREEN_NEGATIVE_NO_FULL125`。

## 发布字段（实现提交后回填）

- 实现Git commit：`e6eb5dc7a63b79cc70811302ff9f84f72da382b0`
- 实现远端OID：`e6eb5dc7a63b79cc70811302ff9f84f72da382b0`（独立核对一致）
- 本地环境：`ssr-gpu`
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- N607 CWD：release内`code`目录
- prediction设备：CPU；`max-workers=2`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`
- scoring root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`
- supplemental scoring root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- 预期artifact：`matrix_index.json`、16个`row_execution_receipt.json`、16个prediction、独立score root、`results_summary.json`

2026-08-23直连只读preflight通过：普通账户、项目根、Python、checkpoint、feature/scoring roots均存在；release/run/log及release archive目标均不存在。GPU0–2有其他负载，GPU3–7空闲；本轮CPU预测不占用GPU。

## 预登记执行命令

prediction：

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m28_local_flip_risk_matrix.py --run-id erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1 --matrix-kind screen --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/predictions --device cpu --max-workers 2

truth-last scorer：

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m28_local_flip_risk_matrix.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/scores --bootstrap-repeats 2000

汇总：

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m28_local_flip_risk_matrix.py --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/predictions --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/scores --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/results_summary.json

## 本地实现与验证

变更文件包括M2.8局部风险模型、通用row executor接入、screen/full125 runner、独立truth-last scorer、汇总器、聚焦测试、实现追踪和本报告镜像。

- RED：2个测试模块因`cvsrffi.stage2_m28_local_flip_risk`不存在而按预期collection失败。
- GREEN：M2.8聚焦测试`8/8`通过。
- 相邻回归：M2.5/M2.7/M2.8共`40/40`通过。
- 编译/集成回归：M2.4与M2.8共`29/29`通过。
- 入口smoke：runner、scorer、summarizer的正式模块入口均返回帮助并退出0。
- `git diff --check`：通过。
- 独立P0/P1审查：`PASS`，没有会直接导致真实实验跑错、越权、覆盖输出、不能启动或不能产生合法prediction的问题。

首次直接执行`code/scripts/*.py`的帮助命令因Python模块根不在`sys.path`而失败；按正式入口从`code`目录使用`python -m scripts.<module>`复验全部通过。该事件是错误调用方式，不是实现缺陷。

## 正式执行闭环

### 发布、smoke与启动读回

- release archive本地路径：`E:/type10-7/local_artifacts/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1_fe349c3f.tar.gz`
- N607 release archive：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1.tar.gz`
- 本地/远端archive SHA-256：`9531eafc11ca1265003aabbd270ea6fd860ce71f34def7ec5bb151ec9557a01b`，一致
- 远端编译：通过
- 真实checkpoint无query smoke：严格加载195个tensor，missing/unexpected/skipped均为0；75个tensor输出均为有限值；`query_input_count=0`、truth未打开
- prediction父PID：`1810409`；启动后CWD、cmdline、run root、2个worker和日志增长均读回一致
- prediction结束后父子进程自然退出，未发现确定性异常指纹

第一次smoke通过交互PTY调用时，远端程序已完成计算但连接仍等待stdin关闭；只中断了该连接所绑定的精确进程`1807229`，并确认其已消失。随后使用正式stdin重定向重新执行，smoke完整通过。该事件属于连接方式问题，不是checkpoint、数据或模型失败，也没有触发prediction重复启动。

### prediction与truth-last闭合

prediction完成后、truth打开前的严格检查结果如下：

|检查项|结果|
|---|---:|
|`matrix_index.status`|`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`|
|方法row|16/16|
|paired input identity|4|
|B0/B3/C1/C2|各4行|
|场景单元|48|
|R1注册前/后分歧|0/0|
|fit阶段query行|0|
|query状态更新|0|
|truth已打开|否|
|候选输出来源|仅B0或B3完整分数行|
|候选实际改变判决的B3翻转|全部否决|

确认prediction闭合后才运行预登记独立scorer。16行评分全部通过，随后汇总器输出`status=ANALYZED`和`decision=SCREEN_NEGATIVE_NO_FULL125`。没有使用局部矩阵，也没有把scorer输出反馈给predictor。

## 总体性能结果

以下为3个LEO场景的query-count-weighted同row结果，每个arm包含4440条注册后query。screen只有1个method seed，绝对数值不能替代full125结论。

|arm|`A_o_pre`|`A_o_post`|`A_n`|H|F|`min-old`|`min-new`|相对B0 H|相对B3 H|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B0：去RF32 D92 E0|0.742005|0.649362|0.588896|0.610486|0.092643|0.268919|0.251577|0|−0.005385|
|B3：M2.5稳定双原型残差|0.744482|0.655856|0.592838|0.615871|0.088626|0.281306|0.256532|+0.005385|0|
|C1：MGD类别对后验|0.742005|0.649362|0.588896|0.610486|0.092643|0.268919|0.251577|0|−0.005385|
|C2：MGD局部共形召回|0.742005|0.649362|0.588896|0.610486|0.092643|0.268919|0.251577|0|−0.005385|

C1和C2的最终预测标签、类别准确率、总体指标及floor都与B0完全一致。它们没有造成性能伤害，但也没有保留B3的有效增益，因而不是可晋级候选。

## 预登记晋级门槛

|候选|ΔH vs B0|阈值|ΔH vs B3|阈值|help/harm vs B0|floor约束|结果|
|---|---:|---:|---:|---:|---:|---|---|
|C1|0|≥0.002|−0.005385|≥0.0002|0/0|满足|FAIL|
|C2|0|≥0.002|−0.005385|≥0.0002|0/0|满足|FAIL|

两臂均不满足H和help/harm门槛。裁决严格按预登记规则执行：`SCREEN_NEGATIVE_NO_FULL125`，不启动500行完整125，不把低性能解释为技术失败。

## 条件、receiver与scene分解

|切片|B0 H|B3 H|C1 H|C2 H|B3−B0|
|---|---:|---:|---:|---:|---:|
|K10/new5|0.719264|0.731394|0.719264|0.719264|+0.012130|
|K5/new20|0.564465|0.566995|0.564465|0.564465|+0.002531|
|receiver 3-19|0.526178|0.527226|0.526178|0.526178|+0.001048|
|receiver 8-8|0.694794|0.704515|0.694794|0.694794|+0.009721|
|`leo_clear_weak`|0.670156|0.675919|0.670156|0.670156|+0.005762|
|`leo_low_elev_weak`|0.581155|0.588843|0.581155|0.581155|+0.007688|
|`leo_rain_weak`|0.580147|0.582850|0.580147|0.580147|+0.002703|

B3在两个K/new条件、两个receiver和三个场景上均优于B0，而C1/C2在所有切片都回到B0。这排除了“总体平均掩盖局部正收益”的解释。由于只有1个seed，本节只能说明本screen内方向，跨seed稳定性仍以M2.5完整125为准。

## 四状态与遗忘

|arm|状态|旧类准确率|新类准确率|H|
|---|---|---:|---:|---:|
|B0|`DA0_REG0`|0.709234|N/A|N/A|
|B0|`DA0_REG1`|0.548198|0.430563|0.478404|
|B0|`DA1_REG0`|0.742005|N/A|N/A|
|B0|`DA1_REG1`|0.649362|0.588896|0.610486|
|B3|`DA0_REG0`|0.709234|N/A|N/A|
|B3|`DA0_REG1`|0.548198|0.430563|0.478404|
|B3|`DA1_REG0`|0.744482|N/A|N/A|
|B3|`DA1_REG1`|0.655856|0.592838|0.615871|
|C1/C2|`DA0_REG0`|0.709234|N/A|N/A|
|C1/C2|`DA0_REG1`|0.548198|0.430563|0.478404|
|C1/C2|`DA1_REG0`|0.742005|N/A|N/A|
|C1/C2|`DA1_REG1`|0.649362|0.588896|0.610486|

B3的`F_within=0.088626`、`F_std=0.086149`；C1/C2的`F_within=F_std=0.092643`。因此M2.8没有继承B3在注册前后两侧带来的改善，也没有产生新的DA×registration交互。

## help/harm、类别与margin

注册后4440条query中，B3相对B0共有49次argmax翻转：28次help、5次harm、16次neutral。因此B3相对B0净增加23条正确预测，总体准确率提高0.005180。

C1和C2均未接受这49次中的任何一次。相对B3，它们的help/harm为5/28：确实恢复了B3的5次harm，但同时丢失了全部28次help。相对B0则为0/0。由于最终标签与B0一致，所有逐类别准确率也与B0一致，不存在被总体指标掩盖的类别改善。

|arm|top-2 margin均值|p05|中位数|≤0.05比例|
|---|---:|---:|---:|---:|
|B0|0.566556|0.018540|0.295723|0.125901|
|B3|0.567031|0.018894|0.295723|0.124099|
|C1/C2|0.566972|0.018178|0.295723|0.124550|

C1/C2虽与B0预测标签相同，但margin不完全相同：策略对B0/B3同argmax的query保留完整B3分数行，只在实际B3翻转处回退B0。因此“与B0一致”仅指最终判决和性能，不表示每个score值逐元素等于B0。

中心角距均值为B0 24.9689°、B3 30.5963°、C1/C2 81.9216°。M2.8角距来自target-centered MGD96诊断空间，与B0/B3分类分数空间不同，不能把更大角距解释为更好分类。它只证明目标域中心化确实改变了频域几何，不能证明该几何已校准到B3翻转效用。

## 机制级诊断

### 可观测事实

每个候选覆盖12个receiver×condition×scene拟合记录。query加权诊断为：

|诊断量|C1|C2|
|---|---:|---:|
|rank1 LOO事件均值|72.532|72.532|
|rank2 LOO事件均值|110.892|110.892|
|rank1成功事件|0|0|
|rank2成功事件|0|0|
|类别LOO准确率均值|0.4203|0.4203|
|类别LOO最小值均值|0.0198|0.0198|
|零LOO准确率类别数均值|6.086|6.086|
|非零类别对数均值|138.459|138.459|
|直接类别对事件均值|0.0126|0.0126|
|B3翻转接受率|0|0|

prediction闭合审查进一步确认：注册前/后两条路径合计每个候选有85次B3翻转，两候选共170次，全部被否决。C1和C2虽然阈值不同，但在“成功事件为0”的输入下必然产生相同结果。

### 根因：学习目标与决策目标错位

当前fit阶段先取B0 support预测作为`source`，再取MGD96 LOO top1/top2作为`candidate`；成功事件定义为“MGD候选不同于B0且等于support真类”。query阶段却把这个后验用于判断实际`B0预测→B3预测`类别对是否应该接受。

这两个问题并不相同：

1. fit阶段学习的是“MGD原型候选能否纠正B0”；
2. 部署阶段需要的是“B3相对B0的实际翻转是否有益”；
3. MGD top1/top2不一定包含B3目的类，即使包含，其support事件也不是B3的leave-one-out效用事件；
4. 本screen中所有rank1/rank2成功计数都为0，分层Beta-Binomial后验只能向低值收缩；
5. 后验下界、事件数、共形p值、径向p值和类别稳定度继续串联后，所有翻转必然被拒绝。

因此本轮失败不是“阈值略高”或“target receiver域偏移没有被表示”。目标域中心化是合法且有非等价几何变化的；真正失败的是把MGD候选正确性误当成B3-vs-B0翻转效用。直接放宽阈值会在没有正成功事件的情况下解除保护，无法提供可靠的harm控制。

## 与D92 E0、M2.5和M2.7的纵向比较

不同矩阵的绝对H不可直接比较；下表只把各run的同row差值和机制行为放在一起。

|实验|矩阵|候选H|同row基线H|ΔH|help/harm vs B0|关键结果|裁决|
|---|---|---:|---:|---:|---:|---|---|
|M2.5 B3|full125|0.539228|0.537558|+0.001669|352/98|跨5 receiver、5 seed、3 scene为正；第一个完整125非等价增益|科学分支保留，非部署默认|
|M2.7 V1|4-identity screen|0.611788|0.610486|+0.001302|6/0|接受6/49次B3翻转，精度100%，help召回21.43%|负screen|
|M2.7 V2|4-identity screen|0.610486|0.610486|0|0/0|Phase32全部否决|负screen|
|M2.8 C1/C2|4-identity screen|0.610486|0.610486|0|0/0|MGD类别对成功事件为0，全部否决|负screen|

M2.8没有实现预期的“提高M2.7 V1召回”。相反，它把V1尚能保留的6次有益翻转也全部拒绝。当前证据排序不变：

1. 去RF32 D92 E0/B0继续作为部署默认与主比较基线；
2. M2.5 B3继续作为已有完整125正证据的最佳科学分支；
3. M2.6、M2.7、M2.8均为负screen，不进入full125。

## 资源分析

|arm|编译状态B|注册时间ms|batch head latency ms/row|query head MAC|MAC上界|
|---|---:|---:|---:|---:|---:|
|B0|13921.49|19.32|3.516|5514.38|7,848,960|
|B3|38032.16|11147.89|63.048|9070.70|12,300,869|
|C1|66438.32|13623.90|28.076|11138.59|15,244,229|
|C2|66438.32|13177.79|77.869|11138.59|15,244,229|

C1/C2状态量约为B3的1.75倍、query head MAC约为B3的1.23倍，却没有保留B3性能。CPU计时受并发和缓存影响，不据此比较C1与C2快慢；状态量、MAC和“零性能收益”的方向足以否决当前实现。

## 下一轮优化：从“help准入”改为“harm-only否决”

M2.5完整125中B3 help/harm为352/98，本screen为28/5，说明B3翻转的先验净效用为正。继续默认B0、要求B3逐次证明help，会系统性牺牲召回。下一候选应反转决策基准：默认保留B3，仅在support-only证据明确指向harm时回退B0。

推荐的M2.9最小机制如下：

1. 对每个held-out support样本，在同一fold内分别构造B0-LOO和B3-LOO预测；不得复用包含该样本的原型或残差状态。
2. 直接定义三态效用标签：`help=B3正确且B0错误`、`harm=B3错误且B0正确`、`neutral=其余`。
3. 后验类别对必须使用实际`B0-LOO预测→B3-LOO预测`，不再用MGD top1/top2充当目的类。
4. MGD96目标域中心、共形p值、径向位置、B3残差强度和B0/B3 margin只作为效用后验的协变量或分层条件。
5. 使用global→receiver/condition→destination→pair的部分池化，估计`P(help)`和`P(harm)`；决策量采用`P(help)−λP(harm)`或harm后验上界。
6. 默认输出B3；只有harm后验达到预登记高置信阈值时输出B0。无事件、稀疏pair和低K时保持B3，而不是回退B0。
7. 同一4-identity screen至少包含B0、B3、harm-only veto和效用后验两臂；只有同时不劣于B3、减少harm且不损伤floor，才进入full125。

这一修改仍是target-support-only、query只读、去RF32，并保留M2.8的目标域中心与FFT96/MGD96表征；改变的是监督目标和默认决策方向，而不是简单放宽阈值。

## 最终证据位置

- `results_summary.json`：完整机器可读总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、遗忘和资源结果
- `evidence/matrix_index.json`：truth打开前的正式matrix index
- `evidence/receipts/`：16个row execution receipt
- `evidence/truth_blind_diagnostics/`：16个truth-blind诊断
- `evidence/scores/`：独立truth-last逐row评分和配对结果
- `evidence/logs/`：prediction、scorer和summarizer日志
- `evidence/control/`：release传输、smoke、启动读回和prediction闭合检查

最终状态为`ANALYZED / SCREEN_NEGATIVE_NO_FULL125`。本轮实验工程闭合通过，科学候选否决；不启动full125是预登记规则的正确执行，不是实验失败或中断。
