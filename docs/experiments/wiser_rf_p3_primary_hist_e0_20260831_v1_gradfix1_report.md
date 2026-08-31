# WISER-RF P3-Primary历史pilot预登记与技术终止分析

## 最终结论

- 最终状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。真实checkpoint无query smoke在N6训练的Stage2分支内出现`id_backbone.t3.dw.weight`非有限梯度，`set -e`阻止后续pilot启动。
- 本run未打开query、未生成任何prediction/receipt、未连接truth，因此query准确率提升、四状态指标、P3 BA、floor、forgetting和新类性能均为`N/A`，不是0，也不能据此判定方法科学失败或科学晋级。
- `fe9ec142`修复具有局部效果：前一`bindfix1`run在Stage1约12.8秒内失败；本run完成Stage1并进入Stage2后才失败，持续约6小时7分28.6秒。但该修复没有消除整条可微D92/P3链的全部非有限梯度表面。
- 当前证据能确定“剩余数值不稳定”和“观测性不足”，不能唯一确定产生NaN/Inf的首个算子。代码审计把D92内部未安全处理的`sqrt`/RMS路径列为首要假设，把无界对偶变量放大和N6梯度投影前分量污染列为次要假设。

## 状态与修复边界

- run ID：`wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1`；最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；Git提交：`fe9ec1424e8396d1f6a4e8931a8653750cbb74e9`。
- 本提交仅将精确D92的退化方差在`sqrt`前安全屏蔽，并为零`delta`预构造安全分母；不改变正常坐标的前向公式、P3方法、训练预算、arm、scene或晋级门槛。
- 两个前序run分别因历史绑定漂移和退化坐标非有限梯度在无query smoke阶段停止，均为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，未进入pilot、未打开query、未产生prediction，原run/log根永久保留。
- 回归证据：`zero_identity/zero_fft/ill_conditioned`三类D92反向测试先失败后通过；10个D92/P3/pilot/scoring/Target25测试文件完整通过；相关模块`py_compile`通过。

## 冻结协议、输入与矩阵

- protocol=`p2_min_v1`，data status=`VALIDATED_ONCE`；outer=`rx_3_19__seed_713102__k_10__new_5`，receiver=`3-19`，seed=`713102`，K=`10`，new-count=`5`。
- capsule=`d92-e0-full-target125:5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5`；split=`d92-e0-full-target125:rx_3_19__seed_713102__k_10__new_5`。
- arm=`N0,N1,N2,N3,N4,N5,N6`；scene=`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；共21个独立prediction/receipt。训练、选择和插值固定`query_rows_used=0`，全部support状态冻结后只读打开query，prediction完整后由独立scorer连接truth。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`；source summary：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz`。
- P3 config=`configs/wiser_rf_p3_primary_20260831.json`；source binding=`configs/wiser_rf_adv3b02_source_binding.json`。

## release、资源、路径与命令

- release=`wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1_fe9ec142.tar.gz`；远端根=`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1_fe9ec142`。归档大小36272459字节，本地/远端唯一SHA256均为`0199fe792f5c38f2b0dfc010fc1d0a4c14117dfb935346c641fec26dd2d6a931`；远端11个相关模块一次编译通过。
- run root=`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1`；log=`/home/szu2070436088/2510044040/CV-SincNet/logs/wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1/pilot.out`；score=`<run-root>/score`。四个新目标均已确认不存在。
- pilot为单进程顺序工作流，冻结物理GPU0，`CUDA_VISIBLE_DEVICES=0`后程序使用`cuda:0`；启动前再次盘点且每GPU不超过用户授权的3个训练实验。

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-smoke --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <run-root>/smoke --device cuda:0 --runtime-commit fe9ec1424e8396d1f6a4e8931a8653750cbb74e9 --arm N6 --scenario leo_clear_weak
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-pilot --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <run-root>/pilot --device cuda:0 --runtime-commit fe9ec1424e8396d1f6a4e8931a8653750cbb74e9 --arms N0 N1 N2 N3 N4 N5 N6
```

## 停止、artifact与晋级

- 仅因协议/query/truth泄漏、错误split/receiver/seed/K/scene、输出冲突、可微与精确D92不同构、非有限loss/gradient、prediction不完整、scorer绑定错误、进程归属不清或确定性重复异常停止；不得因低性能停止。
- 预期artifact：smoke结果、21个support audit与prediction/receipt、completion marker、独立详细score、资源记录和`pilot_auto_result.json`。
- pilot门槛不变：P3 BA三scene中位提升≥3pp、最差scene≥-0.5pp、P3 floor中位及low-elev floor不下降、P1/P2每scene≥-2pp、zero-id=0、条件数≤基线2倍、至少2/3scene净help为正；N1不得成为冠军。
- 仅当三个scene完整且`full_target25_authorized=true`才发布Target25；否则报告科学未晋级。Target25通过才授权K10扩展，Stage B不在本run自动执行范围内。

## 启动核验

- 远端owner PID=`2958724`（PPID1），控制PID=`2958726`，当前smoke worker PID=`2958727`；worker CWD精确指向本run的`fe9ec142`release，cmdline为预登记`p3-smoke`，输出根为本run的`smoke`。
- worker映射物理GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`，启动采样显存7646MiB；GPU0启动前已有2个训练进程，加入本run后总数为3，未超过用户授权上限。
- 本地启动SSH在取得PID后因远端owner保持连接而主动断开；远端owner已脱离为PPID1且继续存活。首次日志采样为0字节，属于stdout缓冲，进程状态为运行且GPU已建立compute context；后续只读检查日志/artifact增长，不重复启动。

## 终态证据与完整日志解析

### 三次正式run的故障演化

| run | 运行时代码 | 完整日志大小 | 失败位置 | 终态 |
|---|---|---:|---|---|
|`wiser_rf_p3_primary_hist_e0_20260831_v1`|`cc720e43`|1186字节|进入训练前，`capsule_id`配置与历史manifest绑定漂移|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|`wiser_rf_p3_primary_hist_e0_20260831_v1_bindfix1`|`ba704748`|1631字节|N6 Stage1，`id_backbone.t3.dw.weight`非有限梯度|同上|
|`wiser_rf_p3_primary_hist_e0_20260831_v1_gradfix1`|`fe9ec142`|1640字节|N6 Stage2分支，`id_backbone.t3.dw.weight`非有限梯度|同上|

三个stdout日志均已从首字节读到EOF。当前run日志只包含一条完整Python traceback；没有`CUDA out of memory`、`Killed`、非法显存访问、文件系统异常、query打开记录、prediction写入或scorer启动记录。

当前日志文件创建于2026-08-31 17:55:07.311849+08:00，最终修改于2026-09-01 00:02:35.877411+08:00，对应约6小时7分28.6秒。终态只读回查确认PID`2958724/2958726/2958727`全部不存在；run root只保留空的`smoke/`目录，递归文件数为0，`pilot/`和`score/`均未创建。日志修改时间、进程消失和空artifact根三项证据相互一致。

运行期只读采样曾观察到GPU0显存约7668–8620MiB、GPU利用率约39%–43%，worker CPU约100%；失败后本run PID不再出现在compute-app列表。该模式不支持OOM或外部杀进程解释，但这些离散采样不能替代逐步训练遥测。

## 失败代码路径

1. `p3-smoke`直接读取正式`p3_training.stage_steps=[1500,2000,2500]`，没有像旧`smoke`入口那样覆盖成少步预算。N6实际计划包含Stage1的1500步、三个Stage2候选分支各2000步以及Stage3的2500步，总计10000个optimizer step。
2. traceback到达`stage2_wiser_runner.py:684`，说明Stage1训练、支持集安全插值和Stage1返回已经完成，随后在三个Stage2分支之一的`train_branch`内失败。日志没有记录branch、step或Stage1选中alpha，因此不能区分`stage2_time`、`stage2_frequency`或`stage2_joint`，也不能确定失败发生在该分支的第几步。
3. `id_backbone.t3.dw.weight`属于`_P3_STAGE1_TIME_PREFIXES`，在Stage1及三个Stage2分支中始终可训练。它是循环中第一个被命名检查发现非有限值的参数，不等于产生异常的首个数学算子。
4. N6先分别计算`primary_grads=grad(p3.total)`和`auxiliary_grads=grad(auxiliary)`，再执行全局冲突投影并把二者相加到`parameter.grad`。现有检查只发生在相加之后，没有分别检查`p3.total`、class risk、duals、primary gradient、auxiliary gradient和projected gradient。因此当前traceback无法把异常归因到P3主目标、辅助目标或投影。
5. 配置中的`diagnostic_interval=100`只在dataclass中校验，没有在训练循环中消费；`smoke_result.json`也只在全部训练成功后写入。这解释了为何6小时内日志和artifact始终为空，也使精确首错点不可恢复。

## 根因分层判断

### 已证实

- `fe9ec142`没有实现真实N6多步可学习性闭合。它通过了合成`zero_identity/zero_fft/ill_conditioned`单次反向测试，但真实checkpoint在动态训练后仍产生非有限梯度。
- 该修复确实把失败从Stage1推迟到Stage2，说明此前修复的Ledoit-Wolf退化方差路径是一个真实问题，但不是唯一问题。
- 当前smoke同时承担技术检查和完整N6训练，成本过高且缺少中间证据。它不符合“快速确认真实checkpoint能否安全启动”的最小实验目的。

### 高优先级根因假设

1. **H1：D92内部优化器仍有零点不可导路径，优先级最高。**`_d92_exact_metric`以`create_graph=True`展开内部Adam更新，但二阶矩分母使用`sqrt(v)+1e-8`。epsilon位于开平方之后，只保护前向除法，不保护`v=0`处的反向导数；梯度总范数也先直接`sqrt(sum(g²))`。某些坐标在真实动态状态中变为精确零时，可把NaN/Inf沿P3主梯度传播到`t3.dw.weight`。
2. **H2：D92 score RMS退化，优先级高。**`_d92_rms`直接执行`sqrt(mean(centered_score²))`，后续多处用该值作除数，没有在RMS为0或极小时定义明确的同构fallback。Stage1适配后的支持几何可能首次触发该边界。
3. **H3：对偶变量在Stage1累积后放大Stage2主梯度，优先级中等。**N6使用`dual_rate=0.1`、`class_risk_epsilon=0`，对偶变量每步只增不减、无上界，并从Stage1带入全部Stage2分支；外层P3训练也没有梯度裁剪。若某类风险长期高于基线，1500步累积可在Stage2入口显著放大主梯度。
4. **H4：投影链隐藏了首个污染分量，优先级中等。**投影实现对有限极端值有合成测试，但没有先拒绝非有限primary/auxiliary输入。任一输入已被污染时，缩放、点积和反缩放会继续传播异常；最终只报告合并后第一个非有限参数。

H1–H4是代码与运行位置支持的可证伪假设，不是已经由当前日志唯一证明的根因。`torch.linalg.solve`的病态协方差也属于可能路径，但当前没有条件数、Cholesky/solve异常或首错算子证据，不能把它提升为主结论。

### 当前证据不支持的解释

- 不支持“GPU显存不足”或“服务器杀进程”：日志无OOM/Killed，进程正常抛出项目定义的`RuntimeError`后退出。
- 不支持“query污染导致失败”：smoke只加载support，query目录和prediction均未创建。
- 不支持“低性能触发停止”：在任何性能评估前已经技术终止。
- 不支持把`t3.dw.weight`直接写成根因算子：它只是首个被循环检查到的非有限参数。

## 性能与科学结论边界

| 指标或阶段 | 结果 | 原因 |
|---|---|---|
|smoke|FAIL|Stage2分支非有限梯度|
|pilot的21个scene×arm prediction|0/21|pilot未启动|
|query准确率提升|`N/A`|query未打开|
|`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`|全部`N/A`|无prediction、无truth-last评分|
|P3 BA/floor/每类准确率/help-harm|全部`N/A`|无scorer输入|
|Target25授权|`false`|pilot未闭合|
|K10扩展与Stage B授权|`false`|上游门槛未产生|

本run只能声明工程技术失败，不能声明WISER-RF P3-Primary有效、无效、优于基线或未达到科学门槛。前三次run共同证明当前发布流程能阻止绑定漂移和非有限梯度继续污染query结果，但尚未证明方法具备真实checkpoint上的稳定可训练性。

## 下一版最小修复建议

以下建议属于后续新提交/新release/新run ID的设计输入，本报告未实施修复或重跑：

1. 将`p3-smoke`改为真正的有界技术检查：真实checkpoint、真实support、N2–N6关键分支各1–5步，并显式覆盖Stage1→三个Stage2分支→Stage3的首步；正式pilot仍保留完整训练预算。
2. 让`diagnostic_interval`实际写入support-only进度JSONL，至少记录`branch/step/selected_alpha`、P3各损失、6类risk/violation/dual、primary/auxiliary/projected/combined梯度的finite/max/norm、投影点积、学习率、显存和`query_rows_used=0`。失败前最后一条完整记录必须可独立读取。
3. 在D92内部把所有可能到达零点的开平方改成“先选择安全输入、再开平方、最后恢复精确前向值”的分支；对RMS=0明确定义技术失败或预登记fallback。不得用`nan_to_num`、把非有限梯度置零或静默跳步伪造可学习性。
4. 在投影前逐层检查`p3.total`、`auxiliary`、`primary_grads`和`auxiliary_grads`，在投影后再检查`projected`与combined gradient；首错报告必须包含branch、step、参数名和分量来源。
5. 先用真实checkpoint做算子级anomaly/finiteness复现，区分H1/H2与H3/H4。只有在证据指向对偶放大后，才讨论dual cap、归一化或梯度裁剪，因为这些会改变方法动力学，不能作为无证据的技术补丁。
6. 复用现有`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`；该数值修复不改变received IQ、物理ID、receiver/TX、scene、K或support/query划分，不触发数据重验证。

## 交付状态

| 对象 | 结论 | 最高已证状态 | 缺口 |
|---|---|---|---|
|正式run|`FAILED`|`RUNNING`后技术终止|无smoke PASS、无prediction、无分析性能|
|预测artifact|`FAILED`|仅创建空`smoke/`根|0/21 prediction/receipt|
|科学结果|`PARTIAL`|协议与技术停止边界已分析|无query/truth-last指标|
|Target25/Stage B|`PARTIAL`|未授权状态明确|必须先有新run的完整pilot门槛结果|

总体结论：`FAILED`，阻断对象是正式run的真实checkpoint多步梯度稳定性；数据协议、query隔离和不可覆盖输出没有发现违规。
