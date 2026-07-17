# D29逐新类安全释放support-only实验

## 启动前记录

- experiment ID：`d29_pcsr_20260718/support_screen_v1`；日期：2026-07-18；operator：Codex；状态：`LOCAL_IMPLEMENTATION_IN_PROGRESS`。
- 目标：解决D28所有new列共同平移无法改变new-new排序、且平均用约12pp old换约7pp new的问题；用每个新类仅2个标量的有界单侧释放，同时优化旧类遗忘保护和`09f8/f608`弱新类floor。
- 数据工作点保持不变：receiver `20-1`、开发seed `713101`、K=10、6 old+5 seen-new、3个LEO_weak场景×5个held-rank fold；复用同一sealed support capsule，不新增物理IQ、LEO信道view或support行，query不打开。
- 本轮属于用户特许的`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`开发筛选：`formal_launch_authority=false`、`formal_metric_claim_allowed=false`、`performance_claim_allowed=false`、`query_opened=false`、`query_rows_opened=0`、`query_labels_opened=0`、`support_query_disjointness_status=SUPPORT_ONLY_NO_QUERY_CLAIM`。全部准确率、floor、H和遗忘量只能解释为注册support内部held-rank诊断，不是正式query性能、独立确认性能或部署证据。
- D29只能复用既有capsule中完全相同的IQ payload与成员清单；D29候选/method lock不同于D27/D28，打开support前仍须生成并验证绑定D29 method lock的新授权/密封根，同时锁定ADV3B02 checkpoint SHA、历史int8组件SHA、真实Phase1 TX列到注册class handle的逐列映射及D29-A/B/C候选。该步骤不重建IQ、不重新叠加LEO信道，也不重复完整数据准备。
- 这是D27后第3个完成前探索轮；本轮完成后、任何D30启动前必须执行并记录三轮retrospective。

## 方法锁：PCSR

对D27-B全类score，为每个新类`c`计算`z_c=s_c-max(max_old,max_other_new)`，在预计算D27-B support score上执行gate-layer 5-fold shot-rank OOF，以fold内未参与PCSR参数拟合的新类support真实类缺口构造`T_c=1.05×{Q50,Q75,Q90,max}`候选及disabled。除非D27-B基础头也在每个fold内重新拟合，否则该证据不得称为端到端LOO。单侧释放为`p_c=clip((z_c+T_c)/T_c,0,1)`、`a_c=A_c p_c`。

`A_c`不自由扫描，而由旧support闭式安全上界给出：对每个D27正确旧样本，要求`s_c+a_c<正确old score`，得到`A_safe=min_i margin_ic/p_ic`，最终`A_c=rho×min(A_safe,2T_c)`。每个新类独立约束，因此多新类同时释放仍不能翻转任何D27正确旧support。

- D29-A：`rho=0.25`，new overall优先；D29-B：`rho=0.50`，H/平衡优先；D29-C：`rho=1.00`，new class floor优先。
- 固定类序做至多2轮坐标选择；每个候选必须保持所有raw-correct旧support逐样本正确和逐旧类非退化，且new gate-layer OOF overall/floor均不降、至少一项严格提升，否则精确透传D27-B。
- 每个新类只保存`T_c,A_c`；允许`a_c-a_d`改变new-new排序，这是相对D28的新增自由度。old score列bitwise不变，推理仍为逐样本全部注册类一次argmax。
- K=1无合法gate-layer OOF缺口时精确透传D27-B；K=2～4 fail closed。query API不得接收truth、role、quota、batch class count/order/statistics。

上述旧类安全结论只约束注册support：old score列bitwise不变也不能单独证明正式query旧类预测或遗忘不变。完整候选应表述为“D27-B target-old域适应+PCSR seen-new注册”，PCSR不能脱离D27-B单独声明完成域适应。

## 协议字段锁

support audit、candidate lock、resource audit和最终receipt必须使用当前精确字段名，而不是旧字段或泛化描述：

```text
phase2_query_decision_policy=per_sample_all_registered_classes
phase2_query_role_oracle_access=false
phase2_query_true_batch_class_count_access=false
phase2_query_class_quota_access=false
phase2_query_batch_global_assignment=false
phase2_query_post_reception_view_fit_access=false

phase2_sample_view_policy=leo_weak_only_no_clean_access
clean_sample_access=false
clean_derived_signal_access=false
phase2_clean_dataset_reachable=false
phase2_clean_cache_reachable=false
phase2_clean_control_flow_reachable=false

phase2_source_sample_access=false
phase2_source_cache_access=false
phase2_source_label_access=false
phase2_unapproved_source_derived_signal_access=false
phase2_source_replay=false
phase2_external_source_adapter_access=false
phase2_pretrained_artifact_policy=sealed_phase1_deployment_bundle_with_optional_int8_domain_class_prototypes_v1
```

- 字段必须由pre-open package/member allowlist、SHA/provenance验证和runtime opened-file/access audit支持，不能只靠manifest布尔自声明。
- 历史int8组件整体只读，禁止更新、替换或加载全精度prototype、source样本、样本级source feature、source cache及bundle外source-derived artifact。
- 当前Phase1组件证据状态继续继承为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；即使support筛选为正，也必须先重建checkpoint+int8共同封存bundle与正式method lock，才能进入query评估和独立确认矩阵。

## 候选与资源锁

- 固定6候选：Z0、B3、C0、D29-A、D29-B、D29-C，共90行；D27-B与D28负结果由同capsule先验实验作直接参考。
- 基础D27-B为2,016活动参数、25个optimizer step、3,456 head MAC/query。PCSR新增`2×C_new`个FP32标量，5/10/20新类的标量payload为40/80/160B；计入32B状态头后，完整PCSR predictor state分别为72/112/192B。PCSR为0梯度步，query约4个标量操作/新类，无新增原型、FFT、support bank、dense图或矩阵乘法。
- 正式部署上限保持≤80,000活动参数、≤30epoch、≤50 optimizer steps、≤256KB持久化predictor state。最终资源必须报告D27-B+PCSR组合总状态，不得只报PCSR增量；外部选择审计不计入星上predictor state，但须单独完整保存。
- PCSR必须标记`EVAL_ONLY_CLOSED_FORM_ADAPTATION`并保存fold参数、坐标选择、`T/A`、安全上界和回退求解诊断；D27-B的25-step训练部分仍保存完整loss trace。
- `z160+FFT96+RF32`均继承自同一固定已接收LEO_weak IQ。正式资源表须区分head MAC与端到端表征成本，并给出平均/P95 backbone forward count、FFT/表征分支数、端到端时延、峰值RAM/显存及相对identity-only单qKNN的同硬件Pareto变化。

## N607计划

- 远端根：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；output/log分别位于`runs/d29_pcsr_20260718/output/support_screen_v1`与`logs/d29_pcsr_20260718/support_screen_v1.log`。
- 启动前重新执行本地73+新增D29测试、Git/SHA闭包、N607直连preflight/live inventory、远端SHA、`py_compile`、`bash -n`及output不存在门。

### 启动前live状态

- 2026-07-18 05:11 CST直连preflight PASS：host `dell-DSS8440`，项目根可见，8张RTX 3090均为0%利用率、10MiB显存占用。
- `n607_training_inventory.py --direct-only --pretty`：`active_training_processes=[]`、`gpu_compute=[]`、`unknown_training_active=false`；本轮计划使用GPU0，未超过每GPU最多2个训练实验的约束。
- Git版本：`c179907c feat(stage2): add D29 classwise safe release screen`。
- 计划同步映射：`code/scripts/run_d25_support_only_concat.py`→同名远端路径；`code/cvsrffi/stage2_classwise_safe_release.py`→同名远端路径；`code/scripts/launch_d29_pcsr_support_20260718.sh`→同名远端路径。远端已有D27/D25/D24/CIAF/control/diag文件仅做SHA只读验证，不覆盖。
- 计划服务器命令：`D29_GPU=0 bash code/scripts/launch_d29_pcsr_support_20260718.sh`；Python环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；PID、log、output分别为`runs/d29_pcsr_20260718/support_screen_v1.pid`、`logs/d29_pcsr_20260718/support_screen_v1.log`、`runs/d29_pcsr_20260718/output/support_screen_v1`。
- 同步后远端验证PASS：runner/D29 core/launcher SHA分别为`9916f508...cba0`、`68633a72...eae`、`3c3026b6...66a`；D27 core与保留的diag operator分别为`553d6361...f1ff`、`14ec9193...1ca`；远端`py_compile`、`bash -n`通过，`OUTPUT_ABSENT`。

## 本地实现与验证

- 新增`code/cvsrffi/stage2_classwise_safe_release.py`：实现support-only 5-fold shot-rank OOF逐类宽度/幅度选择、全support联合安全复验、K=1精确旁路和逐行推理。
- 扩展`code/scripts/run_d25_support_only_concat.py`：加入D29候选锁v7、90行外层held-rank评估、K=10完整部署审计、组合资源、候选选择、full-K10旧类门、receipt和CLI。
- 新增`code/scripts/launch_d29_pcsr_support_20260718.sh`：固定N607路径、source closure SHA、output不存在门和`d29_v1_pcsr`候选集。
- 独立代码审计发现并修复两项关键缺陷：OOF为正但full refit无严格新类增益，或full refit违反旧类安全时，均原子回退全零D27-B旁路，不再保留无收益状态或中断完整矩阵。
- 严格拒绝浮点/字符串/布尔伪装的`coordinate_passes`、class count和`enabled`状态；`4×C_new`只标为ramp算术，不冒充包含max/top-2/排序在内的完整开销，最终以实测head latency为主。
- 2026-07-18本地验证：`py_compile`通过；核心+相邻D25/D26/D27/D28测试共66项通过；额外500组随机K=5压力验证中248组启用且全部满足full-support安全与严格新类提升，252组禁用且全部与原始score逐字节一致；launcher `bash -n`和`git diff --check`通过。
- 当前source closure：runner `9916f5087844aeb5567ed9eea3b4d5f6069abeb35fd59809afdb899fe409cba0`；D29 core `68633a72a0af8ecec519fde72679a78969cf89c86d5551eca47cd2eb70bf2eae`。本地`stage2_diag_cosine_exploration.py`含无关未提交改动，不同步；launcher继续锁远端已验证版本`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`。

## 完成后补充

- 待记录90行support-held联合诊断、逐场景/逐类floor、释放启用率与`T/A`、D27-B完整loss、PCSR闭式求解诊断、修正资源Pareto、artifact哈希、独立代码/协议审计和三轮retrospective。不得把独立代码/协议审计写成独立确认seed矩阵。
- 若筛选为正，正式阶段须在同run、相同query ID和相同推理View下同时保存注册前D27-B old-only状态与注册后D27-B+PCSR全类状态，报告`old_acc_before_increment`、`old_acc`、`min_old_class_acc`、`seen_new_acc`、`H_old_new`、`average_forgetting`、`old_adaptation_gain`、逐旧/新类结果及old→new、new→old、new→new混淆。
- 本轮1个receiver、1个开发seed、K10、5个seen-new及support-held诊断不足以形成正式结论。正路线只能在K10 support上锁定唯一candidate/超参数后，进入5个target receiver×至少5个独立确认seed×3个互斥LEO_weak场景×嵌套真实5/10/20 seen-new TX，并补齐K1/K5/K10/K20遗忘锚点；query只能用于一次性测试和隔离评分，不得反向调参。

## D29完成后三轮retrospective门

任何D30启动前，必须在本报告中记录以下检查结果：

- 重读活动目标与`项目.md`，并查询项目conversation index确认已探索路线和既有决定。
- 阅读D27、D28、D29完整日志，分别记录静态逐类bias、共同平移evidence gate和逐类安全释放的有效机制、失败原因与拒绝路线。
- 检查D29是否实际启用、是否只在少数场景/类别产生收益，以及`09f8/f608`改善是否以其他新类退化为代价。
- 区分support OOF安全与正式query旧类安全，不把前者外推为正式遗忘结论。
- 重新核验LEO_weak-only、单物理IQ单overlay、无clean/source、无query truth、无角色Oracle、无类别配额和逐样本全注册类决策。
- 确认下一候选仍对域适应和新类注册同等留证，包含同run注册前/后、`old_acc`、`min_old_class_acc`、`seen_new_acc`、`H_old_new`、逐类结果和forgetting。
- 核对组合参数、epoch、optimizer step、状态、MAC、端到端时延、峰值RAM/显存及identity-only单qKNN Pareto。
- 记录经验、拒绝路线、剩余假设和下一轮决定；若D29为正，先共同封存checkpoint+int8并重建正式method lock；若为负，直接淘汰。D30不得利用任何query标签或正式确认结果继续调参。
