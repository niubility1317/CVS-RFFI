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
- 启动前重新执行本地66项D29及相邻回归测试、Git/SHA闭包、N607直连preflight/live inventory、远端SHA、`py_compile`、`bash -n`及output不存在门。

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

## N607完成结果

- 启动命令：`D29_GPU=0 bash code/scripts/launch_d29_pcsr_support_20260718.sh`；PID `3666336`；GPU0；运行18.8302s；状态`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`；90/90行完整。
- 最终选择：`D25-C0-DIM-CONCAT`，`selected_positive_route=false`。D29不是full-K10门失败后的回退，而是在support-held候选筛选中未形成正路线。
- 三个D29候选在外层15折和full-K10三个场景中全部禁用：D29-A/B/C均为`0/15`外层启用、`0/3`full-K10启用，全部`T/A=0`，与D28-A/D27-B逐fold指标完全一致。
- PCSR试验层统计解释了失败原因：每个候选209个安全试验均无新类严格增益；A/B/C分别有22/26/35个新类严格增益试验，但全部违反旧类安全。增大ρ只增加不安全试验，没有产生一个“安全且增益”的释放。

|候选|注册前旧类|注册后旧类|seen-new|H|遗忘|joint floor|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|Z0|71.11%|48.33%|52.67%|48.97%|22.78%|0.00%|identity support对照|
|B3|86.67%|73.33%|73.33%|72.65%|13.33%|23.33%|性能诊断对照，不可选择|
|C0|71.67%|50.56%|54.00%|50.35%|21.11%|0.00%|最终fallback|
|D29-A/B/C|80.00%|67.22%|47.33%|52.82%|12.78%|3.33%|PCSR全禁用，等于D27-B|

|D29场景|注册前旧类|注册后旧类|seen-new|H|遗忘|旧类floor|新类floor|
|---|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|81.67%|75.00%|52.00%|58.12%|6.67%|50.00%|20.00%|
|leo_low_elev_weak|75.00%|61.67%|54.00%|56.31%|13.33%|40.00%|0.00%|
|leo_rain_weak|83.33%|65.00%|36.00%|44.03%|18.33%|50.00%|0.00%|

|角色|TX/短handle|Z0|B3|C0|D29|
|---|---|---:|---:|---:|---:|
|old|20-15|86.67%|90.00%|86.67%|80.00%|
|old|8-20|90.00%|90.00%|90.00%|86.67%|
|old|14-10|13.33%|70.00%|13.33%|56.67%|
|old|14-7|66.67%|66.67%|70.00%|66.67%|
|old|6-15|16.67%|60.00%|23.33%|63.33%|
|old|20-19|16.67%|63.33%|20.00%|50.00%|
|new|09f8|3.33%|40.00%|3.33%|13.33%|
|new|1c2a|100.00%|86.67%|100.00%|76.67%|
|new|b8fb|66.67%|76.67%|73.33%|63.33%|
|new|d3af|86.67%|86.67%|86.67%|66.67%|
|new|f608|6.67%|76.67%|6.67%|16.67%|

### 稳定性与资源

- 完整读取90行`training_log.jsonl`，D29三候选共1,215条训练trace；D27、D28、D29三轮全部loss/gradient/几何数值有限，无NaN、Inf、OOM、Killed、Traceback或Exception。D29的失败不是优化发散，而是support拟合与外层泛化/安全边界冲突。
- D29组合资源：2,016活动参数、25epoch/optimizer step、3,456 head MAC/query、无dense query图、峰值CUDA显存0B；PCSR禁用时额外predictor state仍按结构计72B，组合持久状态30,932～31,004B。
- 完整适配+注册估算为10,889,280 MAC；相对identity-only单qKNN的17,600 MAC/query，D29 head为19.636%，减少80.364%；组合状态相对35,200B identity FP16 sample-state为87.875%～88.080%，减少约11.92%～12.13%。
- D29-A/B/C平均batch1 head latency约0.163～0.174ms，P95约0.171～0.202ms；release禁用时额外ramp算术为0。该时延覆盖D27 score+PCSR apply，但不代表完整backbone/FFT端到端时延。
- 当前只有identity的估算MAC/FP16状态，没有同硬件identity实测时延和端到端RAM/显存；D29的head CUDA 0B也不覆盖backbone、FFT96和RF32。因此完整Pareto证据仍未满足，不能把上述比例扩展成端到端结论。
- support capsule虽授权并验证13,608B int8 active payload，但D29 evaluator/predictor未把该component传入D27-B或PCSR计算；30.93～31.00KB组合状态也未单列int8。故本轮不能声称“int8原型辅助PCSR”，下一轮必须让授权int8旧类锚真实进入旧类适应路径并单列其状态/收益。
- 对identity-only单qKNN，D29 head的MAC和状态比具有轻量优势；但由于准确率与floor明显不达标，不能进入性能-资源Pareto正前沿。

### 协议与artifact

- `support_audit.json`确认每场景110个唯一support物理样本、每样本单一LEO_weak观测、`support_view_count=1`、`support_row_multiplicity=1`、`derived_support_rows=0`、`additional_leo_overlay_count=0`；三场景的physical ID、parent IQ hash和overlay token两两无交集。
- query/clean/source精确字段全部满足support-only边界：query opened/rows/labels均为0，query role/true batch count/quota/global assignment均为false，clean dataset/cache/control-flow不可达，source六项访问均为false。
- Phase1组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，且三个正式claim/authority字段均为false；本节全部数字只属于development support-held机制诊断。

|artifact|SHA256|
|---|---|
|`training_log.jsonl`|`3bb2ba276af46dd1b888ee378faa6273f07c20699e640a6e510f0b1b2af8a869`|
|`selection.json`|`a9912ef4c2966693b625afac101c104ff86a0da3763c9ce87076d5c05bbc3ee1`|
|`support_audit.json`|`9942a4f2f424060d83ec2887e164eb910241f322aeca8edcb2fcdfb436561dea`|
|`resource_audit.json`|`3ffe05126a6431263147bab526f8d05bbafc288858f5d2c8751ca03015abc64b`|
|`geometry_audit.json`|`20e52e3d18fea745dd0cdbb332a425f173d25e45bb3b2970a99f7ac690bd3ed7`|
|`RECEIPT.json`|`9decfd46d0872a05ba40a3f99f2bb4549b0008e9ebe623493e37c6517b725791`|

远端任务结束后已确认`NO_SSH_PROCESS`和`NO_ESTABLISHED_N607_TCP22`。

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

## D27–D29三轮retrospective（已完成）

- 已重读活动目标和`项目.md`的Stage2-B/C、LEO_weak-only、单IQ单overlay、query隔离、资源与K锚点条款；已重建并搜索项目conversation index。索引未返回比当前D27–D29本地报告更直接的新决定，因此以当前Git报告与完整artifact为准。
- 已完整读取D27、D28、D29三轮各90行training log，而不是只看最佳行或日志尾部。D27/D28/D29分别有1,140/1,215/1,215条训练trace，三轮均无非有限值。
- D27结论：逐新类静态bias和10-step新类fit能把遗忘压到12.78%，但seen-new只有47.33%；15-step不再改善新类，反而进一步损伤旧类。继续加step不是主要解法。
- D28结论：E5可分辨新旧support，但对所有new列共同平移无法改变new-new排序；任一能改善new的λ都会明显损伤old，因此15/15折和full-K10均关闭。
- D29结论：逐类单侧boost具有new-new自由度，但仍会提高逐样本`max_new`；所有严格新类增益均触发至少一个旧support安全冲突，ρ从0.25增加到1.0只增加不安全试验。继续放大ρ或扫描更多相同形式宽度属于同机制重复，予以淘汰。
- floor审计：旧类瓶颈为20-19/14-10，新类瓶颈为09f8/f608；low elevation的09f8和rain的f608均为0%。下一轮必须显式保留逐旧/新类floor和场景floor，不能只优化平均H。
- 协议复核PASS：三轮都使用同一固定LEO_weak IQ的z160+FFT96+RF32表征，无clean/source、无query truth、无角色Oracle、无类别配额；但当前仍只是support-only且组件为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。
- 域适应与注册同等性：D27-B继续作为target-old适应底座；任何下一层注册机制都必须同run报告注册前old-only和注册后all-class、old/new/H/逐类floor/forgetting，不能只报告新类校准。

### D30下一假设

不放松旧类保护，改为“max-new包络保持”的逐新类校准。对support-only OOF学到类偏移`b_c`后，逐样本计算`u_c=s_c+b_c`，再令：

```text
s'_c = u_c - max_d(u_d) + max_d(s_d), c in Y_new
```

该变换允许改变new-new排序，但严格保持每个样本的`max_new`不变；old score也不变，因此old-vs-new包络、旧类预测和遗忘在数值上保持，而不再像D28/D29那样以提高`max_new`换取新类收益。它只能修复new-new混淆，不能修复new→old错误；若support-held OOF无增益则原子旁路，并据此转向class-balanced/CVaR表征学习，而不是放松query权限或继续扫描ρ。D30仍采用K=1旁路、K≥5 shot-rank OOF、逐样本全注册类argmax、无query拟合，并增加old→new/new→old/new→new support-held混淆审计。

retrospective门已满足，D30可以进入实现，但不得用任何query标签、正式确认矩阵或K1/K5/K20测试结果调参。
