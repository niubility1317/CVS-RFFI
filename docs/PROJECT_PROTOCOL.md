# CVS项目协议

## 场景定义

CVS的主场景是天基RFFI中的弱标注跨接收机域泛化与在轨跨域少样本适应。模型在地面训练，部署到目标卫星接收机域后只允许推理、prototype更新、轻量校准、阈值微调或小adapter更新。自2026-07-07起，Phase2主线是使用叠加简化LEO星地信道的少量目标域旧类样本和新类样本，完成目标域适应、旧类校准和新类学习；open-set/unknown拒识下沉为Phase3备用项。自2026-07-15起，Phase2数据入口严格为`LEO_weak-only`，Phase2接触不到clean样本。

项目需要处理四个约束：

- 星上算力受限，完整训练放在地面完成。
- 发射机身份标签稀缺，receiver、day、rx_day和信道场景等domain label更容易获得。
- 星地链路中的residual Doppler/CFO、相位噪声、低SNR、低仰角、弱多径和弱Rician/shadowed-Rician fading会破坏raw IQ中的发射机细节。
- 在轨部署会遇到旧类和新类少样本。Phase2主线先处理旧类适应和新类学习；未知类拒识作为Phase3备用安全扩展。

## 集合定义

```text
x = R_d( H_d * T_y(s) ) + n
```

- `T_y`：发射机硬件非理想性，是应保留的身份来源。
- `H_d`：传播/星地信道，是域扰动来源。
- `R_d`：接收机链路响应，是跨接收机偏移来源。
- `n`：噪声。

```text
R_s = {source training receivers}
R_t = {target receiver domain / deployment proxy domain}
intersection(R_t, R_s) = empty

Y_old = ground-training transmitter set
intersection(Y_new, Y_old) = empty
intersection(Y_unknown, union(Y_old, Y_new)) = empty
```

`R_t`可以是单接收机，也可以是多接收机deployment proxy domain。关键条件是`R_t`与`R_s`不相交，并且target-old、target-new的support/query权限都按同一个`R_t`定义。若启用Phase3 open-set备用项，unknown query也必须来自同一个`R_t`，且不能参与Phase2阈值拟合或主线排序。

## 地面阶段

地面训练是weak-label/semi-supervised source-domain DG，不是部署few-shot。训练数据分为：

```text
L_s = {(x_i, y_i, d_i): receiver(x_i) in R_s}
U_s = {(x_j, d_j): receiver(x_j) in R_s, y_j hidden or unavailable}
rho_label = |L_s| / (|L_s| + |U_s|) <= 0.1
```

推荐`rho_label`网格为`{0.005,0.01,0.02,0.05,0.1}`。地面阶段不得使用`R_t`的样本、统计、BN信息、阈值、prototype、adapter、伪标签、验证结果或early stopping信号。

## 表征架构

```text
raw IQ -> CV-SincNet/CVS -> z_id, z_dom
```

- `z_id`用于发射机身份分类、prototype、少样本注册和旧类校准。
- `z_dom`吸收receiver、day、rx_day、channel和satellite-style nuisance，用于域诊断、域监督、adapter gate和泄漏审计。

推荐机制包括物理先验CV-SincNet、`z_id/z_dom`解耦、domain-supervised`z_dom`、GRL/leakage probe、Mean Teacher/FreeMatch/UPS、prototype agreement、MLDG/episodic source split和source-derived satellite strong-view consistency。

## 部署阶段

在轨部署阶段面对目标接收机域`R_t`。Stage2-B/C必须记录正整数`K`、support/query划分、receiver/TX split、threshold scope和satellite/LEO target view。Phase2主线row必须包含target-old和target-new目标域样本，并按简化LEO目标视图构造；unknown/open-set字段只作为Phase3备用或diagnostic metadata。

Phase2的全部Stage2-A/B/C target-old、target-new及可选Phase3-backup unknown support/query、适配集、校准集、注册集、模型选择信号、回滚/排序信号和正式评估输入，都必须已经实际叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`之一。Phase2禁止读取、缓存、恢复、重新构造clean样本，也禁止接收由clean样本派生的feature、logit、prototype、阈值、TTA触发、回滚或晋升信号。clean control只允许存在于Phase1或与Phase2完全隔离的离线参考流程中。

每个launchable Phase2 row必须记录`phase2_sample_view_policy=leo_weak_only_no_clean_access`、`clean_sample_access=false`、实际`leo_*_weak`scenario、satellite seed或等价sample-level overlay provenance。缺字段、目标视图含clean、场景超出三个允许的`leo_*_weak`视图或无法证明实际叠加时，必须标为`LOCAL_PROTOCOL_REPAIR_REQUIRED`并阻断matrix、runner、promotion与正式声明。历史clean-access artifact只可封存为`PROTOCOL_INVALID_FOR_PHASE2`，不得生成新的clean诊断候选。

TTA轻量化必须固定同一物理LEO观测、support/query、checkpoint和adapter后比较1/3/5-view；不得用不同adapter或不同LEO随机扰动制造view数量差异。正式结果使用逐样本可部署决策，并报告backbone前向数、FFT数以及相对5-view的`old_acc`、`seen_new_acc`和`H_old_new`变化。

对任一target query，推理前不得假定其属于旧类、新类或未知类。Phase2/Phase3正式候选必须让每个query面对全部已注册类别及允许的reject/defer机制；禁止使用真实old/new/unknown角色、整批类别数量、每类quota、query排序/分块以及Hungarian或等价配额重排。历史role/quota Oracle artifact仅可标记为`PROTOCOL_INVALID_FOR_DEPLOYMENT`后封存，不得新生成、调参、排名、进入论文主表或形成部署声明。本禁令不影响Phase1源域半监督训练中的伪标签quota审计与采样平衡。

每个launchable Phase2 row必须记录`phase2_query_decision_policy=per_sample_all_registered_classes`，并令`phase2_query_role_oracle_access=false`、`phase2_query_class_count_access=false`、`phase2_query_class_quota_access=false`、`phase2_query_batch_global_assignment=false`。缺字段、任一guard不为false，或命令启用role Oracle、真实query class count/quota、Hungarian、optimal transport、global quota matching或batch reassignment时，必须标为`LOCAL_PROTOCOL_REPAIR_REQUIRED`并阻断matrix、runner、promotion与正式声明。support标签、support enrollment身份、预声明的每类`K-shot`构造及预测完成后的metric-only标签读取仍合法，但不得反向影响决策。

多View压缩允许在地面使用严格`rx_light5`逐View监督训练不超过50k参数、最终状态不超过256KB的小模块，也允许星上逐样本置信度门控的1→3→5-view自适应TTA。门限只能由source validation或注册support确定，禁止用query标签、query真实角色、整批类别比例、每类quota或Hungarian分配；正式结果必须报告平均/P95 backbone前向数和1/3/5-view触发率。仅蒸馏5-view均值或仍对所有query固定执行5次backbone前向，不能单独证明多View计算已经压缩。

经用户明确授权，可另设`performance-relaxed`档，在不放宽无角色Oracle、无类别配额、无query拟合、无dense query图和逐样本决策的前提下，把首选档的参数、适配轮数/步数、持久状态或平均View计算提高50%–100%；绝对上限为100k参数、40epoch、512KB和5次backbone前向。放宽档必须逐项报告实际增幅，并与首选档及identity-only单qKNN做同row Pareto比较。

推荐`K`锚点为`{1,2,5,10,15,20,50}`。`K<=20`可称few-shot/low-shot；`K>20`应称higher-shot、medium-shot或saturation point。

## 可声明与禁止声明

可以声明：

- CVS面向天基RFFI的弱标注跨接收机DG与在轨跨域few-shot适应。
- WiSig/ManySig是terrestrial proxy benchmark / ground-accessible source domain family。
- satellite stress是物理启发部署压力测试。
- Stage2-B是旧类目标域校准。
- Stage2-C是Phase2主线的target-old adaptation + seen-new enrollment，前提是`Y_new`与`Y_old`不相交，且`R_t`与`R_s`不相交。
- Phase3是open-set/unknown rejection备用项，不是当前主线。

禁止声明：

- WiSig/ManySig是真实卫星训练集。
- satellite augmentation等价于真实在轨验证。
- source-only DG等价于few-shot learning。
- 旧类target support提升就是新类识别。
- Stage2-A/B拒识结果就是seen-new identity accuracy。
- open-set/unknown FAR结果就是Phase2主线成功。
- `R_t`与`R_s`重叠后仍称部署泛化。
- 缺少target-old或target-new样本覆盖时仍声称完整Stage2-C。
- 使用target query真实角色、类别配额或跨query批量决策后仍报告为Phase2/Phase3正式性能。
- Phase2读取clean样本或clean派生信号，或先用clean适配/选参再只报告`LEO_weak`结果。
- Phase2使用query真实角色、真实批次类别数量、每类query quota或跨query全局配额重排后仍报告正式性能。

## Git与Markdown同步

任何CVS项目相关改动都必须进入Git可追踪流程。改动前必须运行`git status -sb`，改动后必须检查`git diff`/`git status -sb`、完成必要验证，并提交本次意图明确的变更，除非用户明确要求不要提交。

协作输出规则：对于使用工具或长时间运行的任务，首次工具调用前、关键阶段切换时、重连或上下文压缩恢复后、出现阻塞时，以及持续工作期间至少每60秒，必须发送简洁、基于证据的进度更新；只报告可观察操作、发现和下一步，不披露私有思维链，不倾倒原始日志。仅无工具的简短问答可以省略过程更新。

项目相关Markdown必须随代码、配置、脚本、矩阵、prompt、报告模板或协议改动同步检查：

- 工作流、Git、协作或安全规则改动，更新`AGENTS.md`。
- CVS科学场景、数据协议、receiver/TX划分、`rho_label`、Stage2-A/B/C边界、K-shot、satellite/LEO视图、指标或声明口径改动，先更新本文件。
- README、docs、实验报告或发布说明涉及的用法、结果解释、发布范围和复现边界变化时，更新对应Markdown。
