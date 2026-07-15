# CVS项目相关约定与数据协议

版本：2026-07-15

本文是`docs/PROJECT_PROTOCOL.md`的便携清单，不替代源协议。发生冲突时，CVS科学场景与数据协议以`docs/PROJECT_PROTOCOL.md`为准；工程、Git与发布安全以`AGENTS.md`为准。

## 核心约定

- 主场景是天基RFFI中的弱标注跨接收机域泛化与在轨跨域少样本适应；采用地面训练、天上部署。
- Phase1是source-domain weak-label/semi-supervised DG，`rho_label<=0.1`，不得使用目标接收机域`R_t`的任何数据或选择信号。
- `R_t`与`R_s`必须不相交；`R_t`可包含一个或多个receiver。
- `Y_old`是地面训练已见TX；`Y_new`与`Y_old`互斥；Phase3的`Y_unknown`与`Y_old∪Y_new`互斥。
- Phase2 Stage2-A/B/C严格为`LEO_weak-only`：所有support/query、适配、校准、注册、选择、回滚、排名与正式评估输入必须已叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`。
- Phase2接触不到clean样本或clean派生feature/logit/prototype/决策信号。launchable row必须记录`phase2_sample_view_policy=leo_weak_only_no_clean_access`、`clean_sample_access=false`、`target_channel_view=leo_weak_only`、实际scenario和overlay provenance。
- 违反LEO_weak-only边界的row为`LOCAL_PROTOCOL_REPAIR_REQUIRED`；历史clean-access artifact只能封存为`PROTOCOL_INVALID_FOR_PHASE2`。
- Stage2-A无target label support；Stage2-B只允许target-old support；Stage2-C允许target-old校准与target-new enrollment。unknown query只作Phase3-backup/evaluation-only。
- Stage2-B/C必须显式记录正整数`K`；推荐锚点为`{1,2,5,10,15,20,50}`，`K>20`不得称为strict few-shot。
- 每个query必须独立面对全部已注册类别；禁止query真实角色、类别quota、排序/分块和Hungarian/等价批量配额Oracle。
- launchable Phase2 row必须声明`phase2_query_decision_policy=per_sample_all_registered_classes`，并将`phase2_query_role_oracle_access`、`phase2_query_class_count_access`、`phase2_query_class_quota_access`和`phase2_query_batch_global_assignment`设为`false`；缺失或启用即`LOCAL_PROTOCOL_REPAIR_REQUIRED`。
- Stage2-B/C合法support label、enrollment identity与预注册正整数K-shot support不属于query类别配额；Phase1 source-side quota audit不受影响，query真实标签只能在预测冻结后用于指标计算。
- Phase2主指标是同row的`old_acc`、`seen_new_acc`与`H_old_new`；unknown FAR/FPR95/AUROC属于Phase3备用项。
- 当前target-old优先使用`ManySig.pkl`，target-new优先使用同receiver label下`ManyTx.pkl`中的真实non-`Y_old` TX；跨pkl按receiver label对齐，并逐TX验证support/query样本覆盖。
- WiSig/ManySig与简化LEO增强是terrestrial proxy与physics-informed stress，不得声称真实卫星在轨验证完成。
- 正式结果必须绑定同一candidate/run/split/K/receiver/TX/scenario/seed，禁止跨row拼接单项最优值。

更完整的Phase1划分、Stage2权限、K1/K5/K10/K20极轻型门槛、指标和自动化约束见`docs/PROJECT_PROTOCOL.md`与Git承载面的workflow contract。
