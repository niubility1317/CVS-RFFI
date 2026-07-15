# qKNNv42极轻型Stage2-B/C优化追踪

日期：2026-07-15

目标：以最新版`项目.md`为唯一科学协议来源，把历史qKNN/ADV3B02诊断路线收敛为可部署、逐样本、`LEO_weak-only`的极轻型Stage2-B/C候选；在关闭运行时隔离缺口前保持fail closed。

|ID|来源|硬要求|目标实现/证据|当前状态|验收方式|
|---|---|---|---|---|---|
|QEL-01|`项目.md`4.4、6.1、6.2|Phase2对clean及clean派生信号物理不可达|严格predictor package、pre-open审计、OS隔离、访问账本|partial|真实Linux隔离smoke与post-run ledger均通过|
|QEL-02|`项目.md`4.5、6.2|逐样本面对全部注册类，禁止role、真实query批次类别数、quota和global assignment|12字段合同、exact request schema、独立sealed prediction/scorer|implemented_local|合同负测、预测先密封、scorer只读truth sidecar|
|QEL-03|`项目.md`8.4、10.3.1|固定5个target receiver、6个旧类、真实嵌套5/10/20个seen-new TX|离线sealed target package与逐TX覆盖清单|blocked|逐TX样本数、物理support/query ID及嵌套哈希齐全|
|QEL-04|`项目.md`10.3.1|K10统一开发选参；K1/K5/K20只做锁定确认|K10 development ledger与candidate lock|blocked|锁定前无K1/K5/K20反馈进入超参选择|
|QEL-05|用户目标与`项目.md`10.3.1|K10 old≥92%、旧类floor≥88%、new5/10/20≥92/90/86%；K5下降≤3pp|5receiver×≥5confirm seed×3场景结果|blocked|300个prediction cell、900个同row场景结果和逐类表|
|QEL-06|用户目标|K1适应相对identity非负，且相对strict direct ADV3B02总体及逐receiver≥+2pp，paired CI下界>0|同样本paired统计与receiver分层CI|blocked|固定confirm seed独立scorer统计|
|QEL-07|用户目标|不同K值遗忘不劣于identity-only|K1/5/10/20 matched forgetting ledger|blocked|候选与identity使用相同query token配对|
|QEL-08|用户目标|≤50k训练参数、≤20epoch、≤256KB、无dense query graph|effective8 44,048参数/12epoch候选接入strict runtime|partial|独立参数、持久状态、MAC、时延、显存审计|
|QEL-09|用户目标|默认1-view，低置信度逐样本触发3/5-view|margin/entropy/view disagreement自适应策略|partial|阈值仅来自source validation或support；记录每样本view count|
|QEL-10|`AGENTS.md`|每GPU最多2个训练任务、local-first、报告与Git证据|控制状态修复、快照、报告、commit|in_progress|state current view、diff、测试、commit|
|QEL-11|严格运行时审计|adapter/head/TTA外部provenance、固定输入快照、N607等价隔离|candidate trust root、immutable snapshot、Landlock smoke|blocked|三项pre-run blocker全部关闭前`formal_launch_authority=false`|

## 当前证据边界

- `effective8 v14`满足44,048参数、12epoch和自适应1→3→5-view的资源外形，但target matrix未执行，且尚未接入严格predictor/scorer。
- 历史ADV3B02三方法375行是旧版Stage2-B诊断链，不含target-new，配置暴露`query_per_tx`，不能作为当前正式模板。
- 历史`id_norm_late_feature`固定5-view路线性能有正收益，但参数289,685、固定5-view、K1负收益且仅2个新类，不满足当前目标。
- 当前严格基础设施仅为`LOCAL_DIAGNOSTIC_PASS`；本追踪表不会把字段自声明或本地模拟执行升级为正式协议证据。

## 实施顺序

1. 修复prompt、workflow contract和顶层state中的旧控制语义。
2. 将effective8 adapter/head/TTA绑定外部candidate/plan trust root并接入唯一strict request builder。
3. 离线构造并密封25份真实target package，删除Phase2 runtime中的`query_per_tx`与truth/role路径。
4. 本地负测、完整日志与资源审计通过后，才进入N607只读preflight和单cell严格Linux smoke。
5. K10开发锁定后执行独立确认矩阵；K1/K5/K20不得回流选参。

