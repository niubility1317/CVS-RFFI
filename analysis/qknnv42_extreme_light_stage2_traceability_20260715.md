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
|QEL-08|用户目标|≤50k训练参数、≤20epoch、≤256KB、无dense query graph|effective8 44,048参数/12epoch capsule；strict runtime硬上限已收紧为50k/20epoch/256KiB|implemented_local_partial_runtime|真实adapter序列化字节、持久状态、MAC、时延、显存审计|
|QEL-09|用户目标|默认1-view，低置信度逐样本触发3/5-view|margin/entropy/view disagreement自适应策略|partial|阈值仅来自source validation或support；记录每样本view count|
|QEL-10|`AGENTS.md`|每GPU最多2个训练任务、local-first、报告与Git证据|控制状态修复、快照、报告、commit|in_progress|state current view、diff、测试、commit|
|QEL-11|严格运行时审计|adapter/head/TTA外部provenance、固定输入快照、N607等价隔离|candidate capsule与双TorchScript parity已本地实现；package v3、immutable snapshot、Landlock smoke待完成|partial|全部pre-run blocker关闭前`formal_launch_authority=false`|

## 2026-07-15 candidate capsule实现增量

|实现|输入|输出|本地效果|证据状态|
|---|---|---|---|---|
|`phase2_candidate_capsule.py`|candidate lock、ADV3B02基座哈希、base/candidate TorchScript、effective8 LoRA、source stats、head lock、TTA policy与parity receipt|exact-schema capsule校验结果|固定ADV3B02 ID与SHA；固定8层/16个LoRA张量；重算44,048参数、88,096B FP16 payload及真实序列化delta字节；禁止clean、query拟合、role Oracle和quota|`implemented_local`|
|`export_adv3b02_effective8_torchscript.py`|同一ADV3B02 checkpoint、FP16 effective8 LoRA state、输入长度|base TorchScript、merged candidate TorchScript、数值parity receipt|base用于direct/identity，merged candidate用于适应后特征；要求injected→merged→TorchScript feature/logit最大绝对误差≤1e-4|`implemented_local_not_run_on_real_artifact`|
|`build_cvs_phase2_effective8_candidate_capsule.py`|外部candidate lock、两份TorchScript、LoRA、training manifest、source stats、head/TTA及parity receipt|包外可哈希candidate capsule|区分评测包总大小、FP16理论payload和真实星上序列化增量；只有“预装base＋delta重建且不额外持久化merged副本”时才允许采用增量资源口径|`implemented_local_not_built_from_real_v14`|
|strict runtime资源门禁|adapter execution config|fail-closed或预测继续|参数上限由100k降至50k、epoch由40降至20、持久状态由512KiB降至256KiB|`implemented_local`|

本地联合测试为13/13通过：6项capsule正负测试、1项TorchScript trace一致性测试、6项strict runtime/FFT/nested-K/资源门禁测试。测试只证明实现语义和拒绝路径；本地没有真实v14 adapter/checkpoint artifact，因此尚未生成真实capsule，也没有target准确率结果。

## 2026-07-15 qKNN性能增量

|实现|性能假设|资源影响|当前证据|
|---|---|---|---|
|`consensus67`＋有界partial Gram＋support不确定性|稳健化K1原型、抑制相邻类重叠，同时限制逆Gram弱方向放大|head最大15,740B；每query最大958 MAC|机制与FP16 round-trip测试通过|
|source三重identity保护|source锁定候选不得以平均收益交换最差episode或最低类|仅地面选择开销|选择逻辑测试通过|
|部署匹配multi-view worst-K训练损失|LoRA训练直接拟合部署时的多View support原型，而不是先平均View|参数/epoch/星上推理均不增加；地面step head运算约1.5—2倍|梯度与K覆盖测试通过|
|top1 score gate＋跨View稳定性LCB|低绝对相似度或跨View振荡样本触发额外View，高置信度样本停在1-view|TTA门限状态24B；backbone forward由实际触发率决定|lazy/eager一致性与触发测试通过|
|FFT权重源域消融|`weight=2.0`令FFT能量占80%，可能稀释ADV3B02`z_id`及LoRA收益|零新增参数/状态/forward|`0.5/0.7/1.0/2.0`CLI与能量映射测试通过；真实消融待运行|

当前性能实验顺序改为：FFT权重源域消融→部署匹配loss重训对比→K10 target开发→锁定后的K1/K5/K20确认。协议工作只维持正式结论所需的最低边界，不再占用主要优化周期。

## 2026-07-15 adapt优先级增量

FFT权重诊断已否定“降低FFT即可显著改善K1”的假设。当前性能顺序调整为：ground adapt层组/损失/epoch消融→锁定关键层→6,400参数BP-JG-LoRA target support快速适配→再恢复head与自适应View优化。formal ground LoRA新增`projection_feature`、`feat_joint`和`effective_feature`三种层组；nested worst-K已按physical ID排除support同源场景副本。预注册8路source-only矩阵统一8epoch、≤44,048参数，比较保守loss与K1边界增强loss，不读取target/query/clean、role或quota。该实现仅进入source诊断，target K1与正式准确率仍为blocked。

完整定向回归为85/85通过；该数字覆盖算法与最低运行时合同，不构成真实性能结论。

## 当前证据边界

- `effective8 v14`满足44,048参数、12epoch和自适应1→3→5-view的资源外形，但target matrix未执行，且尚未接入严格predictor/scorer。
- candidate capsule和双TorchScript exporter已落地，但strict package/request仍是v2单backbone结构；在package v3接通前，不能声称effective8已进入正式predictor。
- 历史ADV3B02三方法375行是旧版Stage2-B诊断链，不含target-new，配置暴露`query_per_tx`，不能作为当前正式模板。
- 历史`id_norm_late_feature`固定5-view路线性能有正收益，但参数289,685、固定5-view、K1负收益且仅2个新类，不满足当前目标。
- 当前严格基础设施仅为`LOCAL_DIAGNOSTIC_PASS`；本追踪表不会把字段自声明或本地模拟执行升级为正式协议证据。

## 实施顺序

1. 修复prompt、workflow contract和顶层state中的旧控制语义。
2. 将effective8 adapter/head/TTA绑定外部candidate/plan trust root并接入唯一strict request builder。
3. 离线构造并密封25份真实target package，删除Phase2 runtime中的`query_per_tx`与truth/role路径。
4. 本地负测、完整日志与资源审计通过后，才进入N607只读preflight和单cell严格Linux smoke。
5. K10开发锁定后执行独立确认矩阵；K1/K5/K20不得回流选参。
