# D92 D42 Tail-Pair Code Exchange设计追溯

设计源：`E:\codex\home\attachments\a32ff2e7-5e54-4d07-9697-60c470abe165\pasted-text-1.txt`

三轮回顾：`analysis/d92_three_round_retrospective_20260812.md`

冻结候选：`E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE`

|ID|源要求|冻结实现面|验证证据|状态|
|---|---|---|---|---|
|TPCE-01|相对E0八项严格Pareto|Hard10 analyzer逐outer、scene、TX同排门|八项任一均值平或反向即`REJECT_ROUTE`|DESIGN_FROZEN|
|TPCE-02|提高旧类floor并降低遗忘，不牺牲新类|六旧类固定lower-Q20 true-vs-all tail；pooled新类lower-Q20 true-vs-old tail；双向hinge守卫|实际D42解码support收据；Hard10 truth-side裁决|DESIGN_FROZEN|
|TPCE-03|不重复前三轮失败|禁止旧组统一bias、连续头重编码、codec回缩、BLOCK/LOO/Fisher/Pareto枚举|method-lock与调用负测|DESIGN_FROZEN|
|TPCE-04|直接形成可部署非零更新|从E0正式D42 state直接修改`coef2_qint8`；其他数组byte-exact|state快照、解码SHA、非E0码差|DESIGN_FROZEN|
|TPCE-05|类标签置换等变|固定tail、并列竞争者和成对整数交换均使用全类公共公式|old/new组内标签置换测试|DESIGN_FROZEN|
|TPCE-06|old/new同等优先|旧类逐类严格正tail门、新类pooled严格正tail门、两向cross-group hinge均不得增|四组support守卫及篡改负测|DESIGN_FROZEN|
|TPCE-07|单FULL fit和低注册计算|E0 FULL实际fit=1；TPCE新增fit=0；两次全头support打分加确定性稀疏原子解析贪心|fit inventory、support MAC和wall/peak receipt|DESIGN_FROZEN|
|TPCE-08|query/state不增加|最终仍为单D42 F0头，仅现有数组值改变|query MAC与persistent-state bytes对E0精确相等|DESIGN_FROZEN|
|TPCE-09|query零访问|候选只读取同一row合法support、标签和E0部署state|truth/fit/update/selection/role/quota/global均false|DESIGN_FROZEN|
|TPCE-10|K1/K2边界不变|严格D92 FULL alias，不进入码交换|state/prediction/fit inventory exact测试|DESIGN_FROZEN|
|TPCE-11|数值失败精确回退|非有限、无可移动坐标、无非空Pareto安全子集、码位饱和、零码差或最终真实score守卫失败均返回原E0 state|byte-exact fallback及reason测试|DESIGN_FROZEN|
|TPCE-12|快速困难验证|复用冻结Hard10的10个performance outer加1个K1 liveness、3scene、1arm、8shard|11 job/33 scene-arm manifest|DESIGN_FROZEN|
|TPCE-13|不重复数据验证|复用`VALIDATED_ONCE`、`p2_min_v1`及既有sealed packages|capsule/split/protocol身份|DESIGN_FROZEN|
|TPCE-14|性能最后连接truth|不可变prediction后独立scorer|prediction/COMMIT/score/receipt闭环|DESIGN_FROZEN|
|TPCE-15|资源硬门|query/state精确；wall P90≤150ms且配对中位倍率≤1.50；peak≤E0+512KiB；D92 fit proxy降幅≥80%|N607同排resource receipt|DESIGN_FROZEN|
|TPCE-16|唯一裁决|八项全正但幅度/稳定性不足仅`REVISE_ONCE`；全门通过才建议Target125|analyzer三分支测试|DESIGN_FROZEN|

## 前三轮否证吸收

- FloorBoost证明旧类整体logit上移会以seen-new和new→old为代价，TPCE不允许旧组共享shift。
- NewGuard证明“部署字节不同”不足以产生性能变化，TPCE要求实际解码tail严格正增而非仅非退化。
- Pareto Distill证明连续候选可能被D42重新编码吞没，TPCE直接编辑最终部署码且不再调用量化器。

## v2真实smoke后的冻结修订

v2的K10真实checkpoint smoke显示：全量66个交换在三个场景都提高pooled-new tail、且双向hinge不增，但持续降低至少一个旧类tail，因此同步全量发布被证伪。修订不改变原子定义、tail定义、阈值、fit、矩阵或query边界，只把发布规则冻结为确定性Pareto安全贪心子集：每一步只接受相对E0不降低六旧类tail、pooled-new cross/all且不增加双向hinge的原子；优先覆盖尚未越过tolerance的七个必达tail，其次最大化最差tail、总tail增益，最后按稳定语义handle破同分。达到七tail严格正后停止，最终使用真实D42解码score复核同一守卫；失败精确回退E0。
