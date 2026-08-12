# D92 FULL/BLOCK Pareto蒸馏设计追溯

设计源：`E:\codex\home\attachments\a32ff2e7-5e54-4d07-9697-60c470abe165\pasted-text-1.txt`

冻结候选：`E0_FULL_BLOCK_PARETO_DISTILL`

|ID|源要求|冻结实现面|验证证据|状态|
|---|---|---|---|---|
|PD-01|相对E0八项严格Pareto|Hard10 analyzer逐outer、scene、TX同排门|10个performance outer；任一均值平/反向即`REJECT_ROUTE`|DESIGN_FROZEN|
|PD-02|旧类floor和遗忘显著改善，同时seen-new与双向混淆改善|support词典序旧CVaR20、新q20、双向hinge目标|部署support收据；Hard10 truth-side最终证伪|DESIGN_FROZEN|
|PD-03|不复用FloorBoost/NewGuard失败机制|禁止旧类统一bias、保护回缩、多scale和容差放宽|静态method lock与负测|DESIGN_FROZEN|
|PD-04|只保留D92的低成本互补几何|共享一次旧/新协方差统计；FULL与BLOCK各解一次|`covariance_estimation_count=1`、`full_solve=1`、`block_solve=1`|DESIGN_FROZEN|
|PD-05|禁止LOO/Fisher/Pareto枚举|不构造PRESS、不做K折重拟合、不调用D61/D62残差门|fit inventory与调用负测|DESIGN_FROZEN|
|PD-06|类置换等变|类公共仿射规范、固定bottom-20%并列全纳入、组内同公式|old/new组内标签置换测试|DESIGN_FROZEN|
|PD-07|query零访问|仅同一row注册support参与求解|query truth/fit/update/selection/role/quota/global全false|DESIGN_FROZEN|
|PD-08|真实D42部署闭环|对`Q_D42(theta)`重算全部support约束；最多一次固定code-local修正|量化后约束与byte/hash收据|DESIGN_FROZEN|
|PD-09|无效候选不发布|部署头E0-byte-exact、无真实量化步变化或约束失败时`LOCAL_INVALID`/exact E0 fallback|真实checkpoint no-query smoke|DESIGN_FROZEN|
|PD-10|K1/K2边界不变|严格D92 FULL alias|K1/K2 state/prediction与fit inventory测试|DESIGN_FROZEN|
|PD-11|query/state与E0一致|最终仅持久化一个F0仿射头|query MAC/state byte exact|DESIGN_FROZEN|
|PD-12|注册计算显著低于D92|two-state fit=4、DA1_REG1实际solve=2；额外BLOCK复用共享统计|本地资源receipt；N607 paired wall/peak|DESIGN_FROZEN|
|PD-13|只跑最难矩阵|复用冻结Hard10+1个K1、3scene、1arm、8shard|11 job/33 scene-arm manifest|DESIGN_FROZEN|
|PD-14|不重复数据验证|复用`VALIDATED_ONCE`、`p2_min_v1`及原sealed packages|capsule/split/protocol身份|DESIGN_FROZEN|
|PD-15|性能与truth最后连接|不可变prediction后独立scorer|prediction/COMMIT/score/receipt哈希闭环|DESIGN_FROZEN|
|PD-16|三分支裁决|全方向正但幅度/稳定/目标资源不足仅`REVISE_ONCE`；全门通过才建议125|analyzer分支测试|DESIGN_FROZEN|

## 证据驱动的路线边界

- Hard10原D92相对E0：H`+0.8201pp`、old BA/c_old`+1.2778pp`、floor`+3.6667pp`、seen-new`+0.3417pp`、forgetting`-1.2778pp`、old→new`-0.6389pp`，但new→old`+0.0583pp`。新候选不能简单复刻D92。
- Hard12中D46 LOO、D62 Fisher/Pareto和固定50均未形成全指标正向证据；本轮只保留BLOCK协方差的互补方向。
- FloorBoost牺牲seen-new与new→old换取旧类；NewGuard改变部署头却没有改变任何Hard10指标。两条路线均已关闭。

