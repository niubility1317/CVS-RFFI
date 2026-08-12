# D92 CSOAS需求追溯矩阵

状态：`DESIGN_FROZEN`

## 证据起点

- 目标源：`E:\codex\home\attachments\a32ff2e7-5e54-4d07-9697-60c470abe165\pasted-text-1.txt`。
- 当前基线：`E0_FULL_ONLY`完整Target125 artifact与同排分析。
- 最近反证：`d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v2`仅3/8方向改善，8/9 outer预测完全不变，registration wall P90为336.968ms，路线已拒绝。
- 历史排重：D80/D83 ground covariance或precision loading、D82 Wiener residual、D84–D86共享/反事实中心、D87–D91 sigma/head/OOF、FloorBoost/NewGuard/ParetoDistill/TPCE/TCRA均不得换名复用。

## 需求到实现与证据

|ID|冻结需求|实现位置|验证证据|状态|
|---|---|---|---|---|
|R1|新方法只改变一次FULL注册内部的充分统计量，不做拟合后head补丁|`stage2_d92_cauchy_scatter_oas.py`与D92 probe新mode|core单测、fit inventory|待实现|
|R2|分类均值逐字节沿用E0非加权均值|CSOAS统计构造|均值exact测试|待实现|
|R3|D81同一support的每类Cauchy权重仅用于full288加权scatter|CSOAS统计构造|权重一致、row permutation测试|待实现|
|R4|scatter围绕独立加权中心，分母为`1-sum(a^2)`|CSOAS统计构造|手算fixture、非退化负测|待实现|
|R5|逐类effective-DOF OAS，逐类保trace，无额外全局trace重标定|CSOAS统计构造|公式、trace、rho范围测试|待实现|
|R6|old/new组内类均衡，再固定`0.5/0.5`形成唯一共享协方差|CSOAS统计构造|label permutation与组内等变测试|待实现|
|R7|K1/K2与REG0为byte-exact E0 alias；K>2实际FULL fit恰为1|probe、slim、query receipt|生命周期与fit inventory测试|待实现|
|R8|query不参与fit、更新、选择、truth、role、quota或全局重排|query evaluation与receipt|全部禁用字段负测|待实现|
|R9|发布仍为同一D42仿射state，query MAC与持久state精确等于E0|D42发布与query audit|codec/state/MAC闭包测试|待实现|
|R10|部署head必须非E0且support跨组margin变化至少达到真实codec量子|G0 receipt与runner gate|真实checkpoint truth-free G0|待运行|
|R11|G0三场景active、无fallback、wall P90≤150ms、peak≤E0+512KiB|单一K10 G0|N607 artifact与资源收据|待运行|
|R12|Hard9+K1只跑冻结最难矩阵，K1不入性能均值|独立Hard10 wrapper|manifest、smoke、8shard closure|待实现|
|R13|Hard9八项均值全部严格优于E0，否则立即拒绝，不扫描参数|analyzer|paired rows、逐类、资源与verdict|待运行|
|R14|Hard9通过后才允许完整Target125|主代理裁决|报告状态|待裁决|

## 八项裁决方向

`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy必须严格上升；average forgetting、new→old、old→new必须严格下降。大胆目标分别为`+1.0pp、+1.5pp、+1.0pp、+4.0pp、+0.5pp、-1.5pp、-0.5pp、-0.5pp`，但任何一项持平或反向已足以拒绝路线。

## 允许的唯一回退

非有限、权重/有效自由度退化、OAS分母退化后仍非SPD、solve或codec失败时返回byte-exact E0，并标记fallback。fallback可保安全，不能计作机制或性能成功。
