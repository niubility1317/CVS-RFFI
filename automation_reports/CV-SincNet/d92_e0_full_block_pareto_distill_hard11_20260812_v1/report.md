# D92 E0 FULL/BLOCK Pareto蒸馏Hard11实验报告

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_block_pareto_distill_hard11_20260812_v1`|
|状态|`DESIGN_FROZEN / IMPLEMENTING`|
|时间|2026-08-12 11:58 HKT|
|操作者|Codex `/root`；科学实现与机械执行闭环分属独立owner；N607由唯一runner负责|
|候选|`E0_FULL_BLOCK_PARETO_DISTILL`；candidate=`d92_e0_full_block_pareto_distill`|
|协议|`p2_min_v1`、复用`VALIDATED_ONCE`数据，不重复验证|
|claim scope|`DEVELOPMENT_ONLY_HARD_SCREEN`|
|正式基线|`E0_FULL_ONLY`完整Target125历史artifact；本轮不重跑E0或D92|

本轮目标是在冻结Hard10同排上同时提高H、old BA、`c_old_acc`、old floor和seen-new，同时降低forgetting、new→old与old→new；query MAC和永久state与E0精确相同，注册计算远低于原D92。

## 2.既有证据与假设

Hard10原始D92 score已从原retry2任务只读取回10/10份，未修改远端。与E0同排复算如下：

|候选|H|old BA / c_old|old floor|seen-new|forgetting|new→old|old→new|
|---|---:|---:|---:|---:|---:|---:|---:|
|E0_FULL_ONLY|73.3472%|74.8611%|44.8333%|72.0333%|12.9167%|15.0417%|15.3333%|
|原D92|74.1673%|76.1389%|48.5000%|72.3750%|11.6389%|15.1000%|14.6944%|
|D92−E0|+0.8201pp|+1.2778pp|+3.6667pp|+0.3417pp|−1.2778pp|**+0.0583pp**|−0.6389pp|

原D92不是八项Pareto解：new→old略差，seen-new只在5/10行不降。因此本轮不压缩复刻D92，而只提取BLOCK协方差的互补方向，并在support上联合约束新旧两组。

受控Hard12证据同时排除了三条路线：D46 LOO提高floor但损害H/new/forgetting；D62 Fisher/Pareto净收益近零且计算翻倍；固定50提高floor但其余核心指标下降。FloorBoost和NewGuard也已分别因新类崩损、八项全零而拒绝。

## 3.冻结方法

设计与追溯：

- `docs/superpowers/specs/2026-08-12-d92-full-block-pareto-distill-design.md`
- `analysis/d92_full_block_pareto_distill_traceability_20260812.md`
- Git设计提交：`b925e820`

K>2时，同一份D81变换support只估计一次`Sigma=0.5 Sigma_old+0.5 Sigma_new`。FULL解使用`Sigma`，BLOCK解只取同一`Sigma`的160/96/32块对角，不重估中心或协方差。两头规范化后形成互补方向，单次词典序support求解依次优化旧类固定CVaR20、新类固定q20、双向hinge与最小头扰动；最终只发布一个D42 F0头。

禁止PRESS、K折LOO、Fisher、Pareto枚举、旧类统一bias、NewGuard多尺度回缩和query选择。部署头E0-byte-exact、未跨真实量化步或support约束失败时，本地无效或exact E0 fallback，不进入性能发布。

## 4.冻结矩阵

10个performance outer与NewGuard/FloorBoost Hard10完全相同；另含一个K1 liveness：

|序号|outer|角色|
|---:|---|---|
|1|`rx_7_7__seed_713106__k_10__new_5`|performance / K>2 smoke|
|2|`rx_7_7__seed_713104__k_5__new_20`|performance|
|3|`rx_7_7__seed_713103__k_10__new_5`|performance|
|4|`rx_8_8__seed_713103__k_5__new_20`|performance|
|5|`rx_8_8__seed_713103__k_10__new_5`|performance|
|6|`rx_8_8__seed_713106__k_5__new_20`|performance|
|7|`rx_7_14__seed_713104__k_10__new_10`|performance|
|8|`rx_3_19__seed_713102__k_10__new_5`|performance|
|9|`rx_7_7__seed_713105__k_10__new_20`|performance|
|10|`rx_7_7__seed_713104__k_10__new_5`|performance|
|11|`rx_20_1__seed_713106__k_1__new_20`|liveness|

每个outer固定`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，共11 job、33 scene-arm、1 arm、8 shard。K1不进入性能均值。

历史E0 paired baseline：
`E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis\paired_rows.csv`，SHA256=`6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a`。

## 5.性能与资源门

八项均值必须同时严格优于E0。目标幅度为H≥+1.00pp、old BA≥+1.50pp、`c_old_acc`≥+1.00pp、floor≥+4.00pp、seen-new≥+0.50pp、forgetting≤−1.50pp、两向混淆各≤−0.50pp。任一方向平或反向直接`REJECT_ROUTE`。

资源门：

- K>2 two-state fit=4，`DA1_REG1`实际solve=2；FULL/BLOCK共享一套协方差统计；
- query MAC与永久state精确等于E0；
- wall目标≤120ms且≤1.25×E0，硬门≤150ms且≤1.50×；
- peak≤E0+512KiB；
- 注册计算相对原D92至少下降80%。

全方向正确但幅度、稳定性或目标资源不足且硬门未破，只允许`REVISE_ONCE`；全部通过才`ADVANCE_TO_TARGET125_CANDIDATE`，本轮绝不自动跑125。

## 6.本地实现与验证

当前设计文件已进入Git提交`b925e820`。科学核心与Hard11机械闭环正在TDD实现。发布前只要求：

1. 聚焦协议负测和K1/K2 alias；
2. 真实checkpoint K>2 no-query smoke，并证明部署头非E0-byte-exact；
3. 共享统计资源receipt；
4. 独立P0=0、P1=0；
5. clean Git commit和不可覆盖run路径。

P2不阻塞实验。

## 7.N607预注册

|字段|冻结值|
|---|---|
|普通账号|SSH alias `N607`；禁止管理员账号|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_pareto_distill_source_snapshot_20260812_v1`|
|output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_block_pareto_distill_hard11_20260812_v1`|
|logs root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_block_pareto_distill_hard11_20260812_v1`|
|GPU|GPU0–7各一个shard，`CUDA_VISIBLE_DEVICES=i`内使用`cuda:0`|
|expected|11 job receipt、22正式prediction/COMMIT/fit/resource、11 score、8 summary|

最终exact command、archive/config/launch SHA、commit、PID与CWD在本地门通过后回填。启动前四个run路径必须不存在；唯一runner只启动一次。

health stop仅限协议/安全错误、launcher确定性故障、prediction闭包失败或两个不同outer出现同一pre-prediction确定性异常。不得读取性能决定停止。技术失败保留artifact并标记`NO_PERFORMANCE_RESULT`，不得覆盖或在同一run ID重启。

## 8.风险与完成后检查

最大工程风险是共享统计后的BLOCK solve仍超过150ms/1.50×E0；最大科学风险是support margin改善不能外推为query Pareto。前者先由真实checkpoint资源smoke筛除，后者只能由完整10/10 Hard10一次性证伪。

完成后必须回填：Git差异与测试、真实smoke、同步映射与哈希、exact command/PID/GPU、11/11闭合、完整取回、逐outer八指标、receiver/K/scene/六旧类分解、wall/peak/fallback以及唯一裁决。

