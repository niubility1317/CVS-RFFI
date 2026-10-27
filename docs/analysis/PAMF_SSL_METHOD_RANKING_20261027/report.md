# PAMF 半监督探索与消融：完整测试结果排序

汇总时间：2026-10-27 02:08:33 UTC。本表只读取已冻结公开汇总与逐种子结果，没有新训练、推理或真值评分。

本轮去重后共 **99 组三种子 F0 配置结果**，另有 1 组单种子正式结果。它们包含方法、结构、监督损失及核心机制消融，不代表 99 种独立算法或 297 次全新训练。重复对照和精确复用只计一次。

**按 raw clean 均值降序排列；六景和 DURA 是独立指标。未发现同一配置同时达到 clean≥85.981746%、六景>72.096197% 的原联合目标。**

- 主口径为 F0：cross_rx 126000＋cross_both 42000＝168000 条／视图；cross_date 30000 单列，不混入。各方法 2026092701/02/03 三个 F seed。
- clean 与六景准确率单位为 %；差值单位为百分点（pp）。六景均值先在 seed 内平均，再对三个 seed 求均值。原始未恢复 IQ 称 raw。
- 当前单视图表以 SIG1-PL 为描述参照；历史与结构表以 F21 为描述参照。结构和配方不同的行不能据此作严格因果优劣结论。
- 历史 G-LU 列条件于同一数据契约下固定恢复器，表示三个 F seed，而非三次独立 G＋F 系统复现。G-L 单列保存在 CSV，绝不混入 G-LU。
- 本次是已经曝光的目标基准回顾；不能称为新的独立盲测、显著性结论或据此调参的依据。

## 直接结论

1. 全部结构消融中 clean 最高：REPL_SHUFFLE_TIME 86.710714%，六景 70.763360%，同 G-LU 六景 78.966964%。这是历史时间分支打乱消融，不能替代当前指定的正常单视图 SIG_TIME 基座。
2. 历史原结构相关损失消融中，L_without_raw_CE 的 clean 为 85.935119%，六景 70.516204%；原 REPL 为 85.726786%／70.085516%，普通 PL 为 84.981746%／70.359358%。clean 提升没有带来六景同步提升。
3. raw 六景最高：AllURDrop 72.712401%，clean 84.929167%。相对普通 PL 为 clean −0.052579 pp、六景＋2.353042 pp；其同 G-LU 六景反降至 77.317923%。
4. 单视图中原 REPL 历史重训 clean 84.813294%，但六景仅 69.406151%；新普通 PL 为 84.693849%／70.233234%。新完成的 11 个候选均未超过普通 PL 的 clean，双 H＋跨 RX 配对六景最高 71.647817%，clean 降至 84.180754%。
5. 原 REPL 的完整 11 设置：跨 RX raw 六景 11/11 低于普通 PL；同 G-LU 六景 3/11 提升、8/11 下降。单个 F0 clean 排名不能代表全面泛化优越。

## 当前指定的单视图 SIG_TIME

|序号|方法／配置|clean（%）|六景均值（%）|clean 较普通 PL（pp）|六景较普通 PL（pp）|
|---:|---|---:|---:|---:|---:|
|1|原 REPL 改为单视图 SIG_TIME（历史重训） `REPL_SIG_TIME`|84.8133|69.4062|+0.1194|-0.8271|
|2|单视图 SIG_TIME＋普通 PL `SIG1-PL`|84.6938|70.2332|+0.0000|+0.0000|
|3|PM＋高置信 H 硬伪标签（A） `RXTrust-A`|84.6107|70.5004|-0.0831|+0.2671|
|4|选择性协方差增强，仅 U `CovSel-U`|84.5784|70.2862|-0.1155|+0.0530|
|5|随机协方差增强，仅 U `CovRandom-U`|84.5706|70.1429|-0.1232|-0.0903|
|6|选择 KL 最大的接收机响应风格 `RXMean-MaxU`|84.5446|70.2621|-0.1492|+0.0288|
|7|P＋raw／H 分别校准（PM） `RXTrust-PM`|84.5429|70.5692|-0.1510|+0.3359|
|8|完整协方差增强，仅 U `CovFull-U`|84.5329|70.3334|-0.1609|+0.1001|
|9|接收机均值响应增强，仅 U `RXMean-U`|84.5323|70.3646|-0.1615|+0.1314|
|10|选择性协方差增强，L＋U `CovSel-LU`|84.5272|70.2478|-0.1667|+0.0146|
|11|接收机均值响应增强，L＋U `RXMean-LU`|84.5256|70.3581|-0.1683|+0.1249|
|12|接收机风格＋候选集合监督（P） `RXTrust-P`|84.3679|69.7168|-0.3260|-0.5164|
|13|双 practical 视图＋跨 RX 配对，L＋U `REPL_H2_LU_Pair`|84.1808|71.6478|-0.5131|+1.4146|
|14|单视图 SIG_TIME 监督版 `SIG1-Sup`|84.1169|70.1681|-0.5770|-0.0652|

[单视图两项性能与三种子 SD 对比图](E:/type10-7/exports/unified_figures/20261027-PAMF-SSL-method-ranking-r01/current_sig1.png)。

原 REPL 单视图行来自历史 m201 重训，其余为已完成匹配 SIG1 对照／确认。当前新方法均直接读取 raw；后续路线没有新 DURA 读出，缺失值不等于 0。

三种子 SD 详见 CSV。例：SIG1-PL clean 84.693849±0.692508%、六景 70.233234±1.043286%；双 H＋配对 clean 84.180754±1.087816%、六景 71.647817±0.474676%。SD 不是置信区间或显著性检验。

## 历史原 SIG_TIME_X3：方法与损失消融

|序号|方法／配置|clean（%）|六景均值（%）|clean 较普通 PL（pp）|六景较普通 PL（pp）|同 G-LU 六景（%）|
|---:|---|---:|---:|---:|---:|---:|
|1|REPL：L 去 raw CE `L_without_raw_CE`|85.9351|70.5162|+0.9534|+0.1568|78.1496|
|2|REPL 去 MixStyle `without_style`|85.8476|70.4419|+0.8659|+0.0825|78.0319|
|3|REPL：L 单次前向 `L_single_forward`|85.8312|70.6403|+0.8494|+0.2809|78.0478|
|4|原 PAMF-REPL（接收机外推） `HPL_E20_JOINT_RXE6G`|85.7268|70.0855|+0.7450|-0.2738|77.8977|
|5|REPL：L 开启 As KL `L_As_KL_on`|85.7083|70.3491|+0.7266|-0.0102|78.2755|
|6|REPL 三核心 ABC011（C 为 REPL） `ABC011`|85.6762|69.8976|+0.6944|-0.4617|78.1481|
|7|REPL：L 仅联合层对比 `L_joint_only`|85.6450|69.5589|+0.6633|-0.8004|78.0056|
|8|强 U＋跨 RX 联合特征对比（去外推） `HPL_E20_NAMS_JOINT6G`|85.6417|70.1712|+0.6599|-0.1881|78.0808|
|9|REPL：L 仅中层对比 `L_middle_only`|85.6367|70.0348|+0.6550|-0.3245|78.0522|
|10|REPL 三核心 ABC001（C 为 REPL） `ABC001`|85.6341|69.7911|+0.6524|-0.5683|78.0204|
|11|REPL 去梯度保护 `without_guard`|85.6155|69.9377|+0.6337|-0.4217|77.7915|
|12|REPL：L 仅浅层对比 `L_shallow_only`|85.5998|70.2860|+0.6181|-0.0733|78.1432|
|13|REPL 去联合特征对比 `without_joint`|85.5246|70.1768|+0.5429|-0.1826|77.7303|
|14|强 U＋排除 As 的 MixStyle＋梯度保护 `HPL_E20_NAMS6G`|85.5008|70.2834|+0.5190|-0.0759|77.9146|
|15|REPL 三核心 ABC101（C 为 REPL） `ABC101`|85.4893|69.9363|+0.5075|-0.4231|77.8029|
|16|可靠物理 As 监督 `HPL_E20_JOINT_RAS6G`|85.4488|70.7015|+0.4671|+0.3421|77.8642|
|17|PL 去除 U 的物理 As 视图 `HPL_NO_UAS`|85.4230|70.6572|+0.4413|+0.2979|77.9856|
|18|PL 权重 2＋排除 As 的 MixStyle `HPL_NAMS2`|85.3111|70.5730|+0.3294|+0.2136|77.8250|
|19|REPL：L 去物理 As `L_without_As`|85.3069|41.9298|+0.3252|-28.4295|77.8074|
|20|E20 续训：PL2＋排除 As 的 MixStyle `HPL_E20_NAMS2`|85.2462|70.7302|+0.2645|+0.3709|77.7059|
|21|强 U＋标签平滑＋梯度保护 `HPL_E20_LS6G`|85.1016|69.9360|+0.1198|-0.4233|78.2380|
|22|普通 PL 权重 2 `HPL2`|85.0756|70.1875|+0.0938|-0.1718|78.0787|
|23|PL 权重 2＋标签平滑 `HPL_LS2`|85.0700|70.3003|+0.0883|-0.0590|78.1316|
|24|E20 续训：PL2＋标签平滑 `HPL_E20_LS2`|85.0528|70.2430|+0.0710|-0.1164|78.0898|
|25|PL 权重 2＋MixStyle `HPL_MS2`|84.9964|70.3283|+0.0147|-0.0310|78.0759|
|26|PAMF＋普通 PL `F21`|84.9817|70.3594|+0.0000|+0.0000|78.0746|
|27|恢复软标签实验的普通 PL 对照 `PLControl`|84.9671|70.3349|-0.0147|-0.0245|78.0616|
|28|互补负类监督 `HPL_E20_JOINT_COMP6G`|84.9589|70.2528|-0.0228|-0.1066|76.4464|
|29|AllURDrop：全部 U 加入 RDrop `HPL_E20_AS_ALLRD`|84.9292|72.7124|-0.0526|+2.3530|77.3179|
|30|信道可靠性 PL／CE 重分配 `ChannelPL`|84.8790|70.5894|-0.1028|+0.2301|77.9922|
|31|伪类／接收机 CE 贡献均衡 `HPL_E20_AS_BCE6G`|84.8786|72.3230|-0.1032|+1.9637|77.4406|
|32|原始 IQ 软伪标签 `RawSoft`|84.8728|70.2627|-0.1089|-0.0966|78.0118|
|33|PL＋完整 U 监督机制 `HPLT1U`|84.8671|71.9747|-0.1147|+1.6153|77.5089|
|34|U 原生 As 监督 `HPL_E20_JOINT_AS6G`|84.8319|72.2644|-0.1498|+1.9050|77.4284|
|35|路径互补＋可靠 As `HPL_E20_PATH_RAS6G`|84.8210|70.7584|-0.1607|+0.3990|76.3863|
|36|路径互补监督 `HPL_E20_JOINT_PATH6G`|84.8187|70.3020|-0.1631|-0.0574|76.2970|
|37|同伴教师＋完整 U 机制（权重 2） `PeerT1U2`|84.8050|72.2796|-0.1768|+1.9203|77.4413|
|38|同伴一致教师＋普通 PL2 `PeerPL2`|84.8048|70.3234|-0.1770|-0.0360|77.9449|
|39|同伴教师＋完整 U 机制（权重 1） `PeerT1U1`|84.7893|71.8086|-0.1925|+1.4493|77.6909|
|40|PAMF 监督版 `F20`|84.7853|70.5962|-0.1964|+0.2368|77.8549|
|41|REPL 三核心 ABC010（C 为 REPL） `ABC010`|84.6391|70.9695|-0.3427|+0.6102|78.0575|
|42|U 视图梯度均衡 `UVB`|84.6117|71.6754|-0.3700|+1.3161|77.5453|
|43|同包 U 物理配对监督 `PAIREDU`|84.5438|71.9783|-0.4379|+1.6189|77.5076|
|44|原生 As＋分支梯度保护 `HPL_E20_AS_BG2`|84.5270|72.4817|-0.4548|+2.1224|77.0329|
|45|原生 As＋优化步保护 `HPL_E20_AS_SG2`|84.4857|72.4150|-0.4960|+2.0557|77.0139|
|46|原生 As＋梯度范数上限 2 `HPL_E20_AS_CAP2`|84.4611|72.4163|-0.5206|+2.0569|77.0643|
|47|物理视图特征锚定 `VIEWANCHOR`|84.3796|71.9470|-0.6022|+1.5876|77.5218|
|48|U MixStyle `UMIX`|84.3629|72.0009|-0.6188|+1.6415|77.5541|
|49|优化步保护＋取消旧范数上限 `HPL_E20_AS_SGNC`|84.3478|72.4886|-0.6339|+2.1293|76.8797|
|50|U 类条件矩匹配 `UCMR`|84.3365|71.9441|-0.6452|+1.5848|77.1752|
|51|U 完整梯度保护 `UGP`|84.2954|71.8183|-0.6863|+1.4589|77.4200|
|52|U 浅中层跨 RX 对比 `UCRIC`|84.2907|71.8662|-0.6911|+1.5068|77.1860|
|53|SoftMatch＋接收机原型 `RXPSM`|84.2724|71.8536|-0.7093|+1.4942|77.4202|
|54|U 双锚点 `UDAP`|84.2548|71.8053|-0.7270|+1.4460|77.4297|
|55|物理 SoftMatch `PSM`|84.2226|71.8973|-0.7591|+1.5379|77.3528|
|56|DURA 恢复 IQ 软伪标签 `DURASoft`|83.6845|70.0562|-1.2972|-0.3032|76.4049|
|57|REPL 三核心 ABC100（C 为 REPL） `ABC100`|83.2593|70.4584|-1.7224|+0.0990|76.8148|
|58|REPL 三核心 ABC000（C 为 REPL） `ABC000`|82.6024|70.4453|-2.3794|+0.0860|76.3757|

m201 核心开关：A 为 L 跨 RX 对比，B 为 L 成对一致性，C 为 REPL；这与最初 own-EMA／选择门控的 C 不同。

## 最初完整半监督与三核心消融

|序号|方法／配置|clean（%）|六景均值（%）|clean 较普通 PL（pp）|六景较普通 PL（pp）|同 G-LU 六景（%）|
|---:|---|---:|---:|---:|---:|---:|
|1|原三核心 ABC011（C 为旧半监督） `original_ABC011`|84.7276|70.9898|-0.2542|+0.6305|78.0573|
|2|原三核心 ABC010（C 为旧半监督） `original_ABC010`|84.6185|70.9560|-0.3633|+0.5967|78.0422|
|3|原完整 PAMF 半监督／T1 `original_ABC111`|84.3944|70.6013|-0.5873|+0.2420|77.6468|
|4|原三核心 ABC100（C 为旧半监督） `original_ABC100`|83.3867|70.4502|-1.5950|+0.0908|76.9214|
|5|原三核心 ABC101（C 为旧半监督） `original_ABC101`|83.1411|70.4980|-1.8407|+0.1387|76.6349|
|6|最初匹配多视图监督骨架 `original_ABC000`|82.7133|70.4518|-2.2685|+0.0924|76.4103|
|7|原三核心 ABC001（C 为旧半监督） `original_ABC001`|82.5232|70.4849|-2.4585|+0.1255|76.1642|

本组 A＝跨 RX 同 TX 对比；B＝ordinary/raw 成对对称 KL；C＝原 own EMA／选择门控／伪标签／熵整体。110 即 F20 监督版，已精确去重；其 clean 84.785317%、六景 70.596197%。000 是匹配多视图 CE 骨架，不是另一个严格单视图 CE 实验。

## 结构与严格 CE 消融

|序号|方法／配置|clean（%）|六景均值（%）|clean 较普通 PL（pp）|六景较普通 PL（pp）|同 G-LU 六景（%）|
|---:|---|---:|---:|---:|---:|---:|
|1|REPL：打乱时间分支 `REPL_SHUFFLE_TIME`|86.7107|70.7634|+1.7290|+0.4040|78.9670|
|2|REPL：协方差三视图替代 `REPL_COV_X3`|85.7873|70.4464|+0.8056|+0.0870|77.9882|
|3|REPL：错误参考响应 `REPL_WRONG_REFERENCE`|85.2460|70.1170|+0.2643|-0.2423|77.7822|
|4|REPL：去时间分支 `REPL_NO_TIME`|85.1877|70.1394|+0.2060|-0.2199|77.8801|
|5|REPL：去参考响应 `REPL_NO_REFERENCE`|84.4058|69.7844|-0.5760|-0.5750|77.6614|
|6|严格 CE：打乱时间分支 `CE_SHUFFLE_TIME`|83.5798|71.3980|-1.4020|+1.0386|77.2873|
|7|严格 CE：单视图 SIG_TIME `CE_SIG_TIME`|83.1405|70.0831|-1.8413|-0.2762|75.9331|
|8|严格 CE：去签名分支 `CE_NO_SIGNATURE`|82.7238|70.4615|-2.2579|+0.1022|75.5583|
|9|严格 CE：完整结构 `CE_full`|82.6653|70.9105|-2.3165|+0.5512|76.2727|
|10|REPL：去功放分支 `REPL_NO_PA`|82.6367|62.2767|-2.3450|-8.0827|75.6958|
|11|严格 CE：协方差三视图 `CE_COV_X3`|82.4940|71.2655|-2.4877|+0.9062|75.9423|
|12|严格 CE：去时间分支 `CE_NO_TIME`|82.4877|70.3568|-2.4940|-0.0025|75.6735|
|13|去 SIG-TIME 后：打乱参考模板相位 `NoSIG_WRONG_REFERENCE`|82.0712|69.9952|-2.9105|-0.3642|—|
|14|严格 CE：错误参考响应 `CE_WRONG_REFERENCE`|81.9222|70.6671|-3.0595|+0.3077|75.6795|
|15|严格 CE：去参考响应 `CE_NO_REFERENCE`|81.6337|70.3974|-3.3480|+0.0380|75.5986|
|16|REPL：去签名分支 `REPL_NO_SIGNATURE`|81.2940|67.1671|-3.6877|-3.1923|72.1917|
|17|去 SIG-TIME 后：再去参考响应支路 `NoSIG_NO_REFERENCE`|81.0714|70.1441|-3.9103|-0.2152|—|
|18|去 SIG-TIME 后：再去功放支路 `NoSIG_NO_PA`|78.7510|62.4836|-6.2308|-7.8758|—|
|19|严格 CE：去功放分支 `CE_NO_PA`|78.5502|64.0948|-6.4315|-6.2646|72.9791|
|20|去 SIG-TIME 后：同时去功放和参考响应支路 `NoSIG_NO_PA_REFERENCE`|76.6730|61.3005|-8.3087|-9.0589|—|

以上结构变化不属于当前固定基座下的新半监督方法。NoSIG 追加消融的恢复器仅 G-L，故 G-LU 留空，G-L 数值见 CSV。严格 CE 不包含完整监督机制，不能用它冒充 PAMF 监督版。

## 六景性能降序：前 10

|序号|方法／配置|clean（%）|六景均值（%）|clean 较普通 PL（pp）|六景较普通 PL（pp）|同 G-LU 六景（%）|
|---:|---|---:|---:|---:|---:|---:|
|1|AllURDrop：全部 U 加入 RDrop `HPL_E20_AS_ALLRD`|84.9292|72.7124|-0.0526|+2.3530|77.3179|
|2|优化步保护＋取消旧范数上限 `HPL_E20_AS_SGNC`|84.3478|72.4886|-0.6339|+2.1293|76.8797|
|3|原生 As＋分支梯度保护 `HPL_E20_AS_BG2`|84.5270|72.4817|-0.4548|+2.1224|77.0329|
|4|原生 As＋梯度范数上限 2 `HPL_E20_AS_CAP2`|84.4611|72.4163|-0.5206|+2.0569|77.0643|
|5|原生 As＋优化步保护 `HPL_E20_AS_SG2`|84.4857|72.4150|-0.4960|+2.0557|77.0139|
|6|伪类／接收机 CE 贡献均衡 `HPL_E20_AS_BCE6G`|84.8786|72.3230|-0.1032|+1.9637|77.4406|
|7|同伴教师＋完整 U 机制（权重 2） `PeerT1U2`|84.8050|72.2796|-0.1768|+1.9203|77.4413|
|8|U 原生 As 监督 `HPL_E20_JOINT_AS6G`|84.8319|72.2644|-0.1498|+1.9050|77.4284|
|9|U MixStyle `UMIX`|84.3629|72.0009|-0.6188|+1.6415|77.5541|
|10|同包 U 物理配对监督 `PAIREDU`|84.5438|71.9783|-0.4379|+1.6189|77.5076|

## 历史同 G-LU 六景降序：前 10

|序号|方法／配置|clean（%）|六景均值（%）|clean 较普通 PL（pp）|六景较普通 PL（pp）|同 G-LU 六景（%）|
|---:|---|---:|---:|---:|---:|---:|
|1|REPL：打乱时间分支 `REPL_SHUFFLE_TIME`|86.7107|70.7634|+1.7290|+0.4040|78.9670|
|2|REPL：L 开启 As KL `L_As_KL_on`|85.7083|70.3491|+0.7266|-0.0102|78.2755|
|3|强 U＋标签平滑＋梯度保护 `HPL_E20_LS6G`|85.1016|69.9360|+0.1198|-0.4233|78.2380|
|4|REPL：L 去 raw CE `L_without_raw_CE`|85.9351|70.5162|+0.9534|+0.1568|78.1496|
|5|REPL 三核心 ABC011（C 为 REPL） `ABC011`|85.6762|69.8976|+0.6944|-0.4617|78.1481|
|6|REPL：L 仅浅层对比 `L_shallow_only`|85.5998|70.2860|+0.6181|-0.0733|78.1432|
|7|PL 权重 2＋标签平滑 `HPL_LS2`|85.0700|70.3003|+0.0883|-0.0590|78.1316|
|8|E20 续训：PL2＋标签平滑 `HPL_E20_LS2`|85.0528|70.2430|+0.0710|-0.1164|78.0898|
|9|强 U＋跨 RX 联合特征对比（去外推） `HPL_E20_NAMS_JOINT6G`|85.6417|70.1712|+0.6599|-0.1881|78.0808|
|10|普通 PL 权重 2 `HPL2`|85.0756|70.1875|+0.0938|-0.1718|78.0787|

## 单种子正式结果

PL-History 仅 seed2026092701：主 168000 raw clean **85.101190%**，六景 **70.737599%**，G-LU 六景 **78.499603%**。不与三种子均值混排；其原比较引用历史 X3 普通 PL，不能当作已完成当前单视图匹配三种子确认。

## 全部 11 设置的 REPL 泛化复核

以下均为跨 RX 子集，不是前述主 168000 口径；差值为 REPL−普通 PL，单位为 pp。

|设置|raw clean 差|raw 六景差|G-LU clean 差|G-LU 六景差|
|---|---:|---:|---:|---:|
|F0|+0.5257|-0.2766|+0.5415|-0.2298|
|F1_rx_repartition_r01|+0.4627|-0.8570|+0.5069|-0.2757|
|F2_rx_repartition_r01|+0.6444|-0.0492|+0.6307|-0.0046|
|ManyRx|-0.7711|-0.7836|-0.6894|-0.1895|
|label01|-1.2902|-1.6338|-1.3048|-1.1328|
|label05|+0.0127|-0.5863|+0.0598|+0.2448|
|m1-natural|-0.7074|-0.4295|-0.9062|-0.4193|
|m2-natural|+3.6630|-0.6806|+3.6787|+0.3409|
|m3-natural|+0.2944|-1.4217|+0.3157|+0.7318|
|m4-natural|+0.8182|-0.2462|+0.8815|-0.0175|
|m6-natural|-1.2907|-1.2246|-1.2917|-0.9738|

此前 86.0384%→86.5640% 是 F0 cross_rx 126000 的 PL→REPL clean；本榜 84.9817%→85.7268% 合并了额外 cross_both 42000，两个口径不能混用。

## 仅完成源域探索：不进入目标性能排名

### 早期 clean／恢复视图的 30 个源域候选

全部只完成单种子源域筛选，均未达到各自预登记源门槛，没有正式目标测试。下面显示固定留出接收机、三种恢复场景均值增益范围；它不是目标六景收益。

|路径|具体方法|源三恢复场景增益范围（pp）|状态|
|---|---|---:|---|
|AsymmetricHG-sourceHold-m4-r01|HJoint, HAnchor, GJoint, BothJoint|+0.1358 至 +0.1790|全部未通过，未测 target|
|FeatureTask-sourceHold-m4-r01|GJointHard, GFeatureHard, HFeatureHard, HGFeatureHard|+0.0864 至 +0.1420|全部未通过，未测 target|
|ResidualMix-sourceHold-m4-r01|MixBase, MixDose, PairBase, PairDose|+0.0802 至 +0.1296|全部未通过，未测 target|
|RestorationBridge-sourceHold-m4-r01|SeparateRG, BridgeRG, BridgeHG, BridgeGG|+0.1173 至 +0.2037|全部未通过，未测 target|
|RXReference-sourceHold-m3-r01|RawReference, RestoredReference, PairedReference|+0.0679 至 +0.1173|全部未通过，未测 target|
|TaskRefDose-sourceHold-m4-r01|GTask2x, GTask3x, GUniform3x, RawTask3x|+0.0494 至 +0.2099|全部未通过，未测 target|
|TaskScaledRef-sourceHold-m3-r01|GUniform, GTaskScaled, RawTaskScaled|+0.0926 至 +0.2222|全部未通过，未测 target|
|URealizations-sourceHold-m4-r01|GTaskMulti, GGCLMulti, HGCLMulti, GJointMulti|+0.1481 至 +0.2099|全部未通过，未测 target|

### 后续源域方法及诊断

|路径|已完成结果／状态|正式目标测试|
|---|---|---|
|NoSIG 图传播 RGPL|最终认证 V 为 17 条、14 条正确，82.3529%；覆盖和精度均未过门槛|无，未训练学生|
|RawLabeler／GLabeler 专家|源 V 认证精度分别 93.8299%／95.1220%；交集 33/33，但覆盖不足|无|
|NoisyCE／NoisyScaledCE／NoisyGCE／NoisyGCE2Raw／NoisyGCEG／NoisyGCEHG|6 个单种子训练完成；源 clean 比 NoSIG-PL 高 0.2963 至 0.3370 pp，但留出 RX 的 G 三景降 0.2099 至 0.2901 pp；全部失败|无|
|NoSIG-Sup／NoSIG-PL 源对照|源 clean 97.7593%／97.5630%；不能与目标准确率混排|无|
|PL-Negative／PL-Temporal／PL-History|源 clean 98.3704%／98.3111%／98.3222%；只有 History 完成 seed01 目标|History 单 seed，见上|
|PL-VAT|源 clean 98.3852%；源 raw 三景 −0.2173 pp，G 三景 −0.0654 pp，未过门槛|无|
|单视图 REPL_NoStyle／NoStyleNoRXE／RXOffset|源 clean 98.4481%／98.4074%／98.3963%；各自未过源门槛|无；不同于上面 X3 去 style 的正式结果|
|REPL_G_U／REPL_G_L／REPL_G_LU|U-only 源 clean 97.7889%、G3 91.0136%；L-only 98.0185%、G3 91.2062%；均未过门槛；LU 随路线取消中止|无|
|REPL_GH_U／REPL_GH_U_Pair|人类撤下 DURA 路线后停止；只有部分训练，不能给最终性能|无|
|CleanH：H2_U／H2_U_Pair／H2_LU_Pair|源 clean 98.3296%／98.3222%／98.3481%，H3 76.8580%／76.7889%／76.7778%；按原源规则选 LU_Pair|只有 LU_Pair 三 seed，见主表|
|LAnchorH|源 clean 98.3259%，H3 76.5802%，未过门槛|无|
|H2_LU_NoPair|源 clean 98.3407%，H3 76.7259%，未过门槛|无|
|RB_RX／RB_RXTX|源 clean 98.3407%／98.3741%，H3 76.7370%／76.7642%，均未过门槛|无|
|RXConsensus-A／B|困难 V 新认证精度 9/11＝81.8182%／9/13＝69.2308%，未过覆盖与精度门槛|无，未训练|
|BoundedResponse 三冻结模型|只做源域有界输入校正，三个 seed 均未过固定规则|无新训练或目标测试|
|RXTrust-G／接收机响应几何／CovMask／L-anchor 漂移／DURA 多实现等|静态源几何、置信、梯度或输入可变性诊断；不是新识别器性能|无新增目标方法成绩|
|FastTrust／GuardVAT 准备|仅诊断或未发布准备，不能写成已训练方法|无|
|HQC-Fixed／HQC-Soft／HQC-Gate／UHardCE|当前四个单种子源探索，E30 核验通过；截至本汇总没有完整最终源／目标结果|尚无；不冒充三 seed|
|RXTrust-AL|未找到完成的正式目标结果；归入未完成，不用中间日志填成绩|尚无完整成绩|

未完成／取消／技术恢复条目不按 0% 排名。诊断前后变化不能替代训练后目标性能。源域精度只是可见 V 的代理，实际 U TX 真值隐藏，不能将 V 接受精度说成已知 U 精度。

## 文件、去重和核验

- [全部三种子配置：完整数值、单景、SD、来源和排序](E:/type10-7/docs/analysis/PAMF_SSL_METHOD_RANKING_20261027/all_three_seed_methods.csv)。
- [单种子结果](E:/type10-7/docs/analysis/PAMF_SSL_METHOD_RANKING_20261027/single_seed_methods.csv)；[11 设置配对差](E:/type10-7/docs/analysis/PAMF_SSL_METHOD_RANKING_20261027/REPL_11_settings_cross_rx_deltas.csv)。
- [30 个完整源域候选](E:/type10-7/docs/analysis/PAMF_SSL_METHOD_RANKING_20261027/historical_source_only_30.csv)；[重复名称／复用行映射](E:/type10-7/docs/analysis/PAMF_SSL_METHOD_RANKING_20261027/deduplicated_aliases.csv)。
- [结构化证据与核验结果](E:/type10-7/docs/analysis/PAMF_SSL_METHOD_RANKING_20261027/evidence.json)。

逐种子可用时，独立复算三 seed、每视图物理数、均值和样本 SD；复用别名必须与原值一致。m201 的两个互斥子集只按计数加权均值，不拼接 SD。全部负值保留，无 target 重新预测或评分。

### 本轮报告覆盖索引

以下覆盖从 2026-10-23 起盘点到的 121 份 PAMF 探索、诊断、恢复及汇总报告。索引条数不是算法数；技术恢复和重复对照已在结果层去重。

|实验／分析记录|登记方法|原始报告|
|---|---|---|
|20261023-phase1-PAMF-core-factorial-recovery-r01|CPU_core_factorial_recovery|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-core-factorial-recovery-r01/report.md)|
|20261023-phase1-PAMF-core-source-diagnostic-recovery-m2-r01|Frozen_source_diagnostic_recovery|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-core-source-diagnostic-recovery-m2-r01/report.md)|
|20261023-phase1-PAMF-cpu-balance-control-r01|four independent CPU scorers;1s scheduling poll|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-cpu-balance-control-r01/report.md)|
|20261023-phase1-PAMF-paired-U-m3-r01|PAMF paired U:raw+Practical student; own weak EMA;L-only cross-RX prototypes;fixed paired CE/prototype/feature weights|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-paired-U-m3-r01/report.md)|
|20261023-phase1-PAMF-physical-softmatch-m6-r01|PAMF PSM, PAMF RXPSM|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-physical-softmatch-m6-r01/report.md)|
|20261023-phase1-PAMF-physical-softmatch-m6-r02|PAMF PSM, PAMF RXPSM|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-physical-softmatch-m6-r02/report.md)|
|20261023-phase1-PAMF-physics-device-control-r01|bounded_technical_handoff|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-physics-device-control-r01/report.md)|
|20261023-phase1-PAMF-physics-device-recovery-r01|WRONG_REFERENCE|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-physics-device-recovery-r01/report.md)|
|20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01|CPU_permission_factorial|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01/report.md)|
|20261023-phase1-PAMF-PL-Gfactor-control-r01|Healthy_owner_handoff|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-PL-Gfactor-control-r01/report.md)|
|20261023-phase1-PAMF-PL-Gfactor-control-r02|Healthy_owner_handoff|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-PL-Gfactor-control-r02/report.md)|
|20261023-phase1-PAMF-SSL-DURA-eval-m6-r01|PAMF PSM, PAMF RXPSM|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-SSL-DURA-eval-m6-r01/report.md)|
|20261023-phase1-PAMF-SSL-DURA-reference-m9-r01|PSM frozen default head, raw + fixed DURA v1.0 Graph-RR, RXPSM frozen default head, raw + fixed DURA v1.0 Graph-RR, VIEWANCHOR frozen default head, raw + fixed DURA v1.0 Graph-RR|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-SSL-DURA-reference-m9-r01/report.md)|
|20261023-phase1-PAMF-supervised-physics-timing-recovery-r01|Nine structures; raw and DURA-L;5 measurement rounds|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-supervised-physics-timing-recovery-r01/report.md)|
|20261023-phase1-PAMF-U-CMR-m3-r01|confidence-protected pseudo-class conditional cross-source-RX mean/std matching|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-CMR-m3-r01/report.md)|
|20261023-phase1-PAMF-U-CRIC-m3-r01|PAMF RXPSM plus detached soft/confidence-weighted hidden-U cross-RX contrast at native time_down and t2.pool|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-CRIC-m3-r01/report.md)|
|20261023-phase1-PAMF-U-DualAnchor-m3-r01|U gradient projection against both full T1 L and existing raw-L true CE|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-DualAnchor-m3-r01/report.md)|
|20261023-phase1-PAMF-U-GradProtect-m3-r01|one-sided source-L anchored projection and bounded weighted U gradients|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-GradProtect-m3-r01/report.md)|
|20261023-phase1-PAMF-U-MixStyle-m3-r01|confidence-protected same-predicted-TX cross-source-RX temporal moment mixing on U shallow/middle features|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-MixStyle-m3-r01/report.md)|
|20261023-phase1-PAMF-U-ViewBalance-m3-r01|per-view dual projection and fixed equal gradient-norm budget|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-ViewBalance-m3-r01/report.md)|
|20261023-phase1-PAMF-viewanchor-m3-r01|PAMF RXPSM + physical view feature anchor0.10 fromE31|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-viewanchor-m3-r01/report.md)|
|20261023-phase1-PAMF-viewanchor-m3-r02|PAMF RXPSM + physical view feature anchor0.10 fromE31|[报告](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-viewanchor-m3-r02/report.md)|
|20261024-phase1-PAMF-HardPL-AllURDrop-m3-r01|HPL_E20_AS_ALLRD|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-AllURDrop-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01|HPL_E20_AS_BCE6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01|HPL_E20_AS_BG2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-Complement-m3-r01|HPL_E20_JOINT_COMP6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-Complement-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-E20plus30-m6-r01|HPL_E20_LS2, HPL_E20_NAMS2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-E20plus30-m6-r01/report.md)|
|20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01|HPL_E20_LS6G, HPL_E20_NAMS6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01/report.md)|
|20261024-phase1-PAMF-HardPL-Joint160-m3-r01|HPL_E20_NAMS_JOINT6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-Joint160-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-LS-m3-r01|HPL_LS2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-LS-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-MixStyle-m3-r01|HPL_MS2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-MixStyle-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-MixStyle-m3-r02|HPL_MS2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-MixStyle-m3-r02/report.md)|
|20261024-phase1-PAMF-HardPL-NativeAs-m3-r01|HPL_E20_JOINT_AS6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-NativeAs-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-NativeAsCap2-m3-r01|HPL_E20_AS_CAP2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-NativeAsCap2-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01|HPL_NAMS2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-noUAs-m3-r01|HPL_NO_UAS|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-noUAs-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-PathComplement-m3-r01|HPL_E20_JOINT_PATH6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-PathComplement-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01|HPL_E20_PATH_RAS6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01|HPL_E20_JOINT_RXE6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01|HPL_E20_JOINT_RAS6G|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-StepGuard-m3-r01|HPL_E20_AS_SG2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-StepGuard-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-StepNoCap-m3-r01|HPL_E20_AS_SGNC|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-StepNoCap-m3-r01/report.md)|
|20261024-phase1-PAMF-HardPL-T1U-m6-r01|HPL2, HPLT1U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-T1U-m6-r01/report.md)|
|20261024-phase1-PAMF-REPL-completion-m201-r01|ABC000, ABC001, ABC010, ABC011, ABC100, ABC101, CE_COV_X3, CE_NO_PA, CE_NO_REFERENCE, CE_NO_SIGNATURE, CE_NO_TIME, CE_SHUFFLE_TIME, CE_SIG_TIME, CE_WRONG_REFERENCE, CE_full, L_As_KL_on, L_joint_only, L_middle_only, L_shallow_only, L_single_forward, L_without_As, L_without_raw_CE, REPL, REPL_COV_X3, REPL_NO_PA, REPL_NO_REFERENCE, REPL_NO_SIGNATURE, REPL_NO_TIME, REPL_SHUFFLE_TIME, REPL_SIG_TIME, REPL_WRONG_REFERENCE, ordinary_PL, supervised, without_extrapolation, without_guard, without_joint, without_style|[报告](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-REPL-completion-m201-r01/report.md)|
|20261025-diagnostic-PAMF-DURA-dose-noise-m1-r01|source-only restored dose/noise diagnostic|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-diagnostic-PAMF-DURA-dose-noise-m1-r01/report.md)|
|20261025-diagnostic-PAMF-DURA-dose-noise-m1-r02|source-only restored dose/noise diagnostic|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-diagnostic-PAMF-DURA-dose-noise-m1-r02/report.md)|
|20261025-diagnostic-PAMF-DURA-paired-source-m3-r01|frozen F/G source paired diagnostic|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-diagnostic-PAMF-DURA-paired-source-m3-r01/report.md)|
|20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01|frozen matched source geometry/error diagnosis|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01/report.md)|
|20261025-diagnostic-PAMF-RXReference-dose-m1-r01|three matched identity-reference gradient diagnostics|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-diagnostic-PAMF-RXReference-dose-m1-r01/report.md)|
|20261025-phase1-PAMF-ChannelPL-m33-r01|PAMF-ChannelPL fixed candidate|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-ChannelPL-m33-r01/report.md)|
|20261025-phase1-PAMF-ChannelPL-source-m33-r01|fixed ChannelPL source proxy|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-ChannelPL-source-m33-r01/report.md)|
|20261025-phase1-PAMF-PeerSupervised-control-r03|PeerPL2, PeerT1U1, PeerT1U2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-control-r03/report.md)|
|20261025-phase1-PAMF-PeerSupervised-label01-m3-r01|PeerT1U2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-label01-m3-r01/report.md)|
|20261025-phase1-PAMF-PeerSupervised-m9-r01|PeerPL2, PeerT1U1, PeerT1U2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-m9-r01/report.md)|
|20261025-phase1-PAMF-PeerSupervised-m9-r02|PeerPL2, PeerT1U1, PeerT1U2|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-m9-r02/report.md)|
|20261025-phase1-PAMF-PL-consensus-source-m6-r01|frozen default-head source consensus diagnostics|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PL-consensus-source-m6-r01/report.md)|
|20261025-phase1-PAMF-RestoredPL-m9-r01|DURASoft, PLControl, RawSoft|[报告](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-RestoredPL-m9-r01/report.md)|
|20261026-analysis-PAMF-FastTrust-source-r01|FastTrust-style diagnostic|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-FastTrust-source-r01/report.md)|
|20261026-analysis-PAMF-NoSIG-RawAnchor-labels-r01|PLAnchor, SupAnchor|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-NoSIG-RawAnchor-labels-r01/report.md)|
|20261026-analysis-PAMF-SIG1-Covariance-Fonly-m4-r01|CovFull-U, CovRandom-U, CovSel-LU, CovSel-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-SIG1-Covariance-Fonly-m4-r01/report.md)|
|20261026-analysis-PAMF-SIG1-CovSupport-L-s2026092701-r01|CovSupport-L|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-SIG1-CovSupport-L-s2026092701-r01/report.md)|
|20261026-analysis-PAMF-SIG1-rejected-pairs-m1-r01|frozen_cached_default_head_pair_analysis|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-SIG1-rejected-pairs-m1-r01/report.md)|
|20261026-analysis-PAMF-SSL-retrospective-r01|retrospective-plus-NoSIG-RGPL-design|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-SSL-retrospective-r01/report.md)|
|20261026-diagnostic-PAMF-DURARealizations-source-m1-r01|FrozenE30-DURA-realizations|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-DURARealizations-source-m1-r01/report.md)|
|20261026-diagnostic-PAMF-LAnchorDrift-m5-r01|Frozen_L_anchor_drift|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-LAnchorDrift-m5-r01/report.md)|
|20261026-diagnostic-PAMF-LSetGeometry-m1-r01|Frozen_L_relation_geometry|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-LSetGeometry-m1-r01/report.md)|
|20261026-diagnostic-PAMF-NoSIG-REPL-RXGeometry-m1-r01|L-balanced RX offsets minus L class subspace|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-NoSIG-REPL-RXGeometry-m1-r01/report.md)|
|20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-m1-r01|Frozen-NoSIG-RGPL-graph|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-m1-r01/report.md)|
|20261026-diagnostic-PAMF-RXReference-alignment-m1-r01|frozen source identity-gradient comparison|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-RXReference-alignment-m1-r01/report.md)|
|20261026-diagnostic-PAMF-RXResponse-LU-source-r01|fixed_LSTF_mixed_response_crossed_diagnostic|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-RXResponse-LU-source-r01/report.md)|
|20261026-diagnostic-PAMF-SIG1-CovMask-L-s2026092701-r01|frozen-L-correlation-mask-fit|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-SIG1-CovMask-L-s2026092701-r01/report.md)|
|20261026-diagnostic-PAMF-SIG1-LU-restored-views-m1-r01|frozen_SIG1_E20_LU_views|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-SIG1-LU-restored-views-m1-r01/report.md)|
|20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-m1-r01|singleSIG_TIME L-balanced identity-protected RX geometry|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-m1-r01/report.md)|
|20261026-diagnostic-PAMF-SIG1-RXTrust-source-s2026092701-r01|RXTrust-G-Instance-U-LU|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-SIG1-RXTrust-source-s2026092701-r01/report.md)|
|20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01|wholemodel task-scaled reference dose|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01/report.md)|
|20261026-diagnostic-PAMF-URealizations-cache-m3-r01|Frozen-DURA-U-realization-cache|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-diagnostic-PAMF-URealizations-cache-m3-r01/report.md)|
|20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01|BothJoint, GJoint, HAnchor, HJoint|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01/report.md)|
|20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01|GFeatureHard, GJointHard, HFeatureHard, HGFeatureHard|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01/report.md)|
|20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-m6-r01|NoisyCE, NoisyGCE, NoisyGCE2Raw, NoisyGCEG, NoisyGCEHG, NoisyScaledCE|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-m6-r01/report.md)|
|20261026-phase1-PAMF-NoSIG-RestorationLabeler-source-m2-r01|GLabeler, RawLabeler|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-NoSIG-RestorationLabeler-source-m2-r01/report.md)|
|20261026-phase1-PAMF-NoSIG-RGPL-base-source-m2-r01|NoSIG-PL, NoSIG-Sup|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-NoSIG-RGPL-base-source-m2-r01/report.md)|
|20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r01|PAMF supervised noSIG BASE, PAMF supervised noSIG NO_PA, PAMF supervised noSIG NO_PA_REFERENCE, PAMF supervised noSIG NO_REFERENCE, PAMF supervised noSIG WRONG_REFERENCE|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r01/report.md)|
|20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r02|PAMF supervised noSIG BASE, PAMF supervised noSIG NO_PA, PAMF supervised noSIG NO_PA_REFERENCE, PAMF supervised noSIG NO_REFERENCE, PAMF supervised noSIG WRONG_REFERENCE|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r02/report.md)|
|20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01|MixBase, MixDose, PairBase, PairDose|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01/report.md)|
|20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01|BridgeGG, BridgeHG, BridgeRG, SeparateRG|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01/report.md)|
|20261026-phase1-PAMF-RXReference-sourceHold-m3-r01|PairedReference, RawReference, RestoredReference|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RXReference-sourceHold-m3-r01/report.md)|
|20261026-phase1-PAMF-SIG1-AllUVAT-source-s2026092701-r01|PL-VAT|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-AllUVAT-source-s2026092701-r01/report.md)|
|20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r01|CovRandom-U, CovSel-LU, CovSel-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r01/report.md)|
|20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r02|CovRandom-U, CovSel-LU, CovSel-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r02/report.md)|
|20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r03|CovRandom-U, CovSel-LU, CovSel-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r03/report.md)|
|20261026-phase1-PAMF-SIG1-Covariance-source-m4-r01|CovFull-U, CovRandom-U, CovSel-LU, CovSel-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-Covariance-source-m4-r01/report.md)|
|20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01|SIG1-PL, SIG1-Sup|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01/report.md)|
|20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01|SIG1-PL, SIG1-Sup|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01/report.md)|
|20261026-phase1-PAMF-SIG1-REPL-CleanH-source-m3-r01|REPL_H2_LU_Pair, REPL_H2_U, REPL_H2_U_Pair|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-CleanH-source-m3-r01/report.md)|
|20261026-phase1-PAMF-SIG1-REPL-GH-source-m2-r01|REPL_GH_U, REPL_GH_U_Pair|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-GH-source-m2-r01/report.md)|
|20261026-phase1-PAMF-SIG1-REPL-RestoredRoles-source-m3-r01|REPL_G_L, REPL_G_LU, REPL_G_U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-RestoredRoles-source-m3-r01/report.md)|
|20261026-phase1-PAMF-SIG1-REPL-RXOffset-source-m3-r01|REPL_NoStyle, REPL_NoStyleNoRXE, REPL_RXOffset, Reference_PL_X3, Reference_REPL_SIG1|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-RXOffset-source-m3-r01/report.md)|
|20261026-phase1-PAMF-SIG1-UHistory-confirm-s2026092701-r01|PL-History|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-UHistory-confirm-s2026092701-r01/report.md)|
|20261026-phase1-PAMF-SIG1-UHistory-source-m3-r01|PL-History, PL-Negative, PL-Temporal|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-UHistory-source-m3-r01/report.md)|
|20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01|GTask2x, GTask3x, GUniform3x, RawTask3x|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01/report.md)|
|20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01|GTaskScaled, GUniform, RawTaskScaled|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01/report.md)|
|20261026-phase1-PAMF-URealizations-sourceHold-m4-r01|GGCLMulti, GJointMulti, GTaskMulti, HGCLMulti|[报告](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-URealizations-sourceHold-m4-r01/report.md)|
|20261027-control-PAMF-Covariance-threeSeed-queue-r01|original28job-queue-only|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-control-PAMF-Covariance-threeSeed-queue-r01/report.md)|
|20261027-diagnostic-PAMF-BoundedResponse-SIG1-source-m3-r01|frozenSIG1-E20-raw-vs-bounded|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-diagnostic-PAMF-BoundedResponse-SIG1-source-m3-r01/report.md)|
|20261027-diagnostic-PAMF-CovMask-s0203-source-r01|natural_support_mask_fit, reuse_actual_seed01_mask|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-diagnostic-PAMF-CovMask-s0203-source-r01/report.md)|
|20261027-diagnostic-PAMF-SIG1-RXConsensus-m1-r01|RXConsensus-A/B-frozen|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-diagnostic-PAMF-SIG1-RXConsensus-m1-r01/report.md)|
|20261027-diagnostic-PAMF-SIG1-sourceConfidence-m12-r01|RXMean-LU, RXMean-U, SIG1-PL, SIG1-Sup|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-diagnostic-PAMF-SIG1-sourceConfidence-m12-r01/report.md)|
|20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01|CovFull-U, CovRandom-U, CovSel-LU, CovSel-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01/report.md)|
|20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r01|RXMean-LU, RXMean-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r01/report.md)|
|20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r02|RXMean-LU, RXMean-U|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r02/report.md)|
|20261027-phase1-PAMF-RXMeanMax-SIG1-threeSeed-m3-r01|RXMean-MaxU|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXMeanMax-SIG1-threeSeed-m3-r01/report.md)|
|20261027-phase1-PAMF-RXTrustA-SIG1-threeSeed-m3-r01|RXTrust-A|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustA-SIG1-threeSeed-m3-r01/report.md)|
|20261027-phase1-PAMF-RXTrustAL-SIG1-threeSeed-m3-r01|RXTrust-AL|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustAL-SIG1-threeSeed-m3-r01/report.md)|
|20261027-phase1-PAMF-RXTrustP-SIG1-threeSeed-m3-r01|RXTrust-P|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustP-SIG1-threeSeed-m3-r01/report.md)|
|20261027-phase1-PAMF-RXTrustPM-SIG1-threeSeed-m3-r01|RXTrust-PM|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustPM-SIG1-threeSeed-m3-r01/report.md)|
|20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-m3-r01|REPL_H2_LU_Pair, SIG1-PL, SIG1-Sup|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-m3-r01/report.md)|
|20261027-phase1-PAMF-SIG1-REPL-H2LUNoPair-source-m1-r01|REPL_H2_LU_NoPair|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-H2LUNoPair-source-m1-r01/report.md)|
|20261027-phase1-PAMF-SIG1-REPL-HQuality-source-m3-r01|REPL_HQC_Fixed, REPL_HQC_Gate, REPL_HQC_Soft|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-HQuality-source-m3-r01/report.md)|
|20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01|REPL_H2_LU_LAnchorH|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01/report.md)|
|20261027-phase1-PAMF-SIG1-REPL-RxBalance-source-m2-r01|REPL_RB_RX, REPL_RB_RXTX|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-RxBalance-source-m2-r01/report.md)|
|20261027-phase1-PAMF-SIG1-REPL-UHardCE-source-m1-r01|REPL_UHardCE|[报告](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-UHardCE-source-m1-r01/report.md)|
