# 已有较好开放方案：DURA 接入前后配对测试

Run `20261031-phase3-dura-best-s2026092701-r01`。用户要求先选前面性能好的方法，核实并补测DURA。固定既有P1 signal r03的 known_support 与 density_ratio 两方案，唯一主density_ratio；不重训、不拟合统计/normalizer、不重校准阈值、不引入prepared的预测类条件密度。

## 为什么选这两个方案

两个方案共享同一P1 signal网络和known90统计，密度比另用既有OE32背景。已完成raw-open r01的P1两控制为：known A_K89.948148/C_K97.522222/R_U46.888889/H61.643672%；ratio A_K89.770370/C_K97.107407/R_U60.888889/H72.561330%。相同原F0 P1闭集90.981481%，总体损失1.033333/1.211111pp，原known12通过，known42失败；R_U仍未达90%。选用依据为源侧既有已知保护和拒识结果，不按本轮目标结果选择。

上述两个源侧结果**本来已经接入DURA**。本轮精确复用它们作为source_DURA控制，不能将该60.888889%说成未恢复成绩或新的恢复收益。旧跨RX clean测试绕过DURA，且开放网络为其它历史方法，不能直接与本网络的新DURA结果作因果配对。因此本轮对相同网络、统计、阈值和物理记录同时计算raw/DURA路径。

## 冻结模型与分数

原身份为Tri-SIG-TIME SSL F0 E50/11100/default6，232993参数，sourcecontract392005；独立P1开放signal为signal-open r03 / actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`，headless232033＋MLP102688＝334721，复制特征祖先11100＋已有2220＝13320，新MLP已有2220。其原source L/OE训练合法，目标未参与祖先。

源控制取raw-open r01 / code `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb` 的P1_known12/P1_ratio12列0/1、P1_ratio12模型11数组和原两套六阈值。实际加载网络身份匹配该父 `signal_reference_identity.model_identity`，不是其P0主identity，绝不加载另训P0 raw-open checkpoint。

known_support为既有90个class×RX×day Gaussian mode的全类log likelihood；density_ratio为同known likelihood减既有OE32-mode背景log likelihood。FP64规范unit160、共享白化、均值/方差/权重/Jacobian均保持原状态。两分数由同一P1 signal前向计算，身份始终由原F0默认头给出。按该路径F0预测类路由原六阈值，margin≥0接收，ties不拆分。**所有路径共享原P1源星地模型、聚合统计和阈值**；raw只是固定恢复后模型绕过G的诊断，不代表独立重训raw路线的能力。

## 固定六路径

| 数据条件 | raw诊断 | DURA主测试 | 数量 |
| --- | --- | --- | --- |
| 源域已知接收机 | 同固定星地received→原F0/P1signal | 同received→既有冻结DURA→原F0/P1signal | V27000＋source novel3150 |
| 跨接收机clean | 现有clean public→原F0/P1signal | 同clean→冻结DURA→原F0/P1signal | Q864，两路径共享同记录 |
| 跨接收机星地 | 现有固定G10 public→原F0/P1signal | 同G10→冻结DURA→原F0/P1signal | Q864，两路径共享同记录 |

全部三个DURA主测试都实际包含冻结DURA v1.0。源DURA复用已验证同G输出的原预测/特征；两个跨RX条件实际运行同一G，clean也不绕过。每一路径的原F0闭集基线由它实际接收的同输入得到；不能用DURA闭集替raw，也不能以历史clean绕过成绩替代DURA(clean)。比较DURA−raw时两侧身份预测允许改变，正确接收增减必须分别使用两侧closed，另列原闭集纠错/损伤。

DURA固定R69 v1.0/Graph-RR，reuse=True/graph=True/native=False，FP32/TF32off；所有模型、BN、跨query状态、精度和求解预算不更新，不额外RMS或EQ。P0是星地received而非clean；clean仅冻结测试例外，不进入Phase3训练、适应、发现或注册。

## 数据与边界

源source V旧六TX各4500，每TX×RX×day300；novel为19-15/19-16/7-1各1050，每TX×RX×day70。source RX索引1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23均已见。原L6300/OE6134/Us56700仅为既有模型/统计祖先，本轮新读0。source V曾用于父阈值风险校准，本轮不再校准；已知风险不是独立已知盲测。source novel已历史曝光，真值仍只在全部预测后由独立scorer连接。

两个Q复用现有RX24-1(index26)开发evaluation公共三文件及capsule/split，03-15 future日，U为03-08且物理互斥，本轮U不打开。Q旧六类各96、未知三类各96仅builder/scorer知道，预测不见真实TX、K或配额。clean与star是相同物理角色的两条件，raw/G又是同观测的计算路径，不计为3456个独立物理数据。开发Q已曝光，单RX/seed不能称新盲测；正式RX24-18继续保留。现有VALIDATED_ONCE数据只核匹配handles，不新造信道或重复全IQ验证。

G10MIX01沿600km/25MHz/2.45GHz、residual/post_sync/noadditionalEQ/refSNR30dB/reflect、郊区城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001，进入信道前一次unitRMS、输出不再RMS。跨方法/路径共享该固定实现，不从clean向star提供逐条参考或拟合状态。

## 顺序、指标与科学判定

严格加载实际原F0、DURA和P1 signal checkpoint，首先以两行合成G10执行G→F0/signal无query smoke并验证state保持；PASS立即继续。全部11项源聚合和两套原阈值冻结后，才打开source原始V/novel或任何Q。源DURA两列model/τ/raw/margin/defaultidentity精确复用；source_raw只读已有received值，并对相同冻结F0/signal前向。两个Q每批同received分别raw/G，G输出共供F0及signal，不根据任何前路径结果改后路径。

六路径全部预测完成后，一个CUDAempty独立CPU scorer先验证聚合/两τ/单位特征能重建所有12列raw/margin及原source_DURA控制，再读取source novel truth、clean truth、star truth各一次；每个truth同时评价对应raw/DURA两路径。无partial评分、调τ、query统计、候选更换、选择性重跑或历史query训练。源侧失败不取消其它已登记测试。

每路径主报逐TX宏A_K、R_U和H-score，并报C_K、原F0闭集、同路径原闭集到开放损失pp、AUROC/OSCR、逐TX/RX/day/scene与覆盖。重点输出每种方法在source/clean/star的DURA−raw配对：闭集正确与错误转移、正确接收gain/loss/net、未知拒对gain/loss/net、接收/误拒变化。不能只以R_U提高称恢复收益，也不能用两侧不同closed的旧固定身份配对函数计算A_K收益。

source_DURA的known12风险沿原父判定复核，sourceRU90单列开发诊断。主density_ratio的clean_DURA整体R_U≥95%、star_DURA整体及预登记六场景R_U≥90%，两者C_K≥90%、同DURA输出原F0 closed−A_K≤2pp和必需TX覆盖；缺类时N/A，不认证。当前小Q已知有场景覆盖不足，保留事实，不事后重抽或把整体通过当全景通过。raw三路只诊断，不替代DURA主测试或救主；known_support控制不替主。

development gate只在唯一主两个跨RX DURA条件和所需主景/保护/覆盖全部通过时为真。仍非完整Phase3成功：历史开发池、单模型/RX，且本轮没有零标签DA、发现或未来匿名注册，A_N/H_OW为N/A、完整目标ACTIVE。prepared条件密度未在本轮运行。

## 成本和执行

新NN/身份头/signal/projection/G/密度/normalizer/阈值更新全部0；L/OE/Us/U/正式RX新读0。source_raw前向30150＋四个Q路径4×864＝33606行，原F0和P1 signal各处理33606真实行；两个Q的G各864，G合计1728。synthetic三网络各2；源DURA复用，不重复G或NN。独立物理记录仅source30150及共享Q864，重复条件/前向不计新增物理记录。原F011100、P1 signal已有2220、DURA原训练和既有源统计拟合成本单列。

活跃F0＋一个signal为567714参数，加G算术834665；一份11数组364340B、两套六FP64τ96B，逐条source bank不部署。各阶段batch时间、文件/显存离散观察不冒充B1推理时延或峰值。

N607单GPU producer→单独CPU scorer，启动actualfree≥4096MiB、每卡新增本任务≤2，不停止/修改/重启其它owner。一次actualcode commit、focused已知控制/同θτ/G路径/变化closed配对/真值隔离、一次直接P0/P1review、最小登记、不可覆盖root/单preflight/archive比较/compile、PID/CWD/argv/GPU/log增长和终态。非法来源/角色、非有限、碰撞或无法合法预测时保留，不评分partial、不自动重启；UNKNOWN先核原PID/root，科学低分不停止健康工作。
