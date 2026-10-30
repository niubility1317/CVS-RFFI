# Phase3 当前开放表征的无标签匿名注册（2026-10-31）

Run `20261031-phase3-signal-register-s2026092701-r01`，单模型种子2026092701；唯一主 `predicted_REG1`，固定四列 `pooled_REG0/predicted_REG0/pooled_REG1/predicted_REG1`。本轮检验 REG 的独立效应，推进完整开放世界链路；不虚构无变化的 DA 对照，完整零标签域适应目标继续保留。

## 冻结基底与源半径

采用已选原 r03 开放分支与原 pooled/predicted ratio：F0 Tri-SIG-TIME SSL E50/11100/default6、r03 source-only signal完整334721参数及DURA v1.0/R69 E20/4940保持冻结。原 Gaussian90/背景32/白化器/两组六阈值均逐值复用；不加载 BOUNDARY、SSL、偏移、联合特征或扰动积分候选。新类别只增加匿名聚合 Gaussian，不替换默认身份头或重拟合原已知/OE密度。

一次实际三 checkpoint 合成G10两条无query smoke通过后，仅读既有source星地P1的L6300，并用原r03前向。L为六旧类×5RX×3day×70；在原白化160空间内，每个源class/RX/day格排self取第7邻欧氏距离，全部6300距离的95%分位定义唯一ε，下界1e−6。不扫描ε或邻居数。源半径冻结后才打开目标U。

源数据合同split392005/VALIDATED_ONCE保留：旧类14-10、14-7、20-15、20-19、6-15、8-20；RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19；源日2021-03-08/15/23。原OE6134只有outside-six及公共RX/day成员监督，原U_s56700隐藏标签。本轮不新读OE/U_s；L占原L+U_s的10%，V27000为额外known校准标签。

## 完整 U 发现与联合竞争

只用已合法固定的star-G10 U384，RX24-1/index26/2021-03-08，接同一DURA→原F0/r03，不读TX、真实类数、配额或逐条clean参考。所有U参与DBSCAN（ε来自source、min_samples8、opaqueID稳定排序）。簇≥16个unique ID且原predicted_ratio拒识成员比例严格超过1/2才注册；任意簇数包括0均为合法科学结果，不强迫分出真实类别数。保留符合簇的全部成员，包括原来被误接收的成员。

每匿名类保存原白化空间均值和标量方差。v=α×人口径向方差+(1−α)×原pooled_L_variance，α=n/(n+160)，floor1e−8。编号anon-0001等由最小opaqueID稳定排序生成，不表示运营名称。原known/OE模型不变；新类条件密度不乘1/K。

对每条输入，旧类最高分g来自原默认头预测类别及其原密度比阈值；匿名类分数为logN_j−logpOE−τ_reg。旧类和全部匿名类共同竞争，最大分<0拒识，同分优先旧类、匿名同分取固定首ID。允许匿名类纠正原先被错误接收为旧类的新设备，也可能覆盖正确旧类，须在未来Q上实测。K=0严格恢复原两REG0，无神经或阈值伪适应。

全部U匿名状态冻结后，仅读原source V27000（每旧类4500）为两种原拒识器各拟合一个全局τ_reg；旧两组六阈值不改。选择使known12（overall＋5RX＋6scene）宏C_K≥90%、同DURA原F0闭集−正确旧类接收≤2pp成立的最小整ties阈值，另30RX×scene诊断。阈值候选来自新旧分数交叉点，以单调二分选择，未知/U真值/NV/Q均不参与。无源可行余量时保留保守点及明确失败，不取消后续预登记测试。

## 未来测试与独立评分

全部注册状态和阈值冻结后，预测source V/NV、crossRX clean+DURA和crossRX star+DURA。source NV3150为历史19-15/19-16/7-1各1050、仍是源已见RX；只有评分器读取其TX。两种Q是RX24-1/2021-03-15相同864物理记录，576旧类（六类各96）＋288非旧类（三类各96），U/Q与源全部物理角色互斥。Q复用已经由同DURA输出生成的原r03特征和F0默认身份，不重跑IQ、不更新BN或跨Q统计。

观测沿既定G10 MIX01：600km/25MHz/2.45GHz、residual/post_sync/noEQ、referenceSNR30dB、reflect，郊区/城区各0.5、原六分量覆盖面积概率，receiver2027/scene2026103001/channel2026104001。一条物理记录一份固定received，clean仅冻结测试例外，不参与学习。历史NV/devQ曝光、单seed/RX、日期与源重合均披露；正式RX24-18不访问。

四列×三DURA条件全部预测完成后，独立CUDAempty CPU scorer才连接U真值、sourceNV真值与两Q真值。匿名ID→TX的Hungarian一一匹配只用U真值，匹配不回流，绝不用Q重排或选择映射。全部真实U到达新类进入未来A_N分母，未发现/未匹配计错；未匹配匿名预测是接受错误身份，不当拒识。分别报告U簇数/纯度/旧类混入/合并拆分与独立未来Q识别，不能用聚类回测冒充注册。

REG0报告原宏A_K/C_K/R_U/H（R_U分母为全部非旧类）；REG1报告旧类宏A_K/C_K、到达新类A_N、旧/新H_KN、注册类宏正确率/接收率/FRR及H_OW。REG1主R_U/H_score使用后续U未见设备分母，另报实际未注册类分母，并保留同later集合的前后拒识配对；不同分母不混作同一拒识收益。

主验收沿最新用户v3：两future条件overall A_N≥80%，旧A_K较对应REG0下降≤2pp，且较同DURA原F0闭集下降≤2pp；旧及注册类宏接收≥90%；later拒识clean≥95%/star≥90%。各预登记主景要求对应类别覆盖及旧类保护/接收/later拒识约束，新类逐景A_N报告而不新增80%逐景门槛。source known12保护必须通过，其新类/RU仅诊断。high未知0/4/2缺19-15时，相关A_N/注册类宏指标为N/A，不能认证完整六景，不重抽或删场景。

## 成本、发布与停止

新神经更新0；真实signal新增L6300＋U384=6684，F0/G各384（仅U），三模型合成各2。原source星地P1恢复、F0/r03/G训练及原Gaussian/Q缓存成本继承。新增sourceε、U匿名聚合和两个sourceV全局阈值；原τ96B保留。部署原834665神经参数加实际K匿名聚合，source/U逐行bank均不部署；完整成本按真实数组及进程计，不把进程耗时当B1时延。

N607唯一owner `codex-phase3-signal-register-20261031`，不可覆盖run/release/output，实际命令/路径/六类seed见experiment.json；每卡新增本任务≤2、实际free≥4096MiB，单producer GPU和独立CPU scorer，保留其他owner。代码commit、聚焦数学/权限/实际toy贯通、一次直接P0/P1审查、一次preflight/归档SHA比较/compile、启动PID-CWD-argv-GPU-log核实。非法角色/祖先、非有限、输出碰撞或无法合法预测时保留，不partial truth评分/自动覆盖重启；UNKNOWN先核原PID/root。科学未达标仍完成矩阵并如实报告，完整Phase3目标ACTIVE。
