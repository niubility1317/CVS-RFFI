# PAMF 源域类别联合偏移的无标签适应

Run `20261031-phase3-class-shift-s2026092701-r01`。保留原 DURA＋F0＋P1 signal r03、known90/OE32及两套六阈值，只用合法星地U384的既有特征拟合一组类别相关的域系数。主为 `CLASS_ratio`，不是对上轮条件密度阈值的再选择，不消费其新阈值。

## 固定假设与矩阵

原公共偏移先对六个已知类的源域变化平均，方向相反的类别变化会被抵消。新候选保留六类的联合变化，让同一组潜在域系数产生六个不同偏移。其有效性是待验证假设，不因源SVD可计算、U目标改善或非零偏移认定成功。

六列固定为 `DA0_known,DA0_ratio,SHARED_known,SHARED_ratio,CLASS_known,CLASS_ratio`，唯一主索引5。DA0精确复用dura-best r01/code9fc23a4cb78111aa96cc72fde0d35c036a876e5d。SHARED精确复用mixture-uda r01/code d72a7b7c301fec2ff3736e4221dfe5047ab78efc的DA_M，不重新拟合或按本轮Q选取其迭代；CLASS新增固定20次MAP-EM。两个适配器总拟合预算均20步，但本轮新增仅CLASS20，SHARED20为继承成本。

源域已见RX、跨RX clean、跨RX星地三条件均通过冻结DURA v1.0；全部六列和三条件在同一run完成后独立评分。source失败不取消跨RX测试。raw绕过G和另训P0网络均不进入该候选。

## 源类与域联合先验

令mu[c,d]为原白化坐标中known90的均值，c=六类，d=五RX×三day。逐类减去域均值后，将六类域变化拼成15×960矩阵T[d,:]=concat_c(mu[c,d]−mean_d(mu[c,d]))/sqrt(6)，再减列均值去除浮点残差。取非零SVD的行方向，重塑并乘sqrt(6)得到B[q,6,160]；其sum_c B[:,c]B[:,c]^T/6=I。

源方差lambda=s²/15，半径r=max_d ||T[d]||。秩按FP64 eps×max(shape)×最大奇异值固定，不用U/Q选维度。设计期只读原11聚合得到rank14、r=5.263391448463303；这是源端派生统计，不是目标拟合或新增source IQ读取。与原公共先验r约2.370508的差异来自保留类别相关变化，不能把结果解释为单独增加半径的匹配消融。

目标系数a定义六个delta[c]=sum_j a[j]B[j,c]，背景delta_OE=mean_c delta[c]。约束||a||≤r等价于六类偏移的RMS≤r，不是每类最大范数≤r。各已知mode的目标均值变为mu[c,d]+delta[c]，全部背景mode加delta_OE；原source数组保存不变，目标适配状态单独版本。六个delta使用同一a，不是分别用真实目标标签拟合六个类。

输入坐标w=(unit(signal_z)−origin)W只计算一次，不对偏移后的坐标再次单位化/白化。known likelihood对90个移动mode求和，背景对32个移动mode求和，权重、方差、Jacobian和组内权重保持。六类全部参与同一固定分数，不以真实TX或F0预测类选择适应偏移；原F0预测类仅路由既有tau。原F0身份logits、closed和闭集基线完全不变。

## U-only固定优化

目标为sum_U log[0.5 p_K(w|a)+0.5 p_OE(w|a)]−0.5 sum_j a[j]^2/lambda[j]，中性组先验0.5/0.5与旧SHARED相同，不读取真实known比例、未知类数或配额。完整U384全部使用，不使用F0硬伪标签、U真值或clean U。

a从0开始，固定20次EM，没有早停、迭代选择或补步。E步使用122个Gaussian的软责任。令S_c为类别c内责任/variance之和，v_c为相应加权(w−mu)残差，OE类同理。M步H=diag(1/lambda)+sum_c S_c B_c B_c^T+S_OE B_OE B_OE^T，b=sum_c B_c v_c+B_OE v_OE，解(H+eta I)a=b。若无约束解超源RMS球，以固定60次二分解非负eta；上界max(1,||b||/r)来自正定H，不增加搜索候选。

记录iteration0至20的objective、coefficient_norm、shift_rms、class_shift_max、known_posterior_mass及constraint_multiplier，要求有限及数值非降（1e-7容差）。posterior质量不是TX纯度。全部NN/source密度/normalizer/tau更新0，目标移动mode的潜在统计拟合实际为20步，不能称目标密度完全未拟合。

## 数据和访问顺序

原F0为Tri-SIG-TIME SSL E50/11100、232993参数和原默认六类头；P1 signal r03/code dc01a402582f93e452dbaa3ef59a9fbb44151818为headless232033＋MLP102688＝334721，继承原特征11100＋已有2220/MLP已有2220。DURA固定R69 v1.0、266951参数、E20/4940，reuse/graph、非native、FP32/TF32off。source合同392005及祖先保持；不加载不同数据合同/另训P0或目标污染权重。

唯一新拟合输入是既有mixture-uda `adaptation/U_features.npz`：opaque IDs/default_logits/closed/signal_features，算法仅用z160。其星地U为RX24-1、2021-03-08的384条discovery记录，VALIDATED_ONCE，capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`、split `83c5cc1fe2f2a1512dbc1287`；原导出经同DURA/F0/signal，真实标签/K/配额未进入旧导出/SHARED拟合。公共received文件及U truth本轮都不读。缓存复用不增加物理观测或新的神经前向。

首先核父best/旧mix完成元数据、原11数组/原tau、旧SHARED prior/adapter/diag及合法U句柄，仅读聚合，不打开source/Q行。实际三个checkpoint syntheticG10两行经G→F0/signal、状态保持的无query smoke PASS后立即继续。新source prior先冻结，再打开U特征，拟合CLASS20并冻结；然后才首次打开父source V/NV及任一Q prediction NPZ。所有读取的旧source/Q预测只用于本轮固定推理，不计算跨Q均值、责任、normalizer或阈值，不调用父score/results。

source V27000为旧六TX各4500、NV3150为19-15/19-16/7-1各1050，source RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23已见。source其他RX不绑定目标adapter，因此delta0、六列重复父两列。源L6300/OE6134/原Us56700只为祖先，本轮新读取0。原V曾用于父阈值；本轮不再校准。

clean/star Q均为RX24-1(index26)、2021-03-15的同864物理记录，旧六类各96/未知三类各96仅builder/scorer知；U/Q物理互斥、跨日，日期已在源出现，不是严格新日期。两个Q使用同一star-U拟合状态；clean仅冻结测试，不为clean另拟合。source NV及开发Q均历史曝光，单开发RX/模型不是新盲测。正式RX24-18及未来保留query不访问。

G10沿MIX01的600km/25MHz/2.45GHz、residual/post_sync/noEQ、refSNR30dB/reflect，郊区城区各0.5及原六面积权重，receiver/scene/channel seeds=2027/2026103001/2026104001，一次input RMS、output不再RMS。不重抽信道、不从clean给star逐条参考。父三条件均确实包含同DURA；SHARED中两列逐值复用旧mix4:6，DA0前两列精确复用best，原F0全部身份/闭集保持。

全部6×3预测固定后，一个CUDAempty独立scorer先检查source prior和adapter/21trace、两组精确控制、原tau和所有raw/margin，之后连接source NV、cleanQ和starQ truth各一次。不传U truth、不重新EM、不用真值重拟合/换主/选择性重跑；source失败不取消已登记测试。

## 指标、成本和停止

主报宏A_K/R_U/H，伴随C_K、同DURA原F0闭集及损失pp、AUROC/OSCR、逐TX/RX/day/scene及覆盖，主CLASS_ratio与DA0_ratio及SHARED_ratio配对正确接收/接受/未知拒对gain/loss/net。适应成功须star A_K较DA0正增且clean R_U≥95%、star总体及六主景R_U≥90%，两目标条件C_K≥90%、原F0 closed−A_K≤2pp及必需TX覆盖全部满足。源known12保护保持，source未知RU90仅诊断。小Q high已缺19-15，保持N/A、不认证，不事后抽样或删景。

本轮无未知发现或匿名注册，A_N/H_OW为N/A；完整多RX/seed/新类未来准确率80%、旧类注册损失2pp及后续未知保护目标ACTIVE。控制不能救主，源/clean单项通过或U目标改善不能替完整验收。

新真实IQ及G/F0/signal前向均0，U特征行读取384，synthetic真实三checkpoint各2。新增目标统计CLASS20，继承SHARED20以及旧G/F0/signal训练/缓存成本单列；原source11密度364340B/两套六FP64tau96B不变。活跃神经834665参数。新prior数组约107768B，CLASS六delta＋背景delta为8960B、系数14×8B；单主部署需要原source聚合与一组CLASS适配聚合，不部署逐条source/U bank。实际NPZ/JSON字节及批量墙钟据完成产物报告，不冒充B1时延。

默认N607一个GPU producer（actualfree≥4096MiB，每卡本任务新增≤2）后独立CPU scorer，不因其他owner存在而禁整卡，不停止/修改/重启其工作。一次actualcode commit、focused数学/角色/冻结顺序negative及真实CP无query smoke、一独立直接P0/P1 review、最小预登记/不可覆盖root/单preflight/archive比较/compile、PID/CWD/argv/GPU/log增长及终态。非法祖先/角色、非有限、明显objective下降、输出碰撞或无法合法预测时保留，不partial评分或自动重启；UNKNOWN核原PID/root，科学无收益不停止健康任务。
