# PAMF 源模式局部协方差：固定单候选

Run `20261031-phase3-local-cov-s2026092701-r01`。沿用用户选取的原 P1 signal pooled ratio 和 predicted ratio，不改变原身份头或 DURA。唯一主 `local_predicted_ratio`，控制为原 pooled/predicted ratio 与新 local pooled ratio。当前没有新性能，完整目标 ACTIVE。

## 假设与固定改变

原 known90 每个 TX×RX×day 模式仅保存球形方差，即使共享全协方差白化也不能表示各模式不同的相关方向。本轮检验局部相关形状是否改善开放分离；不把问题预先归因于阈值，也不重复已失败的全局硬投影或目标共享偏移。

对原源 L 的每个 70 条模式，在原白化坐标估计 population 协方差 S_j（分母70），固定 α=70/(70+160)，C_j=αS_j+(1−α)pooled_L_variance I。原90均值、共享原点/白化/行列式及OE32全部冻结。trace(C_j)/160应在重复前向的浮点容差内等于原标量variance；不增加可扫描超参数。评分使用完整Gaussian距离和logdet，known模式权重仍1/90，预测类条件为15模式各1/15。原F0预测身份仅作条件，不用真实query TX。

这是局部协方差形状对照，不是新身份分类头、目标适应或匿名注册。收缩协方差的统计依据参照 [Friedman 原论文](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752)，本文固定收缩系数且不复现其分类器/交叉验证；[RMD](https://arxiv.org/abs/2106.09022)是相关背景，已有共享协方差RMD不等同本轮90个局部协方差。文献不证明本项目性能。

## 数据、继承与隔离

父完整三DURA预测为 `20261031-phase3-class-density-s2026092701-r01` / code `4af0668331682636ee212077b38cd912857a1b31`。P1 signal r03 code `dc01a402582f93e452dbaa3ef59a9fbb44151818` 为纯源L/OE的2220步、headless232033＋MLP102688；原F0为Tri-SIG-TIME SSL E50/11100/default6/232993，原sourcecontract392005及VALIDATED_ONCE角色保持。DURA v1.0 R69/266951/E20/4940/reuse=True/graph=True/native=False、FP32/TF32off保持。明确排除最近TARGET_SSL的U适配权重和P0开放权重，继承来源未变。

本轮仅重新前向原L6300的既有DURA星地P1缓存，六旧设备各1050，每TX×RX×day70；五源RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，三个日期2021-03-08/15/23。L真旧标签合法。OE6134仅继承原背景聚合，原21设备真TX/K不读；Us56700新读0。

V27000旧六设备各4500，仅在全部局部协方差冻结后校准两个新六阈值，规则为原known12（overall＋5RX＋6scene），其余30交叉组完整诊断；另两个旧六阈值/校准报告逐值复制。没有新增标签预算或使用novel调阈值。

源novel3150为19-15/19-16/7-1各1050。clean/star Q是同RX24-1(index26)、2021-03-15的同864物理记录，旧六/未知三设备各96；同RX的U384来自03-08、物理互斥且本轮未读。source/V/NV/两个Q只复用父固定特征/默认输出；所有三测试的原F0与signal曾接收同一DURA输出，clean也含DURA。本轮不重读Q IQ、不重跑G/F0/Qsignal，重复评分不新增物理记录。

学习输入只为固定G10星地P1，600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区城区各0.5及原六面积概率，receiver2027/scene2026103001/channel2026104001；一次输入unitRMS、输出不再RMS。clean仅冻结测试，不进入训练/校准/适应。目标真TX/K/配额、U、正式RX24-18和未来query均不进入拟合。

## 顺序、矩阵与判定

唯一producer先执行实际F0/signal/G各两条synthetic no-query smoke，PASS立即继续。读原源模型/父完成身份，再仅开L P1导出6300 signal行，拟合并保存90个局部协方差聚合；冻结后才V校准，全部四套tau冻结后才源novel和两个Q特征。保存四列×三条件全预测；独立CUDAempty CPU scorer验证固定状态、原两控制逐值、全部raw/margin后才连接sourceNV/cleanQ/starQ真值各一次。score不拟合、重调tau或连接Utruth。

三条件无条件完成，不因源RU未过省略Q。主报逐TX宏A_K/R_U/H，另列C_K、同输入原F0 closed及其减A_K、AUROC/OSCR、逐TX/RX/day/scene/42组与配对gain/loss/net。主对原predicted、原pooled和local-pooled固定比较；不改主。clean RU≥95%、star RU≥90%、C_K≥90%、同G原F0闭集损失≤2pp及必需主景覆盖联合验收，主star A_K和R_U还须不低于原predicted。小Q的high未知0/4/2缺19-15，保持N/A、不补抽或认证六景。source为已见RX，NV/Q历史曝光、单模型/开发RX保持；不称新盲测或正式晋级。

仅新协方差90、两个新六tau；神经/optimizer/原密度/normalizer/背景新拟合0，源OE/U/新Q IQ读取0。真实signal前向6300，F0/G真实0；synthetic各2另计。原模型训练和DURA缓存/旧Q预测成本继承，部署原834665神经参数不变，另报局部矩阵/原密度/tau聚合实际字节，不部署L逐行bank。报告批处理时长，不以参数数目推定逐包延迟。A_N/H_OW仍N/A，不以拒识局部收益替代无标签DA/发现/未来注册。

N607唯一owner、独占新root，每卡新增本任务≤2且启动实际free≥4096MiB，不停止/修改/重启其他owner。代码实际commit、精确runtime/命令/CWD/路径/GPU及预期产物写本run experiment.json/report；聚焦负测/一次真实CPsmoke/一次直接P0/P1审查/单preflight与单archive SHA比较/remotecompile/一次PID-CWD-argv-GPU-log读回。非法角色/祖先、非有限、输出碰撞或无法合法预测时保留，不partial评分或自动重启；UNKNOWN先核原PID/root。科学负结果不停止健康任务或降低门槛。
