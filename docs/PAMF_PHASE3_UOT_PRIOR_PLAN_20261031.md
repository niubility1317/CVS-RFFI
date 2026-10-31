# Phase3 非平衡传输混合先验适应（2026-10-31）

Run `20261031-phase3-uot-prior-s2026092701-r01`。沿用户选定的 `predicted_ratio` 及冻结 `pooled_ratio` 对照，检验无标签星地 U 能否帮助选择更合适的源 RX/day 与 OE 模式先验。当前无新性能，完整目标 ACTIVE；不自动替换所选主基线。

## 方法与固定预算

[UniOT](https://papers.neurips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html)使用部分对齐处理通用域适应，并研究目标私有类别结构。本轮是借鉴该方向的 PAMF 独立变体，**不是 UniOT 完整复现**，不继承其性能、目标表征训练或自动发现声明。计算采用[非平衡传输缩放算法](https://arxiv.org/abs/1607.05816)的对数域形式；原身份网络、默认头、开放分支和 DURA 均不更新。

源 known90（六类×五RX×三日）/OE32 的均值、逐模式方差、白化和原11数组保持。用 source-only r03 的既有星地 U384 单位特征，经原白化得到 w，成本 `C[j,i]=||w_i-mu_j||²/(160*variance_j)`，共122×384。源参考质量 `a=(0.5*known_weights,0.5*OE_weights)`，目标 `b_i=1/384`；0.5是固定算法先验，不是U真实已知比例。

优化 `sum(P*C)+epsilon*sum(P*(logP-1))+rho_s*KL(P1|a)+rho_t*KL(P.T1|b)`。固定 epsilon=0.05、rho_s=rho_t=1、200次广义Sinkhorn交替更新，theta=rho/(rho+epsilon)，logK=−C/epsilon，logu/theta=loga−logsumexp(logK+logv)，logv同理。不扫描超参数、借Q早停或读取真实TX/K/配额。保留有限性、固定预算、KKT驻点残差与logmass等数值诊断；不把原始传输质量当作设备比例。

由已知90行质量按六类求和为15个源RX/day质量后归一化；OE32行另归一化。两组均取 `0.5*原先验+0.5*新估计`，保留源支持。所得15＋32＝47个FP64权重（两个单纯形自由度45）是目标U派生状态；不改变中心、方差、特征或归一化，不重新拟合源GMM。部署只需这些聚合，不保留逐U行bank或coupling。

固定四列 `pooled_ratio/predicted_ratio/UOT_pooled_ratio/UOT_predicted_ratio`，唯一主为最后一列。新pooled按15域权重/6覆盖90模式，新conditional仅按原F0预测c选对应15模式，新背景用OE32权重；身份依然是原默认六类argmax，真实query类/RX/day不路由。所有源RX用原先验，源结果重复旧控制；两个目标Q共用同一star-U适配权重。既有两组六τ全部复用，四列共24值192B、唯一12值96B、单主6值48B；无新阈值校准。

## 数据、权限和评估顺序

复用VALIDATED_ONCE的原G10星地U：RX24-1/index26、2021-03-08、384条，与源全部角色及future Q物理互斥。输入只取原d72缓存中的适配前source-only r03公开特征/opaque IDs；不继承旧adapter/prior/score结果，不用U logits、TX、真实类别数、known比例或配额。U capsule/split为 `p3dev-star-g10-6e1292b62ab09286ebf1a0df` / `83c5cc1fe2f2a1512dbc1287`。

实际原F0/r03/G各2条无query合成smoke首步PASS后立即继续→核原源状态/阈值/cache→冻结source→读U并执行唯一200步先验拟合→冻结47权重→source V27000/NV3150和clean/star Q各864四列预测→全部4×3预测固定后唯一独立CUDAempty CPU scorer连接源NV、cleanQ、starQ真值各一次，Utruth读取0。scorer重建分数，不重新拟合UOT。源科学指标不通过也完成预登记Q。

祖先源L6300/隐藏U_s56700（L占训练池10%）、OE6134；本轮源IQ新读0。V27000是额外已知校准监督祖先，本轮只复用其原校准及预测，不宣称独立known盲测。源RX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。旧六TX14-10/14-7/20-15/20-19/6-15/8-20；NV未知19-15/19-16/7-1各1050，真值仅评分。

future clean/star Q为RX24-1、2021-03-15同864条物理记录：六旧类各96＝576，三非旧类各96＝288，两个条件均先经过同一冻结DURA。clean仅冻结测试，不用于任何学习；同物理跨条件处理不增加独立样本。G10 MIX01固定600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、原六景权重、郊区城区各0.5、receiver2027/scene2026103001/channel2026104001，不重抽信道。

历史sourceNV/Q曝光、单模型seed2026092701/单开发目标RX、源日期重合保持；practical_high未知0/4/2缺19-15，宏RU/H为N/A，不能认证六景完整通过。正式RX24-18未访问。本轮推进目标零标签统计适应，尚无未知发现/匿名注册；AN/HOW为N/A，不能以先验权重解释真实设备身份或已完成注册。

## 门槛、成本与执行

主报逐真实TX宏A_K/R_U/H及C_K、同DURA原F0闭集−A_K、AUROC/OSCR、逐TX/RX/day/scene、全部源42组和coverage、同ID配对gain/loss/net。source known12、cleanRU≥95%、starRU≥90%、CK≥90%、同路径原F0闭集损失≤2个百分点、必需主景完整覆盖保持；还要求主starAK严格高于原predicted且RU不低于原predicted。H无替代单项约束的独立门槛，额外30源交叉组仅诊断。失败保留全部四列，不换较高分诊断、不依Q调参/重排/选择性重跑。

新增神经更新、源GMM/normalizer/阈值拟合、真实F0/G/signal前向和源IQ均0；新增target先验拟合1次/200步/47数值权重，整个适配状态不是source-only，原NN和源聚合仍source-only。原F0 E50/11100/232993、r03已有2220/headless232033＋MLP102688、DURA R69 E20/4940/266951，共834665参数，原density364340B、源训练与缓存成本均继承；三CP合成各2条另计，不把过程耗时写为B1延迟。

一个代码/配置Git commit固定执行，实际commit、精确命令、环境/CWD、不可覆盖run/release roots、六类seed在登记experiment.json唯一维护。入口 `experiments.pamf_phase3_open_world.star_uot_prior_launch start --runtime <release>/runtime.json`，单seed单候选自动predict→CPU score→同row报告。N607既有Python，GPU0至7中实际剩余≥4096MiB、每卡新增本任务worker≤2，本轮一个producer；不干预其他owner。

完成聚焦协议负测/真实checkpoint无query smoke、一次直接P0/P1、一次preflight/单release包SHA比较/远端compile，启动后核所属PID/CWD/argv/GPU/log增长。非法角色/祖先、非有限成本/权重、输出碰撞或无法形成合法prediction时保留；partial predictions不得接truth，不覆盖/自动重启，UNKNOWN先查原PID/root。科学低分不停止健康任务。原所选基线及延后native-energy草稿状态保持，完整目标ACTIVE。
