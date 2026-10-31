# 仅适应已知密度的无标签适应结果（2026-10-31）

**结论：NO_PROMOTION。** 唯一主 `selective_predicted_ratio` 相对原 predicted 在星地多正确接收 13 条，但未知拒对净减 40 条，A_K +2.2569 pp、R_U −13.8889 pp、H −5.2531 pp。相对 SHARED 的局部拒识改善也未达绝对门槛。原 `predicted_ratio` 重点候选和 `pooled_ratio` 冻结对照保持，完整目标 ACTIVE。

Run `20261031-phase3-selective-uda-s2026092701-r01`，实际代码 `0f1fda4c8c543394a938bd75dcb5cadbc29c6491`。固定公式与权限见[计划](PAMF_PHASE3_SELECTIVE_UDA_PLAN_20261031.md)；全分组、三配对和两份完整轨迹见[评分](../local_artifacts/pamf_phase3_selective_uda_20261031_r01/score/results.json)及[私有分析](../local_artifacts/pamf_phase3_selective_uda_20261031_r01/analysis.json)。

## 方法与合法输入

冻结原 F0 Tri-SIG-TIME SSL E50/11100/default6、source-only r03、DURA v1 E20/4940、known90/OE32 GMM、normalizer 和两组六阈值。本轮拟合 p_U(w)=0.5p_K(a⊙w+b)prod(a)+0.5p_OE(w)，a>0，无变换后单位化。每条 U/Q 同时计算两分支，不按真实 known/unknown 路由；OE 对新域不变仅是待检验的统计近似。

SELECTIVE 从 a=1/b=0 独立做 20 次 MAP-EM，λ=160 固定单位先验。M 步使用 known 软责任及 N_eff，而非全部 N；N_eff 不是真实已知样本数。只有 known 分支有 affine Jacobian，因此主比值为 ℓ_c(a⊙w+b)+Σlog a−ℓ_OE(w)，该 Jacobian 不能消去。F0 默认身份 c 始终不变。

SHARED 精确复用上一 diagonal/73f 的 DIAG 状态、20 步诊断及对应预测，只作同 320 参数/20 步/λ160 控制，不作新初始化或教师。source 旧 RX 保持恒等，六列重复原两列；新 RX 的 clean/star 共用同一个 star-U SELECTIVE 状态，clean 不参与学习。整个部署状态有 U 祖先，不能因原神经模型 source-only 而称系统 source-only。

只读 old73f←d72 中适应前 r03/DURA 的 U 四字段 ids/default_logits/closed/signal_features，F0 logits 不拟合，真实 TX/K/配额隐藏；旧 Q/score 不进拟合。三实际 checkpoint synthetic 各2→原状态/阈值冻结→U384 拟合新状态并冻结→六列×三条件完整预测→独立 CPU 三真值。U 真值读取 0，无新 V 校准。

## 全部 18 运行点

逐真实 TX 宏平均百分比；A_K=旧类正确接收，C_K=接收，R_U=三非旧类拒识，H=harmonic(A_K,R_U)。source 重复行是接收机条件路由的恒等结果，不是源域适应收益。

| 条件 | 方法 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|
| source | pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| source | shared_pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | shared_predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| source | selective_pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | selective_predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| clean_DURA | pooled_ratio | 68.9236 | 85.4167 | 99.6528 | 81.4874 |
| clean_DURA | predicted_ratio | 70.8333 | 87.3264 | 100 | 82.9268 |
| clean_DURA | shared_pooled_ratio | 73.2639 | 90.7986 | 94.4444 | 82.5167 |
| clean_DURA | shared_predicted_ratio | 73.6111 | 90.6250 | 98.2639 | 84.1695 |
| clean_DURA | selective_pooled_ratio | 75.1736 | 92.5347 | 91.6667 | 82.6049 |
| clean_DURA | selective_predicted_ratio | 74.8264 | 91.6667 | 98.2639 | 84.9583 |
| star_DURA | pooled_ratio | 63.0208 | 84.8958 | 76.3889 | 69.0639 |
| star_DURA | predicted_ratio | 64.0625 | 85.5903 | 77.0833 | 69.9723 |
| star_DURA | shared_pooled_ratio | 66.8403 | 89.9306 | 60.4167 | 63.4663 |
| star_DURA | shared_predicted_ratio | 66.6667 | 89.2361 | 61.8056 | 64.1441 |
| star_DURA | selective_pooled_ratio | 66.8403 | 89.9306 | 59.0278 | 62.6916 |
| star_DURA | selective_predicted_ratio | 66.3194 | 88.8889 | 63.1944 | 64.7192 |

## 完整拟合与三组配对

新 SELECTIVE 和继承 SHARED 的 iteration0 至20共42条记录均完整读取，全部有限、目标非下降。新目标从181695.05281440576到183843.67566550476，最终尺度0.6481322740055696至1.3438724975871188、offset norm1.8977789113737333。known posterior mass由0.5075638714到0.6516592685，只是模型软质量，不是真实 known 比例，不能据此猜测吸收了哪些未知 TX。

SHARED 记录为上一 run 已存20步，目标终值186962.9954438197，并非本轮重拟合。两个目标对应不同统计模型，不能仅按目标绝对值排序候选；新 SELECTIVE 不从 SHARED 初始化，也未依据 Q 早停或选迭代。

主对原 predicted：clean 正确23增/0失、接收+25（2条错误）、未知拒对0增/5失；star 正确13增/0失、接收+19（6条错误）、未知拒对1增/41失、净−40。旧默认头不改，这些是原本被拒的正确样本重新接收，不是闭集身份识别率提高。

主对 matched SHARED：clean 正确8增/1失、净+7，接收净+6，未知4增/4失、净0；star 正确2增/4失、净−2，接收−2，未知19增/15失、净+4。即星地 R_U +1.3889 pp、A_K −0.3472 pp；相对较弱控制的局部权衡不能替代对原基线和绝对门槛的验证。

主对 selective pooled：clean 正确1增/3失、净−2，接收−5，未知20增/1失、净+19；star 正确3增/6失、净−3，接收−6，未知19增/7失、净+12。源域三配对分别为恒等或既有 pooled/predicted 差异，全 gain/loss 及逐组结果保留，控制不能替换唯一主。

主 clean 接收528/576、正确431/576，含97条错误旧身份；star 接收512/576、正确382/576，含130条错误旧身份。未知拒对283/288、182/288。A_K、C_K 和未知拒识必须合并评估，不能把接收变多或相对 SHARED 的 R_U 微升单列为成功。

## 绝对门槛、场景与范围

同 DURA 原 F0 闭集 source/clean/star为90.9815%/80.0347%/72.9167%；主 clean/star 损失仍5.2083/6.5972 pp，均超2。clean C_K91.6667%、R_U98.2639%单项达标仍不够；star C_K88.8889%、R_U63.1944%失败。source known12通过、42组未全过，且原 V 为校准复用数据，不代替未来测试。

主 star 六景（high/mid/low suburban、high/mid/low urban）A_K为72.8571/73.0556/70.5876/75.3307/77.3611/55.1180%，R_U为N/A/60.1852/71.4715/86.6667/70/47.4462%。high缺19-15，R_U/H为N/A，完整主景覆盖未达；不能删景、补零或改用present-only认证。

主 star 的19-15/19-16/7-1拒识为63/96、51/96、68/96。AUROC86.1437%、OSCR64.1674%只描述固定排序，不能证明现有阈值达到90%拒识或用于挑新阈值。全部TX/RX/day/scene、source42及逐类Wilson区间保留，区间不构成单开发RX之外的泛化置信证明。

源契约392005/VALIDATED_ONCE原L6300/U_s56700/OE6134为祖先，本轮新读0；V27000为额外已知标签的原校准/源诊断，NV3150三未知各1050。源5RX/三日、旧六类及G10 MIX01信道沿[计划](PAMF_PHASE3_SELECTIVE_UDA_PLAN_20261031.md)。U为RX24-1/03-08，future Q为同RX/03-15的同864物理记录，old576/nonold288、九类各96，与U互斥；三条件均DURA、clean仅测试，不重抽观测或新增物理样本。

## 实际成本与执行证据

新增目标统计拟合20步、320有效参数，U384全量20次更新为7680条次，另有iteration0诊断。SHARED的20步已计入上一 run 两臂40步研究成本，继承而不重计；本轮不是40次新拟合。新NN/源Gaussian/normalizer/τ拟合及真实G/F0/signal前向、sourceL/OE IQ均0，实际三CP synthetic各2另计。

SELECTIVE数组2740 B、继承SHARED2716 B；原density364340 B、NN834665参数不变。旧12个唯一τ96 B无新V校准，六列重复存36值288 B、单主6值48 B；不部署source/U逐行bank。原F0 11100/r03+2220/MLP2220/G4940、源密度及源/U/Q缓存成本继承。

pipeline ANALYZED_DEVELOPMENT，两job COMPLETE/returncode0；predict/CPUscore/总过程10.401474/5.277220/15.790489 s，非逐包时延。三份完整生产stdout共7行，warning/error/traceback为0。三次观察捕获driver1290055/predictor1290081的CWD/完整argv/CUDA1及GPU UUIDa2d039ff的256/406 MiB两次样本，非峰值；predictlog492→522→636 B，CPU1290333未独立捕获live，仅pipeline gpu=None、代码CUDAempty与完成产物。终态live/ownedGPU空、foreign0，不补遥测重跑。

9 unique＋29 sub PASS：worker7＋22sub/8.17s有focused_tests.log（696个旧trapz提示为合成测试），root launcher2＋7sub/0.032s为已记录console证据；AST7/镜像8、唯一P0/P1 PASS。单包677files/data0，SHA `cb160f20d2db10d480322bd6d3588795624a43a33813a89a5de95f6086bb2079` 比较、compile/runtime VERIFIED。

NV/Q历史曝光、单RX/seed、日期与源重合，formal RX24-18未读；没有发现或匿名注册，A_N/H_OW为N/A。本固定选择性仿射未通过联合标准，不推广为所有选择性UDA无效；原重点/对照保留、完整目标ACTIVE。本次整理未重新拟合、评分、补读真值或遥测。
