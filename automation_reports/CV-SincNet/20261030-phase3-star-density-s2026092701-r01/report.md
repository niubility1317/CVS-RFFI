# PAMF 多中心密度候选：源侧预登记

Run `20261030-phase3-star-density-s2026092701-r01`，实际执行代码 `b8163a03e8d3d299127c7914272a2d866f92c89a`，唯一owner `codex-phase3-density-20261030`。方法固定于 `docs/PAMF_PHASE3_DENSITY_PLAN_20261030.md`，矩阵唯一维护于 experiment.json：一个原模型seed、known_only匹配无OE控制、唯一主density_ratio，均使用相同源L拟合的已知分布。背景capacity32/25Lloyd/seed2026103019、90已知域mode、共享协方差shrink.1、径向modevariance收缩160；概率质量/体积/尺度与白化行列式真实参与似然，不是旧8NN换名。

L6300（原六类各1050、每TX×5RX×3day70）提供已知真标签，OE6134只提供outside-six角色与公共RX/day。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；OE真实TX/K不进入算法。knownV27000仅冻结分布后固定known风险42/六阈值校准；novel3150（19-15/19-16/7-1各1050）在两套预测全固定后独立评分。Us/target U/Q/正式RX24-18不读；sourceRX已见、novel已有开发曝光，不称未见RX或新未知盲测。

全部数据沿已有G10MIX01/P1：600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect，receiver2027/scene2026103001/channel2026104001；不生成新IQ、不抽第二份信道、P0不是clean。原F0为Tri-SIG-TIME SSL E50/11100/default6head/232993，source split392005、原L6300/Us56700/V27000祖先不变。严格加载原F0后，两条合成G10无query smoke为首步，PASS立即继续源缓存拟合；原参数/buffers/默认身份保持，G不加载。

源特征复用protected-cache实际code7dd90671968ddf58aab5a5b302b4d56371c8b9c9中原F0 rawfused160；V/novel特征/defaultlogits复用local-support code84cdc012f7d9641b283c407f6048bfe4e6e21cf6的original_F0/predict。两来源的原F0/DURA与source角色句柄完全一致；不加载旧拒识raw_scores/scores/τ或source评分，local-support不作为训练父。fit/query同FP64单位化。新真实F前向0、合成F2、G0；原预训练和缓存成本保留。

分布参数只拟合L/OE；control先冻结且没有OE参数，主再读取OE并冻结。随后V只用既定known风险校准，两个τ全冻结后才读取novel特征。已知身份直接复用原F0默认头，与同P1 baseline逐条保持。只导出聚合o/W/mode均值/尺度/权重/六τ，不部署source逐行ID/feature/IQ。

独立CUDAempty/BLAS1 CPU scorer在全部两policy预测完整后唯一读取novel truth，给出宏AK/RU/H、CK、原F0损失、AUROC/OSCR、逐TX/RX/day/scene与配对。源晋级只为known42且整体source宏RU≥90%的早筛，其他切片完整诊断；它不代表已达到最终未见RX/逐主场景门槛。控制不能替主、结果不回流任何分布/阈值/seed/预算。通过后才另预登记跨RX星地开发验证，不自动扩多seed。完整无标签适应/发现/未来注册目标仍ACTIVE，AN/HOW均N/A。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-density-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_density_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-density-s2026092701-r01-release/runtime.json`。
- 独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-density-s2026092701-r01`；一个GPU worker仅原F0 smoke与随后源统计/预测，actualfree≥4096MiB、每卡新增own≤2；独立CPU评分。其他用户停止/修改/重启0。
- 聚焦实际验证：{"core": {"unique_tests_verified": 7, "subtests": 13, "first_invocation_passed": 6, "first_invocation_seconds": 1.17, "targeted_floating_sum_assertion_recheck_passed": 1, "targeted_seconds": 0.58, "production_algorithm_changed_for_assertion": false}, "run_score": {"tests": 11, "subtests": 19, "seconds": 6.34, "legacy_np_trapz_deprecation_warnings": 528}, "launcher": {"tests": 8, "seconds": 3.01}, "AST_files": 8, "real_IQ_weights_truth_read_for_local_tests": 0}；八AST及真实runtime校验通过，唯一直接P0/P1审查PASS。一次N607预检已PASS，只读metadata/文件/显存、IQ/private truth值0。
- 一个release archive、一次local/remote SHA比较与远端compile；实际首步checkpoint smoke、启动后PID/CWD/argv/GPU/log及终态独立读回。统计拟合25次聚类不冒称神经更新或训练epoch。
- 预期launch/pipeline/config/log、smoke、聚合state/fittrace、六阈值、两policy source预测、独立score/results/complete及observations。数据/权重/原始log/预测不入Git。
- 非法来源/角色、非有限、碰撞或技术失败保留；不对不完整预测评分，不自动重试或调参，UNKNOWN先核原PID/root；不操作其他owner。此处仅预登记，尚无实际新性能结果。


## 完整源侧结果


**执行 VERIFIED，科学结论 NO_PROMOTION，完整 Phase3 保持 ACTIVE。** 主density_ratio保住已知类：A_K90.374074%、C_K98.325926%、原F0损失0.607407pp；未知R_U28.095238%、H42.864791%，未通过源端90%早筛。没有扩展跨未见RX测试或多seed，也没有执行本轮无标签适应、发现或注册。

Run `20261030-phase3-star-density-s2026092701-r01`，实际执行代码 `b8163a03e8d3d299127c7914272a2d866f92c89a`。固定方法见 `docs/PAMF_PHASE3_DENSITY_PLAN_20261030.md`；一行计算矩阵中的两个policy为known_only控制与唯一主density_ratio，不按结果换主。当前PAMF的原Tri-SIG-TIME SSL/defaulthead及DURA v1.0身份不由本轮辅助负结果更换。

## 数据条件与标签权限

| 角色 | 物理记录数 | 拟合/校准/评价用途 |
| --- | ---: | --- |
| source L | 6300，旧六TX各1050 | 已知分布拟合；真旧类标签与公共RX/day可见 |
| source OE | 6134，来自21辅助设备 | 仅主背景密度拟合；outside-six成员监督、公共域可见，真实TX/K隐藏 |
| known V | 27000，旧六TX各4500 | 模型全冻结后，仅固定known风险及六predicted-class阈值校准 |
| source novel | 3150，19-15/19-16/7-1各1050 | 两policy全部预测固定后独立truth-last；不拟合分布或阈值 |
| 原Us | 56700，原Phase1祖先 | 本轮不读、不解封TX标签 |
| target U/Q、正式RX24-18 | 本轮0 | 无新目标访问、训练、适应或评分 |

四个当前source角色物理互斥，共42584个既有记录，没有新增物理样本。源RX索引1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。每L/novel设备×RX×day70条，每V旧设备×RX×day300条；OE依原实际容量，不按隐藏设备配额均衡。源RX/日期已经见过，source novel已有历史开发暴露，不能称跨未见RX、未来Q或独立新设备盲测。

所有Phase1后source观测均为既有固定G10 MIX01星地P1：600km/25MHz/2.45GHz、residual/post_sync/no additionalEQ、referenceSNR30dB、reflect；郊区/城区各0.5及原六分量面积概率。receiver2027/scene2026103001/channel2026104001保持，信道前一次unitRMS、输出不逐样本RMS。P1为同一received经冻结DURA；P0始终是星地received，不是clean。本轮只读取缓存特征/默认输出，不生成IQ、不重抽信道。

L/OE原F0 rawfused160来自protected-cache code7dd90671968ddf58aab5a5b302b4d56371c8b9c9；V/novel原unit160/defaultlogits来自local-support code84cdc012f7d9641b283c407f6048bfe4e6e21cf6。两个来源的原F0/DURA及source角色句柄匹配。旧raw_scores/scores/τ没有被索引读取；local-support不是训练父。原F0为Tri-SIG-TIME SSL E50/11100/default6head/232993，source split392005、原L/Us/V和祖先保持；本轮新神经/optimizer更新0。所有fit/query输入使用同一FP64单位化。

本方法依赖L真标签、OE成员角色及原Phase1有标签知识，只能称目标端零人工TX标签路线，不能称整个系统完全无标签。

## 实际方法与执行

只在L上计算六类×五RX×三day90个中心、pooled格内协方差及10%各向同性收缩，Cholesky白化。已知mode各1/90质量，径向方差按70/(70+160)向L共同方差收缩。每格仅70条，没有各自拟合160维完整协方差。

主模型再读取OE，在同一空间按15个公共RX/day域各1/15质量加权，固定32容量、seed2026103019、25次weighted kmeans/Lloyd。背景mode质量取样本域平衡总权重，径向方差按effective-count/(effective-count+160)向L方差收缩。真实OE设备数/K、novel标签或V标签均不参与这些参数。32个统计mode不等于32个未知设备。

实际似然计算mode权重、`D log variance`体积项、平方距离/variance与白化行列式；known_only取log p_L，主取log p_L−log p_OE。两模型的全部已知聚合数组逐值一致，控制没有OE参数。source统计量冻结后，V按同一既定42组known风险/common-tail规则分别校准两个policy的六阈值；两套阈值全部冻结后才消费novel特征。身份始终为原F0默认argmax，与同P1 cached baseline的27000条V逐条一致。

独立CPU评分器在全部预测完整后读取唯一novel truth；它从聚合模型重建raw分数与条件margin，所有冻结数组和计数一致。仅部署中心/白化/尺度/权重/六τ，不部署source逐行ID、特征或IQ。

## 主指标与匹配收益

原F0同P1闭集宏准确率90.981481%（24565/27000）。A_K对真实旧TX等权，包含误拒损失；R_U对三未知TX等权；H=2 A_K R_U/(A_K+R_U)。所有比例按0至1计算后转百分比。C_K只表示未拒绝，不等于身份准确率。

| policy | A_K% | R_U% | H% | C_K% | 原F0损失pp | 宏AUROC% | 宏OSCR% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| known_only | 90.285185 | 21.015873 | 34.095309 | 98.433333 | 0.696296 | 88.451313 | 81.898517 |
| density_ratio | 90.374074 | 28.095238 | 42.864791 | 98.325926 | 0.607407 | 91.705905 | 84.743671 |

两policy的known42均通过；整体、所有RX/scene的未知拒识均远低于90%。本轮晋级仅是known42＋整体source R_U90早筛，RX/scene/day/cross完整诊断；source通过也不等于跨RX/逐正式主场景最终达标。本轮主早筛失败，不扩大测试。

主正确接收24401、总接收26548、拒绝452；相对原F0少164条正确接收，余下被拒者288条原本身份错误。未知拒对885/3150、误接2265。对照正确接收24377、总接收26577、未知拒对662。主−控制正确接收gain89/loss65/net24（A_K+0.088889pp），接收gain136/loss165/net−29，未知拒对gain411/loss188/net223（R_U+7.079365pp），H+8.769482pp。两policy独立按同一规则校准，收益不是单独更换raw score而保持同一τ的消融。

## 逐真实设备

| policy | 未知TX | 分母 | 拒对 | R_U% |
| --- | --- | ---: | ---: | ---: |
| known_only | 19-15 | 1050 | 336 | 32.000000 |
| known_only | 19-16 | 1050 | 129 | 12.285714 |
| known_only | 7-1 | 1050 | 197 | 18.761905 |
| density_ratio | 19-15 | 1050 | 412 | 39.238095 |
| density_ratio | 19-16 | 1050 | 125 | 11.904762 |
| density_ratio | 7-1 | 1050 | 348 | 33.142857 |

收益主要来自19-15和7-1；19-16由12.285714%降到11.904762%，仍最困难。不能把总体改善扩展成每个未知设备均改善。

| policy | 旧TX | 分母 | 原闭集% | A_K% | C_K% | 额外失去正确数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| known_only | 14-10 | 4500 | 89.044444 | 88.288889 | 98.111111 | 34 |
| known_only | 14-7 | 4500 | 80.333333 | 79.577778 | 98.355556 | 34 |
| known_only | 20-15 | 4500 | 95.911111 | 95.444444 | 99.200000 | 21 |
| known_only | 20-19 | 4500 | 85.466667 | 84.333333 | 97.577778 | 51 |
| known_only | 6-15 | 4500 | 96.333333 | 95.577778 | 97.933333 | 34 |
| known_only | 8-20 | 4500 | 98.800000 | 98.488889 | 99.422222 | 14 |
| density_ratio | 14-10 | 4500 | 89.044444 | 88.377778 | 97.977778 | 30 |
| density_ratio | 14-7 | 4500 | 80.333333 | 79.644444 | 98.288889 | 31 |
| density_ratio | 20-15 | 4500 | 95.911111 | 95.400000 | 98.733333 | 23 |
| density_ratio | 20-19 | 4500 | 85.466667 | 84.422222 | 97.800000 | 47 |
| density_ratio | 6-15 | 4500 | 96.333333 | 95.933333 | 98.066667 | 18 |
| density_ratio | 8-20 | 4500 | 98.800000 | 98.466667 | 99.088889 | 15 |

## RX、场景和日期

以下仍是已见sourceRX分组，不是独立目标跨RX实验。day1/2/3对应2021-03-08/15/23；六practical为本固定MIX中的切片，不是另抽六次完整观测。

| policy | 分组 | known分母 | novel分母 | A_K% | R_U% | H% | C_K% | 原F0损失pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| known_only | rx/1 | 5400 | 630 | 92.851852 | 25.396825 | 39.884459 | 99.462963 | 0.240741 |
| known_only | rx/3 | 5400 | 630 | 83.555556 | 16.507937 | 27.569092 | 95.870370 | 1.592593 |
| known_only | rx/4 | 5400 | 630 | 91.277778 | 15.714286 | 26.812551 | 99.259259 | 0.351852 |
| known_only | rx/6 | 5400 | 630 | 92.629630 | 28.253968 | 43.300409 | 99.648148 | 0.203704 |
| known_only | rx/8 | 5400 | 630 | 91.111111 | 19.206349 | 31.725020 | 97.925926 | 1.092593 |
| known_only | scene/practical_high | 1103 | 133 | 97.485730 | 30.782720 | 46.790554 | 98.959966 | 0.427150 |
| known_only | scene/practical_mid | 1983 | 256 | 96.796526 | 25.805728 | 40.748106 | 98.626098 | 0.646137 |
| known_only | scene/practical_low_suburban | 10304 | 1203 | 94.849133 | 25.527036 | 40.227517 | 98.456412 | 0.755785 |
| known_only | scene/practical_high_urban | 1171 | 130 | 96.203758 | 23.535241 | 37.818566 | 98.486196 | 0.748644 |
| known_only | scene/practical_mid_urban | 1979 | 236 | 90.851379 | 19.508547 | 32.119964 | 98.311418 | 0.862452 |
| known_only | scene/practical_low_urban | 10460 | 1192 | 82.992210 | 14.312124 | 24.414017 | 98.335798 | 0.630349 |
| known_only | day/1 | 9000 | 1050 | 90.466667 | 28.095238 | 42.875197 | 98.677778 | 0.566667 |
| known_only | day/2 | 9000 | 1050 | 90.377778 | 19.142857 | 31.593843 | 98.744444 | 0.522222 |
| known_only | day/3 | 9000 | 1050 | 90.011111 | 15.809524 | 26.895186 | 97.877778 | 1.000000 |
| density_ratio | rx/1 | 5400 | 630 | 92.851852 | 44.126984 | 59.823434 | 99.500000 | 0.240741 |
| density_ratio | rx/3 | 5400 | 630 | 84.055556 | 19.365079 | 31.478099 | 95.962963 | 1.092593 |
| density_ratio | rx/4 | 5400 | 630 | 91.185185 | 18.888889 | 31.295050 | 98.722222 | 0.444444 |
| density_ratio | rx/6 | 5400 | 630 | 92.537037 | 37.301587 | 53.170286 | 99.407407 | 0.296296 |
| density_ratio | rx/8 | 5400 | 630 | 91.240741 | 20.793651 | 33.868673 | 98.037037 | 0.962963 |
| density_ratio | scene/practical_high | 1103 | 133 | 97.247635 | 39.917105 | 56.601194 | 98.309256 | 0.665245 |
| density_ratio | scene/practical_mid | 1983 | 256 | 97.008269 | 30.778876 | 46.730920 | 98.481781 | 0.434394 |
| density_ratio | scene/practical_low_suburban | 10304 | 1203 | 94.965540 | 33.253844 | 49.258842 | 98.474262 | 0.639378 |
| density_ratio | scene/practical_high_urban | 1171 | 130 | 96.372465 | 30.039638 | 45.802481 | 98.476428 | 0.579938 |
| density_ratio | scene/practical_mid_urban | 1979 | 236 | 91.057264 | 27.905983 | 42.719790 | 98.416643 | 0.656567 |
| density_ratio | scene/practical_low_urban | 10460 | 1192 | 83.027589 | 20.836800 | 33.313233 | 98.107015 | 0.594969 |
| density_ratio | day/1 | 9000 | 1050 | 90.511111 | 30.285714 | 45.385194 | 98.544444 | 0.522222 |
| density_ratio | day/2 | 9000 | 1050 | 90.400000 | 30.761905 | 45.903474 | 98.622222 | 0.500000 |
| density_ratio | day/3 | 9000 | 1050 | 90.211111 | 23.238095 | 36.956352 | 97.811111 | 0.800000 |

主30个RX×scene的R_U范围9.722222%至60.317460%，全部低于90%。全部required TX/候选覆盖完整，无缺类填0或剔除失败设备。

| RX×scene | known分母 | novel分母 | 主A_K% | 主R_U% | 主C_K% | 主原F0损失pp | 控制R_U% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rxscene/1/practical_high | 202 | 20 | 98.569157 | 60.317460 | 99.509804 | 0.000000 | 23.809524 |
| rxscene/1/practical_mid | 437 | 46 | 98.334486 | 53.972498 | 99.765258 | 0.234742 | 40.297937 |
| rxscene/1/practical_low_suburban | 2042 | 245 | 97.065636 | 53.284314 | 99.608999 | 0.193436 | 29.436275 |
| rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 57.619048 | 100.000000 | 0.000000 | 28.095238 |
| rxscene/1/practical_mid_urban | 401 | 44 | 93.945919 | 47.845292 | 99.771689 | 0.000000 | 29.691877 |
| rxscene/1/practical_low_urban | 2082 | 253 | 86.175921 | 31.406819 | 99.227012 | 0.388317 | 18.511352 |
| rxscene/3/practical_high | 208 | 25 | 92.065690 | 31.481481 | 94.909606 | 1.737253 | 10.185185 |
| rxscene/3/practical_mid | 407 | 54 | 92.671440 | 28.289474 | 94.555241 | 1.500000 | 17.872807 |
| rxscene/3/practical_low_suburban | 2007 | 239 | 90.713526 | 21.605605 | 96.266596 | 1.182282 | 21.055412 |
| rxscene/3/practical_high_urban | 241 | 29 | 90.600777 | 30.092593 | 94.110689 | 1.228307 | 16.203704 |
| rxscene/3/practical_mid_urban | 388 | 44 | 86.627523 | 9.722222 | 96.514268 | 1.508192 | 13.194444 |
| rxscene/3/practical_low_urban | 2149 | 239 | 74.149658 | 14.285028 | 96.152696 | 0.729647 | 12.637854 |
| rxscene/4/practical_high | 213 | 26 | 97.207456 | 29.629630 | 98.076923 | 0.641026 | 39.351852 |
| rxscene/4/practical_mid | 403 | 48 | 97.880771 | 11.111111 | 99.184735 | 0.268817 | 18.888889 |
| rxscene/4/practical_low_suburban | 2049 | 248 | 95.623372 | 24.151593 | 98.787408 | 0.431200 | 19.575797 |
| rxscene/4/practical_high_urban | 238 | 30 | 97.378864 | 17.777778 | 99.098124 | 0.901876 | 22.777778 |
| rxscene/4/practical_mid_urban | 392 | 40 | 92.208995 | 20.185185 | 99.126984 | 0.297619 | 6.111111 |
| rxscene/4/practical_low_urban | 2105 | 238 | 84.079795 | 14.730802 | 98.477701 | 0.466103 | 9.048155 |
| rxscene/6/practical_high | 240 | 24 | 100.000000 | 46.296296 | 100.000000 | 0.000000 | 46.296296 |
| rxscene/6/practical_mid | 365 | 47 | 98.543783 | 45.989011 | 100.000000 | 0.000000 | 43.131868 |
| rxscene/6/practical_low_suburban | 2104 | 240 | 96.497423 | 41.628403 | 99.459677 | 0.248093 | 34.752657 |
| rxscene/6/practical_high_urban | 233 | 27 | 98.413186 | 39.814815 | 99.621212 | 0.378788 | 34.814815 |
| rxscene/6/practical_mid_urban | 408 | 57 | 92.960665 | 40.460526 | 100.000000 | 0.000000 | 23.554625 |
| rxscene/6/practical_low_urban | 2050 | 235 | 85.621622 | 28.903946 | 99.035365 | 0.470817 | 16.828220 |
| rxscene/8/practical_high | 240 | 38 | 97.619492 | 30.606431 | 98.767753 | 0.793651 | 28.042328 |
| rxscene/8/practical_mid | 371 | 61 | 97.272834 | 27.228327 | 98.609640 | 0.260417 | 20.390720 |
| rxscene/8/practical_low_suburban | 2102 | 231 | 94.822138 | 24.166188 | 98.177553 | 1.146897 | 20.774299 |
| rxscene/8/practical_high_urban | 223 | 22 | 95.911730 | 12.121212 | 99.185053 | 0.387597 | 14.646465 |
| rxscene/8/practical_mid_urban | 390 | 51 | 89.617117 | 25.393877 | 96.504447 | 1.544374 | 23.433093 |
| rxscene/8/practical_low_urban | 2074 | 227 | 85.487777 | 14.046998 | 97.829619 | 0.901974 | 14.865256 |

## 风险绑定、排序与拟合轨迹

主common-tail q=77/4570，下一档38/2253将令RX3/practical_high的原F0损失由1.737253pp升到2.143757pp；额外被拒正确数由4升5，该组有208条known，各TX分母不同，不能直接用5/208替代其宏损失。当前全组最低C_K94.110689%，绑定的是局部身份2pp保护，并非总体接收90%或accept-all回退。控制下一档在RX3/practical_mid由1.749029pp升到2.026807pp而失效。这里仅评价既定common-tail路径，不证明所有六阈值配置不可能。

主宏AUROC91.705905%不证明存在同时达到R_U90和2pp保护的运行点。设B=0.909814814815，正确known子集至少需接收(B−.02)/B=0.978017504580；R_U90/95对应正确子集AUROC必要下界0.880215754122/0.929116629351。由冻结OSCR/B得到0.931438677432，虽高于必要界，必要条件仍非充分条件。没有根据novel truth扫描或选择新阈值，也没有以AUC替换固定运行点验收。

完整25次Lloyd weighted SSE由141.299655441降至123.470173179，0次上升，25次始终32个活跃mode、空mode0。15个域总质量各1/15，mode权重0.007562466至0.125935448、原始格内记录31至434。L pooled径向方差0.454202；实际known方差0.340875338至0.686046222，背景方差0.259832056至0.964698315，白化矩阵条件数22.506828。这些稳定拟合事实不代表未知拒识已成功；全部25点及42组风险、TX/RX/day/scene配对已完整读取并存于私有analysis。

## 成本、验证与下一步

已知聚合322868B，主364340B（新增41472B），两policy12阈值96B；每个模型仅8/11个聚合数组，无逐source物理行。原F0新神经参数0，固定DURA未加载；原预训练和缓存成本保留。新真实F前向0、合成F2、G0；原F0实际checkpoint两条合成G10无query smoke先于真实source特征访问、状态保持、PASS继续。两policy CPU拟合4.535895s、分数计算7.436353s/60300 policy行（30150既有V/NV记录），完整pipeline26.898966s；不是B1部署时延，也未测峰值显存或新组合端到端G+F。

code `b8163a03e8d3d299127c7914272a2d866f92c89a`；493文件/data0 release archive单SHA ddf5742eb42c565c6c53457d45751cc29a4ed66d4482a4f970b30b3c6cb44a9c 比较、remote compile/runtime PASS。core7个唯一测试与13sub经首轮6通过/单项浮点断言修正后定点复核通过；run/score11+19sub/6.34s、launcher8/3.01s，唯一直接P0/P1 review PASS。local测试528条既有np.trapz弃用警告如实保留，实际run三份stdout无warning/error/nonfinite。

独立抓到driver371424及predict371513的live PID/CWD/完整argv/CUDA6/GPU350MiB，日志从0至485/550B增长；350MiB为离散观察而非峰值。独立CPU score373362未抓到live，仅由jobs完整argv/PID、日志及终态产物核实，不为补观察重跑。终态无本run live/GPU；其他用户停止/修改/重启0。聚合字段/字节/FP64/两policy已知数组相等、两套分数阈值路由、27000条原身份及全部计数均独立数组读回验证；root未读取novel私有标签或重新选τ。

本轮证实概率密度建模相对匹配control有实际收益，但在同一融合后160维特征上仍不能达到拒识目标。下一轮增加开放分支的互补信息，优先检验融合前时频/PA/reference/signature四贡献：原身份仍走原fused160/defaulthead，开放分支固定逐分支单位化后拼接，与当前fused160密度作为匹配信息对照。新方向尚未实现/预登记/验证；需要按实际构造器确认合法导出和统计维度成本，不从本轮novel真值挑分支、权重、中心数、方差或阈值。不会按失败结果直接晋级目标适应，也不会把同池未知近邻结构当注册成功。

跨未见RX星地C_K/R_U90、A_K相对同输入原F0损失≤2pp，进一步高拒识目标及零目标TX标签域适应、发现和未来匿名注册继续；未来A_N80/REG旧损≤2pp未执行，A_N/H_OW为N/A。完整目标ACTIVE，当前只完成本轮源侧反证矩阵。
