# PAMF 独立度量残差＋OE软结构：源侧预登记

Run `20261030-phase3-star-metric-residual-s2026092701-r01`；实际执行code `310588cbd8c79bc4c687939ab49202ea422e8a2c`，唯一owner `codex-phase3-metric-residual-20261030`。固定网络/损失/采样见 `docs/PAMF_PHASE3_METRIC_RESIDUAL_PLAN_20261030.md`，唯一两row矩阵维护于experiment.json。主M_LOE，匹配M_L；两组相同zero-last 640→128GELU→160/102688、init2701、L采样、2220步/constantlr1e−4/freshAdamWwd1e−4/clip5。F0/default6head不训练、不替换；open z=unit(u+g(c))，没有0.1硬残差上界、另训identityhead或maxmember关系头。

源L每步192、six×15cells各2＋每类两轮转格，Lrng3017；M_LOE再取OE64，每15域4＋四轮转格，OErng3018独立。L监督度量KL只跨RX且跨day、正target按合法同旧TX域等权；主另加OE frozenF0 softneighborhood KL权1、T.1、teacher stopgrad，candidate与anchor域平衡，不输入OE TX/K、不假定其它OE为异设备。M_L embedding不打开OE值，但后续两者density均用OEoutside-six，不能称全流程无OE或同FLOP。

共同prepare首先严格加载原F0 actualcheckpoint、两条syntheticG10 noquery smoke，PASS即导出既有L/OE特征对。SourceL6300/OE6134来自protected-cache code7dd90671968ddf58aab5a5b302b4d56371c8b9c9原F0，原u160＋branchunit c640；无R1/640density权重继承。新aux网络随机zero-last起点，训练前只做不更新参数的feature梯度smoke，不占2220steps。每row固定终点保存新CP、严格重载＋synthetic_open2；映射L/OE→D160density90/32/25Lloydseed3019，共享covshrink.1。source/model全部冻结后才V风险校准，τ后才novel；原F0/defaulthead身份逐条保持。

V27000（旧六各4500）/novel3150（19-15/19-16/7-1各1050）的u160/c640/defaultlogits复用contribution-density codeedc28ad00baace5e4b9567fbf3391154bf6c34b4；固定F0density reference为其中col0原样复用，只比较不影响新fit/cal。L6300旧六各1050、每TX×RX×day70；sourceRX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，day2021-03-08/15/23。sourceRX/日期已见、novel历史曝光，非newRX/TX盲测。原Us56700、target U/Q和正式RX24-18不读，不回收历史query。

原F0为Tri-SIG-TIME SSL E50/11100/default6head/232993，sourcecontract392005和DURA v1.0祖先保持。全部postPhase1输入原固定G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、receiver2027/scene2026103001/channel2026104001，原一次RMS/单received不变，P0不是clean。无新物理记录/信道生成/clean/P0实值/真实F0/G前向；仅common syntheticF0两行，各arm另有OPEN feature smoke及真实训练/映射计数，均单列不混为原F0续训。

每row训练终点立即自动eval/独立truthlast，不等另一row；无模型/V/NV/目标早停。两row结果不回流健康训练。独立CPUscore只用保存的learned160重建density，全部本row预测完整后才唯一读novel truth；主报宏AK/RU/H、CK、原同P1 F0损失、AUROC/OSCR、完整TX/RX/day/scene与reference配对。M_LOE known42＋整体sourceRU90只是进入下一次跨RX预登记的必要早筛；控制不能替主，source通过不等于最终目标。完整跨RX星地/更高拒识/零标签DA/发现/匿名注册继续ACTIVE，AN/HOW本轮N/A。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-residual-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_metric_residual_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-residual-s2026092701-r01-release/runtime.json`。
- 独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-residual-s2026092701-r01`；commonprepare后最多两个GPUtrain/eval，启动actualfree≥4096MiB、每卡新增本任务≤2。单独CUDAempty/BLAS1 CPUscore一次最多1，其他用户不停止/修改/重启。
- 实际focused验证：{"model_train": {"tests": 9, "subtests": 16, "seconds": 5.92}, "eval_score": {"unique_tests_verified": 7, "unique_subtests": 11, "bridge_test_passed": 1, "bridge_seconds": 2.82, "integration_first_passed": 5, "integration_first_subtests": 10, "integration_first_seconds": 10.2, "collision_assertion_targeted_recheck_passed": 1, "targeted_subtests": 4, "targeted_seconds": 3.65, "subtests_overlap_not_double_counted": true, "production_changed_for_assertion": false, "legacy_np_trapz_deprecations": 648}, "launcher": {"tests": 5, "seconds": 3.11}, "AST_files": 10, "real_data_weights_truth_read_in_local_tests": 0}；10AST/runtime、唯一直接P0/P1 review PASS。一次N607preflight PASS，只读已验证角色/来源metadata、文件存在及freeVRAM，IQ/private truth值0。
- 单archive SHA比较/远端compile及首步actualcheckpoint smoke通过后立即继续；独立PID/CWD/argv/GPU/log-growth与最终artifact读回；不增加hash/receipt/审批链。
- 预期commonderived cache、完整2220loss/indices/first-two-grad与finalCP、各row实际CP smoke、density/25fit/τ/knownrisk、全部predictions和独立score以及observations。weights/IQ/sourcefeaturebanks/原log/预测不入Git。
- 非法来源/角色、非有限、输出碰撞、无法合法预测或系统技术错误保留；不评分不完整预测，不自动重启，UNKNOWN先核原PID/root。科学负结果不停止健康peer、不过早终止固定预算。此时仅预登记，尚无本轮实际新训练/性能。


## 实际启动与独立观察

VERIFIED／RUNNING（依据所存live观察时刻）。实际code310588cbd8c79bc4c687939ab49202ea422e8a2c；512文件/data0 archive SHA8a9ab6ac6e72cd1928eef956605fc3ce1fd044402329cdedcd0fdbd5c958d288单次比较、remote compile/runtime PASS。独立观察driver754875，M_L train755051/CUDA4/GPU340MiB，M_LOE train755054/CUDA2/GPU344MiB，真实CWD/完整argv/GPU及训练日志增长；离散显存不当峰值。其他用户停止/修改/重启0。

共同prepare actual原F0 syntheticG10两行noquery smoke先于L/OE派生缓存，status PASS后继续；原F0真实新增前向0、G0。此时两组真实NN进度分别800/2220与500/2220，原F0参数没有进入新训练，全部更新仅102688参数的开放网络；按固定预算继续，训练终点后自动每row eval/truthlast，未得到新性能结果，不按训练loss宣布成功。完整目标ACTIVE。


## 完整源侧结果


**执行VERIFIED，科学结论NO_PROMOTION。** 两臂各2220开放网络更新、全部源预测/独立评分完成。主M_LOE的A_K90.418519%、C_K98.411111%、原F0损失0.562963pp通过known保护，但R_U23.174603%、H36.893313%均低于固定F0 density参考；主也没有优于匹配M_L的R_U23.682540%。不换主、不晋级目标或多seed，完整目标ACTIVE。

Run `20261030-phase3-star-metric-residual-s2026092701-r01`，actualcode `310588cbd8c79bc4c687939ab49202ea422e8a2c`。固定设计见 `docs/PAMF_PHASE3_METRIC_RESIDUAL_PLAN_20261030.md`；一份experiment.json维护两row矩阵。原Tri-SIG-TIME SSL/default6head和DURA v1.0默认身份未由本轮辅助负结果更换。

## 数据和权限

| 角色 | 物理记录数 | 本轮使用 |
| --- | ---: | --- |
| source L | 6300，旧六TX各1050 | 两arm开放embedding的真旧类度量监督，随后known密度 |
| source OE | 6134，21辅助设备 | 仅M_LOE embedding软结构；两arm冻结后background密度，真实OE TX/K隐藏 |
| known V | 27000，旧六TX各4500 | 全部模型冻结后，仅既定known风险/六阈值 |
| source novel | 3150，19-15/19-16/7-1各1050 | τ冻结后预测，全部本row预测完整后独立truth-last |
| 原Us | 56700，Phase1祖先 | 本轮不读、不解封TX |
| target U/Q及正式RX24-18 | 本轮0 | 无新目标适应/发现/注册/评分 |

源RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。L/novel每TX×RX×day70、V每TX×RX×day300；四source角色物理互斥，共42584既有记录，无新增物理样本。OE保持原容量，不按隐藏TX均衡。sourceRX/日期已见、novel历史开发曝光，不是跨未见RX、新TX盲测或未来Q。

全部postPhase1输入沿固定G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，郊区/城区各0.5及原六分量面积概率；receiver2027/scene2026103001/channel2026104001，原一次unitRMS/单received不变。P0是星地received，不是clean。本轮只消费既有原F0特征/输出，不生成新IQ或信道、不读P0/clean实值、不运行G。

L/OE原u160/四贡献来自protected-cache actualcode7dd90671968ddf58aab5a5b302b4d56371c8b9c9，V/novel的u160/c640/defaultlogits来自contribution-density actualcodeedc28ad00baace5e4b9567fbf3391154bf6c34b4；固定参考为该父的原F0 density列0/model/τ，全部公共数组逐值复用。原F0 E50/11100/232993、sourcecontract392005和DURA祖先匹配，不加载R1或640密度作为新NN训练父。

本路线利用源L真标签、原Phase1监督知识和postfit OE成员角色，只能称目标端零人工TX标签。M_L仅embedding训练不读OE，不能把整个控制称为无OE。

## 实际网络、目标与训练

新增开放网络640→128 Linear/GELU→160 Linear、102688参数，最后层权重/bias零初始化；z=unit(u+g_theta(c))，无0.1硬残差界。原F0/default6head没有进入训练容器，新NN不输出六类身份，不存在额外identityhead或maxmember关系头。旧身份始终复制同P1原F0默认argmax。

两臂同seed2701/freshAdamWlr1e−4/wd1e−4/clip5/constantLR/2220steps。L每步192：六类×15域各2＋每类两个轮转格各1，Lrng3017；M_LOE每步另OE64：15域各4＋四轮转格各1，OErng3018独立。L索引全部2220步在两臂逐值相同，格内无放回、每step无重复；独立metadata重放/域配额验证全部通过。两臂unique L6300，主unique OE6134，M_L embedding OE0。

L监督度量KL只在跨RX且跨day candidates上计算，同旧TX positive按域等权；candidate质量按公共域均衡。主另加T.1的OE frozen-u软邻域KL，权重1、teacher stopgrad；没有未知硬同类边或把其它OE都当异设备。两目标合成loss公式及全部梯度范数已从完整trace核对。

全部4440步连续/有限，原F0新增native更新0；合成梯度probe不更新参数，不冒充真实训练steps。首步hidden梯度0（last0初始化）、last≈0.780197，第二步hidden≈0.00241；两臂hidden非零2219步、last非零2220步，clip实际0次。因此没有以断梯度、少跑预算或静默跳过OE解释本轮负结果。

| arm | trace量 | 全2220步均值 | 首222步均值 | 末222步均值 |
| --- | --- | ---: | ---: | ---: |
| M_L | loss | 0.452326303 | 0.617783955 | 0.413177004 |
| M_L | L_metric | 0.452326303 | 0.617783955 | 0.413177004 |
| M_L | OE_KL | N/A | N/A | N/A |
| M_L | L_feature_drift | 0.256668920 | 0.114083588 | 0.254029425 |
| M_L | OE_feature_drift | N/A | N/A | N/A |
| M_L | gradient_norm_before_clip | 0.249032305 | 0.436157101 | 0.219844376 |
| M_L | hidden_gradient | 0.110728292 | 0.083752918 | 0.134110293 |
| M_L | last_gradient | 0.218402853 | 0.425025791 | 0.172433478 |
| M_LOE | loss | 0.504020246 | 0.646341558 | 0.463329172 |
| M_LOE | L_metric | 0.480311219 | 0.627012549 | 0.440966072 |
| M_LOE | OE_KL | 0.023709027 | 0.019329010 | 0.022363099 |
| M_LOE | L_feature_drift | 0.134310675 | 0.079630801 | 0.162501897 |
| M_LOE | OE_feature_drift | 0.069202954 | 0.045050594 | 0.096089602 |
| M_LOE | gradient_norm_before_clip | 0.252175697 | 0.389668212 | 0.260189104 |
| M_LOE | hidden_gradient | 0.119977832 | 0.077755139 | 0.164046064 |
| M_LOE | last_gradient | 0.217149782 | 0.378644967 | 0.200583405 |

L_metric降低不代表拒识改善。主OE_KL首末均值0.019329→0.022363，并未趋零；OE feature drift0.045051→0.096090也不能证明未知身份保持。两个目标不同，未按总loss排序选模或宣称充分收敛；固定终点不早停/追加训练。

## 自动冻结预测与主指标

共同prepare actual原F0 syntheticG10两行noquery smoke PASS后立即生成L/OE派生feature cache。各row训练终点严格保存/重载真实102688参数checkpoint，synthetic_open2推理smoke后投影L/OE到z160；再同族D160 Gaussian density90known/32background/25Lloydseed3019/covshrink.1。全部统计冻结后，V仅known42/common-tail六predicted-class τ；τ固定后才novel特征映射与预测。独立CUDAempty CPUscorer只从保存z重建density/raw/margins，不重载NN、fit或选τ。

原F0同P1闭集90.981481%=24565/27000，所有新row默认身份与它逐条相同。A_K按旧六TX宏平均、误拒计错；R_U按三未知TX宏平均；H=2 A_K R_U/(A_K+R_U)，C_K不是身份正确率。

| 系统 | A_K% | R_U% | H% | C_K% | 原F0损失pp | 宏AUROC% | 宏OSCR% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定原F0 density | 90.374074 | 28.095238 | 42.864791 | 98.325926 | 0.607407 | 91.705905 | 84.743671 |
| M_L | 90.388889 | 23.682540 | 37.531544 | 98.329630 | 0.592593 | 88.611874 | 81.910218 |
| M_LOE | 90.418519 | 23.174603 | 36.893313 | 98.411111 | 0.562963 | 86.840988 | 80.545005 |

主正确接收24413、接收26571、拒绝429；相对原正确24565少152，另277个被拒者原本身份错误。未知拒对730/3150、误接2420。M_L正确24405/接收26549/拒对746；固定F0参考正确24401/接收26548/拒对885。

主−M_L：正确接收gain38/loss30/net8，接受净+22，未知拒对gain206/loss222/net−16。主−固定F0：正确净+12、接受净+23、未知拒对净−155。A_K微小收益来自拒绝集合变化，不是原身份分类器被改善。两row known42均通过，但sourceRU90失败；sourcegate只是进入下一次跨RX预登记的早筛，当前无eligibility，控制不能替主。

## 逐真实设备、RX、日期和场景

| arm | 未知TX | 分母 | 拒对 | R_U% |
| --- | --- | ---: | ---: | ---: |
| M_L | 19-15 | 1050 | 369 | 35.142857 |
| M_L | 19-16 | 1050 | 198 | 18.857143 |
| M_L | 7-1 | 1050 | 179 | 17.047619 |
| M_LOE | 19-15 | 1050 | 288 | 27.428571 |
| M_LOE | 19-16 | 1050 | 150 | 14.285714 |
| M_LOE | 7-1 | 1050 | 292 | 27.809524 |

| arm | 旧TX | 分母 | 原闭集% | A_K% | C_K% | 额外失去正确数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M_L | 14-10 | 4500 | 89.044444 | 88.333333 | 97.955556 | 32 |
| M_L | 14-7 | 4500 | 80.333333 | 79.733333 | 98.422222 | 27 |
| M_L | 20-15 | 4500 | 95.911111 | 95.400000 | 98.777778 | 23 |
| M_L | 20-19 | 4500 | 85.466667 | 84.466667 | 97.666667 | 45 |
| M_L | 6-15 | 4500 | 96.333333 | 95.866667 | 97.977778 | 21 |
| M_L | 8-20 | 4500 | 98.800000 | 98.533333 | 99.177778 | 12 |
| M_LOE | 14-10 | 4500 | 89.044444 | 88.422222 | 98.022222 | 28 |
| M_LOE | 14-7 | 4500 | 80.333333 | 79.777778 | 98.511111 | 25 |
| M_LOE | 20-15 | 4500 | 95.911111 | 95.377778 | 98.622222 | 24 |
| M_LOE | 20-19 | 4500 | 85.466667 | 84.533333 | 98.111111 | 42 |
| M_LOE | 6-15 | 4500 | 96.333333 | 95.844444 | 97.955556 | 22 |
| M_LOE | 8-20 | 4500 | 98.800000 | 98.555556 | 99.244444 | 11 |

以下RX均已见，day1/2/3为2021-03-08/15/23；六scene为同固定MIX切片，不是新抽六次信号。全部required TX与分组覆盖完整，不剔除难设备或缺类填0。

| arm | 分组 | known分母 | novel分母 | A_K% | R_U% | C_K% | 原F0损失pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M_L | rx/1 | 5400 | 630 | 92.944444 | 36.031746 | 99.592593 | 0.148148 |
| M_L | rx/3 | 5400 | 630 | 83.944444 | 14.920635 | 95.925926 | 1.203704 |
| M_L | rx/4 | 5400 | 630 | 91.129630 | 22.857143 | 98.592593 | 0.500000 |
| M_L | rx/6 | 5400 | 630 | 92.629630 | 30.317460 | 99.574074 | 0.203704 |
| M_L | rx/8 | 5400 | 630 | 91.296296 | 14.285714 | 97.962963 | 0.907407 |
| M_L | scene/practical_high | 1103 | 133 | 97.290637 | 31.420373 | 98.440971 | 0.622243 |
| M_L | scene/practical_mid | 1983 | 256 | 96.966315 | 24.302919 | 98.495943 | 0.476348 |
| M_L | scene/practical_low_suburban | 10304 | 1203 | 95.004051 | 28.518297 | 98.524045 | 0.600867 |
| M_L | scene/practical_high_urban | 1171 | 130 | 96.291949 | 27.678682 | 98.393228 | 0.660453 |
| M_L | scene/practical_mid_urban | 1979 | 236 | 91.108546 | 19.967949 | 98.202873 | 0.605285 |
| M_L | scene/practical_low_urban | 10460 | 1192 | 83.036762 | 17.930082 | 98.104604 | 0.585797 |
| M_L | day/1 | 9000 | 1050 | 90.522222 | 30.000000 | 98.555556 | 0.511111 |
| M_L | day/2 | 9000 | 1050 | 90.411111 | 23.333333 | 98.700000 | 0.488889 |
| M_L | day/3 | 9000 | 1050 | 90.233333 | 17.714286 | 97.733333 | 0.777778 |
| M_LOE | rx/1 | 5400 | 630 | 92.925926 | 29.523810 | 99.611111 | 0.166667 |
| M_LOE | rx/3 | 5400 | 630 | 84.037037 | 17.301587 | 96.074074 | 1.111111 |
| M_LOE | rx/4 | 5400 | 630 | 91.296296 | 16.507937 | 98.907407 | 0.333333 |
| M_LOE | rx/6 | 5400 | 630 | 92.611111 | 34.285714 | 99.500000 | 0.222222 |
| M_LOE | rx/8 | 5400 | 630 | 91.222222 | 18.253968 | 97.962963 | 0.981481 |
| M_LOE | scene/practical_high | 1103 | 133 | 97.247635 | 39.582337 | 98.747620 | 0.665245 |
| M_LOE | scene/practical_mid | 1983 | 256 | 97.009412 | 28.311486 | 98.530272 | 0.433251 |
| M_LOE | scene/practical_low_suburban | 10304 | 1203 | 95.023472 | 26.739359 | 98.532222 | 0.581446 |
| M_LOE | scene/practical_high_urban | 1171 | 130 | 96.529697 | 27.255048 | 98.884157 | 0.422705 |
| M_LOE | scene/practical_mid_urban | 1979 | 236 | 91.057264 | 21.955128 | 98.356426 | 0.656567 |
| M_LOE | scene/practical_low_urban | 10460 | 1192 | 83.067770 | 16.426638 | 98.177169 | 0.554789 |
| M_LOE | day/1 | 9000 | 1050 | 90.588889 | 28.285714 | 98.677778 | 0.444444 |
| M_LOE | day/2 | 9000 | 1050 | 90.400000 | 22.380952 | 98.666667 | 0.500000 |
| M_LOE | day/3 | 9000 | 1050 | 90.266667 | 18.857143 | 97.888889 | 0.744444 |

| arm | RX×scene | known分母 | novel分母 | A_K% | R_U% | C_K% | 原F0损失pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M_L | rxscene/1/practical_high | 202 | 20 | 98.569157 | 55.555556 | 99.019608 | 0.000000 |
| M_L | rxscene/1/practical_mid | 437 | 46 | 98.377657 | 46.396741 | 99.808429 | 0.191571 |
| M_L | rxscene/1/practical_low_suburban | 2042 | 245 | 97.209469 | 44.215686 | 99.750943 | 0.049603 |
| M_L | rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 24.761905 | 100.000000 | 0.000000 |
| M_L | rxscene/1/practical_mid_urban | 401 | 44 | 93.945919 | 39.398836 | 99.543379 | 0.000000 |
| M_L | rxscene/1/practical_low_urban | 2082 | 253 | 86.274833 | 26.047899 | 99.422595 | 0.289405 |
| M_L | rxscene/3/practical_high | 208 | 25 | 91.827145 | 9.259259 | 96.417092 | 1.975799 |
| M_L | rxscene/3/practical_mid | 407 | 54 | 92.393662 | 14.692982 | 94.542014 | 1.777778 |
| M_L | rxscene/3/practical_low_suburban | 2007 | 239 | 90.662738 | 19.364416 | 96.265838 | 1.233071 |
| M_L | rxscene/3/practical_high_urban | 241 | 29 | 90.600777 | 21.759259 | 94.517193 | 1.228307 |
| M_L | rxscene/3/practical_mid_urban | 388 | 44 | 86.627523 | 6.250000 | 96.276173 | 1.508192 |
| M_L | rxscene/3/practical_low_urban | 2149 | 239 | 74.021757 | 11.710460 | 95.975064 | 0.857548 |
| M_L | rxscene/4/practical_high | 213 | 26 | 97.207456 | 36.111111 | 97.613960 | 0.641026 |
| M_L | rxscene/4/practical_mid | 403 | 48 | 97.880771 | 25.925926 | 99.184735 | 0.268817 |
| M_L | rxscene/4/practical_low_suburban | 2049 | 248 | 95.574923 | 26.658053 | 98.687606 | 0.479650 |
| M_L | rxscene/4/practical_high_urban | 238 | 30 | 96.476988 | 35.833333 | 98.196248 | 1.803752 |
| M_L | rxscene/4/practical_mid_urban | 392 | 40 | 92.208995 | 29.629630 | 99.126984 | 0.297619 |
| M_L | rxscene/4/practical_low_urban | 2105 | 238 | 84.078294 | 14.185435 | 98.378926 | 0.467604 |
| M_L | rxscene/6/practical_high | 240 | 24 | 100.000000 | 48.518519 | 100.000000 | 0.000000 |
| M_L | rxscene/6/practical_mid | 365 | 47 | 98.543783 | 28.113553 | 100.000000 | 0.000000 |
| M_L | rxscene/6/practical_low_suburban | 2104 | 240 | 96.549904 | 35.259740 | 99.706719 | 0.195612 |
| M_L | rxscene/6/practical_high_urban | 233 | 27 | 98.791974 | 38.611111 | 100.000000 | 0.000000 |
| M_L | rxscene/6/practical_mid_urban | 408 | 57 | 92.960665 | 19.477671 | 99.474548 | 0.000000 |
| M_L | rxscene/6/practical_low_urban | 2050 | 235 | 85.763262 | 25.492503 | 99.274726 | 0.329176 |
| M_L | rxscene/8/practical_high | 240 | 38 | 98.016317 | 15.873016 | 99.164578 | 0.396825 |
| M_L | rxscene/8/practical_mid | 371 | 61 | 97.272834 | 22.283272 | 98.609640 | 0.260417 |
| M_L | rxscene/8/practical_low_suburban | 2102 | 231 | 94.914730 | 15.709688 | 98.135723 | 1.054304 |
| M_L | rxscene/8/practical_high_urban | 223 | 22 | 95.911730 | 12.121212 | 98.757702 | 0.387597 |
| M_L | rxscene/8/practical_mid_urban | 390 | 51 | 89.858663 | 12.026144 | 96.463507 | 1.302828 |
| M_L | rxscene/8/practical_low_urban | 2074 | 227 | 85.436653 | 11.478769 | 97.674277 | 0.953099 |
| M_LOE | rxscene/1/practical_high | 202 | 20 | 98.569157 | 45.238095 | 100.000000 | 0.000000 |
| M_LOE | rxscene/1/practical_mid | 437 | 46 | 98.569227 | 45.289025 | 100.000000 | 0.000000 |
| M_LOE | rxscene/1/practical_low_suburban | 2042 | 245 | 97.210763 | 31.813725 | 99.702931 | 0.048309 |
| M_LOE | rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 34.761905 | 100.000000 | 0.000000 |
| M_LOE | rxscene/1/practical_mid_urban | 401 | 44 | 93.945919 | 34.917044 | 99.771689 | 0.000000 |
| M_LOE | rxscene/1/practical_low_urban | 2082 | 253 | 86.180364 | 22.648698 | 99.327031 | 0.383874 |
| M_LOE | rxscene/3/practical_high | 208 | 25 | 92.065690 | 35.185185 | 97.043234 | 1.737253 |
| M_LOE | rxscene/3/practical_mid | 407 | 54 | 92.418914 | 20.614035 | 94.580494 | 1.752525 |
| M_LOE | rxscene/3/practical_low_suburban | 2007 | 239 | 90.715986 | 18.718589 | 96.321573 | 1.179823 |
| M_LOE | rxscene/3/practical_high_urban | 241 | 29 | 90.600777 | 22.222222 | 95.439128 | 1.228307 |
| M_LOE | rxscene/3/practical_mid_urban | 388 | 44 | 86.382425 | 8.333333 | 95.767662 | 1.753290 |
| M_LOE | rxscene/3/practical_low_urban | 2149 | 239 | 74.196460 | 15.144196 | 96.205977 | 0.682845 |
| M_LOE | rxscene/4/practical_high | 213 | 26 | 96.800952 | 42.129630 | 97.670419 | 1.047530 |
| M_LOE | rxscene/4/practical_mid | 403 | 48 | 97.880771 | 12.592593 | 99.184735 | 0.268817 |
| M_LOE | rxscene/4/practical_low_suburban | 2049 | 248 | 95.814122 | 19.853575 | 99.023704 | 0.240451 |
| M_LOE | rxscene/4/practical_high_urban | 238 | 30 | 97.883914 | 18.611111 | 99.603175 | 0.396825 |
| M_LOE | rxscene/4/practical_mid_urban | 392 | 40 | 92.208995 | 22.962963 | 99.404762 | 0.297619 |
| M_LOE | rxscene/4/practical_low_urban | 2105 | 238 | 84.170591 | 9.849850 | 98.667261 | 0.375308 |
| M_LOE | rxscene/6/practical_high | 240 | 24 | 100.000000 | 58.888889 | 100.000000 | 0.000000 |
| M_LOE | rxscene/6/practical_mid | 365 | 47 | 98.543783 | 52.179487 | 100.000000 | 0.000000 |
| M_LOE | rxscene/6/practical_low_suburban | 2104 | 240 | 96.547473 | 40.014542 | 99.560051 | 0.198043 |
| M_LOE | rxscene/6/practical_high_urban | 233 | 27 | 98.791974 | 45.648148 | 100.000000 | 0.000000 |
| M_LOE | rxscene/6/practical_mid_urban | 408 | 57 | 92.960665 | 31.080542 | 99.712644 | 0.000000 |
| M_LOE | rxscene/6/practical_low_urban | 2050 | 235 | 85.716446 | 21.386222 | 99.198437 | 0.375992 |
| M_LOE | rxscene/8/practical_high | 240 | 38 | 98.016317 | 24.338624 | 99.164578 | 0.396825 |
| M_LOE | rxscene/8/practical_mid | 371 | 61 | 97.272834 | 23.565324 | 98.609640 | 0.260417 |
| M_LOE | rxscene/8/practical_low_suburban | 2102 | 231 | 94.728113 | 21.802556 | 97.985950 | 1.240922 |
| M_LOE | rxscene/8/practical_high_urban | 223 | 22 | 95.911730 | 15.151515 | 99.185053 | 0.387597 |
| M_LOE | rxscene/8/practical_mid_urban | 390 | 51 | 89.858663 | 15.947712 | 96.987539 | 1.302828 |
| M_LOE | rxscene/8/practical_low_urban | 2074 | 227 | 85.443117 | 13.482429 | 97.680741 | 0.946635 |

主下一校准档在RX3/practical_high因宏2pp身份保护失效，非总体C_K90绑定。实际当前/下一档损失及下一档接受率如下，仅描述既定校准路径，不能推广成所有其他配置不可能；未根据novel真值扫描/选新τ。

| arm | 绑定组 | 当前损失pp | 下一档损失pp | 下一档C_K% |
| --- | --- | ---: | ---: | ---: |
| M_L | rxscene/3/practical_mid | 1.777778 | 2.034188 | 94.285603 |
| M_LOE | rxscene/3/practical_high | 1.737253 | 2.124850 | 96.655637 |

## 完整拟合、成本和运行证据

| arm | Lloyd首步weightedSSE | 第25步weightedSSE | 上升次数 |
| --- | ---: | ---: | ---: |
| M_L | 123.477057540 | 109.354467749 | 0 |
| M_LOE | 97.571137083 | 85.842212030 | 0 |

两次25步均finite/32活跃mode/0空mode。训练loss和SSE下降不能替代验证R_U。每密度数组payload364340B，每arm保存固定参考＋新模型共728680B、阈值96B；无逐source物理bank部署。每实例原F0＋open算术335681参数，若加固定G为602632，仅算术组合，不是本轮实际端到端G+F时延证明。

真实新F0前向0、common syntheticF02、G0。新open训练feature行994560（两臂L各426240＋主OE142080），冻结后映射85168，合计1079728；合成open训练12＋eval4=16，optimizer实际4440、native0。没有新增物理记录，原F0/缓存成本保留。pipeline199.695379s，非B1部署延迟；未测峰值显存。

actualcode `310588cbd8c79bc4c687939ab49202ea422e8a2c`；512文件/data0 archive单SHA8a9ab6ac6e72cd1928eef956605fc3ce1fd044402329cdedcd0fdbd5c958d288比较、remote compile/runtime PASS。model/train9tests+16sub/5.92s；eval/score7unique+11unique sub经严格NNCP桥接与碰撞断言定点修正后通过，生产未为断言改动；launcher5/3.11s，10AST/runtime/唯一直接P0/P1 review PASS。local648条既有np.trapz弃用警告保留，实际8份完整stdout无warning/error/nonfinite。

独立抓到driver754875、M_L train755051/CUDA4/GPU340MiB、M_LOE train755054/CUDA2/GPU344MiB的live PID/CWD/完整argv/GPU及step800/500后到完整2220日志增长；另抓到M_LOE eval757731/CUDA4/GPU332MiB。显存是离散观察，不当峰值。短prepare/M_L eval/两个CPUscore未全抓live，仅由jobs/完整log/终态产物核实，不为补观察重跑。终态无本run live/GPU，其他用户停止/修改/重启0。

root独立读回4440连续steps/匹配Lindices、两row聚合维度/FP64/字节、27000条原身份、source margin路由、固定control所有公共数组与计数；NNweights/IQ/sourcefeaturebanks未下载，独立分析只用公共metadata重放采样，不读novel原私有truth、不重fit/score/扫τ。原log、预测/特征保持私有，不入Git。

## 下一设计的具体缺口

当前L_metric仅涉及L–L，OE_KL仅涉及OE–OE，没有显式约束L–OE距离。单看这些表示层几何项，分别旋转L和OE可保持组内余弦，却改变两组相对位置；共享网络及零残差初始化提供耦合，但不等同于显式成员分离。这是下一假设的结构理由，不把本轮拒识下降归为已证实的单一原因。

下一候选将把**L–OE有效异设备关系**纳入度量候选：合法OE均在原六类之外，所以L与OE可作负关系；不把OE彼此当异设备，不输入其TX/K。同时保留OE内部软结构的匹配有/无对照，保护后续未知发现所需的类内结构。此项只使用已授权source outside-six弱监督，不新增未知设备真标签或目标数据。

还将预登记“绝对known支持密度”和“known/OE密度比”的读出对照：OE训练分布不等于所有未见设备分布，密度比高不能单独解释为落在known支持域。唯一主、网络/损失/预算与所有阈值规则须在新run固定，再执行同源反证，不由本轮novel分数挑参数/分支或改阈值。下一候选尚未实现或验证有效。

完整跨未见RX星地拒识、known保护、无标签DA/发现和未来匿名持续注册仍未完成；最新v3及更高拒识研究目标保持，未来AN80/REG旧损≤2pp待实测，AN/HOW本轮N/A。目标ACTIVE。
