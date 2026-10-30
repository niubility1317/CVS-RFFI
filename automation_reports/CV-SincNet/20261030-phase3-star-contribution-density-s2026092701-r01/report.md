# PAMF 融合前四贡献密度：源侧预登记

Run `20261030-phase3-star-contribution-density-s2026092701-r01`，实际执行代码 `edc28ad00baace5e4b9567fbf3391154bf6c34b4`，唯一owner `codex-phase3-contribution-density-20261030`。完整固定方法见 `docs/PAMF_PHASE3_CONTRIBUTION_DENSITY_PLAN_20261030.md`，矩阵唯一维护于experiment.json。一行/两policy：唯一主contribution640_density_ratio，固定控制fused160_density_ratio，控制精确复用父density codeb8163a03e8d3d299127c7914272a2d866f92c89a主model/六τ/预测，不能重fit/recal或按新NV换主。

原F0为完整Tri-SIG-TIME SSL E50/11100/default6head/232993；source split392005、原L6300/Us56700/V27000与DURA v1.0祖先保持。新主只读protected-cache code7dd90671968ddf58aab5a5b302b4d56371c8b9c9原F0四贡献，不加载R1/unused adapter。Source L6300旧六类各1050，OE6134只outside-six/publicRXday，无真实OE TX/K；knownV27000仅模型冻结后known风险42校准；novel3150三个设备各1050仅最终独立truth。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。sourceRX/日期已见、novel历史曝光，不能称新RX或新未知盲测。Us/target U/Q/正式RX24-18新读取0。

全部输入沿已有固定G10 MIX01 P1：600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect、receiver2027/scene2026103001/channel2026104001，原单记录单received与DURA缓存不变；P0不是clean。没有重抽信道、clean/P0实值或G前向。

首步actual原F0 checkpoint syntheticG10两条无query smoke，同pass直接native hooks捕四贡献/原logits/CFO mask，finally清hooks、状态保持，PASS即继续。每分支FP64unit、zero分支保0、全zero报错，concat全局unit→FP32[640]；core统一FP64再归一化。主L/OE拟合D640收缩Gaussian mixture，90known/32background/25Lloyd/seed3019/covshrink.1，真实mode质量/体积/尺度及Jacobian进入densityratio。新D640与固定D160控制的方差收缩、自由度/存储/计算不同，不能声称纯单项融合因果或同成本。

主模型冻结后V27000仅前向/固定known42阈值；主τ冻结后才novel3150前向。真实新增F30150+synthetic2，G/NN/optimizer更新0。新pass默认身份/ID/metadata与控制逐条核对，父control τ/model/score不改变、不影响主拟合。仅保存聚合state，不部署source逐行物理bank。全部两policy预测完整后独立CUDAempty/BLAS1 scorer唯一读novel truth，重建statraw/margins并报宏AK/RU/H、CK/原F0损失、AUROC/OSCR及全部TX/RX/day/scene/配对。

源晋级=known42 AND 整体sourceRU≥90%，是早筛；其他source切片完整诊断，source通过不替代最终跨未见RX/逐主场景验收。控制不能替主，不按NV调分支/权重/容量/预算/τ。完整拒识、无标签适应/发现/未来匿名注册目标ACTIVE，AN/HOW均N/A。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-contribution-density-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_contribution_density_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-contribution-density-s2026092701-r01-release/runtime.json`。
- 独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-contribution-density-s2026092701-r01`；一个GPU forwardworker、actualfree≥4096MiB、每卡新增own≤2，随后独立CPU评分；其他用户停止/修改/重启0。
- 实际focused验证：{"core": {"tests": 11, "subtests": 23, "seconds": 1.11, "includes_original160_regression": true}, "run_score": {"unique_tests_verified": 8, "subtests": 13, "first_invocation_passed": 7, "first_seconds": 10.6, "matrix_roundoff_targeted_regression_passed": 1, "targeted_seconds": 4.21, "legacy_np_trapz_deprecation_warnings": 432}, "launcher": {"tests": 8, "seconds": 2.92}, "AST_files": 9, "real_IQ_weights_truth_read_for_local_tests": 0}；9AST/runtime PASS，唯一直接P0/P1 review PASS。一次N607preflight PASS，仅metadata/文件/显存，IQ/private truth值0。
- 单release archive SHA比较/远端compile，首步实际checkpoint smoke后立即继续，独立PID/CWD/argv/GPU/log和最终产物核验；不增加成员hash/receipt/审批。
- 预期launch/pipeline/config/log、smoke、models/25步fit、τ/calibration、known/novel预测、独立score complete/results/observations；大体积特征/原log/权重不入Git。
- 非法来源/角色、非有限、身份错位、碰撞或技术失败保留，不评分不完整预测、不自动重试；UNKNOWN先核原PID/root。此处仅预登记，没有新性能结果。


## 完整源侧结果


**执行VERIFIED，科学结论NO_PROMOTION。** 主contribution640的宏R_U16.158730%，低于固定fused160控制的28.095238%；H由42.864791%降到27.418664%。A_K90.440741%、C_K98.470370%和原F0损失0.540741pp通过已知保护，不能抵消未知拒识下降。主不能被控制替换，没有晋级跨未见RX/多seed/无标签适应或注册；完整目标ACTIVE。

Run `20261030-phase3-star-contribution-density-s2026092701-r01`，实际执行code `edc28ad00baace5e4b9567fbf3391154bf6c34b4`。原固定方法为 `docs/PAMF_PHASE3_CONTRIBUTION_DENSITY_PLAN_20261030.md`；控制逐值复用已完成density父codeb8163a03e8d3d299127c7914272a2d866f92c89a的主模型、阈值和预测，没有按新结果挑选或重fit/recal。

## 数据、域和标签

| 角色 | 数量 | 本轮用途 |
| --- | ---: | --- |
| source L | 6300，旧六类各1050 | 主640分布拟合；旧TX真标签、公共RX/day可见 |
| source OE | 6134，来自21辅助设备 | 主背景密度拟合；仅outside-six/publicdomains，真实TX/K隐藏 |
| known V | 27000，旧六类各4500 | 模型冻结后的真实P1前向、主固定风险阈值校准 |
| source novel | 3150，19-15/19-16/7-1各1050 | 主τ冻结后真实P1前向；两policy全预测后独立truth-last |
| 原Us | 56700，Phase1祖先 | 本轮不读、不解封标签 |
| target U/Q及正式RX24-18 | 本轮0 | 无新目标访问/适应/发现/注册 |

四source角色物理互斥共42584个既有记录，无新增物理样本。源RX索引1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。L/novel每TX×RX×day70条、V每TX×RX×day300条。OE保持实际容量，不按隐藏TX均衡。源RX/日期已见，source novel已有历史开发曝光；本轮不是跨未见RX、未来Q或新未知设备盲测。

全部Phase1后输入沿已有G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/no additionalEQ、referenceSNR30dB/reflect；郊区/城区各0.5和六分量面积概率。receiver2027/scene2026103001/channel2026104001，信道前一次unitRMS、输出不逐样本RMS保持。P1为单份received经冻结DURA，P0不是clean。本轮不新生成received、不重抽信道、不读clean/P0实值、不运行G。

原F0为完整Tri-SIG-TIME SSL E50/11100/default6head/232993，source split392005及原L/Us/V祖先不变。L/OE四项贡献复用protected-cache code7dd90671968ddf58aab5a5b302b4d56371c8b9c9原F0同次特征；未加载R1或其他候选adapter。控制source-only density父与当前F0/DURA/rolehandles匹配，原数据和历史成绩保持身份。

本方法使用源L真标签、OE成员角色及原Phase1有标签知识；目标端零人工TX标签不等于整个系统无源监督。source novel标签从未进入模型/阈值/分支选择。

## 实际方法和对照

原reader同次native hooks捕获时频identity160、原gain加权PA160、原response_gain加权reference160、signature160；参考29坐标27保持屏蔽。四贡献分别FP64单位化，零分支保0，拼接后全局单位化转FP32[640]；core在fit/score统一FP64重归一化。没有学习分支权重。V/NV所有分支均非零，整体范数最大误差分别1.64555591375e-08/1.44168670335e-08，没有零特征补齐。

原默认身份由同次reader.classify_features(fused160)输出，不安装unused openparts、不训练另一个身份头。真实原checkpoint两条合成G10首步smoke通过，PASS继续L/OE拟合；所有真实V/NV前向的ID/metadata/defaulthead与固定控制核对。全部F0参数/buffers保持，实际剩余hook数未另存运行测量；清理由finally实现及聚焦测试覆盖，不能补称实际观测到hook数0。

主D640模型只在L拟合六类×五RX×三day90个mode，共享格内pooled covariance＋10%各向同性收缩、Cholesky白化；mode质量1/90、径向方差按70/(70+640)向L共同方差收缩。OE按15域各1/15质量，固定32容量、seed2026103019、25次weighted Lloyd，背景方差按n_eff/(n_eff+640)收缩。似然实际包含mode质量/体积/尺度/白化Jacobian，score=logp_L−logp_OE。

固定D160控制沿原统计模型及原τ。D640不仅多了贡献信息，还改变协方差几何、统计自由度、径向方差收缩α（0.304348→0.098592）与资源成本。该实验不能单独证明“融合前信息无用”或某一个分支导致拒识下降；结论仅针对本次固定表示与统计规则组合。

主L/OE统计先冻结，V只按既定42组known风险/common-tail校准六predicted-class阈值；主τ冻结后才消费novel P1。控制模型/τ/所有公共预测字段及分数列均与父产物逐值相等。全部两policy预测完成后独立CPU唯一连接novel truth，重建raw/条件margin并按冻结预测评分。

## 总体与配对

原F0同P1闭集宏准确率90.981481%=24565/27000。A_K按旧六TX宏平均、误拒计错；R_U按三个真实未知TX宏平均；H=2 A_K R_U/(A_K+R_U)。C_K只表示接收，不能替代正确身份。

| policy | A_K% | R_U% | H% | C_K% | 原F0损失pp | 宏AUROC% | 宏OSCR% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fused160_density_ratio | 90.374074 | 28.095238 | 42.864791 | 98.325926 | 0.607407 | 91.705905 | 84.743671 |
| contribution640_density_ratio | 90.440741 | 16.158730 | 27.418664 | 98.470370 | 0.540741 | 90.182108 | 83.377045 |

主正确接收24419、接收26587、拒绝413；原正确样本额外拒绝146，另267个被拒样本原本身份错误。主未知拒对509/3150、误接2641。控制正确接收24401/接收26548/未知拒对885。主−控制正确接收gain66/loss48/net18（+0.066667pp），接收净+39（+0.144444pp），未知拒对gain177/loss553/net−376（−11.936508pp），H−15.446127pp。

两policy均通过known42；主source整体未知失败。source gate只表示known42＋整体sourceRU90早筛，sourceRX/scene等切片完整报告；source通过也不替代跨未见RX或逐正式主场景验收。当前无source晋级，更无完整目标通过。

## 逐真实TX

| policy | 未知TX | 分母 | 拒对 | R_U% |
| --- | --- | ---: | ---: | ---: |
| fused160_density_ratio | 19-15 | 1050 | 412 | 39.238095 |
| fused160_density_ratio | 19-16 | 1050 | 125 | 11.904762 |
| fused160_density_ratio | 7-1 | 1050 | 348 | 33.142857 |
| contribution640_density_ratio | 19-15 | 1050 | 137 | 13.047619 |
| contribution640_density_ratio | 19-16 | 1050 | 217 | 20.666667 |
| contribution640_density_ratio | 7-1 | 1050 | 155 | 14.761905 |

19-16多拒对92条，但19-15少275、7-1少193，整体净少376。不能凭单个未知设备改善选择分支、权重或替换主模型。

| policy | 旧TX | 分母 | 原闭集% | A_K% | C_K% | 额外失去正确数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fused160_density_ratio | 14-10 | 4500 | 89.044444 | 88.377778 | 97.977778 | 30 |
| fused160_density_ratio | 14-7 | 4500 | 80.333333 | 79.644444 | 98.288889 | 31 |
| fused160_density_ratio | 20-15 | 4500 | 95.911111 | 95.400000 | 98.733333 | 23 |
| fused160_density_ratio | 20-19 | 4500 | 85.466667 | 84.422222 | 97.800000 | 47 |
| fused160_density_ratio | 6-15 | 4500 | 96.333333 | 95.933333 | 98.066667 | 18 |
| fused160_density_ratio | 8-20 | 4500 | 98.800000 | 98.466667 | 99.088889 | 15 |
| contribution640_density_ratio | 14-10 | 4500 | 89.044444 | 88.444444 | 97.911111 | 27 |
| contribution640_density_ratio | 14-7 | 4500 | 80.333333 | 79.800000 | 98.622222 | 24 |
| contribution640_density_ratio | 20-15 | 4500 | 95.911111 | 95.577778 | 99.155556 | 15 |
| contribution640_density_ratio | 20-19 | 4500 | 85.466667 | 84.333333 | 97.511111 | 51 |
| contribution640_density_ratio | 6-15 | 4500 | 96.333333 | 95.866667 | 98.133333 | 21 |
| contribution640_density_ratio | 8-20 | 4500 | 98.800000 | 98.622222 | 99.488889 | 8 |

## RX、日期和六场景

源RX均已见，day1/2/3对应2021-03-08/15/23。六场景是同固定MIX中的切片，不是六份新观测。全部required TX与分组覆盖完整，无缺类组被剔除或补0。

| policy | 分组 | known分母 | novel分母 | A_K% | R_U% | C_K% | 原F0损失pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused160_density_ratio | rx/1 | 5400 | 630 | 92.851852 | 44.126984 | 99.500000 | 0.240741 |
| fused160_density_ratio | rx/3 | 5400 | 630 | 84.055556 | 19.365079 | 95.962963 | 1.092593 |
| fused160_density_ratio | rx/4 | 5400 | 630 | 91.185185 | 18.888889 | 98.722222 | 0.444444 |
| fused160_density_ratio | rx/6 | 5400 | 630 | 92.537037 | 37.301587 | 99.407407 | 0.296296 |
| fused160_density_ratio | rx/8 | 5400 | 630 | 91.240741 | 20.793651 | 98.037037 | 0.962963 |
| fused160_density_ratio | scene/practical_high | 1103 | 133 | 97.247635 | 39.917105 | 98.309256 | 0.665245 |
| fused160_density_ratio | scene/practical_mid | 1983 | 256 | 97.008269 | 30.778876 | 98.481781 | 0.434394 |
| fused160_density_ratio | scene/practical_low_suburban | 10304 | 1203 | 94.965540 | 33.253844 | 98.474262 | 0.639378 |
| fused160_density_ratio | scene/practical_high_urban | 1171 | 130 | 96.372465 | 30.039638 | 98.476428 | 0.579938 |
| fused160_density_ratio | scene/practical_mid_urban | 1979 | 236 | 91.057264 | 27.905983 | 98.416643 | 0.656567 |
| fused160_density_ratio | scene/practical_low_urban | 10460 | 1192 | 83.027589 | 20.836800 | 98.107015 | 0.594969 |
| fused160_density_ratio | day/1 | 9000 | 1050 | 90.511111 | 30.285714 | 98.544444 | 0.522222 |
| fused160_density_ratio | day/2 | 9000 | 1050 | 90.400000 | 30.761905 | 98.622222 | 0.500000 |
| fused160_density_ratio | day/3 | 9000 | 1050 | 90.211111 | 23.238095 | 97.811111 | 0.800000 |
| contribution640_density_ratio | rx/1 | 5400 | 630 | 93.018519 | 19.206349 | 99.685185 | 0.074074 |
| contribution640_density_ratio | rx/3 | 5400 | 630 | 84.037037 | 12.063492 | 96.074074 | 1.111111 |
| contribution640_density_ratio | rx/4 | 5400 | 630 | 91.129630 | 21.111111 | 98.740741 | 0.500000 |
| contribution640_density_ratio | rx/6 | 5400 | 630 | 92.703704 | 21.111111 | 99.740741 | 0.129630 |
| contribution640_density_ratio | rx/8 | 5400 | 630 | 91.314815 | 7.301587 | 98.111111 | 0.888889 |
| contribution640_density_ratio | scene/practical_high | 1103 | 133 | 97.453942 | 27.690100 | 98.165912 | 0.458937 |
| contribution640_density_ratio | scene/practical_mid | 1983 | 256 | 97.007027 | 19.761720 | 98.746483 | 0.435636 |
| contribution640_density_ratio | scene/practical_low_suburban | 10304 | 1203 | 94.995883 | 20.844486 | 98.475609 | 0.609035 |
| contribution640_density_ratio | scene/practical_high_urban | 1171 | 130 | 96.203758 | 25.040258 | 98.305037 | 0.748644 |
| contribution640_density_ratio | scene/practical_mid_urban | 1979 | 236 | 90.953005 | 9.348291 | 98.213349 | 0.760826 |
| contribution640_density_ratio | scene/practical_low_urban | 10460 | 1192 | 83.191725 | 9.772287 | 98.506489 | 0.430833 |
| contribution640_density_ratio | day/1 | 9000 | 1050 | 90.611111 | 20.380952 | 98.744444 | 0.422222 |
| contribution640_density_ratio | day/2 | 9000 | 1050 | 90.477778 | 17.238095 | 98.811111 | 0.422222 |
| contribution640_density_ratio | day/3 | 9000 | 1050 | 90.233333 | 10.857143 | 97.855556 | 0.777778 |

| RX×scene | known分母 | novel分母 | 主A_K% | 主R_U% | 主C_K% | 主原F0损失pp | 控制R_U% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rxscene/1/practical_high | 202 | 20 | 98.569157 | 50.793651 | 98.569157 | 0.000000 | 60.317460 |
| rxscene/1/practical_mid | 437 | 46 | 98.569227 | 28.584161 | 100.000000 | 0.000000 | 53.972498 |
| rxscene/1/practical_low_suburban | 2042 | 245 | 97.259072 | 24.117647 | 99.701340 | 0.000000 | 53.284314 |
| rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 24.761905 | 100.000000 | 0.000000 | 57.619048 |
| rxscene/1/practical_mid_urban | 401 | 44 | 93.945919 | 9.103641 | 99.771689 | 0.000000 | 47.845292 |
| rxscene/1/practical_low_urban | 2082 | 253 | 86.372057 | 12.714251 | 99.665081 | 0.192181 | 31.406819 |
| rxscene/3/practical_high | 208 | 25 | 93.027750 | 8.333333 | 95.871665 | 0.775194 | 31.481481 |
| rxscene/3/practical_mid | 407 | 54 | 92.868021 | 7.017544 | 95.525308 | 1.303419 | 28.289474 |
| rxscene/3/practical_low_suburban | 2007 | 239 | 90.551703 | 16.781907 | 96.255317 | 1.344105 | 21.605605 |
| rxscene/3/practical_high_urban | 241 | 29 | 90.927575 | 20.370370 | 94.843990 | 0.901510 | 30.092593 |
| rxscene/3/practical_mid_urban | 388 | 44 | 86.271023 | 2.083333 | 95.089780 | 1.864692 | 9.722222 |
| rxscene/3/practical_low_urban | 2149 | 239 | 74.111826 | 9.147107 | 96.297074 | 0.767479 | 14.285028 |
| rxscene/4/practical_high | 213 | 26 | 97.207456 | 36.111111 | 97.207456 | 0.641026 | 29.629630 |
| rxscene/4/practical_mid | 403 | 48 | 97.880771 | 29.629630 | 99.184735 | 0.268817 | 11.111111 |
| rxscene/4/practical_low_suburban | 2049 | 248 | 95.426681 | 24.664083 | 98.540058 | 0.627892 | 24.151593 |
| rxscene/4/practical_high_urban | 238 | 30 | 96.476988 | 39.444444 | 98.196248 | 1.803752 | 17.777778 |
| rxscene/4/practical_mid_urban | 392 | 40 | 91.944444 | 13.888889 | 99.140212 | 0.562169 | 20.185185 |
| rxscene/4/practical_low_urban | 2105 | 238 | 84.268387 | 12.800837 | 98.946631 | 0.277511 | 14.730802 |
| rxscene/6/practical_high | 240 | 24 | 100.000000 | 37.407407 | 100.000000 | 0.000000 | 46.296296 |
| rxscene/6/practical_mid | 365 | 47 | 98.543783 | 34.597070 | 100.000000 | 0.000000 | 45.989011 |
| rxscene/6/practical_low_suburban | 2104 | 240 | 96.647847 | 27.642033 | 99.711581 | 0.097669 | 41.628403 |
| rxscene/6/practical_high_urban | 233 | 27 | 98.341523 | 28.981481 | 99.549550 | 0.450450 | 39.814815 |
| rxscene/6/practical_mid_urban | 408 | 57 | 92.960665 | 16.058612 | 100.000000 | 0.000000 | 40.460526 |
| rxscene/6/practical_low_urban | 2050 | 235 | 85.903581 | 10.471171 | 99.652950 | 0.188857 | 28.903946 |
| rxscene/8/practical_high | 240 | 38 | 97.619492 | 14.021164 | 98.767753 | 0.793651 | 30.606431 |
| rxscene/8/practical_mid | 371 | 61 | 96.752001 | 9.218559 | 98.653778 | 0.781250 | 27.228327 |
| rxscene/8/practical_low_suburban | 2102 | 231 | 94.964158 | 9.542644 | 98.087572 | 1.004876 | 24.166188 |
| rxscene/8/practical_high_urban | 223 | 22 | 95.557120 | 6.060606 | 98.403092 | 0.742207 | 12.121212 |
| rxscene/8/practical_mid_urban | 390 | 51 | 89.678601 | 5.937393 | 96.868962 | 1.482889 | 25.393877 |
| rxscene/8/practical_low_urban | 2074 | 227 | 85.679486 | 4.046141 | 98.151941 | 0.710266 | 14.046998 |

## 风险绑定和完整拟合

主既定common-tail校准的下一档在rxscene/4/practical_high_urban失效：正确拒绝由4至5，原F0宏损失由1.803752pp至2.200577pp，超2pp。当前全组最低C_K=94.843990%，绑定是局部身份保护而非总体接收90%。这只描述既定校准路径，不证明所有其他阈值配置不可能；没有依据novel真值扫描/选择新τ。

主AUROC90.182108%是固定分数的排序诊断，不能证明R_U90与known保护可同时成立。其固定OSCR/B约0.916417757876；面积不是实际运行点，也不能代替主R_U16.158730%。原F0同输入身份预测逐条不变，当前A_K差异全部来自接收决策。

25次Lloyd weighted SSE由496.128563351至419.798685780，无上升、始终32活跃/0空mode；原始计数4至575、mode质量0.000604至0.098302，不等于未知TX数。主known方差0.233914401至0.293714376、背景方差0.228088107至0.549969897；均有限，未以训练SSE下降替代拒识效果。全部25点、42组风险、逐TX/RX/day/scene及配对已完整读取于私有analysis。

## 成本、执行证据和下一学习机制

主数组聚合payload3908712B，控制364340B，合计4273052B；两policy阈值96B。每个模型11个聚合数组，无逐source物理bank部署。主新神经参数0、真实F30150＋合成2、G0、NN/optimizer更新0；原模型和缓存的历史成本保留。

CPU拟合13.242429s、density分数计算6.121102s，源过程30.117368s、pipeline49.871021s，均非B1或端到端G+F部署时延；未测峰值显存。源L/OE特征复用，不把V/NV多表示计为更多物理样本。

执行code `edc28ad00baace5e4b9567fbf3391154bf6c34b4`，501文件/data0 archive单SHA ce018419e88a68ad7db02297009a26d5390ce83d30a2195c716d7211b6ec2fe5比较/remote compile/runtime PASS。core含原160回归11tests+23sub/1.11s；新run/score首7+13sub/10.60s及新增FP64矩阵批次舍入回归定点1PASS/4.21s，共8个唯一测试；launcher8/2.92s，9AST/runtime/唯一直接P0/P1 review PASS。本地432条既有np.trapz弃用警告保留；实际三份stdout241行，warning/error/nonfinite0。

独立抓到driver621460、predict622584的live PID/CWD/完整argv/CUDA7/GPU388MiB及日志0→10378→11587B增长；388MiB是离散观察，不当峰值。独立CPU scorer PID由jobs/日志/终态核实，未抓live，不为补观察重跑。真实首步贡献checkpoint smoke、全部30150F进度、状态不变与完整预测/评分齐全。终态无本run live/GPU，其他用户停止/修改/重启0。root独立读回两模型维度/FP64/字节/分数路由/计数及全部control原样复用，未读取原novel私有truth或重新评分选阈值。原始特征/日志/权重保持私有，不提交Git。

下一步从固定拼接转向**独立开放表征学习**：保留原F0/defaulthead身份，另学一个以原unit160为零残差起点、输出仍160维的开放embedding，避免把本轮维数与统计收缩差异混入下一比较。拟采用源L跨RX且跨day的度量监督，主再加入OE的原F0软邻域结构保持；不假定不同OE是不同设备、不读取其TX/K、不用不可靠互邻硬伪标签，不加入最大成员关系头。原0.1硬残差约束与新方案的差异需明确，不能宣称单因素因果。

下一矩阵拟以同结构/初始化/步数的L-only表征控制与L＋OE软结构主对照，冻结原身份及默认头；两者之后使用相同160维密度与既定known风险规则。两者统计背景均可使用OE，因此L-only仅指embedding训练，不得称整个控制无OE。具体网络/损失/预算/随机流在独立最小预登记固定，不依据当前novel标签调参；新学习尚未实施或获得性能。先完成源侧反证，通过后才另登记跨未见RX星地；无标签适应/发现和未来匿名注册仍保持完整目标，不以source负结果改小目标。

完整v3跨RX星地C_K/R_U90、A_K距同输入原F0≤2pp及进一步高拒识要求仍待证明；本轮AN/HOW为N/A，未来AN80/REG旧损≤2pp未执行，目标ACTIVE。
