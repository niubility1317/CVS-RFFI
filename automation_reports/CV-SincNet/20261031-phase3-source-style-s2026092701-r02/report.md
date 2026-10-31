# 同TX跨RX浅中层统计混合实验

ANALYZED，执行 VERIFIED；科学 NO_PROMOTION，完整目标 ACTIVE。

# Phase3 源端统计混合 r02 技术恢复

恢复 run 为 `20261031-phase3-source-style-s2026092701-r02`，替代技术失败且未开始训练的 r01；不从 r01 权重恢复。科学方法、两个候选、数据句柄、原生 F0/pure-r03 初始化、444×2预算、优化器、种子、两层统计混合及8×3正式测试全部沿[原计划](PAMF_PHASE3_SOURCE_STYLE_PLAN_20261031.md)，不依据目标结果改动。

r01 的首个真实 checkpoint 无 query smoke 在浅层梯度检查失败，尚未创建训练目录或进入源缓存／训练循环，未产生 checkpoint、预测或评分。执行堆栈显示失败位于 `_smoke`，原检查对 `time_down.parameters()` 求范数；真实 `time_down` 是没有参数的 AvgPool1d，空集合范数恒为0。原合成 reader 将卷积与pool合并成同一个Sequential，未覆盖该真实结构。

修复只将浅层**参数梯度诊断**改为池化前的 `time_fuse` 卷积参数，保留 `time_down` 和 `t2.pool` 的统计混合位置、实际前向、损失及非零梯度标准。失败异常附带各模块标量范数，便于定位。测试先用分离的 Conv/AvgPool 忠实结构复现旧失败，再定向验证原生初值、hook退出、实际训练与strict回载；不放宽检查或更换数据／模型父。

原 r01 根目录、日志、发布包、配置及未发布的两个预修复包保留。r02 仍从原 F0 和纯源 r03 projection 独立初始化，不继承本次失败产生的模型状态；只有新run拥有新启动。首步骤重新执行真实无query smoke，PASS立即继续训练和全部预测，再独立truth-last。若相同技术指纹在本次针对性恢复后再现，保留产物停止盲重启，先报告证据和必要信息。

新学生U/Q/V/NV/Us训练读取0、旧PCD仅对照、每卡新增≤2/free≥4096MiB、不修改其他用户任务及原目标ACTIVE等约束不变。真实结果未返回前不声称科学成功。实际代码、命令、环境、CWD、GPU和输出见r02登记。


实际code `57d887a1e1ac34e4e98a235ad698ef8f628dfb44`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-source-style-s2026092701-r02-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.source_style_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-source-style-s2026092701-r02-release/runtime.json`。


# Phase3 同 TX 跨 RX 统计混合：源端联合身份与开放表征结果

**r02 执行完成，科学结论为 NO_PROMOTION。** 唯一主方法 `STYLE_NATIVE_predicted_ratio` 在星地开发集上的 A_K/C_K/R_U/H 为 23.730709%/41.296261%/55.697351%/33.281378%。相对匹配的 SOURCE_NATIVE，宏 A_K 与 R_U 分别提高 0.813160、2.171155 个百分点；相对 OLD_PCD，A_K 仍低 4.221014 个百分点，绝对保护和拒识门槛也未通过。原 PCD 研究基线保持，不扩 seed、不换主，完整目标 ACTIVE。

实际 run 为 `20261031-phase3-source-style-s2026092701-r02`，代码 `57d887a1e1ac34e4e98a235ad698ef8f628dfb44`。新两学生和源密度均为纯源，`new_target_DA=False`、`adaptation_success=None`；本轮不是目标无标签适应。证据为[完整分析](../local_artifacts/pamf_phase3_source_style_20261031_r02/analysis.json)、[固定评分](../local_artifacts/pamf_phase3_source_style_20261031_r02/score/results.json)、[原计划](PAMF_PHASE3_SOURCE_STYLE_PLAN_20261031.md)与[技术恢复说明](PAMF_PHASE3_SOURCE_STYLE_RECOVERY_20261031.md)。

## 方法与对照

SOURCE_NATIVE、STYLE_NATIVE 各自从同源原 F0 E50/11100 的完整原生 reader/default6 和合规纯源 r03 E_SIGNAL/2220 的 projection 独立初始化。实际无 query 合成检查的初始身份 logits 与原 F0 最大绝对差为 0；body 来自 F0，不能声称初始开放 z 等于 r03。旧 target-adapted PCD 不作为新学生参数来源，只保留为冻结比较控制。

NativeOpen 只持有 reader232993＋projection102688＝335681 参数，联合训练 body、原默认六类头 W960 和 projection；冻结 Sinc48，实际可训练 335633。reader 外 667729 个 wrapper 参数不进入新学生或部署。一次 reader 特征前向同时生成原生身份 logits 与开放 unit z160，不新增身份头。

每臂 444 步，每步 L anchors128＋同 TX/异 RX 的 L partners128＋OE64。两臂 A/B/OE 轨迹相同、目标与初始化相同，仅 STYLE 安装两处临时 MixStyle hook；SOURCE 无 hook。AdamW lr1e−4、wd1e−4、eps1e−8，ordinary normclip5。L、partner、OE、style 的固定种子分别为 2026103160/3170、2026103018、2026103171。

两处统计混合位于 `core.id_backbone.time_down` 的实数 `[N,48,128]` 与 `core.id_backbone.t2.pool` 的 `[N,96,32]`；各层独立 p=0.5、逐实例 λ~Beta(0.1,0.1)，沿时间维计算 population mean/std，eps=1e−6。统计量 detach，标准化内容保留梯度；A/B 互为同身份跨 RX 伙伴，OE 精确透传。context 只包该次源学生前向，finally 清除 hooks，教师缓存和冻结评估不安装混合；无跨 query 统计。

共同目标为 `crossset_metric(L256,OE64) + OE_topology_KL64 + 10*anchor + CE_L256 + KD_L256`；anchor＝0.5×L 均值＋0.5×OE 均值，父 z 来自冻结纯源 r03，KD 为原 F0 同 P1 教师的 T=2 KL×4。OE 只用既定成员监督和公共 RX/day，不读真实 TX/K。新两臂训练 U、U ID、Us、V/NV/Q、真值读取均为 0；同源重放控制与旧 192L/64OE 或身份 L128 实验不属于完整计算预算匹配。

预测时，新学生用自己的 default-head `closed` 同时路由自己的源拟合密度和阈值。原 F0 同 DURA 的闭集另存为保护基准，`classification_unchanged=False`。E0/OLD_PCD 的模型、旧四读出及阈值精确继承，不能以较低学生闭集替换原 F0 基准。

## 数据与评估边界

沿用 contract392005/VALIDATED_ONCE：L6300、OE6134、Us56700 祖先，源五 RX 为 1-19、18-2、19-2、2-19、3-19，日期 2021-03-08/15/23；旧 TX 为 14-10、14-7、20-15、20-19、6-15、8-20。新训练读取 L/OE 的既有 P1，L 标签和缓存 F0 logits 合法；Us 只保留祖先，不新打开。

V27000 只在新学生和密度冻结后校准阈值，不是独立已知测试。NV3150 含 19-15、19-16、7-1 各 1050 条，仅在全部预测后评分；该源未知集合与目标四未知集合不同。

目标为开发 RX3-2/index34。旧 03-01 星地 U576 本轮不读值或 ID，仅 OLD_PCD 控制继承 U576 祖先；新学生和新密度没有该祖先。future Q 为 03-15 的 5525 条与 03-23 的 4713 条，共 10238：旧类 4508、未注册 5730。六旧类容量为 1491/1182/150/406/1186/93，5-5、10-7、2-19、7-11 分别 1454/1439/1427/1410；后来专有 7-11 整类不在早日 U。

源/U/Q 原角色物理互斥；clean/star Q 同 canonical 和顺序，clean 仅测试。03-23 缺旧 20-19、8-20，完整 A_K/C_K/H 与闭集宏值保持 N/A，不缩减所需名单。新训练没有目标 U 或配额输入，不因 scorer 知道真实名单而扩大训练权限。

三条件均经冻结 DURA v1.0 R69（E20/4940，266951 参数，reuse/graph 开、native 关，FP32/TF32off）。固定 G10 为 600 km/25 MHz/2.45 GHz、residual/post_sync/noEQ/refSNR30dB/reflect，一次 input RMS、输出无 RMS；源 scene/channel2026103001/4001，future2026103133/34，receiver2027。特征统计混合不是新增物理观测，不重抽信道，也不用 clean 参考恢复。

源 RX 已见、源 V 用于校准、NV/Q 历史曝光、单 RX/单 seed、源日期重合、global physical history UNKNOWN 均保留。正式 RX24-18 未读，不能宣称独立盲测或新目标适应；AN/H_OW 为 N/A，无发现或匿名注册。

## 完整 888 步训练

两臂各 1 至 444 步连续、有限，A/B/OE 索引逐步完全相同。分析独立复核两新臂全部索引轨迹、anchor RNG 与 style RNG；同 TX/异 RX 合法性由训练端逐步检查。分析未重新打开源 y/RX/domain 数组，不将 B/OE 两臂一致误称为独立重建全部 partner 候选。

总损失由所有记录的分量重建，SOURCE/STYLE 最大绝对残差为 2.64e−7/3.87e−7。CE、KD、crossset、anchor、总损失从第 1 步即不同，OE_KL 从第 2 步不同；它们不是训练中途才启用统计混合的对照。下表为全记录三等段均值，anchor 列尚未乘 10，KD 已含 T²=4。

| 臂、步段 | crossset | OE_KL | anchor | CE | KD | 总损失 | 裁剪前范数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SOURCE 1–148 | 0.835286 | 0.086599 | 0.088385 | 0.249919 | 0.030349 | 2.085999 | 5.787914 |
| SOURCE 149–296 | 0.735893 | 0.086354 | 0.044832 | 0.221117 | 0.045764 | 1.537450 | 5.814823 |
| SOURCE 297–444 | 0.659578 | 0.084674 | 0.030307 | 0.192678 | 0.055118 | 1.295115 | 5.814539 |
| STYLE 1–148 | 0.776940 | 0.085928 | 0.091321 | 0.213185 | 0.065367 | 2.054629 | 6.033183 |
| STYLE 149–296 | 0.694541 | 0.085553 | 0.049554 | 0.194670 | 0.071642 | 1.541950 | 6.083858 |
| STYLE 297–444 | 0.630947 | 0.085206 | 0.035950 | 0.172506 | 0.081921 | 1.330080 | 5.583616 |

SOURCE/STYLE 总损失首末为 4.341505→1.151757、4.248417→1.222525。STYLE 的源 CE、crossset 更低，但 KD 与 anchor 较高；这说明共同目标之间有不同取舍，不能由某一训练分量降低推定目标泛化成功。初始开放 anchor 约 0.286/0.289 也符合“F0 body＋r03 projection 的开放初值不等价于 r03”，不是据目标成绩推断的实现错误。

浅层、中层、body、原默认 head 和 projection 在全部 444 步均有非零梯度。SOURCE/STYLE 的浅层均值为 2.590767/2.796138，中层 0.987895/1.022225，head 0.596139/0.605614，projection 1.333768/1.321546。普通梯度裁剪分别触发 274/278 步，最大裁剪前范数为 20.571228/24.751869；Sinc 梯度及更新全程 0，physical clip 调用 0，buffers、原模型和 wrapper 外部状态保持不变。

浅层参数梯度统计实际取 `time_fuse`，统计混合 hook 仍位于无参数的 AvgPool `time_down`。二者用途不同，不能把 AvgPool 没有参数解释为没有上游梯度，也不提出“pool 参数限制性能”的假说。r02 合成 smoke 的浅层/中层/默认头/projection 范数分别为 3.314135/1.005454/3.147210/13.113570，初始身份等价、OE 透传、内容梯度和退出后 logits 精确恢复均通过。

STYLE 两层各被调用 444 次，实际启用 233/222 步、混合 L 行数 59648/56832；OE 全程精确透传、统计量全程 detach。各步 λ 均值的全程平均为 0.498465/0.498114；Beta(0.1,0.1) 在有限精度下可出现 0/1 端点，不据此宣称异常。两个 hook 的全批特征均值绝对改变量均值为 8.40e−5/1.39e−4，这不是逐元素扰动范数或效果强度。每步完成与 smoke 均记录 hooks 已清除；评估不安装 context，没有额外逐 query hook 遥测。

每臂 A/B 各曝光 56832 次，OE 曝光 28416 次；A、B unique 均为 6300，L 联集仍为 6300，OE unique5896。每臂真实学生前向 142080 行，合计 284160；新增 optimizer/body/default-head/projection 均为每臂 444 步、合计 888 步，不把多次 L 使用计为更多独立源样本。

## 全部 8×3 结果

以下均为百分比。A_K 是含拒识的旧类宏准确率，C_K 是旧类宏接收率，R_U 是未知宏拒识率，H 为 A_K/R_U 调和均值；均按所需真实 TX 等权，不按记录数加权。原闭集为同 DURA 原 F0，学生闭集不含拒识。

| 方法 | 条件 | 原闭集 | 学生闭集 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 pooled | source | 90.981481 | 90.981481 | 89.770370 | 97.107407 | 60.888889 | 72.561330 |
| E0 predicted | source | 90.981481 | 90.981481 | 89.744444 | 96.759259 | 61.936508 | 73.291437 |
| OLD_PCD pooled | source | 90.981481 | 90.981481 | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD predicted | source | 90.981481 | 90.981481 | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| SOURCE_NATIVE pooled | source | 90.981481 | 91.677778 | 89.900000 | 96.251852 | 68.539683 | 77.779977 |
| SOURCE_NATIVE predicted | source | 90.981481 | 91.677778 | 89.833333 | 96.103704 | 69.968254 | 78.666071 |
| STYLE_NATIVE pooled | source | 90.981481 | 91.644444 | 89.740741 | 96.125926 | 70.126984 | 78.730682 |
| **STYLE_NATIVE predicted** | source | 90.981481 | 91.644444 | 89.770370 | 96.148148 | 69.714286 | 78.481246 |
| E0 pooled | clean | 48.567015 | 48.567015 | 31.187288 | 43.537692 | 62.980616 | 41.716859 |
| E0 predicted | clean | 48.567015 | 48.567015 | 30.999690 | 42.654045 | 64.187968 | 41.808091 |
| OLD_PCD pooled | clean | 48.567015 | 48.567015 | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD predicted | clean | 48.567015 | 48.567015 | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| SOURCE_NATIVE pooled | clean | 48.567015 | 46.389062 | 26.008330 | 36.827867 | 66.768102 | 37.434655 |
| SOURCE_NATIVE predicted | clean | 48.567015 | 46.389062 | 24.778753 | 35.439150 | 68.239475 | 36.356081 |
| STYLE_NATIVE pooled | clean | 48.567015 | 46.501380 | 25.787024 | 36.456645 | 70.276303 | 37.729627 |
| **STYLE_NATIVE predicted** | clean | 48.567015 | 46.501380 | 25.861631 | 36.569007 | 70.309538 | 37.814230 |
| E0 pooled | star | 41.721059 | 41.721059 | 26.867357 | 47.141720 | 51.837083 | 35.391279 |
| E0 predicted | star | 41.721059 | 41.721059 | 26.440597 | 46.197372 | 52.658525 | 35.204508 |
| OLD_PCD pooled | star | 41.721059 | 41.721059 | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD predicted | star | 41.721059 | 41.721059 | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| SOURCE_NATIVE pooled | star | 41.721059 | 40.551703 | 23.814255 | 41.144699 | 51.963130 | 32.660489 |
| SOURCE_NATIVE predicted | star | 41.721059 | 40.551703 | 22.917549 | 40.057260 | 53.526195 | 32.093907 |
| STYLE_NATIVE pooled | star | 41.721059 | 40.954329 | 23.674450 | 40.967175 | 55.963196 | 33.273155 |
| **STYLE_NATIVE predicted** | star | 41.721059 | 40.954329 | 23.730709 | 41.296261 | 55.697351 | 33.281378 |

## 三个控制的配对得失

配对使用相同 ID，“增/失/净”为记录数，不是宏百分点。它们分别统计身份纠错、正确接收与未知拒识，不能混为一个净收益。

| 主相对控制 | 条件 | 闭集正确增/失/净 | 正确接收增/失/净 | 接收净数 | 未知拒对增/失/净 | ΔA_K / ΔR_U（pp） |
|---|---|---|---|---:|---|---|
| E0 predicted | source | 341/162/+179 | 327/320/+7 | −165 | 347/102/+245 | +0.025926 / +7.777778 |
| E0 predicted | clean | 97/190/−93 | 34/150/−116 | −158 | 477/126/+351 | −5.138059 / +6.121570 |
| E0 predicted | star | 110/140/−30 | 61/126/−65 | −165 | 410/236/+174 | −2.709888 / +3.038826 |
| OLD_PCD predicted | source | 341/162/+179 | 342/332/+10 | −162 | 290/119/+171 | +0.037037 / +5.428571 |
| OLD_PCD predicted | clean | 97/190/−93 | 45/152/−107 | −156 | 568/92/+476 | −5.800173 / +8.329891 |
| OLD_PCD predicted | star | 110/140/−30 | 58/129/−71 | −186 | 525/161/+364 | −4.221014 / +6.389652 |
| SOURCE_NATIVE predicted | source | 90/99/−9 | 114/131/−17 | +12 | 75/83/−8 | −0.062963 / −0.253968 |
| SOURCE_NATIVE predicted | clean | 30/50/−20 | 25/51/−26 | −13 | 194/76/+118 | +1.082878 / +2.070063 |
| SOURCE_NATIVE predicted | star | 22/67/−45 | 26/57/−31 | −10 | 211/88/+123 | +0.813160 / +2.171155 |

STYLE 相对匹配 SOURCE 的星地宏 A_K 增加，但正确接收总数减少 31，闭集正确总数减少 45；两者不矛盾。六旧 TX 的正确接收净变化为 `−23,−9,0,+1,−7,+7`，容量仅 93 的 8-20 增 7 条，对等权宏平均的影响较大。不能将宏改善写成“总正确更多”，也不能反向将总条数下降解释为全部类别都退化。

相对 OLD，主星地已知闭集正确 1914、正确接收 1493，闭集正确但被拒 421；OLD 对应 1944/1564/380。主错误接收 954 条，OLD1069，未知拒对增加 364 条，但正确接收减少 71 条。主星地 H 比 OLD 低 2.396734 pp；相对 SOURCE 的 H 增 1.187470 pp，只是本固定源端统计混合的局部比较，不能取代旧 PCD 联合性能。

## 场景、类别、日期及曲线

主星地六场景在合并未来日期后均覆盖旧六类和未知四类，但联合门槛全部失败；没有因缺类而删去困难 TX。

| 场景 | 已知/未知数 | 学生闭集 | A_K | C_K | R_U | H |
|---|---:|---:|---:|---:|---:|---:|
| practical_high | 206/256 | 45.555648 | 23.475703 | 35.311794 | 62.908280 | 34.191896 |
| practical_mid | 293/443 | 48.777999 | 23.056986 | 35.306963 | 64.055457 | 33.908491 |
| practical_low_suburban | 1715/2167 | 45.693635 | 24.633847 | 37.714500 | 58.500774 | 34.669049 |
| practical_high_urban | 197/244 | 52.797699 | 37.716567 | 52.642343 | 62.405360 | 47.016992 |
| practical_mid_urban | 315/397 | 45.595720 | 26.983032 | 41.175029 | 56.234285 | 36.467686 |
| practical_low_urban | 1782/2223 | 32.497795 | 21.595533 | 45.747703 | 49.662229 | 30.101487 |

主逐旧 TX 的 clean A_K/C_K 为 46.478873/57.478203、7.952623/59.137056、0.666667/0.666667、0/0.985222、60.286678/60.286678、39.784946/40.860215%；星地为 45.137492/60.697518、9.390863/62.774958、2.666667/20、1.477833/7.142857、56.829680/59.527825、26.881720/37.634409%（顺序同旧六 TX）。低容量类的高拒绝与身份错误都真实计入宏 A_K，不能用总体接收替代逐类性能。

四未知 5-5/10-7/2-19/7-11 的 clean R_U 为 38.239340/89.715080/74.702172/78.581560%，star 为 33.906465/68.658791/51.997197/68.226950%；无一跨全部条件达到既定联合要求。源 NV 三 TX 的 R_U 为 68.857143/63.428571/76.857143%，不作为目标四类的替代证据。

目标仅 RX34。03-15 的 clean/star A_K 为 25.432955/23.506327%，R_U 为 69.393971/55.271518%；03-23 缺两个旧类，A_K/C_K/H 和完整闭集宏值为 N/A，R_U 可报为 71.233685/56.148479%。源五 RX 的 A_K 为 83.518519% 至 92.703704%，R_U 为 48.888889% 至 81.746032%；源三日 A_K 为 89.511111% 至 89.966667%，R_U 为 66.476190% 至 75.333333%，不能据此宣称未见 RX 稳健性。

主 source known12 通过，但 known42 未全过。源总体原 F0−A_K 损伤为 1.211111 pp；已存下一 common-tail 候选会令 RX8 损伤达 2.018519 pp，因此按原规则拒绝。30 个 RX×scene 诊断中 6 组失败：RX3 的 high/mid/mid_urban，RX4 的 high_urban，RX8 的 low_suburban/low_urban。它们未改成额外校准门槛，也不能被写成 42 组保护已通过。

| 条件 | E0 predicted AUROC/OSCR | OLD_PCD predicted | SOURCE_NATIVE predicted | 主 STYLE_NATIVE predicted |
|---|---|---|---|---|
| source | 94.734250/87.937558 | 94.730673/87.940433 | 94.971954/88.587071 | 95.021899/88.529613 |
| clean | 53.862200/33.089825 | 54.364758/33.693180 | 49.480723/29.009781 | 49.929192/29.352187 |
| star | 51.242710/26.543484 | 50.997190/27.001261 | 47.150329/23.762539 | 48.333619/24.321527 |

曲线仅为固定 margin 的宏加权诊断，不导出新阈值。评分保留逐真实类 Wilson95% 区间，例如星地 20-15 的 A_K4/150，区间约 1.041794% 至 6.655389%；5-5 的 R_U493/1454，区间约 31.518460% 至 36.379283%。这些区间不能代替跨 RX/seed 确认，全部 TX/RX/day/day×scene、42 组与配对细节均见固定评分 artifact。

主 clean/star 原 F0−A_K 损伤为 22.705384/17.990350 pp，远高于 2 pp；C_K36.569007/41.296261% 低于 90%，R_U70.309538/55.697351% 低于 clean95%/star90%。星地 R_U 虽不低于全部控制，但 A_K 未高于 OLD_PCD，且绝对门槛失败，`source_style_success=False`、`scientific_success=False`、`development_gate_passed=False`，不能按 pooled 或较弱控制替换唯一主方法。

## 冻结顺序、成本与部署

新学生冻结后，各用 L6300/OE6134 拟合 known90/OE32、L 白化和固定 10% 收缩、32 背景容量/25 Lloyd/seed2026103019，共 2 套新源密度。神经及密度均先于 V 冻结，V 只拟合新四组六阈值，共 24 新值；E0/OLD_PCD 的 24 旧值精确复用。总 48 个 FP64/384 B，单主六阈值 48 B，不能声称零统计拟合。所有 8×3 预测及状态完整后，独立 CPU 才连接 NV/clean/star 三 truth 各一次；Utruth0，预测阶段真值0。

| 新增计算 | 训练 | 冻结预测 | 合计 |
|---|---:|---:|---:|
| 新学生真实前向行 | 284160 | 126120 | 410280 |
| 冻结纯源 r03 的 L/OE 教师行 | 12434 | 0 | 12434 |
| 新信号前向（含教师缓存） | 296594 | 126120 | 422714 |
| 原 F0 / DURA 真实前向行 | 0/0 | 20476/20476 | 20476/20476 |
| 合成 F0 / r03 / PCD / G 行 | 4/2/2/4 | 2/0/0/2 | 6/2/2/6 |
| 合成新学生行 | 8 | 4 | 12 |

评估新学生源 L/OE/V/NV 共 85168 行、future 两条件共 40952 行；Q 每批共享一次 G 和原 F0，两学生各前向一次，旧 E0/PCD 控制不新增真实前向。源 F0 teacher6300 为缓存；训练新 F0/G 真实0，U/Us/V/NV/Q 训练0，没有新增物理观测。成本字段使用真实生产端 `real_student_forward_rows`，284160 行未因字段修复改变。

主部署为 NativeOpen335681＋G266951＝602632 神经参数；reader 外 667729 参数不部署，原 F0仅作冻结评估基准。无临时 hooks、教师、source/U 行 bank 或训练源配对表部署。原 F0 E50/11100、纯源 r03/2220、旧 PCD/U576、DURA E20/4940、源密度及缓存成本继承，不从本轮计数中抹去。

## r01 故障、r02 恢复与执行证据

r01（执行代码 `7554e…`）在首个真实无 query smoke 的浅层梯度检查失败，仍未进入源缓存值读取或训练循环，optimizer0、无学生 checkpoint、Q/预测/评分/真值0，原 root 以 FAILED_PRESERVED 保留。直接原因是对无参数的 `time_down` AvgPool 求参数梯度范数，空集合恒为0；旧合成 fixture 把卷积与池化合在一个 Sequential，未覆盖真实结构。它不是科学低分，也不证明统计混合切断了梯度。

r02 只把浅层参数诊断改为上游 `time_fuse`，不改变 `time_down`/`t2.pool` 的统计 hook、前向、损失、非零梯度标准、科学矩阵、种子或预算，并以忠实 Conv/AvgPool fixture 复现 RED 后定向 GREEN。r02 从原 F0/pure-r03 独立重建，不复用 r01 训练状态；真实恢复成功。发布前另修复 phase3 包缺外部 SSL 依赖（同一纯 `mix_moments` 本地实现）及训练/预测成本字段交接，均未先启动科学运行。原失败包和未发布包独立保留，下述 r02 成本不反推或覆盖 r01 失败烟测尝试。

r02 train/predict/CPUscore 三阶段 return0、全 COMPLETE，耗时 90.211241/56.474549/22.513978 s，pipeline169.790262 s，终态 ANALYZED_DEVELOPMENT。它们不是 B1 或受控加速测试。4 次观测独立捕获 driver1939267、train1939295 的 CWD/完整 argv；训练 CUDA5、UUID `ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331`，408/834 MiB 为离散样本而非峰值。预测1940298、CPU1941052 只有 pipeline 的 GPU5/gpuNone、命令、return0 和完成产物，未独立捕获 live，不写全阶段都通过 live 核验。

完整四 stdout 共 185 行/9389 B，warning/error/traceback 均为0；训练日志 492→1028→1343 B 有独立增长，预测最终7983 B含两个10238完整进度，但观测未捕获预测日志中途增长。终态 owned live/GPU 空、foreign0。完整 888 steps 与所有 stdout 均已读取，证据见[训练完成](../local_artifacts/pamf_phase3_source_style_20261031_r02/train/complete.json)、[训练烟测](../local_artifacts/pamf_phase3_source_style_20261031_r02/train/smoke.json)、[预测完成](../local_artifacts/pamf_phase3_source_style_20261031_r02/predict/complete.json)、[评分完成](../local_artifacts/pamf_phase3_source_style_20261031_r02/score/complete.json)、[流水线](../local_artifacts/pamf_phase3_source_style_20261031_r02/pipeline_state.json)与[观测](../local_artifacts/pamf_phase3_source_style_20261031_r02/observations.jsonl)。

合成验证共 12 个唯一测试、43 个子项通过；必要 moments/同一 producer-scoring 交接/两个恢复用例只定向重验，不重复计 unique。恢复忠实 fixture 先 RED、后 2 tests＋10 subtests/6.31 s GREEN，不能声称初轮全绿；集成输出的1408条旧 `np.trapz` 弃用警告及本地 pytest cache 权限告警不属于生产日志。一次完整 P0/P1 审查，以及依赖、成本字段、真实 AvgPool 故障各自的一次定点复审均 PASS，未重复全套审查；既有8 AST/4 CLI接口通过，r02实际 runtime/remote compile通过。

r02 单包838文件/data0，SHA `77ac6a0c2e6c990c19ca62d60a2df54f6a4e88d0374a0de23a227ad9f2861cb9`，单次本地/远端比较、compile/runtime为 VERIFIED；r01 archive 独立保留，不混为同一发布。见[测试记录](../local_artifacts/pamf_phase3_source_style_20261031_r02/test_evidence.json)、[恢复 RED/GREEN 日志](../local_artifacts/pamf_phase3_source_style_20261031_r02/recovery_train_tests.log)、[审查](../local_artifacts/pamf_phase3_source_style_20261031_r02/review.json)和[发布核实](../local_artifacts/pamf_phase3_source_style_20261031_r02/release_verification.json)。

本轮仅表明：这套固定两层、同身份跨 RX 统计混合，在匹配源控制上有部分宏指标收益，却未恢复旧 PCD 的联合性能和绝对目标。不能推广为所有统计混合无效，也不能把低目标准确率只归因于 RX；跨池比较同时改变日期、未知集合及容量。目标结果不回流调整层位、λ、损失或阈值；保留原 PCD 研究基线、历史困难结果和完整目标 ACTIVE。
