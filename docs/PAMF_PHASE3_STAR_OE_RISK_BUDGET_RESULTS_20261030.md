# Phase3 G10 OE 风险预算阈值结果

唯一主候选 `joint::oe_risk_budget` **NO_PROMOTION**：宏 A_K=90.9000%、R_U=20.2857%、H-score=33.1692%，C_K=98.2741%，相对原 F0 同 P1 闭集损失 0.0815 个百分点。42 组已知保护和结构保护通过，但总体、5 个源 RX、6 个场景的未知拒识均未达到 90%。本轮只比较固定分数上的校准策略，没有目标域适应或匿名注册。

## 身份、数据与方法

实际 run 为 `20261030-phase3-star-oe-risk-budget-s2026092701-r01`；code `585e10b07be1f6821edea08b4853582b92512aff`，预登记 `643ee789e420d671ae0f8b1987ba21f39da7e3de`。单一归档 SHA256 为 `594571df9184c5b2da01d942e759353a9aa3aa876cdcc10340218cb56a720440`，318 个文件、data0，发布 compile/runtime/SHA 已 VERIFIED。

两臂 `detach_control/joint` 固定继承联合 OVA r02（code `5aa041122396d48c49b8cb3d5912649bdc3217e1`），模型、原六类默认身份头和 966 参数辅助层均冻结。每臂比较 `common_tail/oe_risk_budget`；唯一主为 joint 预算策略，控制分数较高也不能替代。raw evidence 始终为 max(auxiliary six logits)，身份始终为默认头 argmax。

common_tail 逐值复用父 ova_max 六阈值及原校准报告。预算策略用 V27000 的合法旧六标签、原 F0 同输入闭集参照，以及 OE6134 的 outside-six 成员监督和 15 个 RX/day 域，拟合六个“模型预测类”阈值；不使用真实 TX 路由阈值，不读 OE TX/K。它明确使用 OE 成员监督，不能称 known-only 或零源标签方法；V 不拟合 NN/normalizer。

预算算法从有限全接受阈值开始，只移动有 OE 新增收益的完整分数层，收益按每域 1/15 加权。按新增风险/剩余预算代价贪心分配，保持 42 组真实旧类宏 C_K≥90%、原 F0 闭集−A_K≤2 pp；float64/tolerance1e−12，最后用原 exact-Fraction 风险核对。相同分数整层处理，阈值取 nextafter(score,+∞)。它是确定性局部贪心，不保证全局最优、优于 common-tail 或 conformal 覆盖。

本轮仅 OE P1 新做每臂 6134 实物＋2 个合成 G10 P0 checkpoint smoke，合计 12272 个 F 行前向；无新 V/novel/原 F0/G 前向、NN/aux/G/optimizer 更新均 0，L/原 U_s/目标 U/Q/formal 读取均 0。先实际 checkpoint smoke，再 OE/V；全部阈值、校准与 known 文件写定后，才打开父 novel 公共预测，重算 margin，完整预测后独立 CPU truth-last。

缓存的 L6300/OE6134/V27000/novel3150 仍是同一 G10 实现的 42584 条原物理记录，本轮不增加独立样本或重生信道。F0 祖先为 Phase1 L6300/U56700/V27000、E50/11100；训练标签比例 10%，另有 V 校准标签和 OE 成员监督。父 r02 两臂各追加 2220 步、累计13320，本轮继承该成本而不是重新训练。

source RX1/3/4/6/8 对应 `1-19/18-2/19-2/2-19/3-19`，源日期 2021-03-08/15/23；旧类为 `14-10/14-7/20-15/20-19/6-15/8-20`。novel `19-15/19-16/7-1` 各1050，历史已有源开发曝光，且与 known 同属五个已见 RX；不是跨新 RX 或新设备盲测。全部 P1 为固定 G10 received 经 DURA v1.0 的恢复结果，不是 clean。MIX01 的 scene/channel seeds2026103001/2026104001、2.45GHz、receiver2027 等处理保持父缓存身份。

## 四个固定运行点

A_K 为按真实旧类宏平均的“身份正确且被接收”；C_K 只是接收；R_U 为按真实未知 TX 宏平均的拒识。H=2A_KR_U/(A_K+R_U)，不另设 H 晋级门槛。全部 margin≥0 接收，原 F0 同 P1 闭集参照为90.981481%，候选闭集 control91.548148%/joint91.540741%均保持不变。

| arm / policy | A_K % | C_K % | R_U % | H % | 原F0损失 pp | 宏 AUROC % | 宏 OSCR % |
|---|---:|---:|---:|---:|---:|---:|---:|
| detach_control / common_tail | 91.2926 | 99.1444 | 11.6825 | 20.7143 | -0.3111 | 84.9276 | 79.4583 |
| detach_control / oe_risk_budget | 90.3815 | 97.1926 | 20.2540 | 33.0922 | 0.6000 | 76.0478 | 72.2844 |
| joint / common_tail | 91.0037 | 98.4630 | 24.9206 | 39.1267 | -0.0222 | 88.9786 | 83.3740 |
| joint / oe_risk_budget | 90.9000 | 98.2741 | 20.2857 | 33.1692 | 0.0815 | 77.3276 | 73.0664 |

四运行点均 known42/42 通过、unknown12/12 失败；全部42评分组类别覆盖完整，无缺类 N/A。预算控制臂最小组 C_K=94.6735%、最大损伤1.8627 pp；预算 joint 最小组 C_K=95.1527%、最大损伤1.9260 pp，不能用总体接收高掩盖逐组约束。AUROC/OSCR 使用各策略实际 class-specific margin；类别平移改变排序，不能沿用父 raw-score 面积。

逐 ID 核对两臂 known/novel 的所有非 scores 字段与 r02 完全相同，默认 logits、aux logits、闭集身份、原 F0 baseline、IDs 和两套单位特征均未改变；common 列精确等于父 ova_max margin。新两列 margin 与 raw max−τ[predicted class] 重建误差为 0；OE score 与 aux max、OE closed 与默认 argmax 也逐条一致。

## OE 代理与保留 novel

| arm | common OE域宏 %（pooled %） | budget OE域宏 %（pooled %） | ΔOE域宏 pp | Δ保留novel R_U pp | ΔH pp |
|---|---:|---:|---:|---:|---:|
| detach_control | 36.6480 (43.2670) | 40.1161 (47.3916) | +3.468101 | +8.571429 | +12.377863 |
| joint | 56.6686 (64.4441) | 53.1226 (59.3903) | -3.546007 | -4.634921 | -5.957514 |

joint 的预算策略不仅在保留 novel 上退步，训练 OE 域宏代理也低于已保存 common-tail。轨迹从接受全部的目标0升到53.1226%，只说明相对该初态的训练代理收益；common 可行点原本为56.6686%，贪心路径并不保证达到或超过它。控制臂的代理与 novel 同向提高，但主模型不能换成控制臂。域宏 OE 不是未知真实 TX 宏拒识，两者分母和权重不同。

| budget − common | 已知正确 gain/loss（净） | 接收 gain/loss（净） | novel拒对 gain/loss（净） | ΔA_K pp | ΔR_U pp |
|---|---:|---:|---:|---:|---:|
| detach_control | 7/253（−246） | 18/545（−527） | 306/36（+270） | −0.911111 | +8.571429 |
| joint | 20/48（−28） | 53/104（−51） | 73/219（−146） | −0.103704 | −4.634921 |

主候选接收26534/27000，其中正确24543、错误身份1991；3150个未知仅拒对639、误接2511。98.2741% 接收不等于98.2741%身份准确率。joint 相对 common 的 H 降低5.957514 pp，不能用已知保护通过来挽救未知失败。

## 完整轨迹与终止约束

| arm | 完整贪心步数 | 零增量风险步数 | 六预测类 OE prefix | 累计V拒绝 / OE拒绝 |
|---|---:|---:|---|---:|
| detach_control | 2907 | 2599 | [927,254,272,369,667,418] | 758 / 2907 |
| joint | 3643 | 3370 | [748,229,658,212,1413,383] | 466 / 3643 |

完整解析6550条选择记录，step连续、所有数值有限，每步都有正 OE 收益；收益累加分别为0.401160977875和0.531225673808，与保存终点一致。全部初态可行、无 accept-all 科学回退；终点 `maximal=true/global_optimum_claimed=false`。两臂停止时六类下一候选均因 known 风险不合法，并非 OE 层耗尽。

主 joint 六个下一候选的绑定组如下；均为原 F0 总损伤超过2 pp，拒绝率仍小于10%，没有放宽身份保护去追求 OE 收益。

| 预测类index | 绑定 RX/scene | 下一候选原F0损失 pp | 下一候选 FRR % |
|---|---|---:|---:|
| 0 | RX3/mid | 2.0043 | 5.1251 |
| 1 | RX4/highU | 2.3506 | 2.1825 |
| 2 | RX4/highU | 2.2423 | 2.0743 |
| 3 | RX3/midU | 2.0648 | 4.3714 |
| 4 | RX3/high | 2.3325 | 3.5331 |
| 5 | RX3/high | 2.3890 | 3.5896 |

控制臂的下一候选也全部由局部2 pp约束阻止，涉及 RX3 high/mid/mid urban、RX1/4/6 low urban、RX4 high urban；完整多组失败明细保留在 calibration JSON。这里分析的是已保存轨迹，没有新校准、novel阈值扫描或额外候选。

## RX、六景和真实 TX

下表 R_U 均按该组未知 TX 宏平均；source RX 已见。high/mid/lowS/highU/midU/lowU 为郊区高/中/低、城区高/中/低。unknown RX×scene30组仅诊断，42组的逐法完整指标见评分文件。

| 组 | control common / budget R_U % | joint common / budget R_U % | 主 A_K / C_K % | 主原F0损失 pp |
|---|---:|---:|---:|---:|
| overall | 11.6825 / 20.2540 | 24.9206 / 20.2857 | 90.9000 / 98.2741 | 0.0815 |
| RX1 | 5.8730 / 8.0952 | 21.5873 / 11.2698 | 93.2222 / 99.3704 | -0.1296 |
| RX3 | 16.9841 / 29.8413 | 25.0794 / 26.0317 | 85.1111 / 96.3704 | 0.0370 |
| RX4 | 4.1270 / 18.2540 | 26.5079 / 19.2063 | 91.2778 / 98.4074 | 0.3519 |
| RX6 | 15.0794 / 20.7937 | 29.6825 / 20.7937 | 93.1667 / 99.1296 | -0.3333 |
| RX8 | 16.3492 / 24.2857 | 21.7460 / 24.1270 | 91.7222 / 98.0926 | 0.4815 |
| high | 22.1744 / 28.6785 | 42.0373 / 28.5350 | 97.2267 / 98.3022 | 0.6862 |
| mid | 15.4617 / 25.5757 | 33.4590 / 27.2980 | 96.9736 / 98.4903 | 0.4690 |
| lowS | 13.1689 / 21.0810 | 29.5858 / 22.8601 | 95.4734 / 98.6166 | 0.1315 |
| highU | 16.2814 / 23.5439 | 28.9483 / 23.0955 | 96.6179 / 98.7327 | 0.3345 |
| midU | 13.4509 / 19.3697 | 22.3291 / 19.3269 | 91.7618 / 98.3963 | -0.0479 |
| lowU | 7.3490 / 17.1992 | 16.4911 / 15.2128 | 83.7326 / 97.8144 | -0.1101 |

| 已知TX（各4500） | control budget正确接收 | joint budget正确接收 | 主A_K % | 主C_K % |
|---|---:|---:|---:|---:|
| 14-10 | 4020 | 4023 | 89.4000 | 98.6222 |
| 14-7 | 3511 | 3567 | 79.2667 | 98.0444 |
| 20-15 | 4337 | 4332 | 96.2667 | 98.7333 |
| 20-19 | 3804 | 3881 | 86.2444 | 97.3556 |
| 6-15 | 4312 | 4324 | 96.0889 | 97.9333 |
| 8-20 | 4419 | 4416 | 98.1333 | 98.9556 |

| 未知TX（各1050） | control common / budget拒对 | joint common / budget拒对 | 主R_U % [95% Wilson] |
|---|---:|---:|---|
| 19-15 | 151 / 291 | 273 / 229 | 21.8095 [19.4169, 24.4076] |
| 19-16 | 16 / 73 | 174 / 75 | 7.1429 [5.7363, 8.8618] |
| 7-1 | 201 / 274 | 338 / 335 | 31.9048 [29.1558, 34.7856] |

评分保存逐真实类别 Wilson 95%区间；它们只描述这组源记录，不是新 RX/seed/episode 的泛化区间。场景内每类数目不同，宏平均不能换成 pooled。源 novel 跨 RX1NN 完全继承相同单位特征：joint79.968254%、control79.206349%、原F078.793651%，两臂结构 guard 通过；这是源结构代理，不是未来匿名注册 A_N。

## 成本、观察与结论边界

选中神经方法部署仍为 F232993＋aux966＋G266951=500910 参数；新状态只部署六个FP64阈值（48 B/策略），不部署OE行bank/风险轨迹。两臂各保存两策略12τ（96 B），不是新增NN参数。历史 F/G训练、原父增量和恢复缓存成本保留；本轮对同一物理OE的重复前向不增加样本量。

| 阶段 | 实际PID | 进程墙钟 s | 内部source计时 s |
|---|---:|---:|---:|
| control source（CUDA_VISIBLE_DEVICES=4） | 3090989 | 5.388733 | 4.676904 |
| joint source（CUDA_VISIBLE_DEVICES=2） | 3090990 | 5.848338 | 5.053823 |
| control / joint CPU score | 3091219 / 3091220 | 2.656792 / 2.480302 | N/A |

driver3090943，总墙钟10.486376 s，两臂并行，不能相加或当作逐包部署延迟。首次独立观察已核实两个 source 的 live PID、正确 CWD/argv 和 CUDA_VISIBLE_DEVICES4/2，但 nvidia 尚未采到 owned compute PID；第二次已全部退出。**没有独立GPU显存占用样本或峰值证据**，不能把分配卡号当显存观察，也未为补证据重跑。source日志control532→627 B、joint523→600 B为两次观察的起末大小，不代表捕获完整运行期增长曲线。

独立终态 source/CPU score 全部 COMPLETE、pipeline ANALYZED_SOURCE_ONLY，live/owned GPU均空、foreign stop0。21项测试、34子测试 PASS（5.56 s），唯一P0/P1审查PASS；612条已知 trapz弃用警告不影响测试结果，没有新增重测。完整读取22份JSON、6550条内嵌trace、2次observations、7份阶段日志及全部8份当前NPZ；阶段日志无异常/非有限/OOM信号，未读取真实IQ、私有truth文件或权重，也未新增fit/预测/评分。

可复核：[控制评分](/E:/type10-7/local_artifacts/pamf_phase3_star_oe_risk_budget_20261030_r01/rows/detach_control/score/results.json)、[联合评分](/E:/type10-7/local_artifacts/pamf_phase3_star_oe_risk_budget_20261030_r01/rows/joint/score/results.json)、[完整主校准轨迹](/E:/type10-7/local_artifacts/pamf_phase3_star_oe_risk_budget_20261030_r01/rows/joint/predict/calibration/complete.json)、[pipeline](/E:/type10-7/local_artifacts/pamf_phase3_star_oe_risk_budget_20261030_r01/pipeline_state.json)、[独立观察](/E:/type10-7/local_artifacts/pamf_phase3_star_oe_risk_budget_20261030_r01/observations.jsonl)。

本轮证伪了这套固定贪心预算规则能在主模型上改善保留未知拒识的预期；它不证明全部阈值族无效，也不授权继续按 novel 搜索。A_N/H_OW=N/A、目标/正式RX访问0，完整 Phase3 目标保持 ACTIVE；没有换主、扩seed或启动下一实验。
