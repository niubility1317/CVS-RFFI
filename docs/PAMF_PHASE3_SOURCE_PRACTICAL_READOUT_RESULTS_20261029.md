# Phase3 源端星地域读出与冻结跨 RX 开发结果

**两条开发行均为 NO_PROMOTION。** 预定 augmented MLP 在 C0 的未知拒识为 96.8750%，单项超过 clean 95%；但 known 接收率 90.4514%、FRR 9.5486%、相对原 F 总损伤 8.1597 pp，完整保护失败。P1 的未知拒识仅 38.8889%，接收率 93.0556%、总损伤 4.1667 pp，未知和已知保护均失败。source 两 arm 的 MLP 也未通过完整目标，不能以同批 known-V 校准合格宣称独立确认。

run：20261029-phase3-source-practical-readout-s2026092701-r01；执行代码 ca60bdf20090f664fbcf27682138d63cfd2751dc，预登记提交 2a6c4e68dba38c51133dab8fa26c33f55c5a8184。主候选始终为 augmented_readout::mlp16_oe；calibration_only 与其余六方法均诊断，没有按本轮 source／target 分数换选。实际终态 VERIFIED／ANALYZED_DEVELOPMENT_ONLY，总 Phase3 仍未完成。

## 数据、监督与固定比较

原完整六类 F 的 E50 使用 source L6300／隐藏 TX 真值的 U56700，V27000 仅验证；source 五 RX 为 1-19、18-2、19-2、2-19、3-19（index 1/3/4/6/8），三日为 2021-03-08、03-15、03-23。旧类顺序 14-10、14-7、20-15、20-19、6-15、8-20。L/(L+U)＝6300/63000＝10%源训练比例，V27000是另行使用的校准已知标签，不计入该分母。E51 uniform_oe 在 E50 后固定新增 222 步、累计 11322 步，其祖先另有 21 台／6134 条源 OE 池暴露，与本次读出四台 OE 的监督及预算分开。

本次 OE4 为 11-14、14-19、17-20、18-19；拟合只有 known／unknown 成员监督，不用 OE TX/y。novel3 为 19-15、19-16、7-1，各 1050 条；其 IQ／descriptor／logits 先由冻结网络导出，但不被 fit／τ 消费，拒识判决在两 arm 阈值冻结后生成，全部 source 预测完成后才由独立 scorer 连接 TX。V 标签只校准／评价，不拟合读出或早停；原 U_s 本轮读取 0。目标零 TX 标签不等于全流程没有源监督。

| source 角色 | 原物理数 | 四视图曝光 | 用途 |
| --- | --- | --- | --- |
| L | 6300 | 25200 | known 训练 |
| 原四台 OE | 4200 | 16800 | unknown 训练 |
| known V | 27000 | 108000 | known-only 校准／评价 |
| novel3 | 3150 | 12600 | 冻结后预测／独立评分 |
| 合计 | 40650 | 162600 | 重复计算 view，不增加独立物理记录 |

四视图为 clean、low suburban、mid urban、low urban；后三者为 canonical practical residual 后接冻结 DURA。E51、原默认六类头、原 clean-L6300 聚合几何、DURA v1.0 R69 E20/4940 均不更新、不重拟合几何。导出仅 compact descriptor100／static5／logits6 与必要元数据，没有 raw IQ／全分支逐样本 bank；部署只加载聚合量、读出和阈值。

开发 Q 为 RX24-1（index 26）、2021-03-15 的同序 864 条：旧六类各 96＝known576，novel3 各 96＝unknown288。C0 是无 practical／G 的独立 clean；P1 是旧固定 star received 经相同冻结 DURA，不能读取逐条 clean 参考。两条件仍只有 864 条原物理 Q。source 与开发 RX 不交，日期在其他 source RX 已用，非严格未见日期外推；novel3 和此开发 Q 已评价，不是新 TX 盲测。target U=0、formal RX24-18 U/Q=0，无 target 适应、未知聚类或匿名注册。

original state 精确复用旧 C0／P1 D0 和同输入原 F baseline；两 current arm 共本次同一 forward／head／descriptor／static5。实际 C0、P1 相对历史 D0 的 closed 差异计数均为 0；全部七方法 original 的总体／scene／TX 指标与旧评分逐项相同。当前两 arm 的公平比较同时控制了 source 几何、F/G、目标输入和前向数值。

## 完整拟合曲线与 known-only 校准

calibration_only 原 linear／MLP 权重及 mean/std 精确复制，新增拟合 0；augmented_readout 的 linear101／MLP1633 用 L25200＋OE16800，mean/std 仅来自这两训练角色，seed2026092701、AdamW lr0.001／wd0.0001，各固定 200 次 CPU 全批量原 balanced binary loss。

| 读出 | 有限点数 | 首→末 loss | 全程均值 | 首25均值→末25均值 | 上升步数 |
| --- | --- | --- | --- | --- | --- |
| linear_oe | 200 | 0.693147 → 0.227898 | 0.324002 | 0.551308 → 0.232227 | 0 |
| mlp16_oe | 200 | 0.855322 → 0.151404 | 0.258193 | 0.564354 → 0.155238 | 0 |

已读取全部 400 点，两条 loss 均逐步下降且末段仍下降；固定 200 步结束，无 V／novel 早停，不能从训练拟合下降宣称未知泛化收敛。实际二元训练计时 3.6147 s，含加载／保存的 fit 子进程为 6.0401 s。

| arm MLP | global τ | 30组最大 FRR % | 30组最大总损伤 pp | 零拒绝可行／约束过／下一τ过 |
| --- | --- | --- | --- | --- |
| calibration_only | -2.089528561 | 4.2778 | 2.0000 | true／true／false |
| augmented_readout | -2.164471626 | 2.9444 | 2.0000 | true／true／false |

两 arm 全七方法均为 no_rejection_feasible=true、constraints_passed=true，下一候选 τ 均失败；未进入“最低分接收全部”的不可行兜底。校准只用同一 known V 四视图，保护总体／五 RX／四 view／20 个 RX×view 共 30 组：macro FRR≤5%、原 F 同输入 closed−open≤2 pp；Fraction 精确预算、整组 ties、每方法一个最大全局 τ。旧 encoder 0.5 pp 诊断为 false，未追加为硬目标。旧读出重校准只移动工作点；增强读出的 mean/std 和分数尺度已改变，不能直接比较跨模型 τ 数值大小解释机制。

## Source：保护通过，未知拒识仍失败

以下各格为 calibration_only／augmented_readout 的未知宏拒识率（%）；源验证每 view known27000＋novel3150，均为已见 source RX 和重复 novel 开发。五个静态方法两 arm 相同，只有学习读出改变。

| 方法 | clean | low suburban | mid urban | low urban |
| --- | --- | --- | --- | --- |
| msp | 47.8730／47.8730 | 24.7619／24.7619 | 21.0794／21.0794 | 17.4921／17.4921 |
| maxlogit | 60.6032／60.6032 | 25.2063／25.2063 | 22.4444／22.4444 | 16.3175／16.3175 |
| margin | 27.0476／27.0476 | 16.0952／16.0952 | 13.8095／13.8095 | 10.8889／10.8889 |
| energy | 62.9206／62.9206 | 22.5397／22.5397 | 20.5397／20.5397 | 14.9524／14.9524 |
| cosine | 22.3175／22.3175 | 14.7302／14.7302 | 11.8413／11.8413 | 10.7937／10.7937 |
| linear_oe | 34.6984／69.3651 | 27.9048／49.7460 | 26.6349／40.6032 | 26.2540／34.8254 |
| mlp16_oe | 83.3016／78.0317 | 58.7937／51.1111 | 46.9841／41.0476 | 35.8413／32.6984 |

MLP 的四 view known 接收率为 control 97.2074%至99.2259%、aug 97.8593%至99.3074%，各view总损伤均在2 pp内；这只是同批V校准经验保护。增强 MLP 的 source unknown 在四 view 均低于控制，训练拟合下降没有带来本组 novel 拒识提高。

| source RX index：MLP未知 % | clean | low suburban | mid urban | low urban |
| --- | --- | --- | --- | --- |
| 1 | 94.7619／90.1587 | 74.7619／55.8730 | 58.7302／44.9206 | 42.5397／31.7460 |
| 3 | 78.7302／71.5873 | 41.5873／40.1587 | 34.4444／34.7619 | 23.8095／28.4127 |
| 4 | 83.6508／74.6032 | 72.0635／61.1111 | 58.2540／47.3016 | 44.7619／34.9206 |
| 6 | 85.3968／82.2222 | 65.8730／56.1905 | 53.1746／45.2381 | 45.5556／38.8889 |
| 8 | 73.9683／71.5873 | 39.6825／42.2222 | 30.3175／33.0159 | 22.5397／29.5238 |

两 arm 的 MLP 在全部 20 个 RX×view 单元均未达到 clean95／practical90，整体 source goal=false。完整七方法的逐 RX、TX、view、RX×view、TX×view 保护和置信区间见下文索引；跨 view pooled 只作描述，不以重复物理记录计算独立置信区间。

## 开发主候选与全部方法

| 条件 | state／MLP | 未知拒识 | known 接收 | FRR % | 开放正确 | 原 F 总损伤 pp |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | original | 286/288（99.3056%） | 465/576（80.7292%） | 19.2708 | 361/576（62.6736%） | 17.0139 |
| C0 | calibration_only | 277/288（96.1806%） | 500/576（86.8056%） | 13.1944 | 391/576（67.8819%） | 11.8056 |
| C0 | augmented_readout | 279/288（96.8750%） | 521/576（90.4514%） | 9.5486 | 412/576（71.5278%） | 8.1597 |
| P1 | original | 165/288（57.2917%） | 420/576（72.9167%） | 27.0833 | 307/576（53.2986%） | 19.4444 |
| P1 | calibration_only | 100/288（34.7222%） | 515/576（89.4097%） | 10.5903 | 382/576（66.3194%） | 6.4236 |
| P1 | augmented_readout | 112/288（38.8889%） | 536/576（93.0556%） | 6.9444 | 395/576（68.5764%） | 4.1667 |

C0 三 states 的新 F 闭集均为 451/576＝78.2986%，原 F 为 459/576＝79.6875%；P1 均为 414/576＝71.8750%，原 F 为 419/576＝72.7431%。接收率只问是否未拒绝，开放正确还要求身份分类正确，不能将接收增加写成原 head 准确率提高。

相对 calibration_only，augmented MLP 在 C0 多接收 21 条／多开放正确 21 条／多拒未知 2 条，在 P1 分别多 21／13／12 条；这是固定两 current arm 的实际收益。相对历史 original，C0 开放多 51 条但少拒未知 7 条；P1 开放多 88 条却少拒未知 53 条，不能只报 known 收益。

| 条件 | 主景：aug MLP | 未知 % | 接收 % | FRR % | 开放 % | 总损伤 pp | 完整目标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | clean | 96.8750 | 90.4514 | 9.5486 | 71.5278 | 8.1597 | 失败 |
| P1 | practical_low_suburban | 47.9167 | 94.7917 | 5.2083 | 74.4792 | 2.6042 | 失败 |
| P1 | practical_low_urban | 33.3333 | 92.7083 | 7.2917 | 65.6250 | 5.2083 | 失败 |
| P1 | practical_mid_urban | 35.4167 | 91.6667 | 8.3333 | 65.6250 | 4.6875 | 失败 |

P1 各主景为 known192／unknown96，逐TX等量。以下七方法均列出 control／aug，所有比例为 %，总损伤为 pp。C0 linear 的 unknown 单项也超过95，但 known 保护失败；P1 没有方法同时达到未知与已知要求，不能切换主方法救本轮。

| 条件 | 方法 | 未知拒识 | known 接收 | 开放 | 总损伤 |
| --- | --- | --- | --- | --- | --- |
| C0 | msp | 79.5139／79.5139 | 92.1875／92.1875 | 74.6528／74.6528 | 5.0347／5.0347 |
| C0 | maxlogit | 92.7083／92.7083 | 91.4931／91.4931 | 74.4792／74.4792 | 5.2083／5.2083 |
| C0 | margin | 47.2222／47.2222 | 91.3194／91.3194 | 74.3056／74.3056 | 5.3819／5.3819 |
| C0 | energy | 92.7083／92.7083 | 90.6250／90.6250 | 73.4375／73.4375 | 6.2500／6.2500 |
| C0 | cosine | 30.9028／30.9028 | 99.3056／99.3056 | 78.1250／78.1250 | 1.5625／1.5625 |
| C0 | linear_oe | 53.4722／95.1389 | 100.0000／93.4028 | 78.2986／73.7847 | 1.3889／5.9028 |
| C0 | mlp16_oe | 96.1806／96.8750 | 86.8056／90.4514 | 67.8819／71.5278 | 11.8056／8.1597 |
| P1 | msp | 21.5278／21.5278 | 92.1875／92.1875 | 69.4444／69.4444 | 3.2986／3.2986 |
| P1 | maxlogit | 21.8750／21.8750 | 92.7083／92.7083 | 69.9653／69.9653 | 2.7778／2.7778 |
| P1 | margin | 13.5417／13.5417 | 92.8819／92.8819 | 69.4444／69.4444 | 3.2986／3.2986 |
| P1 | energy | 20.1389／20.1389 | 93.0556／93.0556 | 69.9653／69.9653 | 2.7778／2.7778 |
| P1 | cosine | 5.2083／5.2083 | 98.4375／98.4375 | 71.0069／71.0069 | 1.7361／1.7361 |
| P1 | linear_oe | 26.7361／40.9722 | 93.9236／92.5347 | 69.4444／69.0972 | 3.2986／3.6458 |
| P1 | mlp16_oe | 34.7222／38.8889 | 89.4097／93.0556 | 66.3194／68.5764 | 6.4236／4.1667 |

## 逐 TX 与排序边界

旧六类逐TX的完整闭集／接收／开放计数见评分索引；各TX均为96条。C0 14-7 全接收但仅21条分类正确，不能用阈值修复该身份混淆；6-15 仍误拒46条。P1 6-15／8-20 分别误拒20／17条，开放正确69／71条。逐 TX 的剩余误拒与分类问题均保留，没有按 TX 制定专属规则。

| unknown TX：aug MLP | C0 拒识 | P1 拒识 |
| --- | --- | --- |
| 19-15 | 94/96（97.9167%） | 56/96（58.3333%） |
| 19-16 | 95/96（98.9583%） | 22/96（22.9167%） |
| 7-1 | 90/96（93.7500%） | 34/96（35.4167%） |

历史 planned discovery 两类合计 C0 189/192＝98.4375%、P1 78/192＝40.6250%；planned later 7-1 为 C0 90/96＝93.7500%、P1 34/96＝35.4167%，两行 later 保护也失败。本 run 没有读取 U，这些只是 scorer 的历史数据角色，不表示当前已无标签见过或适应。

增强 MLP 的 known-positive AUROC：C0 0.978709→0.982898，P1 0.709262→0.861075（相对 control；重校准本身不改变 AUROC）。本等量池若一标量阈值同时接收 known≥95%、拒 unknown≥90%，至少0.95×0.90＝0.855的known–unknown配对必严格正确，故AUROC≥0.855是必要条件。P1总体虽超过下界，不证明存在合格τ；low urban为0.821994、mid urban为0.843967，仍低于0.855，故当前分数只逐景移动τ也无法同时满足这两个目标。C0超过clean必要界0.9025也不证明完整目标可行。这是本项目数学推断，仅限当前分数／已评分池；未扫描oracle ROC或用V选阈值。

## 执行、成本与可核查证据

本轮一次独立 P0/P1 审查发现两个 dev API 问题并定向修复，最终29 tests＋24 subtests PASS；未重复全审。一次 N607 preflight、发布归档本地／远端比较和 compile 通过。export 首先用实际冻结 checkpoint 对2条合成物理IQ／8视图 smoke PASS（G6条），source／target真实读取0；之后开放source角色。C0/P1也各先真实checkpoint合成smoke再Q；smoke在log／complete，没有另造smoke文件。

| stage（整子进程） | 墙钟 s |
| --- | --- |
| export | 437.2247 |
| fit | 6.0401 |
| calibrate | 6.5357 |
| predict_source | 3.1148 |
| score_source | 6.0830 |
| C0_predict | 5.1413 |
| C0_score | 0.3559 |
| P1_predict | 7.1190 |
| P1_score | 0.3161 |
| 全 pipeline | 472.1788 |

正常导出前向计数：E51 162600、原 F 120600、DURA／channel各121950，另加上述合成smoke；L/OE不跑无用途原 F baseline。开发每条件只一次 current F前向、P1一次G，旧original与Q原Fbaseline复用。以上整进程包含加载、I/O、导出／拟合／写盘，不能当逐包或单模型延迟。

独立live核实driver PID2624334与export PID2624355的CWD／完整命令，export CUDA_VISIBLE_DEVICES=3；owned GPU计算快照为510／834／900MB等采样，最高已观察值900MB，不是峰值。export完整63行log由539B增长至4879B；9jobs最终COMPLETE、owned进程已退出、foreignstop=0。C0/P1赋GPU3来自driver状态，不能把其他owner显存或未捕获的开发GPU运行期说成本run峰值证据。

本分析完整读取46份取回元数据／日志、两条完整200步loss及全部三份评分，核对1638个指标分组的计数／宏指标、逐view／RX／TX加总、D0精确复用与delta一致。只写本报告，没有读取真实IQ、truth、weights，没有重跑或新启动。详细索引（七方法名顺序固定msp/maxlogit/margin/energy/cosine/linear_oe/mlp16_oe）：

- [source 完整评分](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/source/score/scores.json)：arms.<arm>.methods.<method>.{by_view,by_receiver,by_transmitter,by_receiver_view,by_transmitter_view,goals}；包含全部四view／五RX／TX及交叉明细。
- [C0评分](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/rows/C0/score/scores.json)、[P1评分](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/rows/P1/score/scores.json)：states.<state>.methods.<method>.{overall,by_scenario,by_transmitter,planned_discovery_unknown,planned_later_unknown}及deltas／goals。
- [全部训练loss](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/source/fit/training.json)、source/calibration/<arm>/complete.json、各source／row完成标记，记录冻结、可行性与truth-last顺序。
- [最终post-state](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/post_state.json)、[流水线](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/pipeline_state.json)、[完整观察](E:/type10-7/local_artifacts/pamf_phase3_source_practical_readout_20261029_r01/observations.jsonl)与logs/<stage>.log，保留真实运行／退出和观察边界。

这仅一个模型seed、一个开发RX与三个star主景，非多seed、完整RX或七景证明。本轮支持“源增强读出改善部分开发排序和known开放率”的有限结论，没有满足clean完整保护或星地未知目标。下一研究可优先检查source合法星地／G输入下编码器与原默认头的匹配联合增量；这不是本轮已执行或已证明有效的方法，不修改本轮τ／强度救分，不据结果扩seed。当前formal隔离、targetU0、匿名注册0，总Phase3保持未完成。
