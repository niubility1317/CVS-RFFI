# 完整源域 F＋跨 RX：E30 无星地与普通拼接匹配参照

状态：RUNNING，B00 已 ANALYZED；B01 健康训练中，全 run 未完成。采用用户最新“训练预算提升到 E30”，不续训或覆盖旧 E20。唯一矩阵、seed 和路径在 experiment.json。

## 研究问题与固定比较

B00 为原 ordinary IQ 的精确重复；B01 为 ordinary IQ＋完整 practical(同一 ordinary IQ)。两行均 scratch E30，同 seed2026092701、同4L成本和初始化规则、原U/ownEMA/旧门控。B00不是标准2L结果。没有新增Ag、对齐、课程、诊断头或SSL规则。

E20源训练的sat CE仍高，只说明预算下拟合尚不充分，不能保证延长训练有效。E30同时改变阶段长度及cosine时间轴，不能将差异唯一归因于多10轮或某个物理损伤。旧E20正式成绩不用于本轮参数选择。

探索报告目标：B01相对同预算B00 urban至少+3pp、low至少+5pp、clean下降≤.5pp；low绝对>60%、每个suburban距自身clean≤2pp。这些不是保证或目标反馈晋级门槛。两行固定FINAL30后无论分数均自动七视图测试，不按目标成绩调参、挑种子、重跑。

## 模型、数据、预算

F＋CROSS_RX：SIG_TIME_X3、Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、joint160、原默认cos30、900722参数/206状态，骨干和头原流程联合更新。真实代码只有GroupNorm/LayerNorm，无BatchNorm running stats；不采用BN污染解释。

复用原VALIDATED_ONCE句柄/split392005，全部RX1/3/4/6/8、day1/2/3、6TX，L6300/U56700隐藏/V27000，7/63/30。全源RX训练，无heldRX/fold/Ldev，不将V并入L、不解封U TX，source训练不构建target loader。原角色物理ID已完全匹配，未新增数据验证或哈希链。

每行30轮＝20L＋10U，222更新/轮＝6660。AdamW峰值LR.0002/min.000001、weight decay.0001，原native cosine按total30计算，Sinc物理clip5不变。拼接256，两RDrop forward复用同一IQ，共4L；两半各自归一化后各.5，LS.01/KLalpha1。SupCon仅clean半，time_down/t2.pool、temp.2/max.1/warm5/ramp10，无额外前向。原U/ownEMA/旧门控保持，不给U新增As或对比损失。

每行L loader852480包、L网络3409920包，U loader预计567000包；student/teacher/sourceV/正式prediction/合成smoke成本与墙钟实际分列。两行新增60 model epochs/13320更新，无父训练成本；预算耗尽不等于收敛。

legacy E20配置只用于未改模型工厂，make_args明确覆盖30/20/10，不加载旧权重。launcher第一步用原全源B00 FINAL20/4440的真实默认头严格兼容smoke2合成包后丢弃，随后两学生各自scratch；该CP旧20轮成本独立列，不是祖先，不加载E50/E200。新来源只接受本行FINAL30/6660。

## IQ、信道与测试

25MHz/IQ256/equalized1，原裁剪补齐后复RMS/center=false。原ordinary工厂及private stream20261016051保留，普通增强日程依据label_epochs20；B01训练low_suburban/mid_urban/low_urban原均匀nominal。保留ordinary PA/AWGN/multipath等，不能称完全无普通增强。

As原residual/post_sync、EQoff/G1、reference_snr_db30/fc2.45GHz，质量适配/相位跟踪/RXIQ补偿开启、reflect、原AGC；不另加As后RMS，不反演TX。RX硬件按RX/day/eq跨TX稳定，channel seed202610172201随epoch/view刷新，与正式随机流分离。全部合法有标签sat参加CE，无学生置信筛选，不删除困难包或减弱物理信道。实际配置/__file__保存。

每行固定168000目标物理样本，clean＋六practical。原正式received IQ/index/SNR/边界不变。FINAL30冻结默认头→全部七视图prediction完成→独立exec scorer连接truth；无训练对象/teacher存活到scorer。真值打开前核对全视图/ID/logits/argmax。正式无逐包状态旁表时不虚构其LOS/lock准确率。

## 执行、检查与停止

代码 commit：`bafba40e365d0914a2e86f861fb55a49f1cd3997`，已推送且独立远端OID相同。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E30_20261017_r01`。普通N607 CVS-RFFI/FP32/CPU线程2，不用管理员。

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E30_20261017_r01/experiments/cvs_fullsource_concat30_25/runner.py
```

实际本地检查：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/local_checks.py
E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/local_model_checks.py
F:/App/miniconda3/python.exe -m compileall -q E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_fullsource_concat30_25
```

16协议负测、native30/20L10U/cosine端点、6660与旧4440/跳步拒绝PASS。真实F合成1028NN/2更新/0真实IQ或CP，206初态相同、默认头与SIG_TIME/Sinc梯度、4L/同IQ/首对SupCon/deepcopy绑定PASS；compile PASS。已知本地Torch/NumPy导入警告保留，未用ndarray桥。一次限定独立P0/P1 PASS，无重复review或额外gate。

一次N607只读preflight及句柄/路径核对VERIFIED：普通账户、原角色匹配、兼容CP metadata合法、固定formal carrier有效，新run/release/launch路径未占用、磁盘足够。其他owner健康作业保持。一个owner/dispatcher，共享capacity lock，每GPU总进程≤3且free≥12000MB，计入其他owner。发布后实际读回单归档SHA比较/remotecompile/真实CP smoke/PID/CWD/argv/GPU/log增长。

停止：首次本run来源/角色/CWD/输出碰撞/非有限loss或gradient/跳过更新/未6660/执行或prediction闭合/提前truth等技术错误，保留输出、停止进一步本run启动，已启动健康行继续，不自动重试，不改其他任务。低分或负收益不技术停止；恢复先核实同row进程和prediction，不覆盖。

run：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01`；本地：`E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017`。每行source/prediction/scoring/log/config及动态GPU见experiment.json。预期完整30轮/6660步、训练序列/physics/覆盖、骨干头移动、U隐藏、前向成本、FINAL30/freeze、七视图预测评分、逐row/seed结果及汇总；IQ/CP/原始log/机器状态不入Git。

## 路线边界与未完成项

旧3至15dB/locked-degraded有界课程仅极小low增量、clean保持失败；fixed additive clean监督也跨seed失败，本轮不以改RX/系数/预算重启。源noise/fixed瓶颈成立，强状态耦合/whole梯度冲突前提失败，不据此加梯度手术或对齐。

B00结果已完成；B01完成前尚无E30星地增量比较，亦未证明新Ag、四格交互、clean净正、low>60%、suburban接近clean或跨seed联合复现。本轮只建立新预算参照，不把它当联合目标完成。实际执行与结果返回后更新原run。

## 实际启动读回

唯一代码归档传输SHA比较VERIFIED、remote compile PASS。普通账户dispatcher第一步全源E20 CP兼容smoke PASS，2合成NN/0真实IQ/0query/0更新，随后丢弃。两行在GPU3/4实际运行，PID/start_ticks/CWD/argv/GPU归属已独立核对，准确证据在本地postlaunch_first.json；不以launch返回码代替完成。

实际resolved args均为epochs30/label20/pseudo10/nativecosine、baseline_ckpt与teacher_ckpt为空、ownEMA/U开启。full_source_roles逐ID匹配，L6300/U56700隐藏/V27000、全部5RX、held_RX=null/extra_Ldev=false/target_loaders0；实际模型900722参数与代码commit一致。其他健康任务未修改。训练完成后由原launcher自动执行七视图prediction及独立truth-last，本轮没有目标指标或性能结论。

日志增长独立复核 VERIFIED：同一两行PID/start_ticks/CWD/argv/GPU持续一致，B00日志6425→16724字节并完成E2/30，B01为6425→11575字节并完成E1/30；queue无failure、两行均active。证据postlaunch_growth.json。只作执行健康确认，不将早期V分数当作性能或收敛结论。

## 独立完整分析入口

新增独立分析入口 `analysis/cvs_fullsource_E30_results_20261017.py` 与4项聚焦测试。分析只在所请求行全部七视图冻结预测及独立评分完成后打开truth，不读取IQ/checkpoint、不运行网络或更新参数；严格拒绝旧E20、非6660、非20L10U、额外heldRX/角色改变及提前truth等记录。4项测试PASS，包含手算宏F1/混淆矩阵、非法类别、预算与完整性拒绝、第二行不完整时在truth及输出创建前停止；compile PASS。

工具将完整读取每行30轮/6660步、SupCon/Sinc/训练L-U序列与全部physics，复算正式252个分层混淆矩阵、paired修复/新错及固定B00 clean正确集合保持率，分别记录student/teacher/prediction与墙钟成本。图包括正式准确率、30轮ordinary/second CE及KL/原U gate/源V/墙钟、正式RX表现、训练状态/锁定/质量代理/源RX覆盖。训练覆盖图不表示准确率，质量代理不表示总SNR。实际生成后才检查和提供图表；当前没有伪造结果或图路径。训练发布代码仍为 `bafba40e365d0914a2e86f861fb55a49f1cd3997`，新分析器不会热改该发布。

## 原记录时间上下文核查

复用与native相同的既有ManySig.pkl载体反序列化，只检查原源RX1/3/4/6/8、day1/2/3、6TX容器的shape/dtype元信息，没有索引IQ数值、读取U TX真值、加载CP或运行网络。90个equalized=1源容器均为[1000,256,2]/float64（metadata同时看到eq0也相同），原packet实际已经256点；native转换float32并转置后，out_len256的crop/pad分支没有触发，dataset层RMS作用于完整256点、center=false/epsilon1e-12。证据保留在本地source_container_shapes.json，CPU检查约1.939308秒，0NN/更新；没有修改原正式IQ或信道。

因此，使用更长同一物理record无缝截取256点的候选在当前表示下NOT_APPLICABLE，不是识别性能FAIL。sig_i是独立物理signal，不把相邻packet拼成上下文，也不跨L/U/V借用样本；没有构造新的crop增强或启动额外模型。该事实只约束需要真实额外上下文的候选，不从元信息推断其他增强有效或无效。

E30阶段切换的独立prefix核对：无星地B00已完成24/30，前20轮U/teacher计数均0，第21至24轮U loader及ownEMA teacher eval各226800（4×56700）；全部完成epoch的L/4L/U/teacher计数均符合20L10U。其24个完整epoch记录train_pseudo_truth_available均0。B01已完成12/30，仍为label阶段，U/teacher均0。该核对不衡量U伪标签准确率，也不代表最终性能；同一两个PID/start_ticks仍live、无failure。证据source_phase_audit.json与phase_status_after_shape_audit.json。


## B00 已完成：E30 独立正式结果

B00-s2026092701 已完成训练、全部七视图冻结预测、独立 truth-last 评分及分析，状态 ANALYZED；B01 仍为健康 RUNNING，此处没有读取其未完成结果，也未按首行成绩改动任何训练配置。全 run 尚未完成。

训练代码 commit 为 bafba40e365d0914a2e86f861fb55a49f1cd3997，独立分析代码 commit 为 a081a6c047462aa088394e4f233a05fd4f33ce9f。实际分析命令：

```text
python -u <committed analysis/cvs_fullsource_E30_results_20261017.py> --row B00 --truth <existing registered truth.json> --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01/analysis_B00_r01 --commit a081a6c047462aa088394e4f233a05fd4f33ce9f
```

该已提交脚本通过普通账户 Python 启动为独立 CPU 进程，原训练发布目录没有改动。以完成文件、完整 stdout/stderr、已取回图表与逐项重计数读回 VERIFIED；退出码不作为唯一完成证据。分析只打开已冻结且评分完成行的 prediction 与 truth，0 IQ/CP 读取、0 网络前向、0 新更新。126 个 overall/RX/day/TX 混淆矩阵准确率与宏 F1 全部独立复算一致。

以下均为原正式 168000 个物理样本、模型 seed2026092701、FINAL30 checkpoint 内默认分类头结果，单位为 %：

| 测试环境 | 准确率 | 宏 F1 | 最差 RX 准确率 |
|---|---:|---:|---:|
| clean | 85.156548 | 84.891668 | 74.433333 |
| practical_high | 46.513095 | 41.659905 | 37.666667 |
| practical_mid | 44.677976 | 40.281577 | 36.908333 |
| practical_low_suburban | 42.714286 | 38.762439 | 35.320833 |
| practical_high_urban | 45.233333 | 40.902104 | 36.933333 |
| practical_mid_urban | 37.582738 | 34.857566 | 31.529167 |
| practical_low_urban | 31.985714 | 30.218017 | 27.700000 |

urban 三场景均值为 38.267262%。无星地组不构成星地增强收益结论；普通拼接组完成前没有同预算 As 增量、clean 代价或固定 B00 clean 正确集合保持率。此处仅一个模型 seed，没有跨 seed 重现证据。

完整 30 轮/6660 步训练记录已读；L/U 实际物理唯一数为 6300/56700，重复 loader 暴露为 852480/567000 包；L 网络前向 3409920 包。SupCon 记录6660、实际启用5550步，Sinc 更新6660步；默认头和骨干联合更新，U TX 真值始终隐藏。原 stdout 的300处 NaN 文本是未启用诊断字段，训练停止检查没有非有限损失、跳步或错误标志，不据文本计数误报训练故障。

最后源 V 健康准确率98.148148%，最后配对CE0.286571、RDrop KL0.038116；最后10轮旧U门控选中率约1.33%至1.65%，其数值不代表伪标签精确率。该源 V 与正式跨 RX clean85.156548%口径不同，不混用；预算完成不宣称收敛。

实际成本：无父训练，新增30轮/6660更新；student训练3976920包、student评估864002包、ownEMA teacher评估567000包（teacher训练0），源阶段总身份网络前向5407922包。训练墙钟1540.012254秒；正式预测前向1176002包（包含2包执行校验），预测墙钟47.865980秒。最初兼容性检查的历史E20父训练成本单列且权重被丢弃，不是B00的训练继承。

真实输出本地目录：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/analysis_B00_r01。包含 analysis_summary.json、per_row_seed.csv、training_curves.csv、completion.json、figures.json；formal_accuracy、training_curves、receiver_accuracy 的 PNG/PDF 均已生成并逐张查看PNG，未见遮挡或错误标注。原正式无逐样本物理状态旁表，不虚构 LOS/lock 分层准确率；B00 无 As 物理覆盖图属于不适用。

当前唯一未决比较仍是预登记的 B01 普通拼接相对 B00；无新增 Ag、四格交互或联合突破的结论。固定 FINAL30 自动测试继续，目标成绩不回流选模、调参或选择性重跑。
