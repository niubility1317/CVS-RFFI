# 完整源域 F＋跨 RX：E30 普通星地拼接正式结果

状态：ANALYZED。固定的 B00、B01 两行均完成 scratch E30、FINAL30 默认头冻结、clean＋六 practical 全部预测、独立 truth-last 评分及完整同 row 分析。本实验完成，活跃联合增强目标未完成；实际只有模型 seed2026092701，不能声称跨 seed 重现。

主要结论：原完整 practical 训练使 urban 均值从38.267262%升至65.158333%（+26.891071个百分点），low从31.985714%升至53.137500%（+21.151786个百分点），但clean从85.156548%降至82.797024%（−2.359524个百分点）。这证明普通As的组合适应收益，没有证明新的Ag或clean/星地联合突破；low>60%、clean净正及各suburban距clean≤2pp均未全部达到。

## 正式逐行、逐 seed 结果

run：20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01。每行168000个原正式物理样本，7个视图，共2352000个正式决策；全部使用各自checkpoint内原默认分类头，无头重训、诊断分类器或目标拟合。训练模型seed均2026092701；split/data/eval392005、原普通增强private stream20261016051、训练practical stream202610172201均固定，不把不同随机流当成模型seed。

| 环境 | B00 无星地（%） | B01 普通拼接（%） | 差值（百分点） |
|---|---:|---:|---:|
| clean | 85.156548 | 82.797024 | -2.359524 |
| practical_high | 46.513095 | 81.469048 | 34.955952 |
| practical_mid | 44.677976 | 79.041071 | 34.363095 |
| practical_low_suburban | 42.714286 | 76.586310 | 33.872024 |
| practical_high_urban | 45.233333 | 78.176786 | 32.943452 |
| practical_mid_urban | 37.582738 | 64.160714 | 26.577976 |
| practical_low_urban | 31.985714 | 53.137500 | 21.151786 |

| 环境 | B00 宏 F1（%） | B01 宏 F1（%） | B00 最差 RX（%） | B01 最差 RX（%） |
|---|---:|---:|---:|---:|
| clean | 84.891668 | 82.378042 | 74.433333 | 75.012500 |
| practical_high | 41.659905 | 81.085938 | 37.666667 | 73.945833 |
| practical_mid | 40.281577 | 78.672853 | 36.908333 | 71.608333 |
| practical_low_suburban | 38.762439 | 76.221213 | 35.320833 | 69.462500 |
| practical_high_urban | 40.902104 | 77.763079 | 36.933333 | 70.775000 |
| practical_mid_urban | 34.857566 | 63.649699 | 31.529167 | 57.416667 |
| practical_low_urban | 30.218017 | 52.496507 | 27.700000 | 47.995833 |

B01 high/mid/low suburban与自身clean的差距分别1.327976、3.755952、6.210714个百分点，只有high在首轮2pp探索范围内。缩小部分suburban差距不抵消clean损失。预登记urban≥+3pp与low≥+5pp达到，clean下降≤0.5pp未达到；这些只用于终局描述，不回流调参、晋级或选择性重跑。

固定B00 clean正确集合共143063包，B01在high/mid/low urban的保持率为86.394106%、70.494118%、58.020592%。全168000包上，low修复45527包、产生9992个新错，净增35535；clean修复3996包、产生7960个新错，净减3964。该集合是同预算参照的固定集合，两行均scratch，没有训练父模型。

默认头logit间隔除以30后，B00/B01的全样本均值：clean0.147373/0.129577，mid urban−0.026694/0.057697，low urban−0.036008/0.032490。间隔变化与实际识别结果一起报告，不以间隔代替准确率；本实验没有额外间隔、一致性或对齐损失。

clean的最差RX从74.433333%升至75.012500%，但均值下降；原RX1-1从84.275000%降至75.012500%（−9.262500pp），说明最差RX单指标不能代替clean保持。各RX/day/TX全部结果保留在CSV/JSON，不按某个目标RX补训练。

## 源训练证据和收益归因

两行8880条L/U采样序列逐条一致，0条不匹配；每行全部30轮/6660步、SupCon6660记录（5550实际启用）、Sinc6660更新和完整1198行stdout均已读。没有错误标志或跳步。每行300处NaN文本属于未启用诊断字段，不是非有限活动loss；实际活动目标均有限。

B01完整852480条practical记录均被解析，覆盖全部90个TX×源RX×场景以及90个TX×源RX×状态组合。LOS/PARTIAL/BLOCKED为427829/134726/289925；locked/degraded/unlocked为511572/142274/198634。所有源RX/day参加训练，未额外留出RX或L_dev。分层覆盖图表示训练计数，不表示分层准确率。

每条training EQ均关闭，G=1；质量来源均white_input_expected_power_proxy，reflect及noise reflect均保留，62 tap残余核和24点数值延迟不变。AGC限幅79/852480。质量代理均值16.001663dB、实际signal_to_added_noise均值16.030427dB；两者差异范围−20.875675至3.503348dB。两者均不是原录音未知噪声与新增噪声合计的总SNR，true_total_snr_known全部false。不通过改变正式SNR、边界或难包获得收益。

B01卫星CE的第6至10/16至20/21至25/26至30轮均值为0.822579/0.797194/0.791634/0.788171，末轮clean/satellite CE0.346313/0.785204；末轮源V为97.840741%（B00为98.148148%）。后段变化较小，不能把卫星CE仍高直接解释为应继续增加预算，也不把预算结束称为收敛。

原U门控全10轮选择B00为8775/567000（1.547619%）、B01为8430/567000（1.486772%）。U TX始终隐藏，选择率不是伪标签精确率；未给U添加新As、对比损失或教师配方。

此前合法源V的冻结E20默认头控制诊断见 automation_reports/CV-SincNet/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/report.md：固定接收状态移除新增噪声，B01 mid/low源域分别+5.277778/+10.083333pp；残余核平坦化增量较小，CFO/RXIQ消除约零。两个信道draw不是两个模型seed。该证据支持新增噪声是源域主要瓶颈，强接收状态耦合及整体CE梯度冲突前提未通过。该源诊断不能将本次E30全部收益单独归因于噪声；本次两行识别的是完整practical组合训练效应。

正式固定IQ没有逐包LOS/lock/quality旁表，因此没有虚构正式状态准确率。现有内部门控的逐包激活也未保存，不用静态门控参数或局部分支梯度宣称其造成失败。已缓解的可验证退化是完整practical组合退化；各单独物理因素的训练因果增益仍未识别。

## 固定网络、数据和训练预算

当前基座F＋跨RX同TX SupCon：SIG_TIME_X3、三路、Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、真实joint160、原默认cosine30、900722参数/206状态。骨干和默认头正常联合更新；实际归一化层为GroupNorm/LayerNorm，没有BatchNorm running状态。没有部署分支、新分类头或独立拟合的诊断头。

原VALIDATED_ONCE/固定split392005复用，原全部RX1/3/4/6/8、day1/2/3、6TX，L_s6300/U_s56700隐藏/V27000，7/63/30物理ID不变。全部训练阶段保持原角色，V不更新梯度，U TX不解封，训练不创建target loader。禁止加载来源不明、契约不匹配或目标训练/选模权重。

按用户最新E30授权，每行从零30轮＝20L＋10U，222更新/轮＝6660；旧E20/13L7U仅历史记录，不恢复旧权重，不冒充同预算控制。旧工厂里的E20/E200/E50描述字段不是实际运行预算；actual_native_args与本行completion均为30/20/10/6660，baseline/teacher checkpoint均空，ownEMA仅从本行学生建立。

B00为ordinary(x)＋同一ordinary(x)的精确重复；B01为ordinary(x)＋完整practical(同一ordinary(x))。两半各自归一化CE/KL后各权重0.5；两次RDrop前向复用同一生成IQ，总4L/物理包。B00不是标准2L参照，也不是完全无普通IQ增强。LS0.01、RDrop alpha1、原AdamW峰值LR0.0002/min0.000001、weight decay0.0001，原nativecosine按总30轮计算。

SupCon只使用clean半的time_down/t2.pool，same TX且不同物理源RX为正例、温度0.2、最大权重0.1、warm5/ramp10，不增加前向。原U/ownEMA/旧门控不变。实际训练窗口长度256、采样25MHz、equalized=1、float32，dataset层完整256点RMS、center=false。

原source carrier容器metadata核查已表明packet本身256点，crop/pad实际上不触发；载体确实被反序列化，未检查目标统计或IQ数值。不存在额外同一packet上下文，不把相邻sig_i拼接或跨L/U/V借用样本。该路线是表示下不适用，不是识别性能失败。

原ordinary的PA/AWGN/multipath/CFO/shift等实际开关保留为共有配方，不声称都经证明身份保持。它作用于IQ，practical再处理IQ，经完整原网络和头训练。B01训练low_suburban/mid_urban/low_urban原均匀nominal；接收硬件按源RX/day/eq稳定跨TX，训练channel每epoch/view刷新，正式随机流独立。

As为原residual/post_sync、equalization_enabled=false/G1、reference_snr_db30、fs25MHz/fc2.45GHz；质量适配、相位跟踪与RXIQ补偿开启，原reflect/AGC，未在As后另加RMS，全部合法有标签satellite参加CE，无urban学生置信筛选，不反演录音中未知TX硬件。

实际导入__file__已保存并读回，运行的是历史release下 experiments/adv3b02_xuc/code/leo_practical/channel.py、batch.py与同目录cvsrffi/practical_adapter.py、DataAugmentation.py、SSDG/train_ssdg.py；不是仓库顶层code/同名模块。完整绝对路径在每行source/module_files.json，本次没有热改这些依赖。

## 实际成本与执行

无训练父权重，每行新增30轮/6660更新；L loader852480、U loader567000，唯一物理L6300/U56700，L网络前向3409920。每行student训练3976920、student评估864002、EMA teacher评估567000（teacher训练0），源身份网络总前向5407922；正式预测前向1176002（包含2包执行校验）。

| 行/模型seed | 新轮次/更新 | 源前向包数 | 训练墙钟（秒） | 预测前向包数 | 预测墙钟（秒） |
|---|---:|---:|---:|---:|---:|
| B00/2026092701 | 30/6660 | 5407922 | 1540.012254 | 1176002 | 47.865980 |
| B01/2026092701 | 30/6660 | 5407922 | 3026.314576 | 1176002 | 48.893565 |

两行合计60 model epochs/13320更新；上述训练墙钟按模型单独报告，不能把其和当成双GPU并行的全run墙钟。起始兼容性smoke单独加载历史全源B00 FINAL20/4440默认头，2合成包、0query/真实IQ/更新后丢弃；该旧20轮成本不是本次初始化继承。分析/重绘0 IQ/CP/NN/新更新。

训练代码commit：bafba40e365d0914a2e86f861fb55a49f1cd3997。独立结果分析：a081a6c047462aa088394e4f233a05fd4f33ce9f。覆盖图可读性修正：5b709b89e137839eb81e65fe84e4a1525275913f，仅改文字颜色阈值，数值未修改。交付记录commit见本文件Git历史。

普通N607 CVS-RFFI/FP32/CPU线程2，训练分别GPU3/4；原共享容量每GPU最多3个总实验进程并计入其他owner，未修改其他健康任务。当前两行均自然结束，queue active/pending空、failure空；实际source/freeze/prediction/scoring/analysis完成文件和本地取回产物均独立读回VERIFIED。

训练CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E30_20261017_r01。原自动执行命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E30_20261017_r01/experiments/cvs_fullsource_concat30_25/runner.py
```

独立分析实际以已提交脚本源码经普通账户Python运行，参数为：

```text
python -u <Git:a081a6c:analysis/cvs_fullsource_E30_results_20261017.py> --row all --truth <registered truth.json> --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01/analysis_r01 --commit a081a6c047462aa088394e4f233a05fd4f33ce9f
```

分析先核对两行FINAL30/6660/20L10U、完整角色、七视图冻结预测和独立评分，再打开truth。252个overall/RX/day/TX混淆矩阵及准确率/宏F1逐项复算一致；全部8880条L/U序列逐条比较。没有新的数据重验证、哈希链、重复审查或目标反馈。

## 执行过的验证与产物

16项协议负测、实际30/20L10U/nativecosine端点及6660/旧4440拒绝PASS。真实F合成测试1028 NN/2更新、0真实IQ/CP，覆盖206相同初态、骨干/SIG_TIME/Sinc/默认头梯度和联合更新、4L同IQ、首对SupCon及deepcopy绑定；compile PASS。一次限定独立P0/P1 PASS。一个普通账户N607 preflight、一个发布归档及一次本地远端SHA比较、remote compile、真实CP noquery smoke、PID/CWD/argv/GPU/log增长均已核验；没有重复launch或健康任务重启。

独立分析4项聚焦测试PASS：手算宏F1/混淆矩阵、非法类别、旧预算与未冻结/不完整条件拒绝、第二请求行不完整时在truth及输出创建前停止。绘图颜色修正compile PASS并从原analysis_summary重绘，原指标与原图保留；四张原图和修正后的覆盖图均已查看。无需为颜色变化重复训练或评估。

远端不可覆盖run根：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01。本地实际分析：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/analysis_r01；清晰图表：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/figures_readable_r01。两者completion.json已读回，前者ANALYSIS_COMPLETE/252CM，后者FIGURES_COMPLETE/8PNG-PDF/metric_values_modified=false。

逐row/seed/RX/day/TX CSV：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/analysis_r01/per_row_seed.csv；完整30轮曲线CSV：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/analysis_r01/training_curves.csv；机制、采样、物理统计与成本JSON：E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/analysis_r01/analysis_summary.json。四种PNG/PDF分别formal_accuracy、training_curves、receiver_accuracy、training_support_B01。源IQ、checkpoint、原始日志、生成输出和本地机器状态不入Git。

实际本地验证入口（已在发布前执行，未因报表变化重复运行）：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/local_checks.py
E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_concat_E30_20261017/local_model_checks.py
F:/App/miniconda3/python.exe -m compileall -q E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_fullsource_concat30_25
```

分析聚焦测试入口为 analysis/test_cvs_fullsource_E30_results_20261017.py；完整分析与绘图入口为 analysis/cvs_fullsource_E30_results_20261017.py。记录中的Git源码传递方式是实际执行方式，不表示原训练release包含后续分析器。

## 负结果与后续边界

本次没有证明新Ag、B10/B11或四格交互、clean净收益、low>60%、全部suburban接近clean、跨seed复现。现有两行是完整源域同预算参照，不冒充最终联合模型。所有负结果和历史E20产物保留；不用本次目标成绩调整增强幅度、loss或候选排序。

旧3至15dB/locked-degraded有界课程只有很小low增量，clean保持失败；fixed additive clean监督跨seed失败。部分旧实验实际额外留出RX，不能冒充本次完整源域控制；不以修改RX/系数或重命名方式救回旧失败路线。全通合成G0因用户纠偏在launch前撤下，未运行且不赋予科学PASS/FAIL，本轮没有重启。

下一步仍需真实IQ泛化增强的源域证据，保持全部原L/U/V/RX以及原U配方。待检验的单一假设应来自源RX之间可重复、跨TX稳定的轻度观测差异，并与完整As组合；最小先用原L作不拟合分类器的差异检查和冻结默认头身份保持检查，之后才登记E30 B10/B11。若差异主要随TX变化、无法支持身份保持或实际B10/B11没有识别收益，放弃该具体构造。这里不是新方法已实现/启动或已获益的声明。

最终问题的当前答案：完整practical组合退化得到明显缓解，默认头low/mid urban分别+21.151786/+26.577976pp，clean付出2.359524pp；单物理因素的E30训练归因未识别，模型seed仅1个，跨seed收益尚未证明。
