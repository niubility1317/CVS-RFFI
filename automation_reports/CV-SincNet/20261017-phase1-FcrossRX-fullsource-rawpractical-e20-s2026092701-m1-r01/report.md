# 完整源域 F＋跨 RX：只改变 practical 的输入

状态：ANALYZED。三行完整源域对照、全部七视图评分、完整日志及配对分析齐全；未达联合目标，只有一个模型seed。唯一矩阵见experiment.json，仅RAW_PRAC-s2026092701一行。固定参照为正在按原计划运行的完整源域P22 B01普通拼接；未读取任何目标分数。本行依据是实际代码与配置确认：原native先施加旧ordinary（允许PA/CFO/AWGN/multipath/shift），再施加practical。这可能重复损伤，不等于这些旧普通增强已验证身份保持。

唯一差异：原B01为clean=ordinary(x)，satellite=practical(ordinary(x))；本行仅令satellite=practical(x)，x为同批原始合法源归一化IQ。clean、真实TX标签及其权重、原native concat_full、全部物理信道参数、随机流、RX/day session、接收质量适配与补偿、RDrop、clean-only跨RX同TX SupCon均不变。接收状态正常重新计算，不冻结为clean、不弱化原practical、不假定算子可交换。未知TX硬件不反演。

全部原源RX1/3/4/6/8、day1/2/3、6TX，L6300/hiddenU56700/V27000、split392005/7-63-30精确原物理IDs，无额外留RX/Ldev，不合并V或解封U。训练仅为全新scratch E20=13L+7U、222steps/epoch、4440updates、2273280 L网络样本量，默认F三路SIG_TIME/CENTER_UNITS/joint160/cos30正常联合训练，0新部署模块/头/损失。U/ownEMA/旧门控保持；新raw缓存仅L使用。

每个L包clean/sat各0.5，原CE/RDrop都按两半分别平均，复用同一拼接IQ的两次dropout前向；SupCon只读clean128半批。训练环境low_suburban/mid_urban/low_urban，原native场景先验和跨epoch/view刷新；25MHz/2.45GHz,residual/post_sync,EQoff(G=1),reference30,quality/phase/IQ补偿开启,reflect，无星地后外部RMS。训练三场景全部状态支持保留；测试received IQ/SNR/边界与六场景一律不改。

旧的留RX配对As曾有clean失败，原产物和判据保留。本次是用户明确恢复完整原训练域后的直接比较，不用旧留域结果替代完整基线、不加载其权重，也不宣称新方法/Ag成立。P22的默认头/损失/F优化器代码按不可变dc4e47fe0复用，新包只包装同批raw输入。在独立新进程设置模块配置引用，不修改P22代码文件或健康进程。

数据流与7项negative检查PASS：合成128包raw逐点相等、clean不变、标签两半一致、缓存消费一次、stale/U/错shape/预算/role拒绝；0真实IQ/CP/NN/物理channel/更新。编译PASS，一次独立增量P0/P1审查PASS，无未解决P0/P1。原完整F数学不变，复用P22已执行全图检查。一次N607预检PASS。

第一执行步使用已核实完整源契约CROSS_RX FINAL50（原scratch/无祖先/默认头、FINAL50且冻结前无target）的2合成样本严格加载兼容smoke；不初始化新训练或EMA，历史50epoch/11100成本单列。新候选只增加20modelE/4440updates；两项P22参照40modelE不隐藏、不当作本行父训练。

逐row FINAL20固定后自动原clean＋六practical的168000×7预测，全部prediction完成后另exec独立truth-last；只有候选和参照七视图/评分均完成才读取比较数据。没有源/目标性能选模、晋级或选择性重跑。探索目标预先固定相对P22 B01：urban均值+3pp、low+5pp、clean下降≤0.5pp；low>60%和各suburban距自身clean≤2pp为探索目标，不保证达到。不满足时保留负结果，不扫本输入路由的比例、强度、场景、seed或预算救援。一个seed不证明复现，没有新Ag或Ag×As交互结论。

环境/CWD为普通N607 CVS-RFFI Python，/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_rawpractical_E20_20261017_r01；每GPU总进程≤3含其他owner，空闲≥12000MB；独占run root /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01，launch owner/root。精确入口：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_rawpractical_E20_20261017_r01/experiments/cvs_fullsource_rawpractical23/runner.py`；新worker/source/prediction/scorer均使用新包入口。dispatcher/scorer CUDA空。代码commit发布前登记，单一归档传输比较/remote compile；不触碰健康旧任务。

停止：固定预算，不按微小源分数早停；数据角色/标签/来源/非有限loss/grad/预测/技术错误保留产物、阻止未启动row，不自动重启。其他健康任务继续。预算结束不等于收敛。预期实际config/module_files/raw路由证明、物理覆盖、完整step/epoch曲线与成本、FINAL20、七视图prediction、独立score、逐row/seed CSV和同row报告见experiment.json。

实际实现commit：`31d7fc60f75819ff25a698a29df9fc7ec03691c8`，目标分数未读；固定此实现发布，不更改P22健康运行。


## 实际发布与路由生效

实际执行commit `31d7fc60f75819ff25a698a29df9fc7ec03691c8` 已推送，远端OID独立核对一致；一次归档传输比较VERIFIED，remote compile PASS。独立dispatcher/worker已启动，原P22进程保持运行，GPU由实际资源分配（本行GPU5），未停止或热改任何其他任务。真实fullsource CP兼容smoke PASS，2合成NN、0训练更新、无query。

PID/start_ticks/CWD/argv/GPU/日志增长与全部原L6300/U56700/V27000精确角色独立读回，无fold参数。raw输入首批证明：128×2×256，与同批ordinary输出MSE=0.003188482951372862；clean不变、同一L物理元数据、接收状态正常重算、无TX条件信道或未知TX反演。首批LOS46/PARTIAL11/BLOCKED71、locked54/degraded16/unlocked58，与原固定流参照首批一致，EQoff/reference30等实际配置保持。

截至早期完整可解压step记录，本行至少236次真实更新，活跃CE/RDrop/KL有限、nonfinite loss/grad skip=0；这只证明训练和输入路由实际生效，不证明urban增益/收敛/跨seed/Ag。当前没有读取目标分数，没有改变任何健康运行配方。


## 固定七视图实际结果

RAW_PRAC完成scratchE20/4440和固定7视图168000×7预测/独立truth-last。相对预固定普通拼接B01：urban+0.228175pp、low+0.235714pp、clean+0.514286pp；未达预登记urban+3/low+5目标。仅1seed，不能宣称联合突破或复现；完整三行离线日志汇总正在执行，0新增训练。

| 视图 | RAW_PRAC accuracy | RAW−普通拼接（pp） |
|---|---:|---:|
| clean | 82.702976% | 0.514286 |
| practical_high | 80.745238% | 0.339881 |
| practical_mid | 78.275000% | 0.339286 |
| practical_low_suburban | 75.872619% | 0.377381 |
| practical_high_urban | 77.379167% | 0.315476 |
| practical_mid_urban | 63.320238% | 0.133333 |
| practical_low_urban | 52.479167% | 0.235714 |

RAW相对无As：clean仍下降2.247619pp；low52.479167%、mid63.320238%、urban均值64.392857%，未达low60%及clean净增益目标。这是route比较的负结果，不能用相对B01的clean小增量冒充相对无As的clean突破。源训练墙钟1971.131474秒，新增E20/4440，无训练父权重。未热改或重跑任何健康行，不按目标分数扫增强强度或扩预算。


## 三行完整源域配对结论（FINAL20，seed2026092701）

已完成三行训练、固定clean＋六practical预测、独立truth-last、完整日志解析和378个混淆矩阵/accuracy/F1复算。最近路线确有偏离：把普通合成扰动筛选、额外留源RX/内部Ldev作为主线，削弱了与完整源域普通practical拼接的直接比较。历史上做过真实practical训练，不能一概说从未做过；但不同留域契约的源诊断不能替代本次完整源域对照。现已恢复原RX1/3/4/6/8及L6300/hiddenU56700/V27000，不切训练域、不另设Ldev、不合并V、不解封U。所有5994条L/U采样记录在三行间逐条相同，全部90个TX×RX×day单元覆盖。

| 视图 | B00无As | B01普通拼接 | RAW_PRAC原IQ配对 |
|---|---:|---:|---:|
| clean | 84.950595% | 82.188690% | 82.702976% |
| practical_high | 47.655952% | 80.405357% | 80.745238% |
| practical_mid | 45.714286% | 77.935714% | 78.275000% |
| practical_low_suburban | 43.495238% | 75.495238% | 75.872619% |
| practical_high_urban | 46.239881% | 77.063690% | 77.379167% |
| practical_mid_urban | 38.278571% | 63.186905% | 63.320238% |
| practical_low_urban | 32.222619% | 52.243452% | 52.479167% |

| 行 | urban均值 | low宏F1 | low最差RX | 三郊区相对clean差（pp） |
|---|---:|---:|---:|---|
| B00 | 38.913690% | 30.502712% | 28.120833% | 37.294643 / 39.236310 / 41.455357 |
| B01 | 64.164683% | 51.650902% | 47.079167% | 1.783333 / 4.252976 / 6.693452 |
| RAW_PRAC | 64.392857% | 51.940863% | 47.733333% | 1.957738 / 4.427976 / 6.830357 |

B01−B00：urban均值+25.250992pp、mid+24.908333pp、low+20.020833pp，clean−2.761905pp。RAW−B01：urban+0.228175pp、mid+0.133333pp、low+0.235714pp、clean+0.514286pp。RAW−B00：clean仍−2.247619pp，urban+25.479167pp、low+20.256548pp。因此普通practical训练缓解了mid/low urban整体受损观测的识别退化，但没有达到clean与星地双突破；RAW未满足预设urban+3pp、low+5pp目标，也未达low>60%、clean净正、全部suburban差≤2pp。不得把相对B01的clean回升冒充相对B00的净收益。

配对错误：RAW相对B01在low修复3499、破坏3103，净396/168000；mid修复2827、破坏2603，净224；clean修复1817、破坏953，净864。小净增量包含错误重分配，且RX间并不一致。固定B00 clean正确集合142717上，B01/RAW的mid保持率69.491371%/69.686863%，low保持率57.082898%/57.358969%；这是同一参照集合，无训练父权重继承。

| 行 | clean真实类间隔/30均值 | midurban间隔/30均值 | lowurban间隔/30均值 |
|---|---:|---:|---:|
| B00 | 0.135518119 | -0.024825828 | -0.033465017 |
| B01 | 0.114162132 | 0.048493311 | 0.026760858 |
| RAW_PRAC | 0.116623975 | 0.051619768 | 0.029005717 |

间隔统一除以固定cos30尺度，仅作冻结后分析，不把尺度变化当收益，也没有新增margin/embedding/分支/门控对齐损失。未保存每包融合门控激活，不能声称某个门控获得因果改善。

## 完整日志、物理覆盖及代价

三行各20轮/4440步，全部13320个更新记录、60轮曲线、3×5994采样记录已完整解析；活跃目标有限、无nonfinite跳步、全部SupCon锚有合法跨RX同TX正例。每行L暴露568320、U396900，唯一物理ID仍6300/56700。两条As行各完整解析568320条物理记录，全部90个TX×源RX×scene和90个TX×源RX×state组合覆盖。训练LOS285384/PARTIAL89591/BLOCKED193345，locked341069/degraded95159/unlocked132092，AGC限幅61；两行计数相同不等于已完成固定接收状态的受控因果实验。EQ全部关闭(G=1)，人工reflect边界，残余核62点、数值延迟24点，输入256点；精确有效观察区及边界对分类的影响尚未识别。

接收质量源为white_input_expected_power_proxy。实际signal_to_added_noise_db不是总SNR，全部true_total_snr_known=false；B01实际新增噪声比减代理的1%/99%分位为-1.299783/+1.234072dB，RAW为-1.298920/+1.232840dB。极端尾部B01最低-19.387220dB、RAW最低-8.640800dB，不能仅凭这些分布认定状态切换的因果效应。残余CFO的1%/99%为-972.330797/+955.525805Hz；不预设CFO主因。正式received_views缺少逐样本LOS/lock/quality文件，仅有固定IQ/index/manifest，因此正式分层只报告view/RX/day/TX，不伪造状态准确率。

E20末B01/RAW平均clean CE为0.272667/0.268360，sat CE为0.788595/0.697892；源V97.637037%/97.625926%。卫星CE下降并未转化为预设幅度的正式收益，预算用尽不等于困难观测收敛。U真值始终隐藏，门控选择率不是precision。

| 行 | 新增epoch/update | source NN | 七视图NN | source墙钟秒 | predict墙钟秒 |
|---|---|---:|---:|---:|---:|
| B00 | 20 / 4440 | 3661082 | 1176002 | 1052.837400 | 48.098098 |
| B01 | 20 / 4440 | 3661082 | 1176002 | 1999.742716 | 49.415070 |
| RAW_PRAC | 20 / 4440 | 3661082 | 1176002 | 1971.131474 | 47.808523 |

三行均从零训练，父训练epoch/权重成本0。累计60modelE/13320更新，source＋predict合计14511252网络样本量；两run各2个兼容smoke合计4合成NN另计。本地既有1540合成NN/2开发更新另计。真实E50来源只作2样本严格加载兼容，历史50epoch/11100更新明确不是训练父模型，也未初始化EMA。所有本次离线分析0新增NN/真实IQ/CP读取/训练更新。source墙钟为各行自身测量，不把并行墙钟与累计GPU工作时间混淆。

## 实现、验证和产物

执行代码：P22 `dc4e47fe0c2b3e0dcf73dda104cdbd4a7c814f5a`；P23 `31d7fc60f75819ff25a698a29df9fc7ec03691c8`。完整分析代码 `4d451e6b09cab511b318b923b458d2c28ca0dd06`。F/默认头/原旧门控/ownEMA/4L半权重不变；P23唯一训练差异为satellite输入绕过ordinary，实际4440次调用/568320包，clean不变，未增加物理强度/损失/部署参数。实际导入文件是已有release的experiments/adv3b02_xuc/code/leo_practical/channel.py和batch.py，非仓库顶层code/默认推测；完整__file__见各source/module_files.json。25MHz/2.45GHz、residual/post_sync/noeq、reference30、质量/相位/IQ补偿/reflect均按运行记录，测试IQ/信道/SNR未修改。

已执行：P22八项协议负测、真实完整F两次合成更新和clean-only SupCon/默认头联合梯度检查；P23七项raw缓存/角色/预算负测；两run真实匹配CP无query smoke各2合成NN；一次独立P0/P1审查及P22依赖问题一次定向复核；归档一次SHA比较/remote compile/所属进程证据；分析独立混淆矩阵手算、非法类别、未齐七视图/未冻结/真值访问负测共3测试PASS，编译PASS。新增分析入口 `analysis/cvs_fullsource_practical_results_20261017.py`，聚焦验证 `analysis/test_cvs_fullsource_practical_results_20261017.py`；精确训练命令仍是各原run runner.py，不重复启动。

完整产物已读回并检查图像：本地 `E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_rawpractical_20261017/comparison_analysis_r01`，远端 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01/comparison_analysis_r01`。包含analysis_summary.json、per_row_seed.csv（378组row/seed/view/RX/day/TX指标）、training_curves.csv（60轮）、formal_accuracy/training_curves/receiver_accuracy的PNG/PDF及completion.json。固定prediction与checkpoint保留各原row，未重测、未覆盖。

## 未完成与研发方向

仅模型seed2026092701，三行不等于三seed，尚未验证跨seed复现。Ag独立效用、B10/B11及交互项未得到证据；本次没有证明新联合增强成立。完整源域下固定接收状态/补偿参数的源域损伤分解、现有分支/门控/梯度诊断及C2课程/C3单侧间隔/C4独立SSL均未完成，不能用旧留域结果替代。

下一步最值得检验的源域假设是：困难practical观测仍提供较弱或相冲突的身份学习信号，而非简单把重复ordinary损伤去掉就能解决。应先在原合法源V的同批IQ和命名随机流上，用冻结默认头比较完整practical与损伤控制，区分状态重算的总效应和固定状态/补偿的受控效应；读取原有分支/门控/梯度，不另拟合诊断头，不从L/U再留RX。若源证据不支持梯度竞争/可恢复状态差异，就放弃相应课程或迁移约束，不用正式target分数定参数。当前没有据此启动新候选、扩seed或延长E20，goal仍未完成。
