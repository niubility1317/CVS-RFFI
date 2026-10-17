# 完整源域 F＋跨 RX：冻结默认头的 practical 受控退化诊断

状态：ANALYZED，源域冻结诊断全部完成；联合性能目标未完成。唯一矩阵与种子在 experiment.json。本轮只做源机制诊断，不创建或训练新模型，也不重复正式测试。

上轮三行完整源域 E20 对照已经完成。普通 practical 拼接显著改善 urban，但有 clean 代价；将卫星视图输入换成原始 IQ 只有小增量。该背景不授权用目标成绩选择本次物理参数。本轮唯一待检验假设来自源训练记录：困难 practical 观测的身份学习信号仍较弱，可能涉及特定损伤、接收状态切换或局部目标冲突。诊断先确定源证据，不预先归因于多径、CFO 或门控，不直接追加损失。

## 数据与模型

保留原 RX1/3/4/6/8、day1/2/3、6TX 和 L6300／hidden U56700／V27000，split392005，7/63/30。使用已有 VALIDATED_ONCE 源缓存，在原 V 每个 TX×RX×day 的 300 包中按固定 panel seed202610172401 取 20 包，共1800包。所有5个源RX均覆盖；这不是从训练中留出RX，也不另设L_dev，不移动L/U/V角色。

缓存读取包含原 V27000 的 IQ 和 L6300 的 IQ；本诊断只使用 V 的 x/y/rx/day/domain/IDs，L值及旧父模型特征不消费，U值不打开、TX真值不解封。IQ保持 native equalized1、center crop/pad256、center=False、裁剪后逐包复 RMS（epsilon1e-12）、25MHz。数据不重建，不添加哈希、签名或验证链。

三行分别加载上轮 B00、B01、RAW_PRAC 的完整源域 FINAL20 默认头。每个来源均为同一模型seed2026092701的独立scratch E20/4440，无权重祖先，全部原角色精确匹配，固定终轮在正式预测前冻结。它们只作诊断读出；没有初始化新训练、重置头或继承优化器。总历史成本为60 model epochs/13320训练更新，本轮新增训练epoch/update均为0。

F（三路SIG_TIME＋统一Sinc修正）、真实joint160、默认cos30、NoExplicitCFO保持，严格加载900722参数/206状态tensor。程序第一步在尚未打开源IQ缓存时完成三个真实checkpoint的2包合成默认头与现有分支smoke，共6NN，PASS后立即继续。

## 固定观察矩阵

使用 channel seed202610172402、sourceV/residual_diag24/draw0或draw1命名空间；与训练及正式测试随机流分开。两个信道draw不是两个模型seed。RX硬件seed2026092701，按RX/day/eq固定session，不能由TX标签定义。

只使用原训练环境 low_suburban、mid_urban、low_urban。物理参数读取上轮 B01 的真实运行config，除scenario外完整复用：25MHz/2.45GHz、residual/post_sync、EQoff且G=1、reference_snr_db30、质量适配/相位跟踪/IQ补偿开启、reflect边界。正式received IQ、SNR、信道及困难样本完全不变。

每包同一潜在几何、遮挡、传播、硬件和命名随机流，比较11个条件：

- full；
- no_noise、flat_H、no_carrier、no_RX_IQ，各做fixed与total两种；
- actual_power_quality；
- edge_boundary。

fixed保持原接收状态、残差尺度、估计频率、相位跟踪/EQ状态、均衡器和AGC参数；total在移除对应损伤后重算接收状态及补偿。no_noise仅移除新增噪声，其total质量设为200dB表示新增噪声极限；原录音噪声未知，不能叫作总SNR。flat_H保留原白输入核能量、公共相位和数值延迟，仅移除传播选择性。no_carrier移除补偿后的残余载波/相位项。no_RX_IQ只移除本次注入RX的净widely-linear残差，不反演原录音未知TX硬件。

分解器复用既有实现。每包都将full以及原参数固定重放，与实际导入的ResidualChannel逐点比较；固定条件逐项检查补偿参数。edge控制保持同一AGC，检查FIR公共内部区域不变。辅助actual_power_quality检验白输入功率代理与有限波形功率差异；不把它作为新的训练或测试接收机。

共10800个物理draw，11视图一次缓存后供三个模型共用，避免不同模型重抽信道。预计缓存IQ约243MB，所有生成物只在新run下保存，不进入Git。

## 读出、成本与解释条件

正式科学指标是源默认头accuracy、macro-F1、统一除以30的分类间隔、相对full的修复/破坏、固定B00 clean正确集合上的保持率，以及接收锁定/AGC变化。分层用原full的LOS/PARTIAL/BLOCKED、locked/degraded/unlocked、实际signal_to_added_noise_db区间和源RX，保留各层样本量。新增噪声比不是总SNR。

沿真实前向hook读取现有time/frequency/PA/reference/signature分支，记录clean→full的余弦及范数，并保存已学习的静态门控参数；不虚构逐包门控权重，不拟合额外诊断头，不把特征接近当作识别收益。

局部梯度只用第一组90包（覆盖全部90个TX×RX×day单元）的eval-mode smoothed CE，在原有参数上用autograd.grad读取。Sinc梯度换算回Hz单位，报告分组范数和clean/satellite余弦。模型保持eval、无优化器、无.grad累积，并在结束后逐项检查state_dict不变。它不是原RDrop/SupCon随机训练梯度，不能直接认定训练冲突的因果性。

每模型120600预测NN＋630梯度前向NN＋2合成smokeNN＝121232，三模型共363696NN，teacher前向0、训练/头更新0。实际NN、物理调用数和墙钟分别报告。所有真实source输入均在checkpoint smoke后打开。

运行前固定解释条件：某项fixed损伤移除若在两个urban场景、两个信道draw均提高至少5pp，才支持该项是稳定源瓶颈；noise total比fixed在两个urban均多至少3pp，支持接收状态耦合值得进一步检验。局部梯度竞争前提要求B01/RAW的两个urban×两个draw均出现whole CE梯度余弦<0，且sat范数不低于clean。达不到时不把对应前提写成结论。

这些只是源机制解释条件，不是自动训练、目标访问或扩seed许可。冻结后的离分布控制也不等于重训后的因果收益。当前不自动发出C2/C3/C4或Ag候选。

## 执行与保护

一个run、一个owner、一个GPU worker。普通N607 CVS-RFFI环境，FP32，OMP/MKL/OPENBLAS各2。每GPU最多3个总实验进程，计入其他owner，且空闲至少12000MB；使用既有共享capacity lock。不得修改、停止或重启健康任务。精确路径和命令见experiment.json。

停止规则：首次来源/角色/控制重放/冻结补偿/非有限输出或梯度/状态改变/NN预算/技术错误，保留全部输出并停止本作业，不自动重启或按分数补救。没有新训练，因此不追加正式测试；本作业禁止target IQ、truth或成绩读取。

本地已执行4项测试PASS：11协议负测、6个实际native synthetic案例/84次合成物理调用、Hz梯度单位与无.grad累积、指标手算；0NN/真实IQ/CP/更新。此前一次范围负测因CRLF导致guard未插入而失败，已修复并重跑PASS，失败产物保留。编译PASS。一次普通账户/来源元数据/资源预检VERIFIED，未读IQ或CP。一次限定 P0/P1 独立审查 PASS，未发现直接 P0/P1；无额外报告/哈希/审批gate。

预计产物：三个来源的严格smoke/数据角色/真实模块__file__/配置、共同诊断IQ、逐包控制physics、每行源prediction、branch metrics、门控参数/CE梯度、源分层CSV、summary、损伤控制图/质量分层图/梯度图及completion。实验登记、报告和Git交付在证据返回后补全。不得把本诊断完成写成联合性能目标完成。

执行代码 commit：`5dc8882cc1609f3b857749368a75f856b57bcd9d`，已推送并独立核对远端 OID 相同。

实际启动命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_residual_diag_20261017_r01/experiments/cvs_fullsource_residual_diag24/runner.py
```

CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_residual_diag_20261017_r01`。完整矩阵、停止条件和输出路径见 experiment.json。

## 实际执行与结果

发布包一次传输 SHA 比较 VERIFIED，远端 compile PASS。普通账户 GPU0 的唯一 worker PID/start_ticks/CWD/argv/GPU 归属及日志增长均 VERIFIED；三个真实默认头 smoke PASS 后才打开源 IQ。读取完整 worker.log：6 个观察阶段、3 个 MODEL_COMPLETE 和最终 SOURCE_DIAGNOSTIC_COMPLETE，未触发技术停止。三模型 state_dict 均逐项不变；每行121232NN，共363696NN，0训练/头更新/teacher前向。10800次实际 ResidualChannel＋129600次控制renderer，均复用同一命名潜在draw。墙钟280.439972秒。原三行历史训练成本60 model epochs/13320更新独立保留，本轮没有新增epoch。

实际模块 `__file__`：

- channel：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py`
- residual：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/residual.py`

不是顶层 code/leo_practical。三个场景运行值为 residual/post_sync、EQ=false且G=1、fs25MHz/fc2.45GHz、reference_snr_db30、质量适配/相位跟踪/IQ补偿开启、reflect；详细配置在实际产物 actual_engine.json。没有修改正式接收IQ或测试信道。

### 源识别与损伤分解

| 模型 | 源 clean | 源 low suburban | 源 mid urban | 源 low urban |
|---|---:|---:|---:|---:|
| B00 | 98.333% | 54.556% | 47.111% | 38.694% |
| B01 | 97.722% | 89.167% | 75.083% | 61.333% |
| RAW_PRAC | 97.667% | 89.611% | 74.639% | 61.056% |

**本表仅为原源V1800诊断面板，不能把61%左右的源low urban写成正式low urban超过60%。正式三行成绩仍为32.222619%／52.243452%／52.479167%，模型seed均为2026092701。**

以下增益单位为百分点，均相对同模型、同样本、同draw的full：

| 模型/场景 | 移除新增噪声 fixed | 移除新增噪声 total | 平坦 H fixed | 去残余载波 fixed | 去注入 RX IQ fixed |
|---|---:|---:|---:|---:|---:|
| B01/mid_urban | 5.278 | 5.500 | 2.444 | -0.250 | -0.111 |
| B01/low_urban | 10.083 | 11.361 | 1.556 | -0.222 | -0.056 |
| RAW_PRAC/mid_urban | 5.194 | 5.583 | 2.500 | 0.194 | 0.000 |
| RAW_PRAC/low_urban | 9.361 | 10.528 | 1.611 | -0.222 | 0.000 |

noise/fixed 在 B01 的 mid 两draw分别 +5.388889／+5.166667，low +9.500000／+10.666667；RAW 的 mid +5.166667／+5.222222，low +9.388889／+9.333333。两模型均通过预登记的“两个urban×两个draw均至少+5pp”。另外三项损伤均未通过这一条件。支持新增噪声是本面板上较大的可定位源瓶颈；不支持认定主要是残余CFO或注入RX IQ。平坦H仍有约1.56至2.50pp受控收益，不能写成多径无影响。

noise total−fixed 的额外增益：B01 mid0.222222pp／low1.277778pp，RAW mid0.388889pp／low1.166667pp，未达到两个urban至少3pp。移除新增噪声会大量改变锁定状态，但准确率增量不能只由状态切换解释。这里不移除原录音未知噪声，不能叫作总SNR或无噪声上界。

### 退化集中区域

| B01 low urban 原 full 分层 | n（两个 draw） | full 准确率 | 去新增噪声 fixed 增益 |
|---|---:|---:|---:|
| state_lock/LOS/locked | 1060 | 94.623% | 0.755 pp |
| state_lock/BLOCKED/degraded | 414 | 45.411% | 13.768 pp |
| state_lock/BLOCKED/unlocked | 1565 | 34.185% | 17.636 pp |
| quality/<3 | 1127 | 30.701% | 19.521 pp |
| quality/3-8 | 794 | 43.703% | 13.980 pp |
| quality/8-15 | 303 | 74.587% | 5.941 pp |
| quality/>=15 | 1376 | 93.677% | 1.017 pp |

B01 low的BLOCKED/unlocked占1565/3600，full准确率34.185304%；noise fixed修复425、破坏149，净增276，占总体净增363中的76.03%。LOS/locked为94.622642%，净增仅8。普通practical训练的主要收益和剩余困难不能用一个urban均值概括；源RX分层与其他状态在1782行CSV中保留。1至2包的小层不作泛化结论。

质量分层使用原full实际signal_to_added_noise_db，不是总SNR，也没有以该分层删除任何训练/测试样本。是否重采样低质量观测仍需独立训练检验，移除噪声的离分布控制不能证明课程有效。

### 接收质量、噪声与边界

实际波形功率相对白输入核功率代理的q01/q99：mid为−1.231450/+1.038951dB，low为−1.351333/+1.204032dB。切换为实际功率质量代理使mid87/3600、low84/3600改变锁定状态；B01准确率变化−0.250000/0pp，RAW为−0.250000/+0.222222pp。存在状态影响，本面板不支持它是主要准确率缺口；正式/训练接收机未切换。

三个场景残余核均62 taps，数值延迟24；前61个输出受人工前缀边界影响，共195个不依赖该前缀的内部输出。reflect改为edge、保持同一AGC的压力控制，使B01 mid/low下降10.833333/9.250000pp，RAW下降11.305556/10.111111pp，而内部区域已检查不变。这证明对前缀边界敏感，不能证明原reflect造成同等可修复损失，不能据此改动正式IQ或测试边界。

EQ关闭，G=1，所记录的新增噪声PSD动态范围均0dB；没有把EQ噪声着色错误套到本轮。AGC限幅在low为1/3600，另两场景0/3600。mid残余CFO q01/q99约−887.09/+902.12Hz，low约−1122.73/+1051.66Hz，去掉残余载波没有稳定正收益。完整物理统计在readout.json。

### 真实分支、门控与梯度

B01 low的clean→sat余弦：time0.968915、frequency0.984878、PA0.873587、reference0.360041、signature0.355412。RAW接近。观测变化并不均匀，不据此要求所有分支、参考响应或整个embedding复制clean；这些数值也不是身份收益的替代指标。静态160维PA/reference/signature增益已保存，模型不含本诊断虚构的逐包门控。

B01两个urban×两个draw的whole eval-CE梯度余弦为0.141033至0.354815，RAW为0.160257至0.312766，均为正；“whole梯度冲突”预登记前提失败。frequency/reference的局部分组可为负，但不能把局部现象写成全网络主因。Sinc已换算Hz，PA分组仅匹配现有`.pa_`/`.cls_head.gain`名称，其余参数进入remaining，whole含所有实际参数；不能将这个分组称为全部PA链路。梯度只在90包、eval CE/.01 smoothing下读出，不代表原RDrop/SupCon训练梯度，也不证明整个训练过程从无冲突。

### 收益归因、负结果与下一步边界

在完整源域、固定默认头和正式practical条件下，已核实的主要收益仍来自普通星地拼接：相对同4L预算B00，urban均值+25.250992pp、low+20.020833pp，clean−2.761905pp。RAW相对普通拼接urban+0.228175pp、low+0.235714pp、clean+0.514286pp，未达到既定urban+3/low+5探索目标，也没有联合突破。这些正式背景已在上轮报告，不作为本诊断物理参数选择或新的调参依据。

本轮源证据确认了新增噪声下的低质量BLOCKED困难；否定了预登记的强状态耦合和整体梯度冲突前提。未新增一致性、梯度手术、部署模块、诊断头或held RX。当前唯一值得独立检验的假设是：保留普通拼接和clean份额，以有界混合覆盖全部原nominal状态，并逐步提高可恢复urban状态的训练覆盖，可能改善噪声下身份学习；不能由本诊断直接宣称成立。最小后续比较应为全源FcrossRX、scratch E20、与B01同4L成本/同U规则的单独状态课程，不堆叠额外一致性。课程幅度、源规则和放弃条件须在新row启动前固定；不按上述正式成绩搜索，不只采最大损失或删除unlocked。未实现/未启动该后续候选，也未扩模型seed。

### 可核查交付

- 执行实现：`experiments/cvs_fullsource_residual_diag24/`，commit `5dc8882cc1609f3b857749368a75f856b57bcd9d`。13个文件，默认头严格加载、完整源panel、受控重放、真实分支/局部梯度、汇总、单owner容量保护发布。
- 后处理：`analysis/cvs_fullsource_residual_diag_readout_20261017.py`。实际对三个模型共99个总体混淆矩阵独立复算一致，0NN/IQ/CP/更新/目标访问。
- 命令与CWD见上方冻结预登记；发布调用同目录publish.py的transfer和launch两步。
- 本地4项测试PASS、11负测/6合成native案例/84合成物理调用、compile PASS；此前CRLF guard失败已保留并修复。一次P0/P1 review PASS，无重复review。真实3CP无query smoke及远端实际控制重放PASS。
- 本地产物目录：`E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence`；summary.json、per_row_source_strata.csv（1782行）、readout.json、每行result.json、3张PNG/PDF与completion.json，已读回并目视检查图。共同IQ、逐包物理记录和源prediction/branch数组保留在不可变远端run，不进入Git。
- 三图：[损伤控制](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence/source_damage_effects.png)、[质量分层](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence/source_quality_accuracy.png)、[局部梯度](E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence/source_gradient_cosine.png)。
- 未完成：正式low urban>60%、clean净正收益、suburban接近clean和跨模型seed复现。本诊断只有1个模型seed，两个信道draw不能代替跨seed。

本地检查实际命令（原生 Windows Python，CWD 为本工作树，PYTHONPATH 为工作树及 experiments/adv3b02_xuc/code）：

```text
E:/type10-7/local_envs/ssr-gpu/python.exe -m experiments.cvs_fullsource_residual_diag24.checks
F:/App/miniconda3/python.exe -m py_compile analysis/cvs_fullsource_residual_diag_readout_20261017.py
```

后处理使用同一个 N607 普通账户 Python 从 stdin 执行上述 readout 文件内容，诊断根作为 sys.argv[1]，只读源prediction/physics/branch产物，不重复模型前向。最终独立读回 complete.json/completion.json，worker及dispatcher均已退出，failure不存在，状态 VERIFIED。
