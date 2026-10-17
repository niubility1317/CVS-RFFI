# 完整源域 F＋跨 RX：冻结默认头的 practical 受控退化诊断

状态：预登记已固定，等待发布启动。唯一矩阵与种子在 experiment.json。本轮只做源机制诊断，不创建或训练新模型，也不重复正式测试。

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
