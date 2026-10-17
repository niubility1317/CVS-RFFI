# P46：原普通 Tukey 窗的源域反事实已完成，固定解释关闭

**ANALYZED／VERIFIED，科学条件 FAIL，联合目标仍 ACTIVE。** 本轮是冻结默认头源诊断，没有新 IQ Ag 训练或正式 clean／practical 成绩。没有证据把原时域窗当成主要身份损伤来源，不启动去窗训练，不调窗、幅度、draw、RX、阈值救跑。

## 实际问题、代码与数据

已核对当前分支、协议、登记和引用对话，保留 PA/移位旁路、包络、DC、FIR、噪声、课程、L0 等负结论；引用对话的 carrierGate 实验仍由其 owner 负责，本分支不重复启动。唯一新假设：实际普通增强中的时域 Tukey 窗额外损伤当前源默认头身份证据。代码中的 bandedge_taper 先对时域 IQ 两端乘窗，再进入原 general_channel 与 normal PA；它不是接收频带滤波。

实现 commit ce6eee755301e942acc38e7d8937240ff44ef997：experiments/cvs_taper_source46/{design,core,metrics,runner,run,checks,publish,__init__}.py。独立读数 commit 821327a40daf2916f2aec11da3fd8e1705303caa：analysis/cvs_native_taper_source46_results_20261018.py。两者均已推送并独立 GitHub OID identical；交付收尾 commit 见本文件 Git 历史。[experiment.json](experiment.json) 是唯一矩阵，原预登记在 [report.md](report.md)。

冻结 P43 B00（无 As）／B01（有 As）的 FINAL30，使用其训练过的全部合法 L6300；这是 **seen-fit-L 输入反事实**，不是留出 RX 泛化、V 或正式 target 成绩。F/SIG_TIME_X3＋统一 Sinc CENTER_25K/CENTER_UNITS＋NoExplicitCFO＋原内部门控/参考响应＋joint160/defaultcos30 保持，没有新分支、头、拟合或梯度。

全部 5 源 RX=[1,3,4,6,8]、day1/2/3、6 TX、90 格各 70；L6300/U56700 隐藏/V27000、split392005、EQ1/nativeRMS、IQ256/25MHz 保持，无训练 RX 留出或 Ldev。仅用合法 L；旧 cache 其他源角色可反序列化但不消费其 IQ/标签/统计，U 真值不读，V/target 不用。原正式 7 视图 IQ/信道/SNR/困难样本不读不改。

来源固定为 20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01 B00/B01，运行代码19aedc3604fed9c193237dd92366164178af0983。pinned own_model/parent_payload 严格核对各自 scratch20＋own student/EMA 续10、原角色及 source-only fixedfinal；两模型按对照身份预定，未按正式成绩挑父。历史60modelE/13320更新单列，不冒充新增训练。

## 真实运行值与同抽样干预

实际 DataAugmentation.__file__ 来自固定 release 的 cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/DataAugmentation.py，原 SSDG 和 leo_practical channel/residual 同源。完整绝对路径在私有 resolved_config.json；没有按当前 CWD 猜导入。本轮不生成 practical。

| 字段 | base 值 | 实际 E13 值 |
|---|---:|---:|
| p_bandedge_taper | .25 | .06265008348374353 |
| taper_alpha_min | .02 | .005012006678699482 |
| taper_alpha_max | .10 | .02506003339349741 |
| p_time_shift | .35 | .08771011687724094 |
| max_time_shift | 32 | 8 |
| p_pa | .14 | .03508404675089638 |

真实 E13 由 native build_aug_base_cfg/make_augmentor/configure_augmentor_for_epoch 与保存 base 核对，不能把基础概率当运行概率。模型 seed2026092701，split392005；L 顺序 seed202610180462，增强 seed202610180461，各 draw/batch 私有流，不按 TX 定义随机会话。

两个普通流比较 RAW、FULL、NO_TAPER。NO_TAPER 仍消费原 gate/alpha 并调用原窗函数，只将返回值改为 ones；其余原 DC/PA/噪声/多径/CFO 参数和抽样保持，不强制触发。private 流退出后恢复 Python/NumPy/Torch/CUDA RNG，finally 恢复原函数；未触发的 IQ/logits 必须完全相同。没有改变 alpha、缩短256点或反演未知 TX 硬件。

## 全部默认头结果与固定判定

固定 cohort 为 B00 RAW 正确的6290个L样本；B00 RAW正确6290/6300，B01为6260/6300。要求四个 model/draw 保持率差均≥+.2个百分点、整体准确率差均≥0；每模型合并两draw后至少4/5源RX正增，最差RX≥−.25个百分点。实际分别 FAIL/PASS/FAIL/PASS，整体2PASS/2FAIL，固定解释关闭。

| 模型 | draw | FULL正确 | NO_TAPER正确 | FULL保持 | NO_TAPER保持 | 保持率差（百分点） | 修复/新增错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B00 | 0 | 5976 | 5976 | 5975 | 5975 | 0.000000 | 0/0 |
| B00 | 1 | 5985 | 5985 | 5983 | 5983 | 0.000000 | 1/1 |
| B01 | 0 | 5939 | 5939 | 5937 | 5937 | 0.000000 | 0/0 |
| B01 | 1 | 5946 | 5947 | 5944 | 5945 | 0.015898 | 2/1 |

准确率和六类宏 F1 均为百分数；margin 为真实类相对最大竞争类间隔÷原 cosine30，无温度、尺度或额外头拟合。

| 模型 | 输入 | draw | 准确率% | 宏 F1% | mean margin/30 |
|---|---|---:|---:|---:|---:|
| B00 | RAW | -1 | 99.841270 | 99.841232 | 0.243073847 |
| B00 | FULL | 0 | 94.857143 | 94.857733 | 0.199688873 |
| B00 | FULL | 1 | 95.000000 | 94.995159 | 0.193735920 |
| B00 | NO_TAPER | 0 | 94.857143 | 94.857733 | 0.199688873 |
| B00 | NO_TAPER | 1 | 95.000000 | 94.995636 | 0.194007444 |
| B01 | RAW | -1 | 99.365079 | 99.364644 | 0.241042970 |
| B01 | FULL | 0 | 94.269841 | 94.269094 | 0.197794584 |
| B01 | FULL | 1 | 94.380952 | 94.380254 | 0.191792710 |
| B01 | NO_TAPER | 0 | 94.269841 | 94.269094 | 0.197794584 |
| B01 | NO_TAPER | 1 | 94.396825 | 94.396046 | 0.191893419 |

逐源 RX 保持率差（两 draw 合并，同 IQ 重复观察，不是两个模型 seed）：

| 模型 | RX | 每 draw 固定 cohort 数 | 保持率差（百分点） |
|---|---:|---:|---:|
| B00 | 1 | 1260 | 0.000000 |
| B00 | 3 | 1254 | 0.039872 |
| B00 | 4 | 1258 | 0.000000 |
| B00 | 6 | 1260 | 0.000000 |
| B00 | 8 | 1258 | -0.039746 |
| B01 | 1 | 1260 | 0.000000 |
| B01 | 3 | 1254 | 0.039872 |
| B01 | 4 | 1258 | 0.000000 |
| B01 | 6 | 1260 | 0.000000 |
| B01 | 8 | 1258 | 0.000000 |

两模型都只有 RX3 正增；B00 在 RX8 有对应新增错误，其他 RX 基本不变。局部间隔、F1 微变不替代固定条件，没有跨 seed 重现证据。

## 原 gate 触发不等于波形发生损伤

原函数计算 edge=int(alpha*(L−1)/2)，edge≤1 时直接返回全 ones。在 L=256 时，alpha<4/255（约.0156863）走该分支，位于真实 E13 范围内。因此 p_taper>0 不代表每次触发都会抹除 IQ 边缘。

| draw | 观察数 | 原 gate 触发 | 实际 IQ 变化 | B00 logits 变化 | B01 logits 变化 |
|---|---:|---:|---:|---:|---:|
| 0 | 6300 | 256 | 0 | 0 | 0 |
| 1 | 6300 | 512 | 384 | 384 | 384 |

第0流全部窗权重为1，配对MSE为0。第1流有256次观察的2个窗权重<1、128次的4个权重<1，其余5916次均无权重变化（其中含128次 gate 已触发但返回 identity）。第1流平均 MSE=.000140479835，最大=.012489824556。未触发和 MSE=0 样本的 logits 均完全相等。第一流没有因无效而重抽；第二流在 As 模型仅增加1个净正确样本。

已证实：真实窗较弱且离散化后常无效；该固定旁路缺少所需身份证据增益。被否定的是当前主要损伤解释，不是证明所有加窗永远无害。不放大 alpha 制造阳性，也不重开 LA16/FIR。

## 验证、独立复算与成本

本地8项原生/协议反例、8文件及两内嵌远端脚本 AST PASS；唯一直接 P0/P1 审查 PASS，只读且未接触真实 IQ/CP/target。远端同8项检查、compile PASS。主入口两个合法 CP 各2个合成包的 no-query smoke 共4NN，PASS后立即继续。一次正常普通账号 N607 预检、一次归档本远 SHA 比较；无新增签名、receipt、数据复验、报告审查或审批。

独立工具不导入候选、模型或数据加载器，读取完整378000个保存 logit 值、6300个 opaque ID、两draw和90格。重算主计数、修复/新增、6290 cohort、10个RX汇总和全部判据，最大差8.8263e−15，一致确认FAIL。另独立构造150个overall/RX/TX/day混淆矩阵及指标；在线没有输出这150个CM，不能称为与在线150个CM比对。对有在线对应的原主计数和判据逐项比较。两个PNG/PDF组已逐一打开检查可读。

本轮NN=2×(6300RAW＋2draw×2view×6300)＋4smoke=**63004**；原普通 FULL/NO_TAPER 为12600对、25200次增强观察，唯一 IQ6300。0新的 practical 调用；普通增强内原 AWGN/多径照常执行，0As不意味着0普通随机变换。无新模型、teacher、梯度、训练、头拟合；每模型实测eval计数31502、206个状态张量及冻结检查一致。

主过程记录墙钟7.511489秒，范围从main开始至最终NPZ保存前，含模型载入、源读取和前向，不含前置bootstrap/import与末尾NPZ压缩。独立读数/绘图1.423839秒，0IQ/CP/NN/更新。两模型历史source identity NN合计15946568、历史正式含smoke2352004，合计18298572只作来源成本，本轮不复训/重测。训练更新为0，因此没有本轮训练曲线。

## 执行后态、命令与实际产物

主PID1002504/startticks152833409曾独立核实live，CWD/argv与实际/proc GPU2匹配；stdout从0增至144bytes，完整4行含smoke/两draw/关闭。现PID缺失、completion/source NPZ/result/独立产物齐全，stderr空、无failure，VERIFIED自然完成。普通账号GPU2/CPU2，无其他owner任务改变或健康实验重启。

原launch.json的CUDA字段曾静态写空，独立/proc证明实际env=2；保留原产物，仅在821327a40daf2916f2aec11da3fd8e1705303caa修正后续本地publisher元数据，未热改已完成release或重跑。首次preflight因本地PATH找不到whoami在SSH前失败，显式使用已验证env15/System32后通过；未安装依赖、修改环境或系统设置。

已执行入口（CWD是本run不可变release、原CVS-RFFI Python，勿对原root重跑）：

python -u experiments/cvs_taper_source46/runner.py

独立分析从已提交的本地文件SCP到所属run新文件 independent_analysis_r01.py 后只读执行，CUDA隐藏，不修改原模型release：

python ../../runs/20261018-diagnostic-FcrossRX-nativeTaper-L6300-s2026092701-m2-r01/independent_analysis_r01.py --input ../../runs/20261018-diagnostic-FcrossRX-nativeTaper-L6300-s2026092701-m2-r01 --output ../../runs/20261018-diagnostic-FcrossRX-nativeTaper-L6300-s2026092701-m2-r01/independent_analysis_r01

[执行核验](execution_verification.json)／[最终核验](final_verification.json)。本地产物：[E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018)。私有source_predictions.npz不含IQ；[150行源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/source_strata.csv)、[配对主效应](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/paired_effects.csv)、[全RX](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/source_RX.csv)、[独立复算](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/independent_recount.json)。原IQ/CP/日志/NPZ/生成图/原始CSV不提交Git。

实际图：[源保持率与RX PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/source_taper_effects.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/source_taper_effects.pdf)；[触发与波形变化 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/native_taper_observation.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_nativeTaper_L6300_20261018/independent_analysis_r01/native_taper_observation.pdf)。

本轮没有新的urban识别成绩，不能宣称缓解某类urban损伤、默认头正式提高或零clean代价。有效IQ Ag、组合Ag＋As、lowurban>60%、suburban接近clean和跨seed仍未完成。当前唯一被检验的窗解释已关闭，没有据此晋级的新训练候选；保留用户确认的clean监督份额，不扫描已关闭算子，不由正式成绩倒推参数。完整目标继续ACTIVE。
