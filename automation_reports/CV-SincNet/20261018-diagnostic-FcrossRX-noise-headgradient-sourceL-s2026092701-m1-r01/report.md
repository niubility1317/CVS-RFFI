# P40 结果：未发现默认头分类间隔梯度反向冲突

**ANALYZED_SOURCE_ONLY，预登记假设被否定；联合识别目标尚未完成。** 完成两次新信道抽样、全部900包 L 的实际默认头与分支梯度观察，模型及默认头零更新。两次 low urban 的 full 与 no_noise_fixed 均900/900有效，默认头 W 间隔梯度反向均0/900，未达到预登记 full≥20%且去噪减少≥10个百分点的判据。关闭这一条局部默认头噪声梯度冲突解释，不改阈值、不用 CE01 或别的分支救援，不因此加入梯度修正损失或扩种子。

源 L clean 正确率98.444%，宏F1 98.441%，固定 clean 正确集合886包。以下都是源 L 冻结模型的反事实诊断，不是新的正式目标测试成绩；同一900包在两个draw重复观察，不是两个模型seed。

| 源环境 | draw | 条件 | 准确率% | 宏F1% | 固定886包保持率% | margin/30均值 |
|---|---:|---|---:|---:|---:|---:|
| mid_urban | 0 | full | 74.778 | 74.699 | 75.621 | 0.097705 |
| mid_urban | 0 | no_noise/fixed | 78.778 | 78.697 | 79.684 | 0.107895 |
| mid_urban | 0 | no_noise/total | 79.667 | 79.650 | 80.587 | 0.109189 |
| low_urban | 0 | full | 58.778 | 58.271 | 59.255 | 0.058403 |
| low_urban | 0 | no_noise/fixed | 68.444 | 68.039 | 69.074 | 0.075521 |
| low_urban | 0 | no_noise/total | 67.889 | 67.510 | 68.736 | 0.078399 |
| mid_urban | 1 | full | 71.889 | 71.872 | 72.460 | 0.092132 |
| mid_urban | 1 | no_noise/fixed | 78.222 | 78.055 | 79.120 | 0.102929 |
| mid_urban | 1 | no_noise/total | 78.333 | 78.342 | 79.120 | 0.104323 |
| low_urban | 1 | full | 62.778 | 62.544 | 63.093 | 0.065312 |
| low_urban | 1 | no_noise/fixed | 71.111 | 70.702 | 71.445 | 0.082008 |
| low_urban | 1 | no_noise/total | 71.444 | 71.074 | 71.670 | 0.085393 |

low urban 固定接收状态去新增噪声分别提高9.667和8.333个百分点；mid分别提高4.000和6.333个百分点。重算接收状态的low总效应为+9.111和+8.667个百分点；与fixed相比−0.556和+0.333个百分点，不能把全部恢复归因于状态切换。这些是改善输入后的识别差异，不能当作训练收益。没有反演原录音噪声，没有调弱正式测试。

## 机制解释与边界

分类间隔主判据均不反向，无平滑 CE0 的 W 梯度也均不反向；但原 LS=.01 的 full low W 梯度分别565/900（62.778%）与558/900（62.000%）反向，去噪fixed仍538/900与530/900反向。源 clean 已很高置信，标签平滑的校准目标会改变方向；CE01反向数不能直接当作身份破坏。

full low 的 W margin 梯度 cosine 均值为0.651610/0.646990，去噪fixed为0.667480/0.674977。对应 CE0 的梯度范数均值10.975246/10.275009，clean为0.532909。这里只是当前固定模型的 raw-L 局部导数，不是历史训练普通增强、RDrop、SupCon、加权总损失的真实梯度份额，不能据此宣称已找到训练主因。

其他现有分支/门控确有不同响应：full low 的频率门控参数间隔梯度反向432/900与424/900，PA门控156/900与170/900。这些是附带观察，不能移用为本次主判据，也不足以证明门控错误或支持冻结/替换门控。频率门控乘性尺度的逐包均值相对clean平均变化−0.084850/−0.079551；去噪fixed为+0.022538/+0.021625。允许现有融合调整证据，不强迫其复制clean。

本轮没有新 A_g、没有优化器或头拟合，没有正式target预测/评分。故新增部署识别提升为**未测、未产生新模型**，clean训练代价为零更新（不表示未来训练无代价）；跨模型seed复现未验证。两次信道draw仅复现这次源域阴性结论。P39最近真实模型仍只有seed2026092701：UAS相对固定B0 clean85.511→85.245%（−0.266个百分点）、low51.600→52.461%（+0.861个百分点）、mid62.964→63.857%（+0.893个百分点）、urban均值64.132→64.923%（+0.791个百分点）；其C4源判据失败已关闭，不能由本诊断翻案。low>60%、clean正收益和suburban接近clean尚未达成。

## 真实执行与成本

代码 commit：d12e359a2bcfa927954535c297438d79f81d7dfe，独立 GitHub compare identical；一次归档SHA比较 VERIFIED，远程compile PASS。9项focused tests PASS（本地Torch/NumPy ABI有警告，但未走tensor↔numpy接口），一次独立P0/P1审查PASS。真实checkpoint/defaulthead三目标和门控解析梯度烟测PASS；最大参数梯度绝对误差9.54e-7，模型state、requires_grad和.grad均未改变。

实际导入leo_practical位于既有证据责任release下的 experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，而非根code；完整路径保存在本地actual_channel.json。载频2.462GHz、fs25MHz、IQ256、residual/post_sync、EQoff G=1、reference_snr_db30、reflect、接收质量/相位跟踪/IQ补偿开启。full/native和同状态重放最大绝对误差0；15个RX/day/eq硬件session稳定，不由TX定义。no_noise_total的quality=200为无新增噪声控制，绝非总SNR。

数据原L6300/原5RX/3day/6TX，采样90格各10包。center crop/pad后逐包complexRMS（epsilon1e-12），native normalize_input位置保留，无post-As重归一化。缓存序列化载体包含V，但V张量未使用、无前向/求导；忽略旧parent features，U不加载，真值隐藏。Checkpoint来源严格检查fixed B0 own20scratch→own10，30E/6660更新既有成本；未用target选中的权重。

本轮实测11702模型NN（含2synthetic realCP烟测）、1173autograd、native3600/render14400、0epoch、0optimizer/teacher/headfit更新。源观察墙钟57.099s，独立分析14.043s，总墙钟72.702s。独立聚合复算300个混淆矩阵、11700梯度分层格均PASS；该复算0NN/0IQ/0CP。PID/CWD/cmdline/startticks/CUDA空已独立核实，日志增长且stderr为空；完成后owned进程自然退出，未修改健康任务。

实际命令：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noise_headgradient_L_20261018_r01/experiments/cvs_noise_gradient40/runner.py
```

CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noise_headgradient_L_20261018_r01；输出=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-noise-headgradient-sourceL-s2026092701-m1-r01；本地聚合=E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018。没有新训练曲线，因为没有训练。

## 交付

- [逐draw/场景源指标](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/source_metrics.csv)
- [参数/分支/状态/质量分层梯度](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/paired_gradient_strata.csv)
- [原门控变化](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/gate_changes.csv)
- [独立结论](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/summary.json)
- [全部混淆矩阵](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/confusion_metrics.json)
- [主判据与源准确率图](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/paired_noise_gradient_and_accuracy.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/paired_noise_gradient_and_accuracy.pdf)
- [标签平滑校准对照图](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/objective_calibration_control.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/analysis/objective_calibration_control.pdf)
- [复算结果](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_headgradient_L_20261018/aggregate_validation.json)
- [最新P39正式7环境结果](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/report.md)

2PNG/2PDF实际存在，2张PNG已逐张目视检查。IQ、逐包梯度、物理ID、checkpoint未入Git。下一步仍须以真实身份识别的匹配实验验证独立A_g和A_s的联合价值；本次未启动新的训练候选，未以梯度阴性关闭全部噪声研究，也未把门控附带结果替换为新路线。

---

以下保留运行前固定设计，阈值未修改。

# P40：源 L 配对噪声梯度诊断

状态：PREREGISTERED，尚未运行，不承诺识别收益。代码 commit 在发布时记录于 release_commit.txt 与 verification.json。起点 HEAD：0aa84b5e31dbce76a34838c9e71f00fc3095f33e。

用户已确认继续完整 clean 保留训练组织。P39 的额外 urban U 软监督未通过预登记源域判据，关闭该具体方案；正式单种子 modest 收益不能改变源域否定。历史 sameTX 跨 RX IQmix 与孤立 AWGN 训练已否定，本轮不重开。

唯一假设：真实 practical 中新增噪声导致默认头分类间隔的逐样本参数梯度与同包 clean 更常反向；保持同一接收状态去除新增噪声应减少这种反向。主判据是实际 cosine W 的 margin-loss 梯度；CE0 与原 LS=.01 仅用于排除置信度校准混淆。冻结局部梯度不是历史训练冲突或识别收益的直接证据。

模型：固定 P39 B0，源训练 own20(13L+7U)→own10(0L+10U)，共 E30/6660 更新；F/SIG_TIME_X3＋统一 Sinc＋旧门控＋joint160/defaultcos30，900722 参数、206 state。严格核查来源、同一 source role IDs、final_only 冻结；B0 在正式成绩前固定，未选用 target 分数较高的 UAS。诊断零更新，不更换或另训头。

数据：原 L6300（完整5RX=[1,3,4,6,8]，3day，6TX，split392005）中用种子20261018461选每 TX×RX×day 10包，共900。无留出训练 RX、无 L_dev。cache 同时序列化 V，但只使用 labeled 键；V 无前向/求导，旧 parent features 不使用，U 不加载且真值隐藏，target 不读取。Phase2 capsule/split 不适用；复用已验证数据，仅核对现有角色句柄。

矩阵：同一900包 clean；2个独立命名信道 draw×mid/low urban×full/no_noise_fixed/no_noise_total。same-latent fixed 保持 RX lock、残余尺度、CFO、相位补偿、equalizer、AGC；total 重新计算接收状态。没有删样本。每包 native full 与 renderer full/frozen parity≤2e-6。实际导入 __file__ 与 P39 audit 比较。fs25MHz、fc2.462GHz、IQ256、residual/post_sync、EQ关闭 G=1、ref30、reflect、质量适配/相位跟踪/IQ补偿均开启。RX 参数按 RX/day/eq session 稳定，不由 TX 定义。native normalize_input 位置保留；去掉的是额外注入噪声，不反演录音原有噪声；total 控制中 quality=200 是无新增噪声反事实，绝非实际总 SNR。正式测试 IQ 与配置不变。

预登记否证规则：两次 low draw 均需有效梯度≥855/900，full 反向≥180/900，且 full 反向数比 no_noise_fixed 至少多90/900（10个百分点）。所有900留在分母。任一未达到则关闭这条噪声身份间隔梯度冲突解释，不以 CE01、其他分支或换阈值救援。即使通过，也不自动证明新增强或授权按结果挑训练损失。

预算：零 epoch/optimizer/head fit。新 NN=11702=900 clean+10800 satellite+2 synthetic realCP smoke；三目标共1173次autograd；native3600/render14400，CPU2线程，CUDA空。父模型 E30/6660更新单列为既有成本。独立分析零 NN/零 IQ/零 CP。墙钟、模块路径、状态/质量、每 RX/day/TX、门控、分类间隔和固定 clean 正确集合保持率均输出。无新训练曲线，不能伪造。

命令：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noise_headgradient_L_20261018_r01/experiments/cvs_noise_gradient40/runner.py
```

CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noise_headgradient_L_20261018_r01；唯一 run=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-noise-headgradient-sourceL-s2026092701-m1-r01；唯一 owner=codex/root/noise-gradient40；日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-noise-headgradient-sourceL-s2026092701-m1-r01-launch。源码发布入口 experiments/cvs_noise_gradient40/publish.py；先 realCP no-query smoke，通过立即继续源诊断。

技术停止规则：角色/来源/模块/配置/parity/预算不符、非有限数、模型状态改变、输出冲突或运行失败时保留产物并报告 FAILED，无自动重试、无改阈值重跑。科学负结果不停止中途；健康旧任务不修改。每候选仅一次独立P0/P1审查，直接P1修复最多一次定向复审。发布单归档单SHA比较、远程compile；启动后独立核实PID/CWD/cmdline/CPU/GPU可见性/日志增长。

预期产物：spec.json 所列角色、实际模块/config、真实CP烟测、私有900逐包logits/真实分支/参数梯度、物理元数据、分层CSV、独立summary、2组PNG/PDF与completion。只提交代码/配置/聚合报告；IQ、权重、逐包梯度/物理ID不入Git。
