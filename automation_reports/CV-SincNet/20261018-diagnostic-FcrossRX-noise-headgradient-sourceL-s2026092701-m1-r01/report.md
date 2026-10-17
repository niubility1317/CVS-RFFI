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
