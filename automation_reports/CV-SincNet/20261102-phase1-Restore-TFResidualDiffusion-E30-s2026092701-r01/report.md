# 时频联合Transformer条件残差扩散恢复网络

- run_id：`20261102-phase1-Restore-TFResidualDiffusion-E30-s2026092701-r01`
- group_id：`restoration-time-frequency-residual-diffusion-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

全新骨干/条件残差扩散目标/固定15调用采样器，源配对与冻结F0连续特征指导，从零单阶段30；自动原Main七景。系统比较，不是DURA微调或单因素消融。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 新架构与科学检验

本候选整体更换恢复模型，不继承 DURA、coarse、action、CG 或旧 G/IQ 教师。原地面完整 Tri-SIG-TIME SSL/default head/Main split 仍为主基座；用户要求六景逐个距同一 raw clean≤3 个百分点，并保护两个 clean 总体。性能主线、训练效率支线，默认 DURA 不变。

源端已有 RF 扰动诊断说明可观察扰动与随机扰动均可能被先验衰减；源端桥轨迹存在重复迭代损伤。条件均值/单位范数几何也不能保证保留弱特征。这些证据只支持检验不同建模方式，不证明扩散能够恢复丢失的身份信息。本项研究条件分布在多噪声尺度下的残差去噪，避免继续只扫描 MSAT 的损失权重。未按 Main 逐 TX 错误、目标分组或 SoftPower 分数选超参数；基准已有历史曝光，不能宣称新盲测。

时域分支将预条件 residual 与 received 的4个实通道按patch4映射为64×128 token；频域分支对两份complex IQ分别做256点正交FFT，再形成另外64×128 token。拼接128 token，经6层width128/4head/MLP512 Transformer联合注意力；sigma embedding调制每层LayerNorm。两个patch head分别输出时域和频域校正，后者逆FFT，二者除√2求和。没有预测已知设备类别或continuous160的latent head。共1614608参数，无外部预训练或新增依赖。

训练构造 r=clean−received，z=r+σ ε，ε为独立标准高斯，logσ~N(−1.2,1.2)。clean参与加噪训练样本的构造和loss，但clean、ε、真实信道/SNR都不是额外条件输入。EDM预条件为D(z,y,σ)=c_skip z+c_out N(c_in z,y,logσ/4)，σ_data=.5；输出clean估计为y+D。损失为EDM权重(σ²+.25)/(σ·.5)²×逐实数MSE+.05STFT+.1(F0连续特征cos+源方差标准化MSE)，不用latent辅助头、RMS anchor或硬单位功率投影。F(clean)特征及已冻结源方差只作loss，F/default head从不更新，无TX/logits/CE/原型/新身份头。

部署只用同条received与checkpoint保存的固定Gaussian latent，固定8步Heun、rho7、σ从80至.002再到0，最后Euler，共15次同一denoiser。全程FP32/TF32off，无churn/随机抽样、F0引导、best-of、多观测、ID衍生latent或batch统计。固定latent使相同输入逐条输出一份IQ，且不以其他query决定该记录的结果；它不是后验校准保证。零输出头的E0是Gaussian residual prior，**不是received恒等映射**。

从零单阶段30/7410，原Ls6300/Us56700/R07完整20draw+前10/sourceorder2026110102、B256tail24/247steps、AdamW.001/warm247/cos2e-5/WD1e-4/clip1。独立扩散噪声seed2026110203在无更新源2smoke后重置并保存RNG；固定latent CPU seed2026110204随state保存。训练每pair只调用一次denoiser，共1890000core；另有1890000在线F0输出特征和63000 clean特征参考，原F0已训练知识及源尺度成本不隐去。30轮不等于历史系统总成本相同。无V/OE/novel/目标训练、早停或选模，OE仅冻结诊断。

6项聚焦检查及一次独立P0/P1 PASS，覆盖正交FFT往返、Gaussian prior下独立Heun复算/15调用、逐记录与顺序独立/无采样RNG、checkpoint保存latent/严格加载/覆盖拒绝、各固定sigma梯度/target detach/非零head后骨干梯度、数据权限/固定预算、父技术失败不启动。launcher首个神经工作是原Ls2真实F0和G无更新smoke：完整sampler有限、实际loss/输出head梯度、E0严格重载；零head初始骨干梯度为0是预期。PASS后立即全30，不额外训练调试或质量筛选。

CPU队列仅等待健康SoftPower整个Main技术ARTIFACTS_COMPLETE且terminal无非零，不读其prediction/score/结果质量。等待不加载G/F0/IQ、不占GPU，父失败写所属queue_terminal并抛出异常，无自动重试。真实NN准入重检GPU4/6 UUID/free≥12000MiB/原compute≤2/新增1，总≤3；保护其他owner。固定final30后自动原Main126000+42000各七景，全prediction固定后原共享锁独立CPU truth-last，raw/DURA同IQ-ID判决精确复用，新G全量15调用。完整逐TX/RX/day/双向纠错/clean损伤报告，不因源loss或性能跳过。精确commit、路径、命令、数据句柄和技术停止规则见experiment.json。

单denoiser静态176971776实数Conv/Linear/attention MAC，部署2654576640；另每core两FFT+一IFFT，部署45个complex256点变换。静态计数不含F0/逐点算术/归一化/softmax/Heun/内存，也不等于实测时延。实际训练耗时和显存随证据报告；这是架构、损失和采样共同变化的系统比较，不能解释为同初值/同计算量单因素结果。

## 文献依据与适用边界

[EDM论文](https://arxiv.org/abs/2206.00364)提供噪声预条件、损失加权和采样设计；本实现独立应用其数学形式至条件IQ残差。[原作者预条件实现](https://github.com/NVlabs/edm/blob/main/training/networks.py)、[原作者损失](https://github.com/NVlabs/edm/blob/main/training/loss.py)、[原作者采样](https://github.com/NVlabs/edm/blob/main/generate.py)用于核对方程，不复制其源代码或权重。本项的固定8步/条件残差/联合时频骨干/附加F特征损失是本项目假设，不是EDM原论文的RF结论。

[DiT](https://arxiv.org/abs/2212.09748)研究图像latent patch的Transformer扩散骨干，不能把其图像指标移作IQ身份保真证据。[RFdiffusion](https://arxiv.org/abs/2503.05514)给出低SNR Wi-Fi指纹恢复研究动机；其6设备实验不证明本项目星地复合信道、跨RX、冻结原F0或未知设备有效。没有采用其性能数值作为本实验预期或晋级标准。

扩散先验可能合成源域常见但不属于当前发射机的细节；源F(clean)一致率、波形误差、真实TX准确率和未知设备保真分别解释。有限观测下不能保证逐样本无损重建，也不能从准确率保留比例推出“恢复了几成发射机信息”或理论上限。是否让IQ更易识别由原Main真实TX与clean双向损伤测试判断，不靠生成外观或源损失宣布成功。
