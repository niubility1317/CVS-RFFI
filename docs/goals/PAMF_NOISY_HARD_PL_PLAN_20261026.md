# NoSIG 困难样本的有噪声伪标签学习

状态：**六行单seed源学生已实际启动，尚无完整源／目标性能。** 原RGPL、恢复专家及两个raw锚点的固定可信标签规则均已失败，结论保留。本路线不把它们改判通过，也不放宽其门槛。它明确使用有噪声标签，检验鲁棒损失能否利用此前被排除的有效纠错。

## 已知事实与新假设

原NoSIG E30困难V2326条中，RawLabeler适配后正确1508→1820，净增加312；其confidence≥.95的1329条标签正确1247（93.829947%），纠正123／引错38。恢复专家只保留33条原本判对的标签，监督锚点342条只纠正1，PL锚点720条只净纠正6。近乎无误的交叉门控把新增纠错一起排除了。

本版保留RawLabeler的原始.95门槛和原E30 confidence<.95困难池，不继续换阈值或教师。新假设是：**保留其中的噪声，用有界且按学生概率降权的损失限制破坏，同时让真实L监督和普通PL继续发挥作用。** 93.83%是已暴露源V代理，不是U真实精度，也不是足够有效的保证。错误可能有类／RX依赖，文献中的噪声结论不直接保证本任务。

## 固定标签与损失

单学生seed2026092701，固定自己的NoSIG E30及RawLabeler E40（原L5040、排除RX1新增适配、nativeSinc/骨干/原默认头joint CE10轮）。严格核实其NoSIG从零E20＋10祖先及同契约。RawLabeler冻结后对原U56700逐包raw预测；U真值仍隐藏。b_i为原E30 confidence<.95、RawLabeler confidence≥.95且rx_i≠1。g_i为其argmax。这些是**noisy candidates**，不称认证真标签。

其余U保持原student当前raw self-PL（confidence≥.95）；b_i的旧self-PL被替换，避免对同包同时施加矛盾标签。原完整L机制、默认头joint、NoSIG结构、原预算保持。

`Lq(p,g)=(1−p_g^q)/q`，固定q=.7；q=0分支使用普通CE。GCE每包损失≤1/q，logits梯度为CE梯度乘`p_g^q`；学生认为极不相符的标签影响更小。用`−expm1(q*logp_g)/q`稳定计算。采用已有[广义交叉熵原论文](https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf)的q=.7，不作q扫描；本版不是新的鲁棒损失理论。

`L_U=mean_i[(1−b_i)m_i CE(raw_i,y_self_i)+b_i((1−rho)Lq(raw_i,g_i)+rho Lq(extra_i,g_i))]`。

全部U batch归一；rho=0或.25。没有EMA、新头、在线参照、KL／CL或参数投影。GCE和CE改变了梯度幅度，因此另加统一缩权CE控制：对该batch的b_i，CE统一乘`mean(p_student(g_i)^.7).detach()`，区分平均剂量变小与逐样本降权；它不是精确匹配全模型梯度范数。

## 六行最小匹配源矩阵

| 方法 | 困难包raw项 | 第二视图，rho=.25 |
| --- | --- | --- |
| NoisyCE | 普通CE | 无 |
| NoisyScaledCE | 统一缩权CE | 无 |
| NoisyGCE | GCE(.7) | 无 |
| NoisyGCE2Raw | GCE(.7) | 同包raw再次前向 |
| NoisyGCEG | GCE(.7) | 同包G |
| NoisyGCEHG | GCE(.7) | 同包H／G按步交替 |

六行使用完全相同b_i/g_i和U/L顺序。后两行只读此前已完成的三fresh×三scene成对H/G缓存，原物理信道／同一固定G，不新加噪声、不作IQ混合或G后归一化。G-only和HG使用同一view schedule：`view=floor((step−6661)/2)%9`；HG偶数local步H、奇数G。Raw2第二前向包数与G/HG相同，新增视图只对b_i，b为空时跳过。记录真实每视图曝光，不能把视图当独立物理样本。

## 预算、选择与原目标

各行从同NoSIG E30独立＋20至E50／11100，累计13L＋37LU，原L仍联合训练。每行新增4440更新，六行合计120模型轮／26640更新；RawLabeler额外10轮／2220步和U标签推理、G缓存成本另计，不声称与普通PL完全同训练成本。推理保持一个NoSIG F／同一固定G＋F，无教师、缓存或额外头。

逐行final立即原V27000八视图；全部六行完成后才选择。相对已冻结NoSIG-PL源控制，固定要求：新增分支留出RX1 G三景≥+.3pp、overall raw≥−.1pp、原E30困难V的raw净正确较PL控制为正。RX1原基座和G已见过，仍不是独立未见域。源条件不代替目标性能；失败行保留，不选择性重跑。

多行通过时，先取overall源raw最高者，平分取RX1 G三景最高，再平分取额外训练成本较低者。源选择在新目标访问前冻结。若无通过，不扩三seed或以不同目标指标拼接方法。通过后另行固定三个学生seed2026092701/02/03，同配方各自父／RawLabeler，不逐seed调参；原F0目标RX所有日期168000包/seed，raw与同G-LU clean＋六practical全部预测后独立truth-last。

原绝对要求不降低：rawclean≥85.981746%，raw六景>72.096197%，同G六景>78.074636%，并分别超过新NoSIG匹配监督／普通PL及其相应增量标准。新NoSIG目标基线尚无三seed结果，不能用源V数字填充。原goal未完成。

## 执行

N607现有普通账户，FP32／TF32off、CPU2线程；不安装依赖。GPU0至7全局CUDA＋pending<2、free≥10000MiB、计入其他owner。先真实RawLabeler checkpoint源无query smoke及完整隐藏U缓存，再六行真实NoSIG E30／L128-U256联合backward smoke，通过立即并行训练。独立不可覆盖run/release、唯一owner、一次代码提交／P0P1审查／archive比较／远端编译及独立PID-CWD-argv-GPU-loggrowth。技术失败停止新派发，健康行继续；保留所有旧任务、模型与负结果。

## 实际运行

run `20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-m6-r01`，代码`b298bbc5afbb734e91c19c4c88ef39622a3ff6e8`。12项聚焦检查、唯一P0/P1审查、一次archive比较和远端compile通过。实际R40 checkpoint源smoke与U56700缓存完成，得到2858唯一noisy candidates；教师状态不变，U真值／target0。六行真实L128/U256 backward均通过，随后在GPU0/3/4/5/6/7训练到累计E32，PID/CWD/argv/GPU及记录增长独立核实，非有限跳步0。仅一个学生seed，不能称三种子或目标收益。完整状态见[本行报告](../../automation_reports/CV-SincNet/20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-m6-r01/report.md)。
