# NoSIG 有噪声困难伪标签：六行源证伪

run `20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-m6-r01`，单学生seed2026092701。完整固定方法见[计划](../../../docs/goals/PAMF_NOISY_HARD_PL_PLAN_20261026.md)，逐行矩阵和停止规则见[experiment.json](experiment.json)。这是新的噪声鲁棒假设，原图／恢复专家／raw锚点的可信标签条件仍FAIL，不改旧门槛。

每行严格加载同一NoSIG普通PL E30，独立＋20至E50/11100/13L37LU；完整L机制、原6×160默认头joint、原优化和NoSIG结构保持。RawLabeler E40仅作冻结标签来源，来源／同契约／实际NoSIG头与祖先严格检查。原U56700真值隐藏；原E30confidence<.95、R40confidence≥.95、rx≠1构成noisy candidates。其源V代理precision93.829947%，不是U真实精度。其余U保留当前student raw self-PL；候选包替换旧self标签，不双重监督。

六行固定为NoisyCE、NoisyScaledCE、NoisyGCE、NoisyGCE2Raw、NoisyGCEG、NoisyGCEHG。GCE q=.7，loss=(1−p_target^.7)/.7、梯度CE×p_target^.7；ScaledCE统一乘本batch候选的该权重均值并detach。Raw2／G／HG项为.75raw＋.25第二视图，额外前向只对候选；HG按步交替H/G。后两行只用已完成同G三fresh×三scene成对IQ缓存，view=floor((step−6661)/2)%9，不新加噪、不混IQ、不做G后归一化。G和教师不更新、不另训头。

先实际R40/defaulthead源L2 smoke，通过即完成隐藏U标签缓存（F56702/G0、零优化器）。再六行真实E30/L128/U256联合backward smoke，包括实际候选与非候选、raw/G/H分支、默认头有限非零梯度、L768/As KL0、零优化器／U真值／query。PASS即在资源空位并行训练，不另设smoke凭据。

每行新增4440步，六行26640步／120模型轮；每行U曝光1134000、候选曝光=20×唯一候选数，第二视图行额外F同数，实际view计数保存。R40额外10轮训练和cache/G成本单列，不能宣称完整训练成本与PL相等。

每行final立即源V27000八视图。全部六行完成后，相对既有NoSIG-PL控制固定要求：RX1 G三景≥+.3pp、overall raw≥−.1pp、原困难V raw净正确>0。多行通过先源raw最高，再RX1 G最高，再较低成本顺序；源选择在任何新target访问前冻结。RX1是附加机制留出，基座和G已经见过，不称真正未见域。若全失败不扩种子。最终三个学生seed原F0 168000/seed的raw／同G clean＋六practical及原三项实际目标不降低。

执行命令：`PYTHON -u -m experiments.pamf_channel_pl.noisy_hard_source dispatch --config PROJECT/releases/20261026-phase1-PAMF-NoSIG-NoisyHardPL-source-release-r01/runtime.json`。N607现有普通用户Python/Torch2.1、FP32/TF32off、CPU2线程，无安装；CWD同release。唯一owner `codex-pamf-nosig-noisyhard-20261026`，独立不可覆盖run/release。GPU0至7按全局CUDA＋pending<2/free≥10000MiB派发，计入全部owner，保留健康任务。

Git commit为实际代码／配置身份；一次archive比对、远端compile和独立PID-CWD-argv-GPU-loggrowth。任何实际smoke／协议／模型身份／finite／覆盖／状态／技术失败停止新派发，健康行继续，保留输出，不自动重跑或触碰外部进程。源科学失败完整保留，不通过测试集反馈调q／阈值或换seed。
