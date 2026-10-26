# NoSIG 原始／恢复域标签专家：固定源矩阵

run `20261026-phase1-PAMF-NoSIG-RestorationLabeler-source-m2-r01`，单学生seed2026092701的两个匹配标签专家。完整方法、来源、公式和成本在[源计划](../../../docs/goals/PAMF_RESTORATION_LABELER_PLAN_20261026.md)，逐行机器记录在[experiment.json](experiment.json)。本行只源训练／诊断，不做目标测试、不生成U标签、不训练SSL学生。

两个专家严格加载同一新NoSIG普通PL E30（合法scratch NoSIG E20＋10），分别新增10×222步L128至累计E40。新增训练均排除RX1，使用原L中的5040条；RawLabeler输入raw，GLabeler输入同包固定G四实现×三景缓存，`view=(step−1)%12`。真实L CE、LS0.01，整个骨干、原6×160默认头与native Sinc共同更新。原CENTER_25K/CENTER_UNITS AdamW、physicalclip5；显式重置optimizer，LR.0002至1%cosine、WD.0001、eps1e−8、betas(.9,.999)。无U迭代、EMA、PL、KL／CL、新头或标签专家以外优化目标。父和G已见过源RX1，RX1只是新增适配留出。

共享smoke是第一步：真实checkpoint/defaulthead/nativeSinc/G的原始与恢复L128联合反传，原分类头和Sinc梯度有限非零，零optimizer／U／V／target。PASS后立即构建同G L6300×12缓存。原raw／G IQ之外不额外加噪、不归一化、不按F分数挑实现。G读取IQ和物理参数、不读取TX标签／F输出、不更新。专家F允许用原L真标签。

两个final自动完整V预测：RawLabeler raw27000，GLabeler12恢复视图27000；逐行固定后保存logits，全部认证预测固定后再使用V真值计质量。困难池固定为原NoSIG E30 confidence<.95，两个适配专家各confidence≥.95、同类才认证，G专家平均12份softmax。进入后续学生研发的规则固定为：认证≥200、观察precision≥99.5%、相对原E30 raw净纠正≥100。失败完整归档，不放宽参数，不自动启动U标签或学生训练。RX/day/TX以及两个专家单独质量均完整报告。

新增4440优化步、568320条L曝光；两个专家每行5040唯一L、RX1新增优化曝光0，G每视图23680曝光。L IQ缓存154828800float字节／75600G包；含smoke／evalwarmup预计新F919582包、新G399732包，实际按日志核算。它们是额外教师成本，不能冒充免费增益或三学生seed结果。

代码／配置以同次提交的`runtime.code_commit`为唯一身份。命令：`PYTHON -u -m experiments.pamf_channel_pl.restoration_labeler dispatch --config PROJECT/releases/20261026-phase1-PAMF-NoSIG-RestorationLabeler-source-release-r01/runtime.json`。N607现有普通账户Python/Torch2.1、FP32/TF32off、CPU2线程，无依赖安装；CWD为该release。唯一owner `codex-pamf-nosig-labeler-20261026`，独立不可覆盖run/release。

共享smoke/cache后两个专家有资源即并行，每行冻结后立即V预测，全部完成后控制器自动评分。GPU0至7按全局CUDA＋pending<2且free≥10000MiB取空位，保留全部健康外部工作。一次archive传输比对／远端编译及独立PID-CWD-argv-GPU-loggrowth核实。任何真实smoke、合同、native Sinc、默认头、finite、覆盖、状态或技术失败停止新派发，健康行继续；不自动重跑，不修改外部进程，不据超时重发。
