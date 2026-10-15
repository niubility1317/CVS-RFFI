# 固定 RX1：两位等规模教师的源域互补小试

按用户fab17893附件先检验一个预定源RX，不直接训练十位教师或蒸馏学生。两位教师从零E50、原SIG_TIME/默认cosine头端到端CE：SAMPLE在五个源RX分层取4464个L，EXCLUDE_RX1用另四个源RX的4464个L。两者每TX744、每TX/day248，原外层角色/标签总权限不变。当前已发布，两行训练中，无最终教师性能或图。

矩阵、配置、seed、数据/模型来源、精确命令、环境/CWD、资源、stop与预期产物唯一维护于experiment.json和experiments/cvs_crossfit_teacher50/registration.json。代码身份是实际发布Git HEAD，发布时写入preflight；没有额外报告/成员SHA或审批。

源RX1/3/4/6/8、day1/2/3、全部6TX，split/data392005，原L_s6300中固定L_train5580/L_dev720（fold2026101503）。各教师只从该5580取4464，unused1116不训练，L_dev720含RX1的144条且与训练互斥；子集seed2026101505。U_s56700 TX隐藏且不迭代/不前向，V27000只作原生健康遥测，固定final50不选最佳epoch。没有U/V标签转训练、额外标签或不匹配父权重。真实batch base indices/RX/day/TX逐步记录，EXCLUDE的所有CE和训练统计排除RX1。

这是明确source-only教师证伪，不是普通Phase1正式发布比较。每行冻结后自动预测原720源样本的clean及六practical，再自动进行RX1源比较；不读取正式目标IQ/truth，不另跑目标scorer。本例外不改变主线HQ训练结束后自动正式clean+六practical测试的计划。其他聊天E20教师的预算/折不同，不作为本E50的初始化或匹配结果。

固定single SIG_TIME、内部门控/参考响应、默认cosine6×160 scale30；原骨干及默认头joint CE更新，不冻结后另训头。采样25MHz、IQ256中心裁剪、原native RMS；教师训练关闭generic/satellite增强、U、EMA及全部额外lambda损失。每行E50=50L+0U，222步/轮、11100实际更新，batch128，训练前向1420800/行。AdamW lr2e-4→1e-6 cosine、wd1e-4、label smoothing=.01沿原SIG_TIME设置。parent成本0；teacher自身source-validation前向由真实hook计量，源预测5042/行（真实CP smoke2+7×720）、共享原A0烟测2及源门控replay1440单列。E50不是收敛保证。

源practical评价保留原adv3b02_xuc leo_practical residual/post_sync、EQ关闭G1、reference_snr_db30、接收质量适配/相位跟踪/IQ补偿、reflect/FIR24。source eval流2026101506，RX/day/eq session不由TX定义；不反演原录音硬件或噪声。教师训练无该信道调用；source eval新增噪声SNR不代表总SNR，正式IQ不打开或改写。

两个独立教师只传递可比较的类别语义，本小试不对齐独立joint160/分支，不建U logit缓存，不做学生反馈。先比较RX1 clean的纠正/新增错误、true-class概率、默认头间隔/30、高置信度(.95)错误、class-pair和day；全部六种源practical也原样保留。阈值预定为RX1净修复至少3/144、原门控源拒绝区净修复非负、高置信错误率增加≤.02；不允许某项收益抵销另一项失败，不自动扩库/学生。

原固定A0 final50/default头与ownEMA只作真实checkpoint首步smoke和源门控参考，绝不用于教师初始化、EMA或监督。重放调用实际native rx/day quantile、batch_neighbor窗口/min_conf及强噪声/一致函数，不把source TX传给gate。固定shuffle720分为256/256/208，强分支保留原train Dropout/BN行为，torch.func使用克隆的临时buffer，原baseline/EMA状态与模式不变。该source replay不是历史U实际batch或拒绝决定，不能用其源精度声称U伪标签精度。

实际launcher使用N607普通账号、CPU2/loader0、每worker单GPU，共享receiver_residual_capacity4.lock按所有owner及pre-CUDA计数≤3/card、启动前free≥12000MiB，动态借用空闲容量。精确入口是既有解释器 `-u -m experiments.cvs_crossfit_teacher50.dispatch`，CWD/不可覆盖run/release/row路径见experiment.json，唯一owner codex/root/crossfit-RX1-teachers-20261015。源码/配置先Git；一次archive比较、远端compile/preflight、真实CP无querysmoke后继续，独立PID/CWD/argv/GPU/log读回。角色/RX/来源/非有限/预算/覆盖/输出/技术错误退出owned行并保留，不自动重试或新增派发；已经健康行自然结束，HQ和其他owner进程不停止/迁移/重启。

已有4项核心检查PASS1.468s：4464配额与RX物理排除、真实args关闭U/EMA/增强、222步ForbiddenU不被迭代、原头/t2/SIG真实CE梯度。新增4项评价检查PASS.700s，包含真实模型functional train强视图与克隆模型结果一致且原状态不变、native门控/无TX、错误修复/破坏和输出不覆盖。新增pipeline原3项中2项PASS；mock把str当Path导致1项fixture失败，改为Path后该定向PASS.004s，验证U暴露机制在加载CP前拒绝。合计11个唯一检查分批执行，不声称完整11项单次重跑。compileall通过；Torch/NumPy既有ABI警告未改环境。

一次独立P0/P1审查PASS：核对两行4464/TX-day/RX排除、E50/11100纯L、关闭U/EMA/增强、默认头、强视图克隆buffer、源比较、发布和容量。一次archive/compile/preflight/真实CP smoke及独立PID/GPU/日志增长现已VERIFIED；源比较待冻结。后续只需按既定最小流程发布并核实运行，不增加报告审批或数据重验。最高风险是排除RX1损坏更多判断或提高高置信错误；若发生，保留负结果，不因概念合理而扩库。追踪见analysis/cvs_crossfit_teacher_traceability_20261015.md；当前verified5/pending5/deferred1/rejected0/blocked0；发布及初期读回已验证，全E50和源互补仍待完成。

## 实际启动与初期源暴露

VERIFIED：训练commit `f4018c9b2acdfc6f9b013b956a4735af74c435ee` 已push且独立remote OID一致。一次不可覆盖release/run，archive传输比较VERIFIED，远端compile PASS，普通账号preflight VERIFIED；真实原A0默认头无query烟测PASS后两行从零继续。dispatcher PID3176235，SAMPLE PID3176260/GPU3，EXCLUDE_RX1 PID3176337/GPU6。独立/proc cwd/argv/CUDA mask及nvidia PID→UUID→index匹配，日志与teacher_batches流较初次读回增长。

两行已实际完成4轮，各自覆盖4464不同训练样本。SAMPLE实际训练覆盖RX1/3/4/6/8，EXCLUDE只覆盖3/4/6/8；实际args为epochs=label_epochs=50、pseudo_epochs=0、U/EMA/use_aug/concat均false、sat_training_mode=disabled，源coverage的U前向/EMA更新均0。初始化scratch_only，ancestors/checkpoint_sources空、target_contact=false。全程11100预算、最终源预测与互补判定尚未完成，不把初期覆盖或V健康数当最终收益。证据readback_initial.json、readback_launch_verified.json、delivery_r01/submission.json。HQ两行仍按原发布运行，未热改/停止/迁移。
