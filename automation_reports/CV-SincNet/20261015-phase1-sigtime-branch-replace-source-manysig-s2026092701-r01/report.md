# 冻结 B0 五路径源替换诊断

状态：LOCAL_VERIFIED，尚未发布。该阶段落实用户2026-10-15最新同分支对齐附件中的训练前操作性诊断；它不替代完整8配置×3seed要求，不训练任何新模型/分类头，不按结果筛掉F/P/R等分支。

固定来源：原普通clean/satellite拼接B0-s2026092701，从零E50=33+17、11100更新、source-only final50、原默认cosine6×160/scale30。own_model核实实际L/U/V和物理ID、scratch/无祖先/无目标选择、最终冻结；该模型仅诊断，不作为后续内部开发折的初始化或匹配基线。父训练成本已在原B0报告记账，本次新增epoch、optimizer/head-fit更新均0。

输入复用原L_s6300中的既有TX×源RX×day平衡720面板，sourceRX1/3/4/6/8和day1/2/3、6TX、split392005；该面板曾参与B0训练，不能称为独立未见数据。U不打开、V不前向、query IQ/truth/目标指标不读取。六种practical源验证场景沿原独立固定命名流，各720样本；不读取或改写正式固定received IQ。

实际模块已本地核实：experiments/adv3b02_xuc/code/model.py 的CVSincNet，fuse321=160T+160F+rho1，freq_stats_proj已开启。T/F在真正fuse输入读取，F含全部统计增量；P在真实pa_norm输出读取（包含实际pa_local/pa_delta），R在response_projection输出读取，S在sequence_pool.output_norm的gain前160维读取。所有捕获器学生路径保留计算图，教师/诊断值才detach。joint160辅助重组与实际前向数值严格相等。

在同一个冻结B0中生成clean各分支。随后同一个urban输入先做完整原前向，再只替换T/F/P/R/S其中一项，重算原融合/后处理/默认头。T/F仅替换实际fuse输入对应段，保留urban rho和其余段；P/R/S用实际gain前模块输出替换。每次运行断言仅被选分支等于clean，其他四分支及rho逐值等于完整urban；T干预的S必须仍为urban。模型全程eval、requires_grad=false、no_grad，无拟合头，无optimizer，最后检查所有参数/buffer逐值不变、grad=None、hook移除。

信道沿真实 imported leo_practical residual/post_sync/noEQ（G=1）、25MHz、IQ256、referenceSNR30、quality adaptation/phase tracking/IQ补偿true、reflect、输出不RMS；正式IQ和配置不变。各比较共享同一实际source波形、物理参数和命名随机流，源quality分层为signal_to_added_noise_db，不能叫总SNR。

指标：每场景/分支/LOS-PARTIAL-BLOCKED/锁定状态/新增噪声SNR区间的原默认头准确率、替换相对完整urban的差、错误修复数/新错数、归一化默认头间隔、固定同模型clean正确集合保持率。该混合不一定对应真实可获得输入，所以不是训练可达上界，也不能直接选出最佳训练分支；负差和共同适应现象原样保留。

预期前向26640=720clean+6场景×720×(完整urban+5替换)。实际forward_packets/墙钟以results.json为准。唯一run/output root PROJECT/runs/20261015-phase1-sigtime-branch-replace-source-manysig-s2026092701-r01；唯一release cvs_sigtime_branch_replace_source_20261015_r01，owner codex/root/branch-replacement-source-20261015。普通N607账户，CPU-only，CUDA_VISIBLE_DEVICES为空，torch/OMP/MKL/OpenBLAS2线程。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_branch50.replace`，CWD PROJECT/releases/cvs_sigtime_branch_replace_source_20261015_r01。首个真实B0 clean前向actual T/F/P/R/S/J检查后打印SMOKE PASS/query_read=false并立即继续。

本地发布命令：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_branch50.replace_publish --output E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/replace_delivery_r01`，独立同命令`--inspect`回读PID/CWD/argv/CPU日志或完整source-only结果。一次archive传输比较、一次编译/普通账户预检。来源/角色/非有限/跨分支泄漏/模型状态变化/输出冲突等系统技术错误停止该owned只读作业并保留产物，不自动重试、不影响其他健康作业。

本地5项检查PASS：完整F统计/完整P；六种对齐位置真实学生梯度、S到local_projection和t2、aux W/gain零而主CE W非零；单项替换/其余分支rho保持/模型state与hook不变；训练态替换拒绝和真实joint祖先参数195041（71个张量，含三组gain，不含W）；ALL分支权重和=1。命令 `E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_branch50.checks`；stderr在local_artifacts/cvs_joint160_identity_20261015/branch_checks_stderr.txt。实际总参数863730，未增减网络模块。

一次独立限定该诊断代码/发布路径的P0/P1审查完成，指定范围未发现P0/P1；真实融合/default头重算、其他分支/rho逐值保持、冻结和数据边界、独立发布路径一致。预计产物experiment.json、progress.json、results.json、sample_replacements.jsonl.gz及release诊断日志。物理级原始输出仅保存在远端/local_artifacts，不提交Git。追踪记录analysis/cvs_branch_alignment_traceability_20261015.md；新的内部L开发折、统一预热/全局梯度校准、8配置训练和三seed尚未实施，不把此诊断叫完整设计落地。
