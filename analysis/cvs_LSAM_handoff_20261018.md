# L-only SAM 当前交接（2026-10-18）

完整联合目标仍ACTIVE；当前行尚未完成训练、源资格或正式评分，不标complete/blocked。最新用户约束为F＋跨RX同TX、E30、原256点、全部5源RX、完整L6300/U56700隐藏TX/V27000、split392005和默认cos头联合训练，不留出训练RX。包络均值/波动、FIR/clock、梯度投影/dropout/MC2/配对噪声等固定失败路线保持关闭。

当前唯一实跑run为20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01，实际代码提交a71b0094dba1264b7bacfc742dc03ab3f3a59aa2；release cvs_F_LSAM_E30_20261018_r01；package experiments/cvs_lsam30；同row报告[report](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01/report.md)。三行NOAS_REPLAY／AS_REPLAY／AS_LSAM，原REPEAT20／CONCAT20各自student＋自身EMA追加10至E30；As两行严格同父，NoAs的treatment父不同但源契约/初始化规则一致。新增各2220更新、8Lforward/2extraVJP、一次原clip5/AdamW/EMA；只改变L梯度组织，未新增IQ Ag、部署参数、头、loss系数或U规则。原脚本launch-owner为codex/root/F-LSAM-E30。

已独立核实dispatcher871606；worker871633／871706／871786分别GPU3／4／5，PID/CWD/argv/CUDA及nvidia-smi映射一致；源初始化与原roleIDs一致，真实两父CP无query smoke4NN通过。三行E21回调各222新增更新已通过，native基础L113664＋重放113664＝227328实际LNN、extraVJP444，原clip/AdamW计数一次每步。旧reference_progress.L_packets只计基础4forward，不能作为总8forward成本。最新完整gzip可读步数386／271／253，日志增长、无非有限跳步、U真值0；这只是当时的观测，恢复时重新核实原句柄，不由旧RUNNING文字推断live。

私有工具位于本worktree local_artifacts/F_LSAM_E30_20261018：runtime.json／remote.py／monitor.py／partial_steps.py。用既有verified native Python/SSH短连接运行这些只读脚本，不展示私有runtime或服务器账号。monitor只读源进度与进程、不读target scores；partial_steps用zlib解码未关闭gzip的完整JSON前缀，只用于健康计数，最终日志分析读取全部完成记录。不要重复publisher、覆盖run、热改或重启健康任务。观察超时先只读核对原PID和产物。

每行自动train→fixed源V1800资格（全5RX/90cells，clean1＋6practical×2draw=23400NN；21600物理处理、0更新/U/target）→固定7view received预测→独立CPU truthlast。root finish先只读三行资格生成source_matrix_decision，再汇总全部正式分数；Agent在完整源资格冻结前不读取本run目标指标。三行全部固定测试，不根据source或target删行。源判据AS_LSAM vs AS_REPLAY：clean/mid不降，low≥+.5pp，每suburban≥−.25pp；失败关闭固定rho.05/坐标/父/seed/optimizer/预算配方，不调参救跑，性能差不技术停止。

训练结束后必须完整复算3×10epoch/2220step、370输入池/4440 L-U序列、原头更新与父来源，实际8LNN/2VJP/teacher/资格/正式prediction/CPUscore/父训练及synthetic成本分别记录；完整读取L_replay_steps（权重/RNG/buffers恢复、半径、中性控制误差）、reference_steps、unit_steps、actualargs和channel coverage。完成3×7视图378个CM／63分区及3对×7配对、全部RX/TX/day和source资格2draw每RX计数；不能用source loss或梯度改善替代识别收益。保留负结果、缺项；准确率/F1、suburban绝对值与差距、修复/新增错误和clean∩scene联合正确均报告。全部训练/评分产物齐全才标ARTIFACTS_COMPLETE/ANALYZED。

开发成本本地＋远端完整F各9216NN/3syntheticupdate/12extraFVJP，总18432NN/6syntheticupdate/24extraFVJP；源smoke4NN单列。首个NumPy测试seed超32位在任何F前向前失败，额外2随机modelbuild/0NN，保留记录。新增真实E10每行预估2273280LNN、4440extraVJP，实际从training_result取；两份E20父共享一次共40modelE/8880updates＋新增30modelE/6660updates＝唯一70modelE/15540updates，不以每行E30掩盖额外前向。

本候选即使source或单seed目标提高，也不是IQ Ag/factorial交互/跨seed联合目标完成；不从目标成绩自动晋级或扩seed。最终仍欠新的A_g clean独立价值、A_g×A_s四因子和多seed证明，严格匹配B00的clean正净收益、low>60%以及suburban探索≤2pp且不降低clean。

终轮辅助工具已准备：analysis/cvs_LSAM_recount_20261018.py（先完整source及源资格，再正式378CM/63分区/21配对，0IQ/CP/NN/rawtruth），analysis/check_cvs_LSAM_recount_20261018.py（39CM/2220records/8negative合成PASS），analysis/cvs_LSAM_report_20261018.py与analysis/cvs_LSAM_figures_20261018.py。尚未消费本run终轮或目标指标。旧共享analysis/cvs_view_pcgrad_source_logs_20261018.py按单行调用_base_run复用完整源日志，source_qualification要求source_matrix_decision存在；target侧复用已验证metrics函数，不引入额外预测或诊断头。执行后保留private完整recount，再从whitelist metadata生成公开report；检查两张PNG后登记实际结果。
