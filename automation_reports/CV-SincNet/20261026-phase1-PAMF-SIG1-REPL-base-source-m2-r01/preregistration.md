# 单视图完整 PAMF 共享源控制预登记

用户最新指定原完整 PAMF 保留 SIG_TIME，改为单视图。结构为既有 CompositePool SIG_TIME：一组96→6路径投影，含时间坐标／四窗口，112→160投影，18976参数；时频、PA、参考响应、原6×160头joint保持。

固定两行、单seed2026092701：SIG1-PL继承同契约从零单视图普通PL E20，原生+10+20；SIG1-Sup从零20+10+20，不读U。各累计E50/11100，PL13L37LU；完整L768前向、AsKL0、ordinary/raw RDrop、LS.01、原跨RX和Sinc设置。PL门槛.95、权重1，无EMA/另训头。新17760步/80模型轮，共享旧4440步另计。

已核实三seed同结构E20实际张量、完整来源、物理角色与原六类；不用NoSIG或裁切X3权重。只有seed01进入本次源控制，不能称三seed结果。输入源L6300/U56700/V27000，RX[1,3,4,6,8]、days[1,2,3]，U TX隐藏；V不更新任何状态；无target句柄。数据VALIDATED_ONCE复用，无新重验。

代码/config身份为提交的Git版本，由runtime.code_commit记录。CWD为PROJECT/releases/20261026-phase1-PAMF-SIG1-REPL-base-source-release-r01；已有Phase1 Python/Torch2.1，FP32/TF32off、CPU2线程，不安装依赖。入口：PYTHON -u -m experiments.pamf_channel_pl.sig1_repl_base dispatch --config RELEASE/runtime.json。逐行配置、命令、输出见experiment.json。

唯一owner codex-pamf-sig1-repl-base-20261026；独立不可覆盖run/output。先真实单视图E20与固定G的源L2检查，再两行L128和PL U256／Sup U关闭的联合backward；默认头、单视图梯度均非零且finite，0优化步。PASS直接继续，真实smoke不另建许可。

全部GPU可用；派发时计全局CUDA+pending<2且free≥10000MiB，最多2个本run训练/预测worker，保留其他owner健康任务，无静态占卡。一次发布归档比较／远端compile／独立PID-CWD-argv-GPU-loggrowth检查。技术／契约／来源／smoke／非有限／路径冲突停止新派发，活动健康行继续；不因科学低分停止或选择性重跑。

每行固定final后自动V27000八视图raw/Graw/三H/三G，保存完整logits与TX/RX/day/overall分组和全量训练曲线；使用同固定原F0 G-LU，不优化G。两个协同方向共用同一公共E30和PL/Sup，不重复发布。本次不训练改进REPL或历史U候选，不自动扩seed；原三seed实际clean/raw六景/同DURA六景目标保持，源评估不是目标成功。
