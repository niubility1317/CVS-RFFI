# 原RIEI优化算法+普通固定星地拼接+硬伪标签

用户明确要求不带STAR E训练组件。此run完全从零训练，替代riei_pl_loo_ce11_s299_t90_20260920_r02；旧权重不继承，旧产物保留。

沿用STAR对比数据范围：seed299、TX0–9、源RX0/1/2/3/5、目标RX4、200轮，固定clean/Loo配对(2,2048)。不在线重生成Loo、不做sigma课程、极端衰落修复、CFO/IQ。只有固定clean和固定星地样本拼接。

RIEI模型与原CVS优化版相同，ResNet1D18 FED、EC/RC。恢复原启动脚本scripts/launchers/run_cvs_fixed_riei_drift_ratio_sweep.sh配置：Adam heads/FED各LR0.0001，恒定无调度；有标签batch64，CE/MI/IE reduction=sum，feature norm mean且权重0.0001，无gradient clipping。不采用STAR分层采样，普通shuffle；不采用任何EMA或STAR SNN/投影/教师。

每batch两步：第一步clean TX CE+星地TX CE+clean RX CE+0.0001Norm+0.65PL，同时更新heads和FED。clean/星地身份CE均为1。第二步只对有标签clean计算1.2MI−1.2IE+0.0001Norm，冻结heads，只更新FED。RX/MI/IE不扩展到源无标签数据；源U仅提供普通TX伪标签，TX真值在进入loader前清除。各CE/MI/IE的sum归约相对于上一版mean是恢复原法的重要变化。

普通PL阈值0.9、系数0.65，对选中视图mean；E100启用，与本次原版CVCNN一致（已说明采用此默认）。无标签batch256。RX4只eval，不参与训练/BN校准/伪标签。固定E200 student为正式结果，无EMA、best选择或target选模。预测写盘后独立读取连接truth评分。

相对上一RIEI r02：去掉STAR E动态Loo/修复、分层采样、warmup余弦LR、clip10、EMA；LR0.001→0.0001，有标签batch32→64，CE/MI/IE mean→sum，恢复原法仅有标签clean的RX/MI/IE更新；不改变数据、seed、E200、身份CE1:1、MI/IE/Norm系数。PL采用最普通的自身硬标签。

验证：两次真实网络交替更新、CE系数及sum损失分解、PL启用边界、checkpoint读回、STAR E函数未导入/调用、一次独立P0/P1审查。独立输出及launch.lock；用户已授权突破两进程限制，此row允许第三个任务且预留6500MiB显存，保护其他进程。

启动VERIFIED：GPU7，PID187566；完成epoch=10，batch日志数=310。PID/CWD/argv/GPU、配置文件及增长日志已核实，尚未E200完成。证据remote_inspect.json。
