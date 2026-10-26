# 单视图 REPL RX 方向：源必要条件预登记

本次只读已完成的 m201 单视图 ordinaryPL E20/4440 与原冻结 DURA；不训练或重训任何监督、普通PL、原REPL对照，不读取已停止控制的partial，不打开U或target。

单seed2026092701，原L6300/V27000源RX[1,3,4,6,8]、day[1,2,3]、原六TX。原物理ID/顺序和VALIDATED_ONCE契约复用。真实E20 state/default6×160 head、singleSIG_TIME1view/18976参数和scratch来源先核对；实际源L2 raw/G(defaulthead) smoke共F4/G2，0优化，PASS立即全量提取。冻结F/G parameters及全部named buffers前后相同。

每记录1raw＋4信道随机实现×3场景G特征，共432900新F、399600新G包视图，另计smoke4/2。G为同一个原F0 G-LU，reuse=True/graph=True/native=False，FP32/TF32off。12个G特征先分别L2normalize、平均后再normalize；这只作几何，不是单个G场景分类分数。随机namespace含U是沿用名字，实际只对L/V记录生成；U数据0。

用L类别/RX平衡均值估计共享RX偏移，投影掉L类别均值差子空间，不放大近零方向；alpha沿用.25。L跨类留一每次只用其余类别估计偏移/投影，检查被留类RX方向。V仅最后只读测方向迁移，不构建候选b、投影或阈值。

固定必要条件：raw和Gmean12都须protected_RX_fraction>=.05、LOCO平均余弦>.1、LOCO正方向比例>.5、L方向对V的平均余弦>.1。全部结果保留；失败不扫描alpha/阈值或重新放大，成功也不等于分类提高、不自动启动学生或目标测试。

代码/config由Git commit固定并写runtime.code_commit。CWD PROJECT/releases/20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-release-r01；既有N607 Phase1 Python/Torch2.1、CPU2线程，无安装。命令：PYTHON -u -m experiments.pamf_channel_pl.sig1_repl_rx_source dispatch --config RELEASE/runtime.json。唯一owner codex-pamf-sig1-repl-rx-geometry-20261026，新不可覆盖run root。

只派一个frozen-inference GPU worker，0至7全卡准入核实globalCUDA+pending<2/free>=10000MiB，计所有owner。一次archive比较、远端compile和独立PID/CWD/argv/GPU/loggrowth。真实输入/shape/state/nonfinite/smoke/输出冲突等技术失败结束本worker，保留产物不自动重试、不动其他进程。科学FAIL作为完整负结果，不算技术故障。

预期产物：smoke.json、完整L/V双表征FP32缓存及角色metadata、features_complete.json、几何results/complete.json、pipeline/controller完成状态与成本。缓存/逐包数据不提交，公开只提供完整统计与状态。原三seed实际目标clean/raw六景/同DURA六景仍未达成。
