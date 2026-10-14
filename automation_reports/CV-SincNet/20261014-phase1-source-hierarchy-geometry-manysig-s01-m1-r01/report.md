# 连续层级时序签名源域诊断预登记

依据用户附件报告路线二，检查真实源域特征上的4→2→1连续拼接。单行 20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01，固定已完成 moment/area 首种子 SIG_TIME final50/11100，源执行512c10a267b5e654cfd2c983ab1debdeebc7f626。来源为相同物理契约从零旧 batch_neighbor 门控＋EMA＋默认cosine联合训练，额外分类器拟合0；加载前复用 source_proof 核对来源。选择预定义最低seed2026092701，不依据目标分数选父权重。

L_s6300 中每6TX×5RX×3day单元按physical ID取前8，共720包；U_s56700 IQ/真值均不访问，V27000既有cache可materialize后discard，无V forward。split392005、已验证源contract、NoExplicitCFO及内部逐包补偿保持。只读全部eval/frozen模型，0新训练轮、0 optimizer/head更新，不拟合辅助诊断头、不读取目标。

捕获真实t2的32位置和实际final6D Conv+tanh；整段只加入一个global0至1时间坐标。四段各8位置，显式保留7→8、15→16、23→24三个边界增量；用二阶skew area的有序交叉项组合成两个相邻半段和完整路径，与direct16/direct32比较。独立对照为缺边界、四段时间分别重置、仅求和省略有序cross-area。公式依据 [Chevyrev与Kormilitzin原论文](https://arxiv.org/pdf/1603.03788) 的Chen identity，并用合成路径核验。

实际记录3错误对照×5描述子组（完整28、feature increment6、time increment1、feature area15、time area6）×9分层（overall、5RX、3day），共135项。保存私有source_geometry.pt后独立重读复算；源数组、IQ、weights和原日志不上传Git。允许float64组合误差≤1e-11、native float32≤5e-5；参数/buffers/模式不变且无梯度累积。合成数学与真实网络零IQ本地检查已通过；一次独立直接P0/P1审查PASS，无发现。

实现/config Git提交 13728d26334a3523c4225f24a274f410bd462076。命令 python -u -m experiments.cvs_source_hierarchy.probe；普通N607既有CVS-RFFI CPU，CUDA_VISIBLE_DEVICES为空，Torch/OMP/MKL/OpenBLAS2threads。CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_hierarchy_20261014_r01；不可变输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01；owner codex/root/source-hierarchy-20261014。发布仅一次archive本地/远端SHA比较、remote compile和真实源CP两零IQ无查询smoke，PASS直接继续。随后核对PID/CWD/cmdline/log增长及实际完成产物。不占GPU、不改其他健康任务。

停止规则：角色/来源/非有限/形状/模型状态变化或组合误差超限时保留产物并停止，无自动重试、换权重或重采样。预期results.json、resolved_config.json、source_geometry.pt、completion.json以及release preflight/submit/log。登记后逐项核实结果、Git推送并独立比较remoteOID。

边界：本诊断只测描述子几何。局部签名遗漏的跨窗信息不证明整个已有骨干都无法表达该信息；源域非零差异不证明层级训练有效、性能提高或新类迁移。实际网络和固定预算对照另行发布。

实际已完成 ANALYZED／VERIFIED。连续组合 float64 最大误差8.882e-16、float32 7.153e-7；720源包135统计、保存数组复算通过、0训练/分类头更新。详见 final_analysis.md/results.json。
