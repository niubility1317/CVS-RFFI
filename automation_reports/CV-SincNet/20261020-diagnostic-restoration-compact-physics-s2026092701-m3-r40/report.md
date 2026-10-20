# R40：冻结压缩恢复器的源域接收链诊断

状态：LOCAL_VERIFIED，尚未发布。14项聚焦检查、CLI帮助、一次P0/P1通过。方法commit `eebad5b7aaff7181bcdd89e1751f29dcaa471592`。


状态：LOCAL_IMPLEMENTATION，尚未启动。此项比较原 source-only 选定的 R17 与 R38 两档完整链压缩模型，不增加新网络、训练或正式测试。原七场景正式结果保持原产物。本次不选择候选、不改父权重，也不把已暴露的测试成绩用于设计。

问题来自源域证据：R39 的低仰角 urban 解析同 TX 关联在两版压缩模型中均弱于原 R17。该关联本身不证明细节真实性，也不能说明哪一部分产生偏差。R40 进一步拆分粗恢复、复合信道作用和噪声估计，检查原 R17 冻结上下文与 R38 联合优化上下文的差异。

固定输入为原 R07 `cache/validation` 的 1800 个源域 V 物理样本及全部七个视图；复用原始 `validation_actions` 已验证仿真算子和绝对实轴噪声方差。V 只做冻结诊断，0 拟合/训练更新。原 R17 为 E20/4940 的 `unroll_unshared`；R38 为固定 E20/4940 的 `dense_three_quarter` 和 `dense_half`，完整原 L6300/U56700、同数据契约、TX/F/V/target 训练读取均为 0。原 R02/R07/R17 继承按现有记录核对，不继承其他 owner 的 F 蒸馏模型。

CPU 两线程顺序提取三份 checkpoint 的 coarse IQ、复合算子组成和原生噪声方差，共 37800 次上下文估计；不执行 prior、F 或默认分类头。所有 21 个估计单元固定且验证完整后，另起 CPU scorer 才读取 clean/已知仿真算子。0 TX 标签、F 权重/输出、target、额外恢复 IQ 预测、GPU worker 或新校准。推理仍只读取当前 received IQ。launcher 首步使用实际三份最终 checkpoint 和两条 L_s IQ 做 no-query smoke，通过后直接继续。

算子误差按复合算子在配对 clean 和四个共同复高斯探针上的相对输出误差计算；随机种子为 2026102040。不用单独 kernel 参数 MSE，因为 FIR 与时变增益的不同分解可以表示相同作用。NumPy scorer 使用 causal reflect FIR，局部检查与 Torch 的真实 Action 对齐。粗恢复 NMSE 的复功率分母下限为 2e-4，等于此前每实轴功率下限 1e-4；探针算子分母下限为 1e-4。

噪声报告绝对每实坐标方差，不除以 prior 的 `SCALE**2`。保留 log(预测/原生) 及平方、已知算子残差和估计算子残差的实轴 MSE、残差/方差比。clean 的原生方差 1e-8 只是历史仿真缓存的数值下限，clean 实际未加噪，不能要求其残差/方差比为 1。对每个模型、每场景报告 overall、RX、day 的均值、中位数和 P90；0 TX 标签，不报告本次逐 TX 统计。RX/day 来自原源域物理索引。

只解释已知模拟接收链中的偏差，不证明真实在轨信道、唯一物理因子或每个细节属于输入 TX。解析关联、物理保真和冻结 F 识别率分别报告。此项没有源域收益门槛、候选排序或目标反馈，不自动晋级任何模型。停止条件为错误契约/输入、非有限或缺失估计、输出碰撞及技术失败；保留全部产物，不自动重试，不干预健康任务。

代码：`experiments/pamf_compact_physics_probe`。精确命令、commit、环境/CWD、独立输出与产物位置在 run 的 `experiment.json` 登记。无需 target 接口；正式七场景只复用 R17/R38 已完成的同 row 结果，不重复测试。


精确执行信息见[登记](experiment.json)。
