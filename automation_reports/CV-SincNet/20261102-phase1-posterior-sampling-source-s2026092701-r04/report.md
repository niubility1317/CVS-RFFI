# Phase1 后验采样源域预检 r04

r03 真实预检已通过后验计算对照，随后在已有数组一致性比较报错：服务器中 PyTorch `.numpy()` 返回的 ndarray 与当前 NumPy 的 ndarray 无法混合使用。针对相同环境问题已验证 `np.from_dlpack` 产生可与当前 NumPy 正常比较的数组；本次统一替换采样输出、预测保存和独立审计三个边界的 tensor 导出，原数据、随机种子、模型、六成分、选择器、样本预算均保持不变。r03 不可覆盖，未形成固定预测。

代码和配置以本报告所在 Git commit 为准。N607 普通用户进程，工作目录为 CV-SincNet 项目根，CPU 执行、禁用 CUDA、每进程 2 线程，负载超过 72 时等待。入口为 release 内的 `experiments/phase1_posterior_sampling/sample_source_controller.py --commit <本报告所在 commit>`。

先对 28 条固定源记录进行真实模型预检；通过后固定 672 条源预测，再连接合法 L_s 真值评分并独立复核。输出和控制器均使用 `20261102-phase1-posterior-sampling-*-s2026092701-r04` 独立目录；不访问 U_s 身份、V 或目标测试真值，不进行目标地面性能主张。输出包括预检 result、固定预测、源评分和独立复核。系统性技术错误停止并保留原产物；低性能不是停止理由。

本机检查模拟 `.numpy()` 不可用和旧版 DLPack 不支持布尔类型；真实服务器确认先转换为 uint8、导出后恢复 bool 可保持数值与类型。完整真实模型仍由源预检验证。
