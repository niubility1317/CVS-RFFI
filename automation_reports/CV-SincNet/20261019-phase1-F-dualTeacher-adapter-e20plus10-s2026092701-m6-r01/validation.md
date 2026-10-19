# 聚焦验证与唯一审查

本地 18 个 Python 文件编译 PASS，聚合检查 PASS：78 个协议负例，六行合成 hook、真实 DSConvBlock1d/ResidualPhysicalHead 类、严格合成父 206→adapter6、新 CP reload、4768 参数冻结范围、默认头排除、梯度可达、全128 CE/teacherWrong mask、原 U 参数保留。未读实际 CP/IQ，真实训练更新为 0。实际父 CP no-query smoke 是 launcher 首步，PASS 立即继续。

一次独立整包 P0/P1 审查发现两个 P1：继承的 a1_scratch_only 与合法续训冲突；取消它后旧全机制终态门槛会阻断 source completion。已显式关闭 scratch-only，并按既有 continuation 范围只将旧 mechanism_gates_required 设 False、委托原终态函数，其余参数不变，finally 恢复。两项定向复审 PASS，其余已审接口未发现 P0/P1。

REJECTED_EXTRA_GATE：旧全机制晋级要求包含本轮未启用的其他方法，不适用于当前固定适配阶段，不为满足它启用额外机制。新源资格由已登记 S3−S2 与 clean 容差判定。此项是阶段适配，不放宽数据/来源、数值、预算、冻结参数或 prediction legality 检查。

本地 Torch 环境存在已知 NumPy2 ABI 警告；检查使用纯 Torch，实际退出码 0。远端使用既有环境，不安装或修改依赖。
