# AFFINE 诊断 snapshot 传输修复

日期：2026-10-01。状态：`SYNTHETIC_VERIFIED / REAL_EXTRACTION_NOT_RUN`。

Affine collector 的 `remote_operation` 仍将完整 JSON 经 `repr` 嵌入 Python 字符串字面量。主任务已报告同规模 AJLR snapshot 在该方式下遇到 parser `OverflowError`；本次未读取该真实 snapshot 或运行证据，仅对照两个 collector 的传输实现。

[Affine collector](../tools/collect_d92_affine_joint_training_diagnostics.py)现精确复用 AJLR 的传输步骤：完整请求以 `ensure_ascii=False, allow_nan=False` 序列化为 JSON，再编码为 UTF-8、使用 `gzip.compress(..., mtime=0)` 压缩并转为 ASCII base64。远端脚本仅通过标准库 `base64 → gzip → UTF-8 → json.loads` 恢复完整数据；不执行请求内容，不删字段，不改变诊断计算、状态、schema、SSH 参数、操作或读取范围。无请求的 snapshot 阶段保持原样。

[窄合成回归](../tests/test_collect_d92_affine_joint_training_diagnostics.py)沿用 AJLR 测试方式：截获 mock subprocess 的脚本，只执行前两条标准库解码语句，核对 Unicode、转义、空值、有限数和 64 层嵌套完整往返；核对 gzip 时间戳为零；覆盖超过 1 MB 的重复 JSON 压缩及完整恢复、损坏 gzip 拒绝，以及 NaN/正负无穷在 subprocess 调用前拒绝。原有命令和无远端写入检查保留。

子任务没有执行测试、Conda、SSH、Git、launch 或发布，也没有读取真实 snapshot、诊断、评分、全局索引或交接。主任务随后在项目 `ssr-gpu` 环境串行运行该文件的全部测试：**16 passed in 8.97s**（12 个既有测试、4 个新增传输测试）。完整输出保存在 `E:/type10-7/.codex_tmp/pytest_utf8_1790815639763930300.stdout` 与同前缀 `.stderr`。这些是合成与命令边界检查，不是超过 2 GB 的真实 Affine snapshot 提取。

仅修改 collector 的请求传输分支、对应合成测试和本文，未改正在运行的 Affine release。实际 Affine snapshot/extract 尚未启动；独立分析继续使用源版本 `1acc83a584ace40f294a433ace80629472ee2525`。本文不宣称真实提取成功，也不据此扩大方法的科学结论。
