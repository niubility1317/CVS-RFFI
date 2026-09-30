# AJLR 训练诊断请求传输修复

本修复只改变[旧 AJLR 收集器](../tools/collect_d92_anchor_joint_training_diagnostics.py)的 `remote_operation` 请求传输，不改变 Arrays、训练诊断统计、数学、报告、配置或活跃远端。回归检查写在[既有合成测试](../tests/test_collect_d92_anchor_joint_training_diagnostics.py)。

主 Agent 提供的操作证据为：完整 snapshot 有 2,673,911,234 B、3240 个 stage。旧实现将整个 JSON 用 `repr` 嵌入 Python 字符串字面量；远端编译报 `OverflowError: string to parse is too long`，尚未进入 extract，没有输出。这是请求编译层问题，上述字节数不是性能指标；本任务未读取该真实 snapshot 或结果。

请求现在按以下顺序传输：

```python
json_text = json.dumps(request, ensure_ascii=False, allow_nan=False)
encoded = base64.b64encode(gzip.compress(json_text.encode('utf-8'), mtime=0)).decode('ascii')
```

远端初始化只包含 stdlib import 和压缩后的 base64 字面量：

```python
SNAPSHOT_REQUEST = json.loads(gzip.decompress(base64.b64decode(encoded)).decode('utf-8'))
```

编译器处理压缩字面量，运行时恢复完整 UTF-8 JSON。字段、层级、Unicode、转义、有限数值和空值都保留；不删减内容，不按需跳过，不使用 eval/pickle。`allow_nan=False` 继续在调用 subprocess 前拒绝 NaN 和正负无穷。坏 gzip 在 stdlib 解压阶段拒绝。

gzip 的 `mtime=0` 使传输头不引入当前时间；不增加哈希、receipt、authority 链或权限流程。SSH stdin/stdout 形态、远端命令、超时和本机独占输出流程保持原样。无 request 的 snapshot 操作仍发送原脚本。

新增四项纯合成回归：完整深层结构的 stdlib 初始化往返、超过 1 MB 的重复 JSON 使用压缩字面量、坏压缩拒绝、非有限请求在 subprocess 前失败。测试只执行捕获源码的初始化 AST，不执行 collector 主体或真实 SSH，不读取真实 snapshot、NPZ、summary、outer/query。

开发 Agent 仅检查 AST、UTF-8 和链接。主 Agent 统一串行运行 `python -X utf8 -m pytest -q tests/test_collect_d92_anchor_joint_training_diagnostics.py`，再管理部署和真实重试。合成验证不表示真实提取已经成功。

DIAGNOSTIC_TRANSPORT_READY/VERIFIED：旧AJLR只读提取源码literal超过Python解析长度，完整snapshot和分析结果保留，仅改为完整JSON→gzip→base64数据载荷；远端仍stdlib解码，没有字段删减、eval/pickle、新拟合/求解或数学变更。14项相关合成检查通过（5.97s），4项新增、10项既有回归。root在Git交付后仅重试extract，完整snapshot不重采，目标ACTIVE。

实际snapshot为2,673,911,234B；2,650,592,222B是写盘结束前的观察值，完成后已更正。原失败handle51314已退出，未产生diagnostics目录。真实压缩字节、耗时及重试结果尚未测量，不用合成结果代替。
