# D26 compact-diag support-only追溯表

日期：2026-07-18

状态：核心、runner、launcher和55项相关回归完成，待N607 90行support-only执行

边界：复用D25已验证的单LEO_weak物理IQ support；`z160/FFT96/RF32`是同一接收IQ的一条288D拼接表征，不增加物理样本、LEO状态、support行或K。query始终不可达。

|ID|需求|实现|状态|验证|
|---|---|---|---|---|
|D26-01|压缩B3有效结构|shared 288D diagonal+逐类cosine weight，Stage2-B全批次15步|verified|核心focused测试|
|D26-02|轻量Stage2-C|new suffix仅0/10/15步，总步数≤30|verified|config/atomic append测试|
|D26-03|旧头冻结|旧weights、shared diagonal、old raw score prefix逐字节冻结|verified|核心与真实fold测试|
|D26-04|注册遗忘保护|1标量new-group bias仅由support LOO和注册后bias0旧类安全门选择；runner另做注册前后逐类/floor终门|verified|bias audit、fold、full-K10测试|
|D26-05|K=1|bias固定0，不构造伪LOO，不宣称无遗忘|verified|K1测试|
|D26-06|逐样本全类决策|FP32全注册类score+单次argmax，无role/quota/global assignment|verified|API/CLI签名测试|
|D26-07|query/clean/source隔离|fit API和runner无query/truth/scorer/source/clean入口|verified|源面与签名测试|
|D26-08|资源上限|峰值参数≤2,016、总epoch/step≤30、状态≤256KB、无dense query图|verified|resource audit测试|
|D26-09|NumPy2/Torch2兼容|小support训练使用list bridge，无`torch.from_numpy`/`.numpy()`|verified|源码回归与N607前置修复|
|D26-10|矩阵与选择|Z0/B3/C0+D26A/B/C，6×3×5=90；C0硬门，B3仅性能参考|verified|candidate lock/selector测试|
|D26-11|full-K10终门|任一场景旧support逐类或floor退化即撤销D26并回退C0|verified|构造性失败回归|
|D26-12|证据闭包|锁runner、D26/D25/D24/CIAF/D19和实际FFT/RF operator SHA；D26 Git提交独立于Phase1模型提交|verified|独立review后修复与lock测试|

## 验证

- 核心提交：`0a9fbb20e58f1f77c7f9ccc350cc826351ce0d79`。
- runner提交：`e4681cee`。
- `python -m pytest -q tests\test_run_d25_d26_support_only_compact_diag.py tests\test_stage2_multimodal_compact_diag.py tests\test_run_d25_c3_support_only_diag_floor.py tests\test_run_d25_support_only_concat.py tests\test_stage2_multimodal_diag_floor_adapter.py tests\test_stage2_multimodal_concat_fusion.py`：55项PASS。
- `py_compile`、launcher `bash -n`、`git diff --check`：PASS。
- 独立review：未发现协议/算法高严重度阻断；发现并修复D26 Git归因与实际FFT96/RF32 operator闭包遗漏。
- 追溯状态：12/12项verified；实际性能与N607 artifact仍待执行，不在本地实现PASS含义内。
