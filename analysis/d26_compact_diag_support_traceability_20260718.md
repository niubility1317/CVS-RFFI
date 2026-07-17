# D26 compact-diag support-only追溯表

日期：2026-07-18

状态：v1/v2各90行support-only均已完成；v2证明全局标量bias无法同时保护old与new，D26不晋级并转入D27逐新类安全bias

边界：复用D25已验证的单LEO_weak物理IQ support；`z160/FFT96/RF32`是同一接收IQ的一条288D拼接表征，不增加物理样本、LEO状态、support行或K。query始终不可达。

|ID|需求|实现|状态|验证|
|---|---|---|---|---|
|D26-01|压缩B3有效结构|shared 288D diagonal+逐类cosine weight，Stage2-B全批次15步|verified|核心focused测试|
|D26-02|轻量Stage2-C|new suffix仅0/10/15步，总步数≤30|verified|config/atomic append测试|
|D26-03|旧头冻结|旧weights、shared diagonal、old raw score prefix逐字节冻结|verified|核心与真实fold测试|
|D26-04|注册遗忘保护|v2的1标量new-group bias仅由support选择；以Stage2-B old-only逐类准确率和所有正确旧support行为硬门；runner另做held/full-K10终门|verified|strict bias audit、fold、full-K10测试|
|D26-05|K=1|不构造伪LOO，在旧类保护可行bias中选择最接近0者；无安全bias即fail closed|verified|K1安全选择与无解失败测试|
|D26-06|逐样本全类决策|FP32全注册类score+单次argmax，无role/quota/global assignment|verified|API/CLI签名测试|
|D26-07|query/clean/source隔离|fit API和runner无query/truth/scorer/source/clean入口|verified|源面与签名测试|
|D26-08|资源上限|峰值参数≤2,016、总epoch/step≤30、状态≤256KB、无dense query图|verified|resource audit测试|
|D26-09|NumPy2/Torch2兼容|小support训练使用list bridge，无`torch.from_numpy`/`.numpy()`|verified|源码回归与N607前置修复|
|D26-10|矩阵与选择|Z0/B3/C0+D26A/B/C，6×3×5=90；C0硬门，B3仅性能参考|verified|candidate lock/selector测试|
|D26-11|full-K10终门|任一场景旧support逐类或floor退化即撤销D26并回退C0|verified|构造性失败回归|
|D26-12|证据闭包|锁runner、D26/D25/D24/CIAF/D19和实际FFT/RF operator SHA；D26 Git提交独立于Phase1模型提交|verified|独立review后修复与lock测试|

## 验证

- v1核心提交：`0a9fbb20e58f1f77c7f9ccc350cc826351ce0d79`；v2核心提交：`55d69d0efa3d5ef4d43e9702058d15c20e7f95e5`。
- v1 runner提交：`e4681cee`；v2 runner提交：`bbaf5958`。
- v1 N607 90/90行完成、query未打开；D26注册前old达到80.00%，但注册后仅9.44%–23.89%，确认故障集中在bias0注册竞争，v1不晋级。
- `python -m pytest -q tests\test_stage2_multimodal_compact_diag.py tests\test_run_d25_d26_support_only_compact_diag.py tests\test_stage2_multimodal_concat_fusion.py tests\test_run_d25_support_only_concat.py tests\test_stage2_multimodal_diag_floor_adapter.py tests\test_run_d25_c3_support_only_diag_floor.py`：58项PASS。
- `py_compile`、launcher `bash -n`、`git diff --check`：PASS。
- 独立review：未发现协议/算法高严重度阻断；发现并修复D26 Git归因与实际FFT96/RF32 operator闭包遗漏。
- v2 N607 90/90行完成、query未打开、artifact哈希闭环。D26-B从v1的old/new=23.89%/70.67%变为79.44%/8.00%，旧类保护有效但新类被系统性压制；C0继续为回退，D26不晋级。
- 完整loss无非有限值；D26-B Stage2-B loss从0.554833降至0.066382，Stage2-C从0.727749降至0.256986，排除训练不收敛。
- 三轮回顾已复核D25/C3/D26报告、完整日志、活动目标、项目协议和conversation index。下一路线D27用每新类独立安全bias上界及support LOO坐标选择，保持旧raw score、单IQ、query不可达和逐样本全类argmax。
- 追溯状态：12/12项实现与v2执行证据verified；D26为development support-only负筛选，不构成正式query性能或部署成功。
