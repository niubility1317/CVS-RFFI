# ADV3B02-CSIL/MoPC-HR CVS接口优化实验v3

- 实验ID：`adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3`
- 日期：2026-07-24
- 状态：`LOCAL_VERIFIED / RELEASE_READY`
- 操作者：Codex主代理；N607唯一运行代理：`no_leo_n607_release`
- 目标：保持已冻结CSIL/MoPC-HR方法、LEO数据、query边界与400-cell矩阵不变，修复v2 smoke发现的容量契约和machine status链后完成正式实验。

## 历史技术停止

- v1：仓库外CWD导入失败，0 cell/0 prediction/0 score，`NO_PERFORMANCE_RESULT`。
- v2：CSIL base capacity drift；MoPC partial prediction后status drift；smoke receipt=0、authority plan=0、full launch=0，`NO_PERFORMANCE_RESULT`。
- v1/v2均只读保留，不覆盖、不续跑，所有run-owned进程和SSH连接已清理。

## 本轮最小修复

1. CSIL v2 adapter在builder和runner强制`required_total_capacity=26`；MoPC v2 adapter强制31。
2. runner按method统一校验并写入三类status：strict baseline、interface adapter、ordered-arrival diagnostic。
3. status规则覆盖predictor返回、existing/new cell receipt和smoke artifact authority验证。
4. 未修改`adv3b02_official_repo_ci.py`、predictor训练/判决、LEO IQ入口、query打开顺序、loss或优化器。
5. 本地定向+计划+相邻集成：`54 passed`；`git diff --check`通过。
6. 独立复审：`P0=0,P1=0 / APPROVE`。

## 冻结矩阵与容量

| 方法 | 新类数 | K | receiver×seed | capacity | cells | 场景行 |
|---|---:|---|---:|---:|---:|---:|
| `csil_official_repo_corefix_cvs_adapter` | 1、3 | 1、5、10、20 | 5×5 | 26 | 200 | 600 |
| `mopc_hr_official_repo_cvs_adapter` | 25 | 1、5、10、20 | 5×5 | 31 | 100 | 300 |
| `mopc_hr_official_repo_sequential5_cvs_adapter` | 25 | 1、5、10、20 | 5×5 | 31 | 100 | 300 |
| 合计 | — | — | — | — | 400 | 1200 |

## N607预注册

- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3`
- log root：`logs/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v3`
- base26：只读复用v7，SHA256=`635becd7db2d8041a669cb0ef922429c42ba389846f2252c7cfe4e0f3510a07e`
- base31：只读复用v7，SHA256=`306c8dfc767bad93f78f24b675b1c20058629eda8f3153bdf98d03ab6ae26202`
- 必须显式以26/31构建两份pre-smoke plan；smoke PASS后才生成authority plan和完整矩阵。
- 预期每cell闭环prediction、score、predictor/enrollment/cell receipt、loss trace和3条formal rows。

## 健康、停止和结论边界

- P0或两个不同row在prediction前同一确定性异常指纹时，仅停止v3精确process tree并保留制品；不得因性能低停止。
- 新类support/query只使用固定LEO弱信道IQ；query在模型状态锁定后才打开，训练行数0。
- CSIL差异同时包含官方old-old fingerprint mask修复和small-K接口适配，禁止单因归因。
- MoPC single-stage new25是instrumentation parity；sequential5仅为`ORDERED_ARRIVAL_DIAGNOSTIC`，不与同时注册等价。
- 若v3技术失败，不自动创建v4。

## 结果

待运行后填写完整制品闭环及同row old-before/after、seen-new、H、forgetting、min-old、逐类旧类准确率与解释。
