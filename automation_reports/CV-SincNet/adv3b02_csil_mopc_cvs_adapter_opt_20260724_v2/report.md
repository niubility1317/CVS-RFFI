# ADV3B02-CSIL/MoPC-HR CVS接口优化实验v2

- 实验ID：`adv3b02_csil_mopc_cvs_adapter_opt_20260724_v2`
- 日期：2026-07-24
- 状态：`LOCAL_VERIFIED / RELEASE_READY`
- 操作者：Codex主代理；N607唯一运行代理：`no_leo_n607_release`
- 目标：在不改变v1已冻结方法、参数、数据和矩阵的前提下，修复plan runner从仓库外CWD启动时的模块导入错误，并完成LEO正式优化实验。
- 前序run：v1因两个不同smoke在prediction前同指纹`ModuleNotFoundError: paper_reproduction`而合规停止，0 cell/0 prediction/0 score，属于`NO_PERFORMANCE_RESULT`。

## 假设与比较

- 技术假设：在导入共享matrix validator前把仓库根目录加入`sys.path`，可消除启动错误，且不改变CSIL/MoPC-HR数值路径。
- 性能比较：CSIL v2 adapter对比v7历史实现偏差诊断；MoPC single-stage new25作instrumentation parity，sequential5仅作`ORDERED_ARRIVAL_DIAGNOSTIC`。
- 不允许把CSIL差异单因归于small-K batch；其同时包含官方old-old fingerprint mask纠正。

## 冻结矩阵

| 方法 | 新类数 | K | receiver×seed | cells | 场景行 |
|---|---:|---|---:|---:|---:|
| `csil_official_repo_corefix_cvs_adapter` | 1、3 | 1、5、10、20 | 5×5 | 200 | 600 |
| `mopc_hr_official_repo_cvs_adapter` | 25 | 1、5、10、20 | 5×5 | 100 | 300 |
| `mopc_hr_official_repo_sequential5_cvs_adapter` | 25 | 1、5、10、20 | 5×5 | 100 | 300 |
| 合计 | — | — | — | 400 | 1200 |

## 版本与本地验证

- 方法实现与配置沿用v1提交`76bbc9ca2286f80010bcc41da32707c3b50395d9`，不得修改。
- 本轮唯一代码修复：`paper_reproduction/scripts/run_adv3b02_paper_full_ci_plan.py`在导入项目包前注入解析后的`REPO_ROOT`。
- 新增回归测试：从仓库外临时CWD执行runner `--help`必须成功。
- 本地定向测试：`47 passed`；定向+计划+相邻集成：`54 passed`。
- 手工外部CWD验证：从`E:\type10-7`执行runner `--help`成功。
- 独立修复复审：`P0=0,P1=0 / APPROVE`；确认仅修启动/import边界，方法、矩阵、LEO/query和训练参数均未改变。
- import修复提交：`da3e2ca186d06b6308fa2ec5eb31890c1ac145cc`。
- 修复后runner SHA256：`0753c760a54eb5c59af4a524c62e3a1ac9abcdd50bd22580ab2844283f7f23bb`。

## N607预注册

- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v2`
- log root：`logs/adv3b02_csil_mopc_cvs_adapter_opt_20260724_v2`
- 复用只读base26：`runs/adv3b02_official_newcount_scale_20260724_v7/base26/official_repo_base_state.pt`
- 复用只读base31：`runs/adv3b02_official_newcount_scale_20260724_v7/base31/official_repo_base_state.pt`
- 先执行CSIL/MoPC各自真实checkpoint smoke；通过后才生成authority plan并启动完整矩阵。
- 预期每cell产生prediction、score、predictor receipt、enrollment receipt、loss trace和row exit闭环。
- GPU最多每卡两个训练进程；启动时补录实际PID、cmdline、GPU和计划哈希。

## 健康与停止规则

- P0协议/安全错误，或两个不同row在prediction前出现同一确定性异常指纹时，停止该精确run-owned进程树并保留所有制品。
- 不得因accuracy、H、forgetting或其他性能低而停止。
- query必须在全部模型状态锁定后打开，训练行数为0；新类support/query必须使用固定LEO弱信道IQ。
- sequential5必须封存固定类序，stage2起prototype输入哈希匹配上一阶段输出，最终query仍用全注册类classifier logits。
- 技术失败不得覆盖或续跑v2；修复后必须使用新的run ID。

## 结果

待运行后填写同row old-before/after、seen-new、H、forgetting、min-old、逐类旧类准确率、coverage、异常与结论。
