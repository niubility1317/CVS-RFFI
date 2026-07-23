# ADV3B02官方仓库语义CSIL/MoPC-HR对比实验

本文件是Git承载镜像。完整预注册、方法锁、数据矩阵、本地验证、风险和后续结果记录与根目录报告保持一致。

- 实验ID：`adv3b02_officialrepo_csil_mopc_20260723_v1`
- 日期：2026-07-23
- 状态：`LOCAL_IMPLEMENTATION_REVIEW`
- 执行策略：`OFFICIAL_CODE_EXECUTION_SEMANTICS`
- 正式base：8400条source样本
- 正式矩阵：2方法×5receiver×5seed×4K×4新类数=800cell，每cell 3场景
- 正式LEO：新类support/query均叠加固定LEO弱信道
- matched无LEO：仅作`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`
- 详细方法锁：`analysis/official_repo_execution_lock_csil_mopc_hr_20260723.md`

本地验证：

| 检查 | 结果 |
|---|---|
| Python编译 | PASS |
| 官方数值fixture | 8/8 PASS |
| 相关runner/plan回归 | 25/25 PASS |
| `git diff --check` | PASS |

独立base复核曾发现首版误吸收`v4_4`的指纹正交深度实验项。该版本未发布；
修正后采用`v4_3`标准`trainNetwork`交叉熵、默认单次shuffle、保留尾批和
L2=0.01，并新增base尾批fixture。

待N607 smoke完成后补充精确hash、远端路径、命令、PID、GPU、完整结果表及结论。
