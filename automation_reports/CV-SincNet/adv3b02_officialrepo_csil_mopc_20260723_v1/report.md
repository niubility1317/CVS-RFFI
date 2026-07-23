# ADV3B02官方仓库语义CSIL/MoPC-HR对比实验

本文件是Git承载镜像。完整预注册、方法锁、数据矩阵、本地验证、风险和后续结果记录与根目录报告保持一致。

- 实验ID：`adv3b02_officialrepo_csil_mopc_20260723_v1`
- 日期：2026-07-23
- 状态：`LOCAL_VERIFIED / READY_FOR_N607_SMOKE`
- 执行策略：`OFFICIAL_CODE_EXECUTION_SEMANTICS`
- 正式base池：8400条；MoPC-HR全量训练，CSIL按官方下标切5879条train/2521条互斥Fisher validation
- 正式矩阵：2方法×5receiver×5seed×4K×4新类数=800cell，每cell 3场景
- 正式LEO：新类support/query均叠加固定LEO弱信道
- matched无LEO：仅作`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`
- base编码器：`BASE_MODEL_INITIALIZATION_ADAPTATION`，预训练ADV3B02替换原编码器后按各官方trainer在8400条source上训练
- CSIL新10/新20：`CLASS_CARDINALITY_INITIALIZATION_ADAPTATION`，因为新增类数超过官方扩展前旧输出宽度6
- MoPC-HR新2/新20：`CLASS_SCHEDULE_ADAPTATION`，官方main展示的间隔为25/10/5/3
- 详细方法锁：`analysis/official_repo_execution_lock_csil_mopc_hr_20260723.md`

本地验证：

| 检查 | 结果 |
|---|---|
| Python编译 | PASS |
| 官方数值fixture | 8/8 PASS |
| 相关runner/plan回归 | 26/26 PASS |
| `git diff --check` | PASS |

发布锁：

- Git代码commit：`c712a104`
- 旧适配无LEO诊断封存commit：`47df97ae`
- 官方方法锁SHA256：`527deffd2716d2b40c34e76fc31da5fcd44b1fd920c2391c7077b2cb9a2b3ebf`
- 官方语义内核SHA256：`db90b1506fe02a1274a5bf79aea6d18bf034eec118858a612a36a819b7fd5b77`
- base builder SHA256：`c110ad16d2b65c28e8ad43f05e1ce9d9bb3bcc4e1233260ec39af65748e30059`
- 正式LEO predictor SHA256：`55d94bd81f1d1f32d725983bf9052e33382a6a572a3e3bd3891a7deb12ba1846`
- matched无LEO runner SHA256：`f66e00c408428cd488e45c7844d47fda4e1cb6cf09feb082a20f0f4acf8ba499`
- plan builder SHA256：`af7669daaba51b5bacf5575c7fd1ddefbdd6b3499a409bbcbe7db3564711bb06`
- plan runner SHA256：`19e6a6f6560ff35727b1d2b36d3cd6fce7036df888f279ed286a75fdf9d3b128`
- 官方fixture SHA256：`d74fb77ade52b581fc12f7b310f669310d30ef53c7c07127ee89361330be085f`

独立base复核曾发现首版误吸收`v4_4`的指纹正交深度实验项。该版本未发布；
修正后采用`v4_3`标准`trainNetwork`交叉熵、默认单次shuffle、保留尾批和
L2=0.01，并新增base尾批fixture。

待N607 smoke完成后补充精确hash、远端路径、命令、PID、GPU、完整结果表及结论。
