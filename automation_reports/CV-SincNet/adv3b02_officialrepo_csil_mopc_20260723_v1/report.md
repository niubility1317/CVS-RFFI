# ADV3B02官方仓库语义CSIL/MoPC-HR对比实验

本文件是Git承载镜像。完整预注册、方法锁、数据矩阵、本地验证、风险和后续结果记录与根目录报告保持一致。

- 实验ID：`adv3b02_officialrepo_csil_mopc_20260723_v1`
- 日期：2026-07-23
- 状态：`ARTIFACTS_COMPLETE / ANALYZED`
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
| 相关runner/plan回归 | 27/27 PASS |
| `git diff --check` | PASS |

发布锁：

- Git代码commit：`7986887cba62446954ea2a2ec33e56e6a31b8d56`
- 旧适配无LEO诊断封存commit：`47df97ae`
- 官方方法锁SHA256：`813a74eb233a18b662dfbcafd69b189af9765ca520e05aebf3789a13b600eb8f`
- 官方语义内核SHA256：`25cd5ee8a2031d3e80bb82d76b75aebee17293190a5c59eb92d1fa81b5716c67`
- base builder SHA256：`c110ad16d2b65c28e8ad43f05e1ce9d9bb3bcc4e1233260ec39af65748e30059`
- 正式LEO predictor SHA256：`55d94bd81f1d1f32d725983bf9052e33382a6a572a3e3bd3891a7deb12ba1846`
- matched无LEO runner SHA256：`f66e00c408428cd488e45c7844d47fda4e1cb6cf09feb082a20f0f4acf8ba499`
- plan builder SHA256：`af7669daaba51b5bacf5575c7fd1ddefbdd6b3499a409bbcbe7db3564711bb06`
- plan runner SHA256：`19e6a6f6560ff35727b1d2b36d3cd6fce7036df888f279ed286a75fdf9d3b128`
- 官方fixture SHA256：`d5cad8fe44fd3c0ce71f73e4f4db7383dd79dfcb0daaf15368bf93405b5d6c2c`

独立base复核曾发现首版误吸收`v4_4`的指纹正交深度实验项。该版本未发布；
修正后采用`v4_3`标准`trainNetwork`交叉熵、默认单次shuffle、保留尾批和
L2=0.01，并新增base尾批fixture。

低K严格保留官方固定batch与drop-last行为；不足一个完整batch时允许0个优化步，
并记录`official_zero_step_due_to_drop_last=true`，不再使用缩小batch的执行适配。

## N607完整矩阵闭环

- base state：8400条source；CSIL train/Fisher=`5879/2521`且互斥；
  MoPC-HR使用8400全量。
- base SHA256：`19a0c75d8ca71be3a1d6f4d5d7c1ce595d579c4b7f5cdea8c32fc2985250d346`
- 正式授权plan SHA256：`4c5a99c42794d0bd876898cc5cd750fb414ed1e070cad09e933d571f96b6f681`
- 正式LEO：8/8分片、800/800cell、2400/2400场景row、800/800prediction，PASS。
- matched无LEO：8/8分片、800/800cell、2400/2400场景row、800/800prediction，
  永久标记`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`。
- 同row配对：2400/2400；资源记录4800；`failures=[]`。
- query训练步数均为0，prediction先原子封存再评分，loss全部有限。
- 两条件官方零步场景资源各825；`small_k_execution_adaptation=true`为0。
- 最终无GPU/run-owned进程；本地`ssh.exe=0`，N607与bridge的TCP22连接为0。

## 全矩阵总体结果

每行聚合1200条同条件、同方法场景row。

| 条件 | 方法 | old_before | old_after | seen_new | H_old_new | forgetting | min_old |
|---|---|---:|---:|---:|---:|---:|---:|
| 正式LEO | CSIL | 42.83% | 23.17% | 8.65% | 1.18% | 19.66% | 0.82% |
| 无LEO | CSIL | 42.83% | 23.78% | 12.15% | 1.67% | 19.06% | 0.80% |
| 正式LEO | MoPC-HR | 45.32% | 22.14% | 26.61% | 10.85% | 23.19% | 3.89% |
| 无LEO | MoPC-HR | 45.32% | 21.45% | 52.50% | 12.98% | 24.11% | 3.38% |

matched无LEO减正式LEO：

| 方法 | Δold_after | Δseen_new | ΔH | Δforgetting | Δmin_old |
|---|---:|---:|---:|---:|---:|
| CSIL | +0.60pp | +3.50pp | +0.49pp | -0.60pp | -0.02pp |
| MoPC-HR | -0.68pp | +25.89pp | +2.13pp | +0.92pp | -0.52pp |

## 解释与声明边界

1. 去掉LEO未使两种方法恢复到良好联合性能，因此不能把旧实验崩塌完全归因于
   LEO信道。
2. CSIL主要受新类注册能力不足及大量官方零步cell影响；正式LEO
   `seen_new=8.65%`、`H=1.18%`。
3. MoPC-HR无LEO时新类提高25.89pp，但旧类下降、遗忘增加；信道影响主要体现在
   新类可分性，未解决旧新类平衡。
4. 正式结果只能称为
   `official-github-execution-aligned with ADV3B02 and CVS data-interface adaptations`。
   ADV3B02替换原编码器、CSIL新10/20初始化和MoPC-HR新2/20日程均已显式标注；
   不能称为原论文网络逐结构复现。

完整根报告：
`E:\type10-7\automation_reports\CV-SincNet\adv3b02_officialrepo_csil_mopc_20260723_v1\report.md`

完整证据归档SHA256：
`245631566ca55d534b66eaffaf2de2130e147ef18baf89a0a75ca1ebdaa95767`。
