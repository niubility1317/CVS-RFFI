# D7b注册类局部对比边界head追踪

日期：2026-07-17
状态：实现与support-only验证完成；无独立query性能声明
声明边界：固定已提取representation；不修改IQ operator；不提交Git。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D7B-01 | 用户要求 | 每类只由support prototype Gram选择最近1至少数局部碰撞类 | `code/cvsrffi/stage2_local_contrastive_boundary.py` | verified | `test_nearest_rival_comes_only_from_support_prototype_gram` | 不构建query图 |
| D7B-02 | 用户公式 | `s_c=sim(q,p_c)+beta_c*(sim(q,p_c)-max_j sim(q,p_j))`角色对称逐样本修正 | 同上 | verified | 手算score与batch-local测试通过 | 全注册类同一规则 |
| D7B-03 | 用户要求 | beta固定小集合，仅由物理support leave-two-out逐类非退化与floor选择 | 同上、测试 | verified | before全类LTO；after新增类LTO+旧类侵入门 | K10时leave-two |
| D7B-04 | 用户要求 | after严格锁定before旧类prototype/rival/beta；新增类只注册自身局部边界 | 同上 | verified | 单测与三场景head状态SHA/bitwise检查 | 旧类score状态不漂移 |
| D7B-05 | `项目.md`7.1.1/7.2 | 单物理样本单LEO观测、view不增加K；无query角色/配额/标签/全局分配接口 | 同上、测试 | verified | sealed enrollment协议字段、接口签名、物理ID测试 | 输入仅固定representation与support lineage |
| D7B-06 | 星上资源 | 0训练参数、0epoch、局部prototype邻接、状态<=256KB、无dense query图 | 同上 | verified | after 11类：6380B、3212MAC/query、0B dense图 | 每类1 rival |
| D7B-07 | 用户要求 | 当前cache无未评分query时只报告support选择，不声明独立性能 | support-only报告 | verified | `query_package_opened=false`、COMMIT SHA256固定 | 不读取历史query score |

最高风险：旧类rival/beta严格冻结意味着新增类不能进入旧类局部修正项；这保证旧状态不变，但不能单独阻断新类logit侵入旧类，仍需依靠新类自身beta或后续独立注册保护机制。

## 验证与产物

- 单测：`conda run -n ssr-gpu python -m pytest tests\test_stage2_local_contrastive_boundary.py -q`，`7 passed`。
- support-only审计：`E:\type10-7\automation_reports\CV-SincNet\d7b_local_contrastive_boundary_20260717\support_only_current_row_audit\report.md`。
- 不可变提交面：`COMMIT.json` SHA256=`c0c3b34ce4fc9b791856248f98199a352fbb934c39ab486e36b9e62b7d68041d`。
- 场景beta：clear旧6类与新5类均0.20；low-elev旧6类0、新5类0.10；rain旧6类与新5类均0.10。
- support floor仍低：after clear/low-elev/rain分别0.30/0.10/0.00，因此本轮只证明机制与锁定正确，不支持推广。
