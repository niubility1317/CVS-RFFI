# Stage2协议与目标重构追踪表（2026-07-18）

## 范围

本次只完成四项工作：

1. 普查以Phase2域适应/新类注册方法研发为目标的历史目标模式对话，并区分方法研发、验证支线和协议/报告审计。
2. 把Phase2数据协议压缩为可一次校验、跨方法复用的最小版本，同时保留核心数据边界。
3. 从`项目.md`迁出活动性能目标、当前候选路线、资源预算、实验矩阵和工程控制规则，使其只描述项目场景、数据与阶段权限。
4. 汇总历次目标，形成独立、可测量、可直接执行的Stage2方法研发目标。

不修改实验代码，不启动N607，不删除或覆盖历史数据、日志、checkpoint与未归属工作树修改。

## 需求到证据映射

| ID | 用户要求 | 落地文件 | 验证方式 | 状态 |
|---|---|---|---|---|
| R1 | 全面搜索最初qKNN至当前所有Phase2方法研发目标模式对话 | `analysis/stage2_method_goal_history_census_20260718.md` | 会话索引刷新、原始rollout抽查、报告反查、排除表 | 已验证 |
| R2 | Phase2最小数据协议；每个clean物理样本只生成一个随机LEO弱信道观测 | `项目.md`、`docs/PROJECT_PROTOCOL.md`、`docs/PHASE2_DATA_VALIDATION_APPENDIX.md` | 核心约束关键词审计、目标/方法耦合反向检查 | 已验证 |
| R3 | `项目.md`只保留项目场景和数据协议，不混入活动目标 | `项目.md` | 禁止词和章节职责审计 | 已验证 |
| R4 | 综合历次目标，重写独立研发目标 | `docs/STAGE2_METHOD_RESEARCH_GOAL.md` | 指标、矩阵、资源、证据和研发优先级完整性检查 | 已验证 |
| R5 | 多个子agent协作并设置监督agent | 本追踪表“协作审计”及监督结论 | 子agent产出与监督二次验收 | 已验证 |

## Phase2最小协议的不可删核心

| ID | 核心约束 | 验收口径 |
|---|---|---|
| P1 | target/source接收机域不相交，旧类/新类集合不相交 | `R_t∩R_s=∅`、`Y_old∩Y_new=∅` |
| P2 | 单物理样本单LEO接收观测 | 每个稳定物理ID只绑定一个允许的`leo_*_weak`场景、一个随机信道实现和一份固定接收IQ |
| P3 | K-shot由K个独立物理样本组成 | 计算view不增加K；跨场景物理ID互斥 |
| P4 | support/query互斥且query只测试 | query及其派生计算不得更新任何状态或参与选择 |
| P5 | Phase2禁止clean/source运行时访问 | 唯一例外为与Phase1 checkpoint共同封存、只读的int8多样本聚合知识 |
| P6 | 逐query面对全部已注册类 | 禁止query真值、角色、真实批次类数、类别配额和跨query全局重排 |
| P7 | 数据一次验证、跨方法复用 | 方法、adapter、超参数、epoch、prototype规则或method lock变化不使数据capsule失效 |

## 协作审计

- 历史普查agent：负责会话索引、原始rollout和报告三方反查。
- 协议瘦身agent：负责重复项、过度准入根因和最小协议字段分析。
- 监督agent：独立制定验收清单，并在草稿完成后进行第二轮逆向审计。
- 主agent：负责合并、文件修改、验证、Git提交和最终交付。

## 最终验收记录

- 历史普查刷新997条记录，覆盖最初qknn8、qKNNV92、Stage2-B方法前史、两次正式active goal、JG支线、目标增量会话和D1-D36，并单列非研发排除项。
- Phase2运行时正文从860行压缩到158行；Git公开正文为90行。旧协议中data capsule与method lock/checkpoint状态的耦合已删除。
- 数据验证拆成独立`capsule_id/split_id`与`bundle_id`；更换方法或bundle不重验固定IQ数据。
- 根`项目.md`未保留性能阈值、125矩阵、80k/30epoch、D版本、N607/Git流程或当前候选路线；这些内容进入独立目标、实现附录或`AGENTS.md`。
- 监督agent完成两轮逆向审计：T1历史普查PASS、T2最小协议PASS、T3项目文档职责PASS、T4独立目标PASS。
- 验证计数：四项主任务及协作监督共5项已验证；0项延期、0项拒绝、0项阻塞。
