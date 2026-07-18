# Stage2 N607指向与floor通用性修订追踪（2026-07-18）

| ID | 来源要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|
| R1 | 在项目中加入N607，并明确其项目角色 | 工作区`项目.md`、Git`项目.md`、`docs/PROJECT_PROTOCOL.md` | verified | `N607`角色与“非协议例外”关键词检查通过 | N607是主要实验计算/验证承载面，不是source/target数据域或协议例外 |
| R2 | 项目中加入`PROJECT_PROTOCOL.md`指向 | 工作区`项目.md`、Git`项目.md` | verified | 相对链接存在性检查通过 | Git公开协议正文 |
| R3 | 项目中加入`PHASE2_DATA_VALIDATION_APPENDIX.md`指向 | 工作区`项目.md`、Git`项目.md`、`docs/PROJECT_PROTOCOL.md` | verified | 相对链接存在性检查通过 | 一次性builder/validator实现附录 |
| R4 | floor优化必须具有类别无关的通用性 | `docs/STAGE2_METHOD_RESEARCH_GOAL.md`、历史普查边界说明 | verified | 目标中无历史类ID；共享规则与禁止项检查通过 | 历史难类只作诊断，不得触发类ID白名单、专属权重或专属超参数 |

验收计数：4项verified，0项deferred，0项rejected，0项blocked。修改为严格语义落地，不是近似实现；未启动或访问N607，因为本次只修改项目职责、文档指向和算法目标边界。
