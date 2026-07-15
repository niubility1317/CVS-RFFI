# Phase2 LEO_weak-only数据可达性变更追踪

日期：2026-07-15

目标：把“Phase2所有使用的数据样本均已叠加`LEO_weak`信道，Phase2接触不到clean样本”落实为协议、控制面、矩阵字段、机械验证和Git镜像中的一致硬约束。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| P2-LEO-01 | 用户指令；根目录`项目.md`第7、8.5、9节 | Phase2全部support/query、target-old、target-new及可选unknown样本只能使用已叠加`leo_*_weak`的目标域视图 | 根目录`项目.md`；`docs/PROJECT_PROTOCOL.md` | verified | 项目文档token测试通过；反向审计通过 | clean不得作为Phase2 control/reference输入 |
| P2-LEO-02 | `AGENTS.md`自动化禁止项 | 冲突row不得生成、放行、运行、晋升或形成正式声明 | 根目录及Git镜像`AGENTS.md` | verified | prompt/contract/项目文档token测试通过 | 历史artifact只可封存为协议无效证据 |
| P2-LEO-03 | 控制面所有权与Phase2 sample protocol | prompt、contract、manifest、state使用同一机器可读策略 | active prompt；control manifest；workflow contract；optimizer state | verified | 根目录与Git镜像state JSON解析通过；共享控制文件hash一致 | 新策略值为`leo_weak_only_no_clean_access` |
| P2-LEO-04 | Phase2矩阵生成 | 每个Phase2 row显式写入LEO_weak-only与no-clean-access字段 | `tools/spaceborne_fewshot_da_matrix.py` | verified | generator smoke与OA-MSE payload测试通过 | 已把旧target/eval legacy场景改为三个`leo_*_weak`场景 |
| P2-LEO-05 | Phase2矩阵验证 | 缺字段、允许clean访问、目标视图含clean或不属于`leo_*_weak`族时硬失败 | `tools/optimizer_validate_matrix.py` | verified | 两项新增negative test、合法64-row test与py_compile通过 | 同时检查row字段和命令中的clean/legacy场景token |
| P2-LEO-06 | 版本管理 | 根目录非有效Git工作树的改动同步到约定Git承载面并仅提交本次目标文件 | `github_publish/CVS-RFFI-repo` | verified | `git diff --check`通过；本追踪记录与目标文件一并提交 | 保留现有无关dirty worktree |
| P2-ORACLE-01 | 2026-07-15用户追加指令；根目录`项目.md`第7.2节 | Phase2禁止使用query真实old/new/unknown角色Oracle | 根目录`项目.md`；根目录及Git镜像`AGENTS.md`；`docs/PROJECT_PROTOCOL.md` | verified | 项目文档与contract token测试通过；反向审计通过 | support注册角色与标签仍合法 |
| P2-ORACLE-02 | 2026-07-15用户追加指令；根目录`项目.md`第7.2节 | Phase2禁止读取query批次类别数量、每类quota或以此做全局分配 | prompt；contract；manifest；optimizer state | verified | 根目录与Git镜像state JSON解析通过；contract/prompt文本断言通过 | 预注册support K-shot不是query类别配额 |
| P2-ORACLE-03 | Phase2 candidate schema | 每个launchable row显式声明逐样本、全注册类、无角色/配额Oracle | `tools/spaceborne_fewshot_da_matrix.py` | verified | OA-MSE 64-row payload聚焦测试通过 | 五个机器可读字段同时存在于row与top-level protocol |
| P2-ORACLE-04 | Phase2 matrix validator | 缺少guard、guard为true、命令含Oracle/quota/Hungarian/OT/batch assignment时硬失败 | `tools/optimizer_validate_matrix.py` | verified | 缺字段、true guard、row/parameters别名、CLI启用及显式false正负测试通过 | 历史artifact只封存，不得生成新Oracle诊断 |
| P2-ORACLE-05 | 项目约定交付 | 相关约定清单明确禁止项与允许的support K-shot边界 | `docs/CVS_PROJECT_CONVENTIONS_AND_DATA_PROTOCOL_20260715.md` | verified | 根目录与Git镜像文本检查通过 | 未误伤Phase1 source quota、support采样或metric-only标签读取 |
| P2-ORACLE-06 | 版本管理 | 根目录改动同步到Git承载面并仅提交本次目标文件 | `github_publish/CVS-RFFI-repo` | verified | `git diff --check`通过；定向stage/commit | 保留现有无关dirty worktree |

## 遗漏风险

- 只把clean降级为control，仍允许Phase2代码实际读取clean样本。
- 只检查`target_channel_view`包含`LEO`，却不证明support/query已经叠加`leo_*_weak`。
- 只改协议文档，不更新matrix generator与validator，旧row仍可被放行。
- 直接覆盖Git镜像中的较旧prompt/state，误伤未归属差异；应采用定点补丁。
- 只禁用显式`role_oracle`字段，却漏掉`parameters`别名或命令行中的class quota/Hungarian/OT/global assignment开关。
- 把合法的support K-shot类别构成误判为query类别配额，导致Stage2-B/C无法按协议注册支持集。

## 验证记录

- 根目录定向验收：7项通过。
- Git镜像定向验收：5项通过。
- `tools/optimizer_validate_matrix.py`与`tools/spaceborne_fewshot_da_matrix.py`在`ssr-gpu`中`py_compile`通过。
- 根目录与Git镜像`stage2_optimizer_state.json`均通过JSON解析。
- P2-ORACLE定向验收：根目录7项通过，Git镜像7项通过；覆盖规范字段、row/parameters别名、命令行启用、显式false与合法support K-shot。
- 扩展历史相关套件结果为98项通过、11项失败；剩余失败位于既有OPGAC断言、旧H06 bundle、idle state和旧launcher文本假设，不属于P2-LEO-01至P2-LEO-06验收项，未以本次协议改动顺带修改。

## 反向审计结论

P2-LEO-01至P2-LEO-06以及P2-ORACLE-01至P2-ORACLE-06均已落实并有定向验证。共12项verified，0项deferred，0项rejected，0项blocked。实现为严格协议一致性，不是近似实现。当前最高风险是历史Phase2 matrix/artifact缺少新字段、曾使用旧场景，或曾依赖role/class-quota Oracle；这些历史项会按预期失去launchable资格，必须重新生成，或分别标记为`PROTOCOL_INVALID_FOR_PHASE2`、`PROTOCOL_INVALID_FOR_DEPLOYMENT/ROLE_OR_CLASS_QUOTA_ORACLE`。
