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

## 遗漏风险

- 只把clean降级为control，仍允许Phase2代码实际读取clean样本。
- 只检查`target_channel_view`包含`LEO`，却不证明support/query已经叠加`leo_*_weak`。
- 只改协议文档，不更新matrix generator与validator，旧row仍可被放行。
- 直接覆盖Git镜像中的较旧prompt/state，误伤未归属差异；应采用定点补丁。

## 验证记录

- 根目录定向验收：7项通过。
- Git镜像定向验收：5项通过。
- `tools/optimizer_validate_matrix.py`与`tools/spaceborne_fewshot_da_matrix.py`在`ssr-gpu`中`py_compile`通过。
- 根目录与Git镜像`stage2_optimizer_state.json`均通过JSON解析。
- 扩展历史相关套件结果为98项通过、11项失败；剩余失败位于既有OPGAC断言、旧H06 bundle、idle state和旧launcher文本假设，不属于P2-LEO-01至P2-LEO-06验收项，未以本次协议改动顺带修改。

## 反向审计结论

P2-LEO-01至P2-LEO-06均已落实并有定向验证。实现为严格协议一致性，不是近似实现。当前最高风险是历史Phase2 matrix/artifact缺少新字段或曾使用旧场景；这些历史项会按预期失去launchable资格，必须重新生成或标记为`PROTOCOL_INVALID_FOR_PHASE2`。
