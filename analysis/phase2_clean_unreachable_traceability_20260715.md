# Phase2 clean物理不可达协议强化追踪

日期：2026-07-15

目标：把“整个Phase2链路完全无法访问clean”固化为CVS项目最高优先级数据可达性约束，并同步根目录权威协议与Git承载协议。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| PCU-01 | 用户明确要求 | 强调Phase2不是“不使用clean做最终指标”，而是从数据加载到最终决策的全链路对clean物理不可达 | `E:\type10-7\项目.md` | verified | UTF-8读取通过；强约束原文与必需字段检查通过 | 根目录为非Git工作区，已同步Git镜像 |
| PCU-02 | 用户明确要求 | 明确禁止clean样本及clean-derived feature/logit/prototype/teacher/loss/cache/sidecar/selection signal | `E:\type10-7\项目.md`、`docs/PROJECT_PROTOCOL.md` | verified | 两个文件逐项关键词审计通过 | 任何间接决策影响也属于访问 |
| PCU-03 | 项目Phase1/Phase2边界 | 只允许Phase2开始前冻结登记的Phase1 checkpoint作为不可变入口，禁止Phase2读取其clean训练数据或clean辅助artifact | `E:\type10-7\项目.md`、`docs/PROJECT_PROTOCOL.md` | verified | `sealed_phase1_checkpoint_only`及边界段落均存在 | 避免把Phase1合法训练与Phase2运行期clean访问混为一谈 |
| PCU-04 | fail-closed要求 | 增加显式manifest字段、运行前可达性检查和缺证据即阻断规则 | `E:\type10-7\项目.md`、`docs/PROJECT_PROTOCOL.md` | verified | 7个必需字段及`fail-closed`检查通过 | 声明`uses_target_clean=false`不足以证明全链路不可达 |
| PCU-05 | Git与Markdown同步 | Git镜像、追踪记录完成验证并提交 | `docs/PROJECT_PROTOCOL.md`、本文件 | verified | `git diff --check`通过；已进入本协议提交 | 未覆盖现有无关工作树改动 |

## 验证记录

- UTF-8与关键词验证：使用`ssr-gpu`环境Python读取两个协议文件，检查强约束原文、7个必需字段、`fail-closed`和历史artifact封存语义，结果均为PASS。
- Git差异检查：`git diff --check -- docs/PROJECT_PROTOCOL.md analysis/phase2_clean_unreachable_traceability_20260715.md`通过。
- 反向审计：PCU-01至PCU-05均已verified，无deferred、rejected或blocked项。
