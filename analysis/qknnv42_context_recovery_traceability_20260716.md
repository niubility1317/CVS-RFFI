# qKNNV42对话上下文恢复追踪表

## 范围

本表只恢复当前主任务`019f5fe9-b4ed-7c00-b935-91eb4657c1fc`中仍可由原始session、项目报告和Git证据交叉验证的用户引导、turn总结与结果边界。已删除会话`019f6610-86af-7572-b857-2544e7b598ba`及其strict300、EvidenceNorm、JP-R4影响不得恢复为当前证据。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|CTX-01|当前主任务原始session JSONL|恢复2026-07-14至2026-07-15的用户直接引导|`automation_reports/CV-SincNet/qknnv42_context_recovery_20260716/report.md`|verified|主session直接user message逐条提取；报告G01至G23计数为23|未恢复重复internal goal文本|
|CTX-02|原始session缺失turn总结|恢复关键层＋K-shot总结及其同row指标|同上|verified|交叉核对session第27290至27291行与v21/v22/v23报告|已把实测与计划拆开|
|CTX-03|v21适配结果|确认`JG_R8_LR020=88.8354%`及资源、最低类、对照|同上|verified|核对优化报告17.8节、result SHA和adapter SHA|明确为source-only|
|CTX-04|v22/v23 K-shot结果|恢复K1/K5/K10/K20收益与K1负迁移结论|同上|verified|核对优化报告18.2至18.4节|明确K1相对direct为正不等于梯度适配为正|
|CTX-05|当前`项目.md`与active goal|恢复当前最终目标、LEO_weak-only、无Oracle、适配与注册同等重要|同上|verified|完整重读`AGENTS.md`与695行`项目.md`；读取active goal|以当前92/88/92/90/86门槛为准|
|CTX-06|会话污染清理报告|隔离已删除会话及strict300/EvidenceNorm/JP-R4影响|同上|verified|核对清理报告、Git状态和重建后的978条会话索引|已删除内容仅列隔离状态|
|CTX-07|Codex界面恢复边界|说明可恢复证据与无法原位重插聊天气泡的边界|同上|verified|主session存在；污染session的DB、JSONL、可视化与索引记录均为0|Git报告＋当前回复替代界面原位气泡|

## 反向审计结果

- 7项全部`verified`，`deferred=0`、`rejected=0`、`blocked=0`。
- 23条直接用户引导全部在恢复报告中有落点。
- 88.8354%已与`source receiver、K=10、6个source类`绑定。
- 已明确当前没有合法target Stage2-C新类注册实测。
- 已明确K1 target梯度相对P4 identity为负，不能写成K1适配成功。
- 已明确strict300、EvidenceNorm、JP-R4不再参与最强版本判断。

## 验证命令

```powershell
python tools/conversation_index.py build
python tools/conversation_index.py search "88.8354"
rg -n "JG_R8_LR020|88\.8354|K1|BPJG|LOPO" automation_reports/CV-SincNet/qknnv42_extreme_light_optimization_20260715/report.md
git diff --check -- analysis/qknnv42_context_recovery_traceability_20260716.md automation_reports/CV-SincNet/qknnv42_context_recovery_20260716/report.md
```

另以PowerShell断言报告和追踪表存在、G01至G23共23项、CTX-01至CTX-07共7项、88.8354%、source-only、新类未实测、K1负迁移和污染隔离字段全部存在，结果均为`True`。

证据内容恢复为严格一致；聊天界面原位气泡恢复不可用，因此可见层采用Git报告和当前回复替代，属于界面承载近似，不是证据近似。
