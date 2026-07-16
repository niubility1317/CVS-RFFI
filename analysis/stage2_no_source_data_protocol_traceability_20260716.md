# Stage2禁止源域数据协议追踪表

目标：将“Stage2运行时不得使用源域数据”写入项目科研协议与实验报告，并重新界定既有MRIOR-SDA、DADDA-SDA结果的证据等级。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|NSD-01|用户最新要求|Stage2-A/B/C运行时禁止读取源域样本及其缓存|`项目.md`|verified|新增§7.3并检查禁用项|已先更新科研源协议|
|NSD-02|用户最新要求|禁止source label、feature、logit、prototype和统计量进入Stage2|`项目.md`|verified|7个source-access字段检查通过|避免以派生artifact绕过限制|
|NSD-03|现有Phase2预训练边界|仅允许封存Phase1 ADV3B02 checkpoint作为源域训练遗产|`项目.md`|verified|预训练artifact边界检查通过|checkpoint不是Stage2源域数据输入|
|NSD-04|版本同步规则|根目录与Git承载面的`项目.md`保持一致|两份`项目.md`|verified|SHA256一致|根目录非Git，已同步到Git承载面|
|NSD-05|用户最新要求|在`项目实验.md`加入正式Stage2无源域数据协议|`项目实验.md`|verified|§1.1、§4文本及字段检查通过|已说明输入、禁用项和阻断规则|
|NSD-06|证据边界|使用source replay的MRIOR-SDA、DADDA-SDA结果降级为历史诊断|`项目实验.md`、周报|verified|状态标记全文检查通过|不再充当当前正式Stage2基线|
|NSD-07|后续实验设计|MRIOR/DADDA若重跑，必须改成target-support-only变体并重新命名|`项目实验.md`、周报|verified|后续矩阵与基线章节检查通过|不得沿用旧结果冒充无源协议|
|NSD-08|反向审计|报告不得继续把sealed source LEO cache写成正式Stage2允许输入|相关Markdown|verified|全文检索仅命中历史诊断说明|历史方法说明保留并明确失效日期|
|NSD-09|执行控制面|validator、bundle与runner在代码层阻断source成员和访问|后续代码修复|deferred|现有runner仍要求source cache|本次用户要求为项目实验文档更新；代码存在并行未提交修改，未在本次覆盖|

## 反向审计结论

文档与科研协议共8项已验证，执行控制面1项延期：`verified=8`、`deferred=1`、`rejected=0`、`blocked=0`。

当前最高风险不是文字遗漏，而是现有`adv3b02_supervised_da_runner.py`仍为MRIOR-SDA/DADDA-SDA打开source cache。新协议生效后，在validator、predictor bundle、runtime allowlist和runner全部阻断source访问之前，不得启动或晋升新的Stage2正式实验。
