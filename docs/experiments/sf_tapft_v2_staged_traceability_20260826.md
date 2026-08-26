# SF-TAPFT V2高性能上界与逐级瘦身追踪

## 当前边界

用户在本对话中放宽Phase2原型、模型层、target head、Adapter、参数tensor和bundle的修改限制，并取消低于1%可训练参数约束。数据权限、support/query隔离、query只读、truth-last和不可覆盖输出仍保持不变。该任务级授权不修改`项目.md`。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|V2-01|授权边界|允许修改原型、模型、head、Adapter和tensor|设计、配置、runner|verified|用户当前对话明确授权|仅限本路线模型能力|
|V2-02|正确性边界|source/query/truth隔离和`p2_min_v1`绑定不变|adapt、runner、prediction测试|verified|55项聚焦回归通过；prediction签名无truth/role输入|真实query未打开|
|V2-02A|Phase1 bundle绑定|正式bundle固定checkpoint lineage、ordered class registry和int8 component；target样本来自已验证capsule|`sf_tapft_phase1_binding.py`、runner、严格loader、测试|implemented_artifact_blocked|lineage/class/capsule/count错配负测通过|N607无CORE90同谱系正式deployment bundle；旧int8 manifest明确非正式|
|V2-03|R0 averaging|只保存/平均许可trainable delta与target head|`target_only_progressive_adapt.py`、测试|verified|anchor＋float64 mean(delta) fixture；非许可state逐tensor相等|禁止完整floating state averaging|
|V2-04|R0 final refit|OOF选择后从原checkpoint对60条support重拟合|adapt、runner、测试|verified|fresh checkpoint、完整support、`fixed_final_step`测试通过|不得返回fold0|
|V2-05|R0指标|阶段级BA、macro-F1、floor、NLL、recall、margin、flip和移动量|adapt、runner、receipt|verified|含缺类记0的手算fixture通过|A/B/C分别记录|
|V2-06|R0 group|真实group优先；缺失时明确row-stratified|runner、receipt|verified|group/row两条路径测试通过|不伪造物理group|
|V2-07|R0 bundle|新增clean single v2 schema并严格回读|runner、测试|verified|V2严格回读与外部可信target binding篡改负测通过|V1精确allowlist保持不变|
|V2-08|R0 query接口|实现只读prediction接口，不在R0运行真实query|`sf_tapft_prediction.py`、测试|verified|真实模型/head＋合成IQ接口通过；single=batch、重排等价、state不变|实际query推迟到R3|
|V2-09|R0本地验证|聚焦负测、回归、真实checkpoint no-query smoke|测试、报告|blocked_before_remote_smoke|本地55项通过；N607 preflight通过；同谱系正式bundle缺失|不得用异谱系bundle或旧非正式int8组件替代|
|V2-10|R1 P0|head only贡献|配置、OOF runner|deferred|同split指标|等待R0分析|
|V2-11|R1 P1|P0+time norm贡献|配置、OOF runner|deferred|同split指标|不得跳过P0|
|V2-12|R1 P2|P1+time Adapter贡献|配置、OOF runner|deferred|同split指标|不得跳过P1|
|V2-13|R1 P3|P2+完整`t3`贡献|配置、OOF runner|deferred|同split指标|不得跳过P2|
|V2-14|R1 P4|P3+`t2.pw/time_fuse/identity fusion`|adapt、配置、测试|deferred|同split指标|不得同时开频域|
|V2-15|R2 teacher|保存4-fold logit ensemble|ensemble bundle、loader、测试|deferred|logit均值fixture|等待R1结构选择|
|V2-16|R2 strong single|保存全support单模型|single bundle、loader、测试|deferred|全60条refit|不参数平均fold模型|
|V2-17|R3 prediction|DA0/teacher/single统一truth-blind prediction|prediction runner|deferred|prediction完整性|首次真实query|
|V2-18|R3 scorer|独立truth-last同row评分|scorer、报告|deferred|truth侧读回|不得反哺模型选择|
|V2-19|S0等价瘦身|向量化、稀疏验证、delta-only等|adapt、runner、bundle|deferred|`max_abs_logit_delta<1e-5`|等待R3性能上界|
|V2-20|S1单模型|取消4模型部署|部署配置|deferred|teacher/single差值|teacher保留研发参照|
|V2-21|S2阶段删除|删除或条件化C|配置、OOF|deferred|BA/floor/NLL门槛|一次只改阶段|
|V2-22|S3 t3增量|dw→pw LoRA→组合|模型、配置|deferred|BA/floor/NLL门槛|不得跳级|
|V2-23|S4 rank压缩|R32→R16→R8→R4|配置|deferred|逐级非劣|不得同时删除t3|
|V2-24|S5 head压缩|完整head到更轻决策头|模型、配置|deferred|严格非劣|最后执行|

## 当前状态

- 已验证：8项（含授权边界）。
- 已实现但真实工件阻塞：1项（V2-02A）。
- 远端smoke阻塞：1项（V2-09）。
- 延后：15项，按R1→R2→R3→S0→S1–S5顺序解锁。
- 拒绝：0项。
- 阻塞：CORE90同谱系正式Phase1 deployment bundle缺失；不得以checkpoint SHA为`1eb6d07b…307d7`的其他Phase1 bundle替代SHA为`2699eedc…d59c98`的CORE90。
- 当前最高交付状态：`LOCAL_VERIFIED / NO_N607_SMOKE / NO_PERFORMANCE_RESULT`。R1及真实query仍保持延后。
