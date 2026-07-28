# CVS-RFFI全量消融需求追踪

源文档：`paper/ieee_transactions_draft_20260727/experiments/CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md`

状态：`PHASE1_T1_RELEASE_PREFLIGHT_IN_PROGRESS`

|ID|源章节|需求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|G-01|§1|每个arm绑定唯一ID、Git、bundle/capsule/split/support/query/scenario hash和seed|待映射|pending|待验证|无绑定结果仅作历史线索|
|G-02|§2|Phase1/Stage2-A/B/C权限与`p2_min_v1`强制执行|待映射|pending|协议负测试|query/clean/source必须不可达|
|G-03|§2.2|匹配`VALIDATED_ONCE`句柄时跨方法复用数据|待映射|pending|paired manifest测试|方法变化不得重验数据|
|G-04|§4|Phase1/Phase2公共因子、paired随机性与fresh seed/draw注册|`configs/full_ablation_20260728/seed_registry.json`;`code/cvsrffi/full_ablation_spec.py`|verified|精确历史搜索；7项规范测试|历史713101–713106拒绝；screen/confirm不重合|
|G-05|§5|完整指标、同row语义、资源口径和分层统计|待映射|pending|schema与统计测试|不得拼接不同arm极值|
|P1-REF|§3.1|实现并重训当前划分的`P1-FULL`|待映射|pending|5-seed训练与artifact|历史ADV3B02不可替代|
|P1-T1|§6.1|执行`P1-FULL/SUP/A0/B0/C0/D0`|`code/cvsrffi/full_ablation_spec.py`;`code/cvsrffi/phase1_ablation_factory.py`;`code/SSDG/train_ssdg.py`|partial|六arm执行器与30-row矩阵已验证；服务器训练待完成|A0参数量已严格匹配|
|P1-A|§6.2|实现并按漏斗执行A1–A13|待映射|pending|arm diff/指标/资源|含DAC/RCN/GRL等独立开关|
|P1-B|§6.3|实现并按漏斗执行B1–B14|待映射|pending|matched coverage等|B1必须等coverage|
|P1-C|§6.4|实现并按漏斗执行C1–C14|待映射|pending|累加链与内部开关|C1→C2→FULL|
|P1-D|§6.5|实现并按漏斗执行D1–D13和H0–H6|待映射|pending|clean/stress同row|Phase2信道拆解需独立capsule|
|P1-E|§6.6|实现并按漏斗执行E0–E6|待映射|pending|稳定性/数值测试|AMP为诊断|
|P1-LABEL|§6.7|执行标签率0.005/0.01/0.02/0.05/0.10|待映射|pending|曲线与样本计数|关键点5个确认seed|
|P1-BASE|§6.8|执行同权限Phase1基线|待映射|pending|同split/seed/预算|历史异口径结果不入主表|
|P2-REF|§3.2|实现`P2-FULL`联合特征、地面谱、稳健中心、D92、双几何、Fisher、安全门和量化|待映射|pending|真实row executor|K≤2精确回退|
|P2-S2AB|§7.1|独立执行Stage2-A和Stage2-B表|待映射|pending|旧类same-row证据|S2B不借用新类指标|
|P2-BASE|§7.2|执行7个同权限Stage2-C基线|`code/cvsrffi/full_ablation_spec.py`|pending|7个逻辑ID已锁定；executor待实现|外部方法分表|
|P2-A|§7.3|实现并按漏斗执行A0–A8|待映射|pending|feature/normalization测试|A4只在development选一次|
|P2-B|§7.4|实现并按漏斗执行B0–B11|待映射|pending|K阈值/量化谱测试|B10需独立预封存bundle|
|P2-C|§7.5|实现并按漏斗执行C0–C10|待映射|pending|协方差/任务均衡测试|不可按slice选task weight|
|P2-D|§7.6|实现并按漏斗执行D0–D8|待映射|pending|LOO/尺度/融合测试|in-sample仅诊断|
|P2-E|§7.7|实现并按漏斗执行E0–E9|待映射|pending|Pareto/atomic gate测试|非安全arm标NON_DEPLOYABLE|
|P2-F|§7.8|实现并按漏斗执行F0–F8|待映射|pending|state/logit/latency测试|无整数kernel不得声称加速|
|P2-K|§7.9|K1/K2精确闭合，K5/K10完整激活|待映射|pending|逐logit/预测闭合|闭合失败先修实现|
|P2-G|§7.10|执行一次/连续注册、顺序、持久状态和原子rollback|待映射|pending|session与失败注入|顺序预登记|
|P2-R|§7.11|执行分层场景及独立诊断压力|待映射|pending|独立manifest/capsule|R5/R6不冒充安全认证|
|JOINT-01|§8.1|执行Phase1×Phase2最小2×2|待映射|pending|四cell同row统计|bundle必须来自对应P1模型|
|JOINT-02|§8.2|A0/B0/C0/D0 bundle下游传递|待映射|pending|固定P2-FULL screening|不替代Phase1主消融|
|JOINT-03|§8.3|Phase2核心消融固定fresh P1-FULL bundle|待映射|pending|bundle hash一致|不同arm不得换checkpoint|
|T0-01|§9.1|arm单因素配置diff测试|`code/cvsrffi/phase1_ablation_factory.py`|verified|六arm唯一hash、差分与真实解析器测试通过|Phase1 T1已闭合；内部T2仍待实现|
|T0-02|§9.1|参数量匹配测试|`code/model_dual_cvsincnet.py`|verified|8域fixture均1,061,334；真实14域均1,062,306且梯度可达|正式绝对值以row resource summary为准；Conv-A1仍属T2|
|T0-03|§9.1|query不可达、全类逐样本argmax、scorer分离|待映射|pending|协议负测试|发布硬门槛|
|T0-04|§9.1|K1/K2精确fallback闭合|待映射|pending|逐logit/预测测试|发布硬门槛|
|T0-05|§9.1|量化state无FP32 sidecar|待映射|pending|artifact审计|发布硬门槛|
|T0-06|§9.1|同capsule/support/seed paired manifest|待映射|pending|manifest测试|发布硬门槛|
|ART-01|§11|保存完整run/identity/data/prediction/score/resource/exit artifact|`code/cvsrffi/full_ablation_spec.py`;`code/SSDG/train_ssdg.py`|partial|Phase1 split/terminal/completion收据与负测试通过；真实run artifact待验证|Phase2 prediction/score闭环仍待实现|
|RUN-01|§9/§13|统一arm factory、矩阵executor、8GPU×2调度和不可覆盖输出|`code/cvsrffi/full_ablation_spec.py`;`code/scripts/build_full_ablation_plan.py`;`code/scripts/run_full_ablation_phase1_t1.py`|partial|Phase1 30-row/16槽真实执行器和完成收据验证通过；N607待发布|已有任务计入上限；Phase2执行器待实现|
|STAT-01|§5.3/§10|paired CI、分层bootstrap、Holm、论文表图数据|待映射|pending|统计单测与fixture|失败row不得删除|
|REVIEW-01|§13|独立审查达到`P0=0,P1=0`后方可发布|提交`67b41f6d5eecfa90aaf134c891bd43f2a4793997`|verified|`APPROVE_LOCAL_VERIFIED`|仅Phase1-T1代码就绪，不是性能或landed结论|

## 统计

- verified：3
- partial：3
- deferred：0
- rejected：0
- blocked：0
- pending：33

## 最高风险待确认项

`REVIEW-01`：Phase1-T1代码已达到`P0=0、P1=0`。metadata-only新提交须确认相对`67b41f6d`无代码漂移，正式启动仍依赖WiSig SHA、环境、占用、远端checkout和sealed plan。
